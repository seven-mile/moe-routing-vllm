from __future__ import annotations

import triton
import triton.language as tl
import torch


LAYER_RANGE_INACTIVE = -(2**31)


@triton.jit
def _compute_entropy_from_logits_block(
    logits_ptr,
    b,
    s_offsets,
    mask_s,
    V,
    stride_lb,
    stride_ls,
    stride_lv,
    BLOCK_V: tl.constexpr,
):
    # Online reduction for log-sum-exp and expectation of logits.
    # Entropy H = log(Z) - E[z], where z are logits.
    m = tl.full((s_offsets.shape[0],), float("-inf"), tl.float32)
    sum_exp = tl.zeros((s_offsets.shape[0],), tl.float32)
    sum_exp_z = tl.zeros((s_offsets.shape[0],), tl.float32)

    for v_start in range(0, V, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask_v = v_offsets < V
        logits_ptrs = (
            logits_ptr
            + b * stride_lb
            + s_offsets[:, None] * stride_ls
            + v_offsets[None, :] * stride_lv
        )
        z = tl.load(
            logits_ptrs,
            mask=mask_s[:, None] & mask_v[None, :],
            other=float("-inf"),
        ).to(tl.float32)
        z_safe = tl.where(mask_v[None, :], z, 0.0)

        blk_max = tl.max(z, axis=1)
        new_m = tl.maximum(m, blk_max)

        scale_old = tl.exp(m - new_m)
        exp_shift = tl.exp(z - new_m[:, None])

        sum_exp = sum_exp * scale_old + tl.sum(exp_shift, axis=1)
        sum_exp_z = sum_exp_z * scale_old + tl.sum(exp_shift * z_safe, axis=1)
        m = new_m

    log_z = tl.log(sum_exp) + m
    expected_z = sum_exp_z / sum_exp
    entropy = log_z - expected_z
    return entropy

@triton.jit
def fused_logits_to_topk_kernel(
    logits_ptr,        # [B, S, V]
    cfg_ptr,           # [B, K] ascending boundaries
    layer_mask_ptr,    # [B, L] bool: True means keep base_k for that layer
    out_ptr,           # [B, S, L]

    B, S, V, L, K,
    base_k,

    stride_lb, stride_ls, stride_lv,
    stride_cb, stride_ck,
    stride_mb, stride_ml,
    stride_ob, stride_os, stride_ol,

    BLOCK_S: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)
    s_offsets = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)
    l_offsets = tl.program_id(2) * BLOCK_L + tl.arange(0, BLOCK_L)

    mask_s = s_offsets < S
    mask_l = l_offsets < L

    entropy = _compute_entropy_from_logits_block(
        logits_ptr,
        b,
        s_offsets,
        mask_s,
        V,
        stride_lb,
        stride_ls,
        stride_lv,
        BLOCK_V,
    )
    ppls = tl.exp(entropy)

    k_offsets = tl.arange(0, BLOCK_K)
    cfg = tl.load(
        cfg_ptr + b * stride_cb + k_offsets * stride_ck,
        mask=k_offsets < K,
        other=float("inf"),
    )

    comp = (ppls[:, None] >= cfg[None, :]).to(tl.int32)
    k_basic = tl.sum(comp, axis=1)

    layer_mask = tl.load(
        layer_mask_ptr + b * stride_mb + l_offsets * stride_ml,
        mask=mask_l,
        other=False,
    )

    k_vals = tl.where(layer_mask[None, :], base_k, k_basic[:, None])

    tl.store(
        out_ptr
        + b * stride_ob
        + s_offsets[:, None] * stride_os
        + l_offsets[None, :] * stride_ol,
        k_vals,
        mask=mask_s[:, None] & mask_l[None, :],
    )


def _as_cuda_contiguous(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor, got {t.device}.")
    return t.contiguous()


def _next_pow2(n: int) -> int:
    n = max(1, int(n))
    p = 1
    while p < n:
        p <<= 1
    return p


def fused_logits_to_topk(
    logits: torch.Tensor,
    cfg_boundaries: torch.Tensor,
    layer_mask: torch.Tensor,
    base_k: int,
    *,
    block_s: int = 8,
    block_v: int = 128,
    block_l: int = 8,
) -> torch.Tensor:
    """Compute [B,S,L] token top-k directly from logits in one kernel launch.

    Args:
        logits: [B,S,V], bf16/fp16/fp32.
        cfg_boundaries: [B,K] ascending boundaries for bucketize(right=True).
        layer_mask: [B,L] bool, True means keep base_k on that layer.
        base_k: default k for masked layers.
    """
    logits = _as_cuda_contiguous(logits, "logits")
    cfg_boundaries = _as_cuda_contiguous(
        cfg_boundaries.to(dtype=torch.float32), "cfg_boundaries"
    )
    layer_mask = _as_cuda_contiguous(layer_mask.to(dtype=torch.bool), "layer_mask")

    if logits.ndim != 3:
        raise ValueError(f"logits must be [B,S,V], got {tuple(logits.shape)}")
    if cfg_boundaries.ndim != 2:
        raise ValueError(
            f"cfg_boundaries must be [B,K], got {tuple(cfg_boundaries.shape)}"
        )
    if layer_mask.ndim != 2:
        raise ValueError(f"layer_mask must be [B,L], got {tuple(layer_mask.shape)}")

    B, S, V = logits.shape
    b_cfg, K = cfg_boundaries.shape
    b_mask, L = layer_mask.shape
    if b_cfg != B or b_mask != B:
        raise ValueError(
            f"Batch mismatch: logits={B}, cfg_boundaries={b_cfg}, layer_mask={b_mask}"
        )

    block_k = _next_pow2(K)
    out = torch.empty((B, S, L), device=logits.device, dtype=torch.int32)

    grid = (B, triton.cdiv(S, block_s), triton.cdiv(L, block_l))
    fused_logits_to_topk_kernel[grid](
        logits,
        cfg_boundaries,
        layer_mask,
        out,
        B,
        S,
        V,
        L,
        K,
        int(base_k),
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        cfg_boundaries.stride(0),
        cfg_boundaries.stride(1),
        layer_mask.stride(0),
        layer_mask.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_S=block_s,
        BLOCK_V=block_v,
        BLOCK_K=block_k,
        BLOCK_L=block_l,
    )
    return out


@triton.jit
def fused_logits_to_total_topk_kernel(
    logits_ptr,      # [B, S, V]
    cfg_ptr,         # [B, K]
    layer_mask_ptr,  # [B, L]
    out_ptr,  # [B, S + 1, L]

    B,
    S,
    V,
    L,
    K,
    base_k,

    stride_lb,
    stride_ls,
    stride_lv,
    stride_cb,
    stride_ck,
    stride_mb,
    stride_ml,
    stride_ob,
    stride_os,
    stride_ol,

    PAD_S: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
    APPLY_LAST_TOKEN: tl.constexpr,
):
    b = tl.program_id(0)
    l_offsets = tl.program_id(1) * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = l_offsets < L

    layer_mask = tl.load(
        layer_mask_ptr + b * stride_mb + l_offsets * stride_ml,
        mask=mask_l,
        other=False,
    )

    k_offsets = tl.arange(0, BLOCK_K)
    cfg = tl.load(
        cfg_ptr + b * stride_cb + k_offsets * stride_ck,
        mask=k_offsets < K,
        other=float("inf"),
    )

    s_offsets = tl.arange(0, PAD_S)
    mask_s = s_offsets < S

    entropy = _compute_entropy_from_logits_block(
        logits_ptr,
        b,
        s_offsets,
        mask_s,
        V,
        stride_lb,
        stride_ls,
        stride_lv,
        BLOCK_V,
    )
    ppls = tl.exp(entropy)

    comp = (ppls[:, None] >= cfg[None, :]).to(tl.int32)
    k_basic = tl.sum(comp, axis=1)
    k_basic_masked = tl.where(mask_s, k_basic, 0)

    k_vals = tl.where(layer_mask[None, :], base_k, k_basic[:, None])

    tl.store(
        out_ptr
        + b * stride_ob
        + s_offsets[:, None] * stride_os
        + l_offsets[None, :] * stride_ol,
        k_vals,
        mask=mask_s[:, None] & mask_l[None, :],
    )

    base_vals = tl.full((BLOCK_L,), base_k, tl.int32)
    if APPLY_LAST_TOKEN:
        denom = tl.maximum(S, 1)
        mean_k_basic = (tl.sum(k_basic_masked, axis=0).to(tl.float32) / denom).to(
            tl.int32
        )
        agg_vals = tl.where(layer_mask, base_vals, mean_k_basic)
        last_vals = tl.where(S > 0, agg_vals, base_vals)
    else:
        last_vals = base_vals

    tl.store(
        out_ptr + b * stride_ob + S * stride_os + l_offsets * stride_ol,
        last_vals,
        mask=mask_l,
    )


def fused_logits_to_total_topk(
    logits: torch.Tensor,
    cfg_boundaries: torch.Tensor,
    layer_mask: torch.Tensor,
    base_k: int,
    *,
    apply_last_token: bool,
    pad_s: int | None = None,
    block_v: int = 128,
    block_l: int = 8,
) -> torch.Tensor:
    """Compute [B,S+1,L] top-k tensor used by EAGLE speculative proposals.

    This API uses a single Triton kernel that:
    1) computes per-token ``k_basic`` from proposal logits;
    2) writes token-wise ``[B,S,L]`` values;
    3) computes aggregated ``k_agg`` for the trailing ``S+1`` slot.

    ``apply_last_token`` is compiled as a Triton ``constexpr``.

    Args:
        logits: Proposal logits, shape ``[B,S,V]`` on CUDA.
        cfg_boundaries: Per-request cfg boundaries, shape ``[B,K]`` on CUDA.
        layer_mask: Per-request masked layers, shape ``[B,L]`` on CUDA.
        base_k: Default top-k value.
        apply_last_token: Whether to derive the ``S+1`` slot from the mean of
            ``[B,S,L]``; when ``False`` the final slot stays at ``base_k``.
        pad_s: Triton PAD_S parameter. If None, uses next power-of-two of S.
        block_v: Triton BLOCK_V parameter.
        block_l: Triton BLOCK_L parameter.

    Returns:
        Tensor of shape ``[B,S+1,L]`` with dtype ``torch.int32``.
    """
    logits = _as_cuda_contiguous(logits, "logits")
    cfg_boundaries = _as_cuda_contiguous(
        cfg_boundaries.to(dtype=torch.float32), "cfg_boundaries"
    )
    layer_mask = _as_cuda_contiguous(layer_mask.to(dtype=torch.bool), "layer_mask")

    if logits.ndim != 3:
        raise ValueError(f"logits must be [B,S,V], got {tuple(logits.shape)}")
    if cfg_boundaries.ndim != 2:
        raise ValueError(
            f"cfg_boundaries must be [B,K], got {tuple(cfg_boundaries.shape)}"
        )
    if layer_mask.ndim != 2:
        raise ValueError(f"layer_mask must be [B,L], got {tuple(layer_mask.shape)}")

    B, S, V = logits.shape
    b_cfg, K = cfg_boundaries.shape
    b_mask, L = layer_mask.shape
    if b_cfg != B or b_mask != B:
        raise ValueError(
            f"Batch mismatch: logits={B}, cfg_boundaries={b_cfg}, layer_mask={b_mask}"
        )

    total_topks = torch.empty(
        (B, S + 1, L),
        device=logits.device,
        dtype=torch.int32,
    )

    if pad_s is None:
        pad_s = _next_pow2(S)
    else:
        pad_s = max(int(pad_s), 1)
        if pad_s < S:
            raise ValueError(f"pad_s must be >= S, got pad_s={pad_s}, S={S}")
        if pad_s != _next_pow2(pad_s):
            raise ValueError(f"pad_s must be a power of two, got {pad_s}")

    block_k = _next_pow2(K)
    grid = (B, triton.cdiv(L, block_l))
    fused_logits_to_total_topk_kernel[grid](
        logits,
        cfg_boundaries,
        layer_mask,
        total_topks,
        B,
        S,
        V,
        L,
        K,
        int(base_k),
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        cfg_boundaries.stride(0),
        cfg_boundaries.stride(1),
        layer_mask.stride(0),
        layer_mask.stride(1),
        total_topks.stride(0),
        total_topks.stride(1),
        total_topks.stride(2),
        PAD_S=pad_s,
        BLOCK_V=block_v,
        BLOCK_K=block_k,
        BLOCK_L=block_l,
        APPLY_LAST_TOKEN=apply_last_token,
    )
    return total_topks.contiguous()
