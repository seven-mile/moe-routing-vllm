# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.spec_decode.fused_kernel import (
    LAYER_RANGE_INACTIVE,
    fused_logits_to_topk,
)
from vllm.v1.spec_decode.utils import calc_distribution_perplexity


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
def test_fused_logits_to_topk_matches_spec_with_list_layer_range() -> None:
    # Load the existing action implementation from workspace root.
    repo_root = Path(__file__).resolve().parents[4]
    cfg_file = repo_root / "configs" / "ppl_to_ks.py"
    spec = importlib.util.spec_from_file_location("ppl_to_ks", cfg_file)
    assert spec is not None and spec.loader is not None
    ppl_to_ks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ppl_to_ks)

    torch.manual_seed(0)
    device = torch.device("cuda")

    batch_size = 4
    spec_len = 7
    vocab_size = 37
    num_layers = 10
    base_k = 8

    logits = torch.randn(batch_size, spec_len, vocab_size, device=device).to(torch.bfloat16)

    # Compact cfg: only first cfg_lens[b] values are valid.
    cfg_values = torch.tensor(
        [
            [6.0, 1.17, 1.07, 1.07, 0.0, 0.0, 0.0, 0.0],
            [6.0, 1.17, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [6.0, 1.17, 1.07, 1.035, 1.005, 1.005, 0.0, 0.0],
            [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    cfg_lens = torch.tensor([4, 2, 6, 1], dtype=torch.int32, device=device)

    # [start, stop, step], with one inactive row and one out-of-range row.
    layer_ranges = torch.tensor(
        [
            [2, 8, 1],
            [-100, 100, 3],
            [LAYER_RANGE_INACTIVE, 0, 1],
            [9, 1, 1],
        ],
        dtype=torch.int32,
        device=device,
    )

    cfg_boundaries = torch.zeros((batch_size, base_k), dtype=torch.float32, device=device)
    layer_mask = torch.zeros((batch_size, num_layers), dtype=torch.bool, device=device)
    for b in range(batch_size):
        cur_len = int(cfg_lens[b].item())
        if cur_len > 0:
            cfg_boundaries[b, base_k - cur_len :] = cfg_values[b, :cur_len].flip(0)

        start, stop, step = (int(x) for x in layer_ranges[b].tolist())
        if start != LAYER_RANGE_INACTIVE and step != 0:
            s, e, st = slice(start, stop, step).indices(num_layers)
            idx = list(range(s, e, st))
            if idx:
                layer_mask[b, torch.tensor(idx, device=device)] = True

    got = fused_logits_to_topk(
        logits=logits,
        cfg_boundaries=cfg_boundaries,
        layer_mask=layer_mask,
        base_k=base_k,
    )

    class DummyConfig:
        num_experts_per_tok = base_k
        num_hidden_layers = num_layers
        first_k_dense_replace = 0

    ppls = calc_distribution_perplexity(logits)
    expected = torch.empty((batch_size, spec_len, num_layers), dtype=torch.int32, device=device)

    for b in range(batch_size):
        cfg = tuple(float(x) for x in cfg_values[b, : cfg_lens[b]].tolist())
        layer_range = tuple(int(x) for x in layer_ranges[b].tolist())
        # spec_with_list_layer_range returns [L, S]
        # The reference action builds bucketize boundaries on CPU,
        # so run the reference path on CPU and copy back.
        ref = ppl_to_ks.spec_with_list_layer_range(
            cfg,
            layer_range,
            ppls[b].cpu(),
            DummyConfig,
        )
        expected[b] = ref.transpose(0, 1).to(device=device, dtype=torch.int32)

    assert torch.equal(got, expected)
