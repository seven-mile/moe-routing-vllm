# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.llm_base_proposer import (
    SpecDecodeBaseProposer,
    compute_probs_and_sample_next_token,
)

DEVICE_TYPE = current_platform.device_type


def _seed_default_generator(seed: int) -> None:
    set_random_seed(seed)


def _make_sampling_metadata(batch_size: int) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.ones(batch_size, dtype=torch.float32, device=DEVICE_TYPE),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.empty(0, device=DEVICE_TYPE),
        presence_penalties=torch.empty(0, device=DEVICE_TYPE),
        repetition_penalties=torch.empty(0, device=DEVICE_TYPE),
        output_token_ids=[[] for _ in range(batch_size)],
        spec_token_ids=[[] for _ in range(batch_size)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def test_compute_probs_and_sample_next_token_uses_fp64_exponential_race():
    batch_size = 4
    vocab_size = 32
    generator = torch.Generator(device=DEVICE_TYPE).manual_seed(11)
    logits = torch.randn(
        batch_size,
        vocab_size,
        dtype=torch.float32,
        device=DEVICE_TYPE,
        generator=generator,
    )
    metadata = _make_sampling_metadata(batch_size)

    _seed_default_generator(12345)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty(probs.shape, dtype=torch.float64, device=probs.device)
    q.exponential_()
    expected_ids = q.reciprocal_().mul_(probs).argmax(dim=-1).view(-1)

    _seed_default_generator(12345)
    actual_ids, actual_probs = compute_probs_and_sample_next_token(
        logits.clone(),
        metadata,
        use_fp64_gumbel=True,
    )

    assert torch.equal(actual_ids, expected_ids)
    assert torch.allclose(actual_probs, probs)


def test_dense_target_uses_zero_topk_plan_without_loading_fused_kernel():
    proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
    proposer.vllm_config = SimpleNamespace(model_config=MagicMock())
    proposer.runner = MagicMock()
    proposer.runner.get_model.return_value = object()

    token_ids = torch.zeros((2, 3), dtype=torch.int64)
    logits = torch.zeros((2, 3, 8))
    with patch(
        "vllm.v1.spec_decode.llm_base_proposer.is_mixture_of_experts",
        return_value=False,
    ):
        token_top_ks = proposer.get_token_top_ks_from_proposals(token_ids, logits)

    assert token_top_ks.shape == (2, 4, 1)
    assert token_top_ks.dtype == torch.int32
    assert torch.count_nonzero(token_top_ks) == 0
    proposer.vllm_config.model_config.get_num_experts_per_token.assert_not_called()
