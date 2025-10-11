# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F

from vllm.sampling_params import SamplingParams

_SAMPLING_EPS = 1e-5


def is_spec_decode_unsupported(sampling_params: SamplingParams) -> bool:
    """True if request is incompatible with speculative decoding"""
    return (sampling_params.frequency_penalty != 0.0
            or sampling_params.presence_penalty != 0.0
            or sampling_params.repetition_penalty != 1.0
            or sampling_params.min_p > _SAMPLING_EPS
            or sampling_params.logprobs is not None)


def calc_perplexity(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    assert logits.shape[:-1] == token_ids.shape, \
        f"Logits shape {logits.shape} does not match token_ids shape {token_ids.shape}"
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), token_ids.reshape(-1), reduction='none')
    perplexity = torch.exp(loss)
    return perplexity.view(token_ids.shape)
