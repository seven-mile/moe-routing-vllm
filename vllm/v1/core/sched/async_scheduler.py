# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        hf_config = self.vllm_config.model_config.hf_text_config
        base_top_k = int(
            getattr(hf_config, "num_experts_per_tok", 0)
            or getattr(hf_config, "moe_top_k", 0)
            or 0
        )
        num_moe_layers = (
            int(getattr(hf_config, "num_hidden_layers", 0))
            - int(getattr(hf_config, "first_k_dense_replace", 0))
            if base_top_k > 0
            else 0
        )
        self._spec_token_top_k_placeholders: list[list[int]] = [
            [base_top_k] * num_moe_layers
            for _ in range(self.num_spec_tokens + 1)
        ]

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        spec_decode_token_top_ks = (
            scheduler_output.scheduled_spec_decode_token_top_ks
        )
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate a new token plus num_spec_tokens
            # in this scheduling step.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            if (
                cur_num_spec_tokens
                and self._spec_token_top_k_placeholders
                and req_id not in spec_decode_token_top_ks
            ):
                spec_decode_token_top_ks[req_id] = (
                    self._spec_token_top_k_placeholders[: cur_num_spec_tokens + 1]
                )
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders
            if self._spec_token_top_k_placeholders:
                request.spec_token_top_ks = self._spec_token_top_k_placeholders

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        if request.discard_latest_async_tokens:
            # If the request is force preempted in reset_prefix_cache, we
            # should discard the latest async token.
            request.discard_latest_async_tokens = False
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Update the number of output placeholders.
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
