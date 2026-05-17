# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
from vllm.benchmarks.serve import calculate_metrics


class DummyTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        return SimpleNamespace(input_ids=list(text))


def test_calculate_metrics_reports_pure_decode_window():
    input_requests = [
        SampleRequest(prompt="a", prompt_len=5, expected_output_len=4),
        SampleRequest(prompt="b", prompt_len=7, expected_output_len=4),
    ]
    outputs = [
        RequestFuncOutput(
            success=True,
            start_time=1.0,
            ttft=2.0,
            latency=6.0,
            output_tokens=10,
            prompt_len=5,
        ),
        RequestFuncOutput(
            success=True,
            start_time=2.0,
            ttft=2.0,
            latency=5.0,
            output_tokens=20,
            prompt_len=7,
        ),
    ]

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=10.0,
        tokenizer=DummyTokenizer(),
        selected_percentiles=[99.0],
        goodput_config_dict={},
    )

    assert actual_output_lens == [10, 20]
    assert metrics.output_throughput == pytest.approx(3.0)
    assert metrics.pure_decode_duration_s == pytest.approx(3.0)
    assert metrics.pure_decode_output_throughput == pytest.approx(10.0)
    assert metrics.total_token_throughput == pytest.approx(4.2)