import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import promparse  # noqa: E402

TEXT = """
# HELP vllm:time_to_first_token_seconds TTFT
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="m"} 0
vllm:time_to_first_token_seconds_bucket{le="0.5",model_name="m"} 2
vllm:time_to_first_token_seconds_bucket{le="1.0",model_name="m"} 3
vllm:time_to_first_token_seconds_bucket{le="5.0",model_name="m"} 4
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="m"} 4
vllm:time_to_first_token_seconds_sum{model_name="m"} 4.7
vllm:time_to_first_token_seconds_count{model_name="m"} 4
vllm:num_requests_waiting{model_name="m"} 3
vllm:num_requests_running{model_name="m",engine="0"} 2
vllm:num_requests_running{model_name="m",engine="1"} 1
vllm:spec_decode_num_accepted_tokens_total 20
vllm:spec_decode_num_draft_tokens_total 30
"""


def test_parse_and_totals():
    s = promparse.parse(TEXT)
    assert promparse.total(s, "vllm:num_requests_waiting") == 3
    assert promparse.total(s, "vllm:num_requests_running") == 3  # summed over labels
    assert promparse.total(s, "vllm:num_requests_running", engine="1") == 1
    assert promparse.total(s, "nope") is None


def test_histogram_quantiles_interpolate():
    s = promparse.parse(TEXT)
    # 4 observations: p50 -> target 2 = exactly the 0.5 bucket boundary
    assert abs(promparse.histogram_quantile(s, "vllm:time_to_first_token_seconds", 0.5) - 0.5) < 1e-9
    p99 = promparse.histogram_quantile(s, "vllm:time_to_first_token_seconds", 0.99)
    assert 1.0 < p99 <= 5.0
    assert abs(promparse.histogram_mean(s, "vllm:time_to_first_token_seconds") - 4.7 / 4) < 1e-9
    assert promparse.histogram_quantile(s, "missing", 0.5) is None


def test_empty_histogram_is_none():
    s = promparse.parse('h_bucket{le="1"} 0\nh_bucket{le="+Inf"} 0\nh_count 0\nh_sum 0\n')
    assert promparse.histogram_quantile(s, "h", 0.5) is None
