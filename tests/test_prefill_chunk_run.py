# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""tt.prefill_chunk_run = K: a lone mid-prompt request gets K chunks per prefill step.

The run multiplies the step's token budget only when the step would hold
exactly one request that is already past its first chunk and nothing else
could use the budget; every other step keeps ``max_num_batched_tokens`` (the
admission budget), so burst TTFT and decoder cadence are unchanged.
"""

from types import SimpleNamespace

import pytest
import vllm  # noqa: F401  (resolve the platform plugin before importing ours)

from vllm_tt_plugin.config import get_tt_prefill_chunk_run
from vllm_tt_plugin.scheduler import TTScheduler

CHUNK = 2048


def _scheduler(run=4, waiting=(), skipped=()):
    s = TTScheduler.__new__(TTScheduler)
    s._prefill_chunk_run = run
    s.max_num_scheduled_tokens = CHUNK
    s.waiting = list(waiting)
    s.skipped_waiting = list(skipped)
    return s


def _request(prompt_tokens, computed, prefill_chunk=True, mm=False):
    return SimpleNamespace(
        num_prompt_tokens=prompt_tokens,
        num_tokens=prompt_tokens,
        num_computed_tokens=computed,
        is_prefill_chunk=prefill_chunk,
        mm_features=[object()] if mm else [],
        has_encoder_inputs=mm,
    )


def test_lone_continuation_gets_k_chunks():
    s = _scheduler(run=4)
    assert s._prefill_chunk_run_budget([_request(32000, 2048)], []) == 4 * CHUNK


def test_first_chunk_of_a_fresh_request_stays_one_bucket():
    s = _scheduler(run=4)
    assert s._prefill_chunk_run_budget([_request(32000, 0)], []) == CHUNK


def test_running_decoders_keep_the_step_cadence():
    s = _scheduler(run=4)
    decoder = _request(100, 100, prefill_chunk=False)
    assert s._prefill_chunk_run_budget([_request(32000, 2048)], [decoder]) == CHUNK


def test_waiting_requests_keep_the_admission_budget():
    s = _scheduler(run=4, waiting=[_request(100, 0)])
    assert s._prefill_chunk_run_budget([_request(32000, 2048)], []) == CHUNK
    s = _scheduler(run=4, skipped=[_request(100, 0)])
    assert s._prefill_chunk_run_budget([_request(32000, 2048)], []) == CHUNK


def test_two_partial_prefills_share_the_plain_budget():
    s = _scheduler(run=4)
    assert (
        s._prefill_chunk_run_budget([_request(32000, 2048), _request(9000, 2048)], [])
        == CHUNK
    )


def test_run_of_one_and_multimodal_rows_are_untouched():
    assert _scheduler(run=1)._prefill_chunk_run_budget([_request(32000, 2048)], []) == CHUNK
    s = _scheduler(run=4)
    assert s._prefill_chunk_run_budget([_request(32000, 2048, mm=True)], []) == CHUNK


def _steps(prompt, run, budget=CHUNK):
    """The chunk ends a lone request sees, step by step (the base scheduler
    gives a running request min(remaining, budget) tokens)."""
    s = _scheduler(run=run)
    s.max_num_scheduled_tokens = budget
    computed, steps = 0, []
    while computed < prompt:
        b = s._prefill_chunk_run_budget([_request(prompt, computed)], [])
        n = min(prompt - computed, b)
        computed += n
        steps.append(n)
    return steps


def test_32k_and_128k_prompts_step_plans():
    # 32,000 tokens: 16 one-chunk steps today -> 1 + ceil(15 x 2048 / 8192) = 5 steps at K = 4
    assert _steps(32000, 1) == [2048] * 15 + [1280]
    assert _steps(32000, 4) == [2048, 8192, 8192, 8192, 5376]
    assert sum(_steps(32000, 4)) == 32000
    # 131,072 tokens: 64 steps today -> 1 + ceil(63 / 8) = 9 steps at K = 8
    assert len(_steps(131072, 1)) == 64
    assert _steps(131072, 8) == [2048] + [8 * 2048] * 7 + [63 * 2048 - 7 * 8 * 2048]
    assert sum(_steps(131072, 8)) == 131072
    # 1M: 512 -> 1 + ceil(511 / 8) = 65 steps at K = 8
    assert len(_steps(1048576, 8)) == 65


def test_config_getter_validates():
    cfg = SimpleNamespace(additional_config={"tt": {"prefill_chunk_run": 4}})
    assert get_tt_prefill_chunk_run(cfg) == 4
    assert get_tt_prefill_chunk_run(SimpleNamespace(additional_config={"tt": {}})) == 1
    for bad in (0, True, "4", -1):
        with pytest.raises(ValueError):
            get_tt_prefill_chunk_run(
                SimpleNamespace(additional_config={"tt": {"prefill_chunk_run": bad}})
            )
