# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Segmented prefill traces, the scheduler's alternating step loop (doc/ttnn/14b-persistent-stack/segtrace/
runner_design.md section 4, items 7-9; doc/ttnn/12-fast-start/foldin/step_loop.md section 1).

A segmentable chunk [s, e) runs as S segments over S engine steps while requests decode:
- step 0 schedules the chunk minus its last token (stamped set_tt_segmented + segment [0]);
- each hold step is decode-only and carries the next segment, with D - 1 plain decode steps between two segments;
- the release schedules the 1-token tail [e - 1, e) with nothing else prefilling (the segments not run yet + the head).
The decoders advance once per step throughout. No decoders, no segmentation; decoders gone inside the hold, an early
release. Host-only: the real TTScheduler over the real vLLM base scheduler (test_mixed_steps' helpers)."""

from __future__ import annotations

import os
import pickle

import pytest
from vllm.v1.request import RequestStatus

from tests.test_mixed_steps import BUDGET, _commit, _decoding, _request, _scheduler
from vllm_tt_plugin.config import (
    PREFILL_SEGMENT_DECODES_ENV,
    PREFILL_SEGMENT_MIN_TOKENS_ENV,
    SCHED_CTL_FILE_ENV,
    MIXED_DECODE_FOLLOW_ENV,
)
from vllm_tt_plugin.scheduler import get_tt_segment_step, get_tt_segmented, is_tt_mixed_step


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    for k in (SCHED_CTL_FILE_ENV, MIXED_DECODE_FOLLOW_ENV, PREFILL_SEGMENT_DECODES_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(PREFILL_SEGMENT_MIN_TOKENS_ENV, "256")  # the test budget is 512, the served chunk 2,048


def _settle(scheduler):
    out = scheduler.schedule()
    _commit(scheduler, out)


def _steps(scheduler, prompt_id, max_steps=60):
    """run until the prompt's prefill is done: per step (kind, the prompt's scheduled tokens, the segment stamp, the
    step-0 stamp, the decoders' rows)"""
    rows = []
    for _ in range(max_steps):
        req = scheduler.requests.get(prompt_id)
        if rows and (req is None or (not req.is_prefill_chunk and req.num_output_tokens > 0)):
            break
        out = scheduler.schedule()
        n = out.num_scheduled_tokens
        kind = "M" if is_tt_mixed_step(out) else ("P" if n.get(prompt_id) else "D")
        seg = get_tt_segment_step(out)
        rows.append((kind, n.get(prompt_id, 0), seg["segments"] if seg else None, get_tt_segmented(out),
                     sorted(r for r in n if r != prompt_id)))  # fmt: skip
        _commit(scheduler, out)
    return rows


def _kinds(rows):
    return "".join(k for k, *_ in rows)


def test_s3_one_decoder_runs_each_chunk_over_three_steps():
    scheduler = _scheduler(prefill_segments=3)
    _decoding(scheduler, "d1")
    _settle(scheduler)
    before = scheduler.requests["d1"].num_output_tokens
    scheduler.add_request(_request("p", 2 * BUDGET, max_tokens=4))
    rows = _steps(scheduler, "p")
    assert _kinds(rows) == "MDMMDM", rows
    assert [r[1] for r in rows] == [BUDGET - 1, 0, 1, BUDGET - 1, 0, 1]  # chunk - 1, the hold, the 1-token tail
    assert [r[2] for r in rows] == [[0], [1], [2], [0], [1], [2]]
    assert [r[3] for r in rows] == [{"p": 3}, {}, {}, {"p": 3}, {}, {}]  # the connector's step-0 stamp only
    assert all(r[4] == ["d1"] for r in rows)  # the decoder rode every step
    assert scheduler.requests["d1"].num_output_tokens - before == 6
    assert scheduler.requests["p"].num_output_tokens == 1  # the final chunk's release reported the token
    assert scheduler.segment_chunks == 2 and scheduler.segment_steps == 4
    assert scheduler._segment_hold is None


def test_the_step0_stamp_carries_the_device_chunk_end():
    scheduler = _scheduler(prefill_segments=3)
    _decoding(scheduler, "d1")
    _settle(scheduler)
    scheduler.add_request(_request("p", 2 * BUDGET, max_tokens=4))
    out = scheduler.schedule()
    seg = get_tt_segment_step(out)
    assert seg == {"req_id": "p", "segments": [0], "num_segments": 3, "end": BUDGET}
    assert scheduler.requests["p"].is_prefill_chunk and scheduler.requests["p"].num_computed_tokens == BUDGET - 1
    assert get_tt_segment_step(pickle.loads(pickle.dumps(out))) == seg  # survives the executor boundary


def test_d2_puts_two_decode_steps_after_every_segment(monkeypatch):
    monkeypatch.setenv(PREFILL_SEGMENT_DECODES_ENV, "2")
    scheduler = _scheduler(prefill_segments=3)
    _decoding(scheduler, "d1")
    _settle(scheduler)
    scheduler.add_request(_request("p", 2 * BUDGET, max_tokens=4))
    rows = _steps(scheduler, "p")
    # per chunk: seg0 (mixed) | plain | seg1 | plain | release (mixed) | the release's follow-up
    assert _kinds(rows) == "MDDDMD" + "MDDDM", rows
    assert [r[2] for r in rows] == [[0], None, [1], None, [2], None, [0], None, [1], None, [2]]
    assert scheduler.segment_decode_steps == 5  # 2 plain steps per chunk + the first release's follow-up


def test_no_decoders_no_segmentation():
    scheduler = _scheduler(prefill_segments=3)
    scheduler.add_request(_request("p", 2 * BUDGET, max_tokens=4))
    rows = _steps(scheduler, "p")
    assert _kinds(rows) == "PP" and [r[1] for r in rows] == [BUDGET, BUDGET] and all(r[2] is None for r in rows)
    assert not getattr(scheduler, "segment_chunks", 0)


def test_s1_is_today():
    scheduler = _scheduler()
    _decoding(scheduler, "d1")
    _settle(scheduler)
    scheduler.add_request(_request("p", 2 * BUDGET, max_tokens=4))
    rows = _steps(scheduler, "p")
    assert _kinds(rows) == "MM" and [r[1] for r in rows] == [BUDGET, BUDGET]


def test_the_decoders_leaving_inside_the_hold_release_early():
    scheduler = _scheduler(prefill_segments=4)
    _decoding(scheduler, "d1", max_tokens=3)  # its admission step emitted token 1
    scheduler.add_request(_request("p", BUDGET, max_tokens=4))
    rows = _steps(scheduler, "p")
    # step 0 (d1 token 2), the hold's segment 1 (d1 token 3: finished), then no decoder: the release runs 2 and 3
    assert [r[2] for r in rows] == [[0], [1], [2, 3]], rows
    assert _kinds(rows)[-1] == "P"  # a prefill-only release (no decoder to mix with)
    assert scheduler.segment_early_releases == 1


def test_a_request_aborted_inside_the_hold_breaks_it():
    scheduler = _scheduler(prefill_segments=3)
    _decoding(scheduler, "d1")
    _settle(scheduler)
    scheduler.add_request(_request("p", 2 * BUDGET, max_tokens=4))
    _commit(scheduler, scheduler.schedule())  # step 0
    assert scheduler._segment_hold is not None
    scheduler.finish_requests("p", RequestStatus.FINISHED_ABORTED)
    out = scheduler.schedule()
    assert scheduler._segment_hold is None and scheduler.segment_holds_broken == 1
    assert get_tt_segment_step(out) is None and out.num_scheduled_tokens == {"d1": 1}


def test_the_segmented_chunk_is_the_only_prefill_until_its_release():
    scheduler = _scheduler(prefill_segments=3)
    _decoding(scheduler, "d1")
    _settle(scheduler)
    scheduler.add_request(_request("p1", BUDGET, max_tokens=4))
    scheduler.add_request(_request("p2", 300, max_tokens=4))
    seen = []
    for _ in range(3):
        out = scheduler.schedule()
        seen.append(dict(out.num_scheduled_tokens))
        _commit(scheduler, out)
    assert seen == [{"d1": 1, "p1": BUDGET - 1}, {"d1": 1}, {"d1": 1, "p1": 1}]
    out = scheduler.schedule()  # p2 next: its chunk (300 >= the 256 floor) is segmented in turn
    assert out.num_scheduled_tokens.get("p2") == 299 and get_tt_segmented(out) == {"p2": 3}


def test_interleave_mode_segments_through_prefill_only_steps():
    """no mixed steps (a spec twin's form): step 0 and the release are prefill-only steps, the hold decode-only"""
    scheduler = _scheduler(mixed=False, prefill_segments=3)
    _decoding(scheduler, "d1")
    _settle(scheduler)
    scheduler.add_request(_request("p", 2 * BUDGET, max_tokens=4))
    rows = _steps(scheduler, "p")
    assert [(k, n, s) for k, n, s, _st, _d in rows[:3]] == [("P", BUDGET - 1, [0]), ("D", 0, [1]), ("P", 1, [2])]
    assert rows[0][3] == {"p": 3} and not rows[0][4] and rows[1][4] == ["d1"]


def test_a_chunk_whose_last_position_opens_a_block_is_not_segmented():
    scheduler = _scheduler(prefill_segments=3)
    _decoding(scheduler, "d1")
    _settle(scheduler)
    scheduler.add_request(_request("p", 321, max_tokens=4))  # [0, 321): position 320 opens block 5 of 64
    rows = _steps(scheduler, "p")
    assert [r[1] for r in rows] == [321] and rows[0][2] is None
    assert scheduler.segment_skipped_block_edge == 1


def test_a_short_chunk_is_not_segmented():
    scheduler = _scheduler(prefill_segments=3)
    _decoding(scheduler, "d1")
    _settle(scheduler)
    scheduler.add_request(_request("p", 200, max_tokens=4))  # below the 256 floor (the served 512 bucket)
    rows = _steps(scheduler, "p")
    assert [r[1] for r in rows] == [200] and rows[0][2] is None


def test_the_control_file_sets_s_and_d_per_chunk(monkeypatch, tmp_path):
    ctl = tmp_path / "sched.ctl"
    monkeypatch.setenv(SCHED_CTL_FILE_ENV, str(ctl))
    stamp = [10**18]

    def _ctl(text):
        ctl.write_text(text)
        stamp[0] += 10**9
        os.utime(ctl, ns=(stamp[0], stamp[0]))

    scheduler = _scheduler()
    _decoding(scheduler, "d1")
    _settle(scheduler)
    _ctl("prefill_segments=2\n")
    scheduler.add_request(_request("p1", BUDGET, max_tokens=1))
    assert [r[2] for r in _steps(scheduler, "p1")] == [[0], [1]]  # S = 2: step 0, then the release
    _ctl("prefill_segments=3\nprefill_segment_decodes=2\n")
    _settle(scheduler)
    scheduler.add_request(_request("p2", BUDGET, max_tokens=1))
    assert [r[2] for r in _steps(scheduler, "p2")] == [[0], None, [1], None, [2]]
    _ctl("prefill_segments=1\n")
    _settle(scheduler)
    scheduler.add_request(_request("p3", BUDGET, max_tokens=1))
    assert [r[2] for r in _steps(scheduler, "p3")] == [None]


def test_the_counters_line_carries_the_scheduler_segment_counters():
    from vllm_tt_plugin.kv_connector import _COUNTER_KEYS

    for k in ("segment_chunks", "segment_steps", "segment_decode_steps", "segment_early_releases",
              "segment_holds_broken", "segment_skipped_block_edge", "segment_release_retries"):  # fmt: skip
        assert k in _COUNTER_KEYS["sched"]


def test_async_scheduling_one_step_ahead_keeps_the_loop():
    """vLLM's batch queue schedules step k + 1 before step k's output is in: the hold needs no output to advance"""
    scheduler = _scheduler(prefill_segments=3, async_scheduling=True)
    _decoding(scheduler, "d1")
    _settle(scheduler)
    scheduler.add_request(_request("p", 2 * BUDGET, max_tokens=4))
    rows, prev = [], None
    for _ in range(6):
        out = scheduler.schedule()
        seg = get_tt_segment_step(out)
        rows.append((out.num_scheduled_tokens.get("p", 0), seg["segments"] if seg else None,
                     "d1" in out.num_scheduled_tokens))  # fmt: skip
        if prev is not None:
            _commit(scheduler, prev)
        prev = out
    assert [r[0] for r in rows] == [BUDGET - 1, 0, 1, BUDGET - 1, 0, 1], rows
    assert [r[1] for r in rows] == [[0], [1], [2], [0], [1], [2]]
    assert all(r[2] for r in rows)
