# SPDX-License-Identifier: Apache-2.0
"""GLM53_PREFILL_SHORT_FIRST = T (program 2026-10-28T13:27:02Z / 13:32:43Z): with GLM53_PREFILL_ONE_REQUEST a long
mid-prompt prefill owns every prefill step until its last chunk, so a short arrival on its block waits the long's whole
remainder (DP4 load run 1: 255-325 s behind a 1M). The long sits a prefill step out while a short prompt (remaining <=
T) can take it, at most GLM53_PREFILL_SHORT_FIRST_MAX steps in a row; a short <= one chunk run takes its run."""

from types import SimpleNamespace

import pytest
import vllm  # noqa: F401  (resolve the platform plugin before importing ours)
from vllm.v1.core.sched.output import SchedulerOutput

import vllm_tt_plugin.scheduler as SCH
from vllm_tt_plugin.scheduler import TTScheduler

CHUNK = 2048


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("GLM53_PREFILL_SHORT_FIRST", "GLM53_PREFILL_SHORT_FIRST_MAX", "GLM53_PREFILL_SHORT_FIRST_RETRY"):
        monkeypatch.delenv(k, raising=False)


def _req(rid, prompt, computed, chunk=True):
    return SimpleNamespace(request_id=rid, num_prompt_tokens=prompt, num_tokens=prompt, num_computed_tokens=computed,
                           is_prefill_chunk=chunk, mm_features=[], has_encoder_inputs=False)  # fmt: skip


def _sched(running=(), waiting=(), run=4):
    s = TTScheduler.__new__(TTScheduler)
    s._prefill_chunk_run = run
    s.max_num_scheduled_tokens = CHUNK
    s.max_num_running_reqs = 16
    s.running = list(running)
    s.waiting = list(waiting)
    s.skipped_waiting = []
    return s


def test_off_by_default_nothing_is_held():
    long_ = _req("L", 1_000_000, 8192)
    s = _sched([long_], [_req("S", 2048, 0, chunk=False)])
    assert s._short_first_hold([long_]) == []


def test_a_waiting_short_holds_the_long_up_to_the_cap_then_the_long_gets_a_step(monkeypatch):
    monkeypatch.setenv("GLM53_PREFILL_SHORT_FIRST", "32768")
    monkeypatch.setenv("GLM53_PREFILL_SHORT_FIRST_MAX", "3")
    long_ = _req("L", 1_000_000, 8192)
    s = _sched([long_], [_req("S", 2048, 0, chunk=False)])
    got = [bool(s._short_first_hold([long_])) for _ in range(5)]
    assert got == [True, True, True, False, True], got  # 3 held steps, the long's turn, then held again
    assert s.prefill_short_first_holds == 4 and s.prefill_short_first_yields == 1




def test_running_short_partial_holds(monkeypatch):
    monkeypatch.setenv("GLM53_PREFILL_SHORT_FIRST", "32768")
    long_, short = _req("L", 1_000_000, 8192), _req("M", 32000, 2048)
    s = _sched([long_, short])
    assert s._short_first_hold([long_, short]) == [long_]
    s2 = _sched([long_], [_req("B", 200_000, 0, chunk=False)])  # a waiting LONG is no reason to hold
    assert s2._short_first_hold([long_]) == []


def test_a_short_a_held_step_could_not_admit_is_retried_later(monkeypatch):
    monkeypatch.setenv("GLM53_PREFILL_SHORT_FIRST", "32768")
    monkeypatch.setenv("GLM53_PREFILL_SHORT_FIRST_RETRY", "3")
    long_ = _req("L", 1_000_000, 8192)
    s = _sched([long_], [_req("S", 2048, 0, chunk=False)])
    held = s._short_first_hold([long_])
    empty = SchedulerOutput.make_empty()
    s._short_first_after(empty, held)  # the short was not admitted (e.g. a deferred lookup)
    assert s.prefill_short_first_stuck == 1
    assert s._short_first_hold([long_]) == [] and s._short_first_hold([long_]) == []  # the long runs meanwhile
    assert s._short_first_hold([long_]) == [long_]  # retried after 3 steps


def test_the_prefill_step_hides_the_long_and_puts_it_back(monkeypatch):
    monkeypatch.setenv("GLM53_PREFILL_SHORT_FIRST", "32768")
    long_, dec = _req("L", 1_000_000, 8192), _req("D", 100, 100, chunk=False)
    short = _req("S", 2048, 0, chunk=False)
    s = _sched([long_, dec], [short])
    seen = {}

    def base_schedule(self):
        seen["running"] = [r.request_id for r in self.running]
        seen["budget"] = self.max_num_scheduled_tokens
        self.running.append(short)  # the base scheduler admits the short
        out = SchedulerOutput.make_empty()
        out.num_scheduled_tokens = {"S": 2048}
        return out

    monkeypatch.setattr(SCH.AsyncScheduler, "schedule", base_schedule, raising=False)
    monkeypatch.setattr(TTScheduler, "_segment_begin", lambda self, *a, **k: None)
    out = s._schedule_prefill_only()
    assert seen["running"] == [] and out.num_scheduled_tokens == {"S": 2048}  # the long sat the step out
    assert [r.request_id for r in s.running] == ["L", "S", "D"]  # back, ahead of the newly admitted short
    assert s.max_num_running_reqs == 16 and s.max_num_scheduled_tokens == CHUNK


def test_a_short_owner_takes_its_chunk_run_beside_decoders_a_32k_does_not():
    s = _sched()
    dec = _req("D", 100, 100, chunk=False)
    assert s._prefill_chunk_run_budget([_req("S", 8192, 2048)], [dec], short_first=True) == 4 * CHUNK
    assert s._prefill_chunk_run_budget([_req("M", 32000, 2048)], [dec], short_first=True) == CHUNK
    assert s._prefill_chunk_run_budget([_req("S", 8192, 2048)], [dec]) == CHUNK  # off: unchanged
    assert s._prefill_chunk_run_budget([_req("S", 8192, 0)], [dec], short_first=True) == CHUNK  # its TTFT chunk
