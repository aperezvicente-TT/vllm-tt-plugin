# SPDX-License-Identifier: Apache-2.0
"""GLM53_TTFT_PHASES=1 (program 2026-10-28T09:57:04Z ttft2k (c)): the cold first chunk's host span between its
lookup and the backend prefill, split on the engine thread (the scheduler's stamp + the runner's phases), one line."""

import logging
import time
from types import SimpleNamespace

import pytest

from vllm_tt_plugin import model_runner as MR
from vllm_tt_plugin import ttft_phases as TP

from .test_decode_interleave import _scheduler


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(TP, "_MARKS", {})


def _lines():
    got = []
    h = logging.Handler()
    h.emit = lambda rec: got.append(rec.getMessage())
    TP.logger.addHandler(h)
    TP.logger.setLevel(logging.INFO)
    return got, h


def _out(*reqs):
    return SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(req_id=r, num_computed_tokens=n) for r, n in reqs])


def test_a_cold_first_chunk_gets_one_line_with_every_phase():
    sched = SimpleNamespace(requests={"a": SimpleNamespace(arrival_time=time.time() - 0.05),
                                      "b": SimpleNamespace(arrival_time=time.time())})  # fmt: skip
    out = _out(("a", 0), ("b", 1024))  # b resumes at 1,024 computed tokens: not a cold first chunk
    TP.mark_scheduled(sched, out, time.perf_counter() - 0.002)
    assert set(TP._MARKS) == {"a"} and TP._MARKS["a"]["arrival_to_sched_ms"] >= 50
    got, h = _lines()
    try:
        sp = TP.StepPhases(out)
        for name in ("begin_step", "drain", "build", "segment", "load", "forward"):
            sp.stamp(name)
        fin = MR._ttft_timed_finish(sp, lambda grammar: "output")
        assert fin(None) == "output"
    finally:
        TP.logger.removeHandler(h)
    assert len(got) == 1 and got[0].startswith("TT TTFT phases a: "), got
    rep = __import__("json").loads(got[0].split(": ", 1)[1])
    assert {"arrival_to_sched_ms", "sched_ms", "sched_to_runner_ms", "begin_step_ms", "drain_ms", "build_ms",
            "segment_ms", "load_ms", "forward_ms", "forward_to_finish_ms", "finish_ms"} == set(rep), rep  # fmt: skip
    assert rep["sched_ms"] >= 2 and not TP._MARKS


def test_a_step_without_a_cold_first_chunk_logs_nothing():
    got, h = _lines()
    try:
        sp = TP.StepPhases(_out(("c", 0)))  # never stamped by the scheduler (the knob was off there)
        sp.stamp("forward")
        sp.finish()
    finally:
        TP.logger.removeHandler(h)
    assert got == [] and sp.report() is None


def test_the_scheduler_stamps_only_under_the_knob(monkeypatch):
    s = _scheduler()
    s.requests = {"r": SimpleNamespace(arrival_time=time.time())}
    monkeypatch.setattr(s, "_schedule_tt", lambda throttle_prefills=False: _out(("r", 0)))
    monkeypatch.delenv("GLM53_TTFT_PHASES", raising=False)
    s.schedule()
    assert not TP._MARKS
    monkeypatch.setenv("GLM53_TTFT_PHASES", "1")
    out = s.schedule()
    assert set(TP._MARKS) == {"r"} and out.scheduled_new_reqs[0].req_id == "r"
