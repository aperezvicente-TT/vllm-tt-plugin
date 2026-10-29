# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""v3b4 SHORT_FIRST pacing at the device (program 2026-10-28T20:11:59Z (3), the mesh f2i3b3 phase-2 RED).

The served loop is vLLM's EngineCore.step_with_batch_queue (async scheduling: batch queue of 2) over the uniproc
executor, and the TT runner returns a head-less intermediate prefill chunk run as soon as it is enqueued. On the box a
lone 131K's 16 head-less runs were submitted 0.04 s apart and the scheduler never ran while the shorts waited (v3b3's
scheduler-side count was released at the same host pace). Now the runner waits after enqueuing a head-less run for the
run GLM53_PREFILL_SHORT_FIRST_INFLIGHT - 1 behind it (prefill_pacer).

Host-only, on the served path: vLLM's real step_with_batch_queue drives the real TTScheduler under the served async
flags (priority / fcfs, 16 seqs, chunk run 4, ONE_REQUEST, the KV gate, SHORT_FIRST), over a runner stand-in with the
TT runner's timing contract and the real pacing hook, on a virtual-clock device with an in-order queue."""

from __future__ import annotations

import collections
import contextlib
import sys
import types
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

import tests.test_mixed_steps as tm
from tests.test_mixed_steps import _request, _scheduler
from vllm_tt_plugin import prefill_pacer

T = 2048  # GLM53_PREFILL_SHORT_FIRST, scaled (the box: 32768)
RUN = 2048  # one chunk run: prefill_chunk_run 4 x the 512-token step budget (the box: 4 x 2048)
RUN_S = 1.0  # the device time of one chunk run (the box: ~1.6 s for 8,192 tokens of a 131K)
HOST_S = 0.04  # one engine step's host time (the box: 0.041-0.069 s between submits)
DEC_S = 0.05  # one decode step's device time
LONG = 512 + 8 * RUN  # a first chunk, then 8 chunk runs (7 head-less + the head)
SHORT = 512


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GLM53_PREFILL_SHORT_FIRST", str(T))
    monkeypatch.setenv("GLM53_PREFILL_ONE_REQUEST", "1")
    monkeypatch.setenv("GLM53_ADMIT_KV_BUDGET", "1")
    for k in ("GLM53_PREFILL_SHORT_FIRST_INFLIGHT", "GLM53_PREFILL_SHORT_FIRST_MAX", "GLM53_PREFILL_SHORT_FIRST_RETRY",
              "PROMETHEUS_MULTIPROC_DIR"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(tm, "MAX_MODEL_LEN", 32768)  # room for the scaled long


class _Ev:
    def __init__(self, done_at: float):
        self.done_at = done_at


class _Box:
    """the virtual clock and the device's in-order queue; what the engine submitted and when tokens came back"""

    def __init__(self):
        self.t = 0.0
        self.free_at = 0.0
        self.busy: list[tuple[float, float, float]] = []  # the device runs, in queue order: (enqueued, start, end)
        self.submits: list[tuple[float, str, int, bool]] = []  # (host time, request, tokens, head-less)
        self.first_token: dict[str, float] = {}

    def enqueue(self, dur: float) -> None:
        start = max(self.t, self.free_at)
        self.free_at = start + dur
        self.busy.append((self.t, start, self.free_at))

    def unfinished(self, t: float) -> int:
        """the device runs enqueued by host time t that have not finished at t"""
        return sum(1 for q, _s, e in self.busy if q <= t + 1e-9 and e > t + 1e-9)


@pytest.fixture
def box(monkeypatch):
    b = _Box()
    fake = types.ModuleType("ttnn")
    fake.record_event = lambda device, cq_id=0: _Ev(device.free_at)

    def _sync(ev):
        b.t = max(b.t, ev.done_at)

    fake.event_synchronize = _sync
    monkeypatch.setitem(sys.modules, "ttnn", fake)
    return b


class _Runner:
    """the TT runner's timing contract as the engine sees it: execute_model enqueues the step's device work and returns
    (no read-back for a head-less chunk run: the real runner's prefill path, then its pacing hook); sample_tokens reads
    back (waits for the queue) only when a row produces a token (a prompt's head, a decode)"""

    def __init__(self, box: _Box, sched):
        self.box, self.sched = box, sched
        self.runner = SimpleNamespace(mesh_device=box)
        self._rows: list[tuple[str, bool]] | None = None

    def execute_model(self, so, non_block=True):
        f: Future = Future()
        if so.total_num_scheduled_tokens == 0:
            f.set_result(EMPTY_MODEL_RUNNER_OUTPUT)
            return f
        b = self.box
        b.t += HOST_S
        dur, mask, rows = 0.0, [], []
        for rid, n in so.num_scheduled_tokens.items():
            r = self.sched.requests[rid]
            prefill = int(r.num_computed_tokens) - int(n) < int(r.num_prompt_tokens)
            tok = int(r.num_computed_tokens) >= int(r.num_prompt_tokens)  # a head or a decode: a token comes back
            dur += n / RUN * RUN_S if prefill else DEC_S
            rows.append((rid, tok))
            if prefill:
                mask.append(not tok)
                b.submits.append((b.t, rid, int(n), not tok))
        b.enqueue(dur)
        if mask:  # model_runner._execute_model_impl: the prefill forward is enqueued, then the pacing hook
            prefill_pacer.after_prefill_submit(self.runner, SimpleNamespace(prompt_lens=torch.ones(len(mask)),
                                                                            intermediate_prefill_mask=torch.tensor(mask)))  # fmt: skip
        self._rows = rows
        f.set_result(None)
        return f

    def sample_tokens(self, grammar_output, non_block=True):
        rows, self._rows = self._rows or [], None
        b = self.box
        if any(tok for _rid, tok in rows):
            b.t = max(b.t, b.free_at)  # the read-back waits for everything queued before it
        for rid, tok in rows:
            if tok:
                b.first_token.setdefault(rid, b.t)
        ids = [rid for rid, _tok in rows]
        out = ModelRunnerOutput(req_ids=ids, req_id_to_index={rid: i for i, rid in enumerate(ids)},
                                sampled_token_ids=[[9] if tok else [] for _rid, tok in rows], logprobs=None,
                                prompt_logprobs_dict={}, pooler_output=[])  # fmt: skip
        f: Future = Future()
        f.set_result(out)
        return f


def _sched(policy: str):
    from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

    s = _scheduler(mixed=False, async_scheduling=True, prefill_chunk_run=4)
    s.scheduler_config.max_num_seqs = 16
    s.max_num_running_reqs = 16
    s.policy = SchedulingPolicy.PRIORITY if policy == "priority" else SchedulingPolicy.FCFS
    s.waiting = create_request_queue(s.policy)
    s.skipped_waiting = create_request_queue(s.policy)
    return s


def _run(box: _Box, policy: str, arrivals: list[tuple[float, str, int, int]]) -> SimpleNamespace:
    """EngineCore.run_busy_loop's shape: take the arrivals due, then one step_with_batch_queue (vLLM's own code)"""
    sched = _sched(policy)
    eng = SimpleNamespace(scheduler=sched, model_executor=_Runner(box, sched), batch_queue=collections.deque(maxlen=2),
                          batch_queue_size=2, is_ec_consumer=True, is_pooling_model=False, check_for_draft_tokens=False,
                          _should_throttle_prefills=lambda: False, _process_aborts_queue=lambda: None,
                          log_error_detail=lambda so: contextlib.nullcontext(),
                          capture_iteration_details=lambda so: contextlib.nullcontext(None),
                          _attach_iteration_details=lambda outputs, details: None)  # fmt: skip
    step = types.MethodType(EngineCore.step_with_batch_queue, eng)
    pending = sorted(arrivals)
    while True:
        while pending and pending[0][0] <= box.t:
            _t, rid, n, max_tokens = pending.pop(0)
            sched.add_request(_request(rid, n, max_tokens=max_tokens))
        if not (sched.has_requests() or eng.batch_queue):  # EngineCore.has_work
            if not pending:
                return eng
            box.t = pending[0][0]  # the engine blocks on its input queue
            continue
        t0 = box.t
        step()
        if box.t == t0:
            box.t += 0.001  # run_busy_loop's idle yield
        assert box.t < 200, "the engine did not finish"


def _long_runs(box: _Box) -> list[tuple[float, str, int, bool]]:
    return [s for s in box.submits if s[1] == "LONG"]


@pytest.mark.parametrize("policy", ["priority", "fcfs"])
def test_control_without_pacing_the_engine_submits_the_whole_prefill_at_host_speed(monkeypatch, box, policy):
    """the f2i3b3 shape (INFLIGHT=0): every head-less run is submitted within a few host steps, and a short arriving
    mid-prefill gets its first token only after the long's"""
    monkeypatch.setenv("GLM53_PREFILL_SHORT_FIRST_INFLIGHT", "0")
    _run(box, policy, [(0.0, "LONG", LONG, 2), (2.5, "S", SHORT, 4)])
    runs = _long_runs(box)
    assert len(runs) == 9 and runs[-1][0] - runs[0][0] < 10 * HOST_S, [round(r[0], 3) for r in runs]  # the burst
    assert box.first_token["S"] > box.first_token["LONG"], box.first_token


@pytest.mark.parametrize("policy", ["priority", "fcfs"])
def test_pacing_submits_at_the_device_pace_and_a_mid_prefill_short_goes_first(box, policy):
    _run(box, policy, [(0.0, "LONG", LONG, 2), (2.5, "S", SHORT, 4)])
    cap = 2
    runs = _long_runs(box)
    assert len(runs) >= 9 and not runs[-1][3], runs  # (the long takes 512-token chunks beside the short)
    for t, _rid, _n, _headless in runs:  # never more than INFLIGHT runs on the device when a long's run is submitted
        assert box.unfinished(t) <= cap, (round(t, 3), box.busy)
    t_short = next(s[0] for s in box.submits if s[1] == "S")
    full = [r[0] for r in runs if r[2] == RUN and r[0] < t_short][1:]  # (the first follows the 512-token first chunk)
    gaps = [b - a for a, b in zip(full, full[1:])]
    assert gaps and min(gaps) >= 0.9 * RUN_S, [round(g, 3) for g in gaps]  # >= 1 device run apart (was 0.04 s)
    # the short: scheduled behind at most INFLIGHT runs, its first token well before the long's
    wait = box.first_token["S"] - 2.5
    assert wait <= cap * RUN_S + SHORT / RUN * RUN_S + 0.5, wait
    assert box.first_token["S"] < box.first_token["LONG"] - 3 * RUN_S, box.first_token


def test_pacing_costs_a_lone_long_nothing(monkeypatch, box):
    """the device never idles behind the host: the lone long's first token is where the unpaced burst puts it"""
    _run(box, "priority", [(0.0, "LONG", LONG, 2)])
    paced_ttft = box.first_token["LONG"]
    busy = [r for r in box.busy if r[2] <= paced_ttft + 1e-9]  # the prefill's runs (then the stand-in's sync decodes)
    assert len(busy) == 9 and all(b[1] - a[2] <= 1e-9 for a, b in zip(busy, busy[1:])), busy  # back to back: no idle
    monkeypatch.setenv("GLM53_PREFILL_SHORT_FIRST_INFLIGHT", "0")
    burst = _Box()
    _run(burst, "priority", [(0.0, "LONG", LONG, 2)])
    assert abs(paced_ttft - burst.first_token["LONG"]) <= 0.01 * burst.first_token["LONG"], (
        paced_ttft, burst.first_token["LONG"])  # fmt: skip


def test_the_paced_counter_reaches_the_counters_line(box):
    from vllm_tt_plugin.kv_connector import counters_snapshot

    eng = _run(box, "priority", [(0.0, "LONG", LONG, 2)])  # (the counters line holds its sources weakly)
    snap = counters_snapshot()
    assert eng.model_executor.runner._prefill_pacer.prefill_short_first_paced >= 6
    assert snap.get("prefill_short_first_paced", 0) >= 6, snap  # the 7 head-less runs after the first chunk


@pytest.mark.parametrize("case", ["head", "decode", "off", "inflight0"])
def test_steps_that_are_not_paced(monkeypatch, box, case):
    runner = SimpleNamespace(mesh_device=box)
    mi = SimpleNamespace(prompt_lens=torch.ones(2), intermediate_prefill_mask=torch.tensor([True, True]))
    if case == "head":
        mi.intermediate_prefill_mask = torch.tensor([True, False])  # a row's head: the read-back waits anyway
    elif case == "decode":
        mi.prompt_lens = None
    elif case == "off":
        monkeypatch.setenv("GLM53_PREFILL_SHORT_FIRST", "0")
    else:
        monkeypatch.setenv("GLM53_PREFILL_SHORT_FIRST_INFLIGHT", "0")
    for _ in range(3):
        box.enqueue(RUN_S)
        prefill_pacer.after_prefill_submit(runner, mi)
    assert "_prefill_pacer" not in runner.__dict__ and box.t == 0.0


def test_inflight_one_is_synchronous(monkeypatch, box):
    monkeypatch.setenv("GLM53_PREFILL_SHORT_FIRST_INFLIGHT", "1")
    runner = SimpleNamespace(mesh_device=box)
    mi = SimpleNamespace(prompt_lens=torch.ones(1), intermediate_prefill_mask=torch.tensor([True]))
    box.enqueue(RUN_S)
    prefill_pacer.after_prefill_submit(runner, mi)
    assert box.t == RUN_S and not runner._prefill_pacer.pending
