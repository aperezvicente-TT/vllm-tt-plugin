# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""GLM53_PREFILL_SHORT_FIRST_KV_OVERTAKE (program 2026-10-28T16:15:39Z / 16:15:53Z; the DP4 acceptance's blk3):
under GLM53_ADMIT_KV_BUDGET a resident 1M reserves all but ~9.4K tokens of a 1M pool, and vLLM's waiting loop stops at
the first request the gate defers, so one 32K / 131K arrival at the head of the queue held every short behind it for
the 1M's whole prefill (shorts 318-330 s; SHORT_FIRST held the long for a short it could not admit: 142 stuck steps).
Now a gate-deferred request that a LONG resident blocks sits the pass out and the short behind it is admitted; one
blocked by short residents alone keeps the FIFO stop. Host-only: the real TTScheduler over the real vLLM base scheduler
(test_mixed_steps' helpers; pool 255 blocks of 64, max_model_len 4,096: three 4,000-token residents stand in for the
1M)."""

from __future__ import annotations

import pytest

from tests.test_mixed_steps import _commit, _request, _scheduler
from vllm_tt_plugin.scheduler import TTScheduler

T = 1024  # GLM53_PREFILL_SHORT_FIRST, scaled to the test pool


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GLM53_ADMIT_KV_BUDGET", "1")
    monkeypatch.setenv("GLM53_PREFILL_SHORT_FIRST", str(T))
    monkeypatch.setenv("GLM53_PREFILL_ONE_REQUEST", "1")
    for k in ("GLM53_PREFILL_SHORT_FIRST_KV_OVERTAKE", "GLM53_PREFILL_SHORT_FIRST_MAX", "GLM53_PREFILL_SHORT_FIRST_RETRY",
              "GLM53_ADMIT_RESERVE_TOKENS", "PROMETHEUS_MULTIPROC_DIR"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)


def _resident(s, ids, prompt=4000, max_tokens=1000, steps=60):
    """admit `ids` (each reserves min(prompt + max_tokens, 4,096) = 64 blocks) and run until all of them decode"""
    for i in ids:
        s.add_request(_request(i, prompt, max_tokens=max_tokens))
    for _ in range(steps):
        _commit(s, s.schedule())
        if all(s.requests[i].num_output_tokens > 0 for i in ids):
            return
    raise AssertionError({i: s.requests[i].num_computed_tokens for i in ids})


def _first_step_of(s, rid, steps=12):
    for k in range(steps):
        out = s.schedule()
        got = out.num_scheduled_tokens.get(rid)
        _commit(s, out)
        if got:
            return k
    return None


def _sched(policy: str = "priority"):
    """the scheduler under the SERVED server_cmd flags (program 2026-10-28T18:24:59Z: a host-green must never differ
    from the box on a flag): --scheduling-policy priority (the fleet's; the v3 overtake bailed on it), --max-num-seqs 16,
    --max-num-batched-tokens 2048. policy='fcfs' keeps the vLLM default covered too."""
    from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

    s = _scheduler(mixed=False)
    s.scheduler_config.max_num_seqs = 16
    s.max_num_running_reqs = 16
    s.scheduler_config.max_num_batched_tokens = 2048
    s.max_num_scheduled_tokens = 2048
    s.policy = SchedulingPolicy.PRIORITY if policy == "priority" else SchedulingPolicy.FCFS
    s.waiting = create_request_queue(s.policy)
    s.skipped_waiting = create_request_queue(s.policy)
    return s


POLICIES = pytest.mark.parametrize("policy", ["priority", "fcfs"])


@POLICIES
def test_a_short_passes_a_request_a_long_resident_blocks(policy):
    s = _sched(policy)
    _resident(s, ["L1", "L2", "L3"])  # 3 x 64 = 192 of 255 blocks: 63 left
    s.add_request(_request("M", 4000, max_tokens=64))  # 64 blocks: deferred while the longs are resident
    s.add_request(_request("S", 256, max_tokens=64))  # 5 blocks: fits
    k = _first_step_of(s, "S")
    assert k is not None and k <= 2, k
    assert s.requests["M"].num_computed_tokens == 0 and s.requests["M"].status.name == "WAITING"
    assert s.prefill_short_first_kv_overtakes >= 1
    s.add_request(_request("late", 256, max_tokens=8))
    assert [r.request_id for r in s.waiting][:1] == ["M"]  # M keeps its place, ahead of a later arrival


@POLICIES
def test_off_the_gate_keeps_its_fifo_stop(policy, monkeypatch):
    monkeypatch.setenv("GLM53_PREFILL_SHORT_FIRST_KV_OVERTAKE", "0")
    s = _sched(policy)
    _resident(s, ["L1", "L2", "L3"])
    s.add_request(_request("M", 4000, max_tokens=64))
    s.add_request(_request("S", 256, max_tokens=64))
    assert _first_step_of(s, "S", steps=6) is None  # the blk3 behaviour: the short waits behind M
    assert getattr(s, "prefill_short_first_kv_overtakes", 0) == 0


@POLICIES
def test_short_residents_alone_keep_the_fifo_stop(policy):
    """no starvation: a request that would fit beside the long residents (none here) waits for the shorts to drain,
    and no new short overtakes it meanwhile"""
    s = _sched(policy)
    _resident(s, ["s1", "s2", "s3"], prompt=256, max_tokens=3800)  # short prompts, 64-block reserves each
    s.add_request(_request("M", 4000, max_tokens=64))
    s.add_request(_request("S", 256, max_tokens=64))
    assert _first_step_of(s, "S", steps=6) is None
    assert getattr(s, "prefill_short_first_kv_overtakes", 0) == 0


@POLICIES
def test_a_long_in_its_prefill_is_held_for_a_short_that_passes_and_nothing_sticks(policy):
    """the blk3 shape: a long mid-prompt (SHORT_FIRST holds it) + a deferred big arrival at the queue head + a short"""
    s = _sched(policy)
    _resident(s, ["L1", "L2"])
    s.add_request(_request("L3", 4000, max_tokens=1000))
    for _ in range(8):  # L3 mid-prompt (the served 2,048-token step budget: one step can take half of it)
        _commit(s, s.schedule())
        if s.requests["L3"].num_computed_tokens > 0:
            break
    assert 0 < s.requests["L3"].num_computed_tokens < 4000 - T
    s.add_request(_request("M", 4000, max_tokens=64))
    s.add_request(_request("S", 256, max_tokens=64))
    k = _first_step_of(s, "S")
    assert k is not None and k <= 3, k
    assert getattr(s, "prefill_short_first_stuck", 0) == 0
    assert s.prefill_short_first_holds >= 1


@POLICIES
def test_the_hold_ignores_a_waiting_short_the_gate_would_defer(policy):
    s = _sched(policy)
    _resident(s, ["L1", "L2"])
    s.add_request(_request("L3", 4000, max_tokens=1000))
    for _ in range(3):
        _commit(s, s.schedule())
    s.add_request(_request("B", 1000, max_tokens=3100))  # prompt <= T, but its 64-block reserve does not fit (192 + 64)
    partial = [r for r in s.running if r.is_prefill_chunk]
    assert s._short_first_hold(partial) == []  # no short to hold for: no stuck, no empty step


def test_restore_puts_the_hidden_back_at_their_places():
    s = _sched("fcfs")  # (insertion order is an FCFS property; the priority heap orders by arrival)
    reqs = {i: _request(i, 64) for i in "abcde"}
    for i in "abcd":
        s.waiting.add_request(reqs[i])
    before = list(s.waiting)
    s.waiting.remove_requests([reqs["b"], reqs["d"]])
    s.waiting.remove_request(reqs["a"])  # the pass admitted a
    s.waiting.prepend_request(reqs["e"])  # and prepended a preempted e
    TTScheduler._short_first_kv_restore(s, {"waiting": (before, [reqs["b"], reqs["d"]])})
    assert [r.request_id for r in s.waiting] == ["e", "b", "c", "d"]


_GAUGES: dict = {}


def _gauges(monkeypatch, tmp_path):
    """the admit gauges, created once per process (prometheus_client refuses a second registration of a name)"""
    import vllm_tt_plugin.scheduler as SCH

    if "g" not in _GAUGES:
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
        monkeypatch.setattr(SCH, "_ADMIT_GAUGES", {})
        _GAUGES["g"] = SCH._admit_gauges()
    if _GAUGES["g"] is None:
        pytest.skip("prometheus_client gauges unavailable in this process")
    monkeypatch.setattr(SCH, "_ADMIT_GAUGES", {"g": _GAUGES["g"]})
    return _GAUGES["g"]


def test_the_admission_headroom_gauges(monkeypatch, tmp_path):
    """router v6's input (program 2026-10-28T16:15:39Z / 16:15:53Z): glm53_admit_{pool,reserved,headroom,queued}_tokens
    = the gate's reserve rule in tokens, set from the scheduler (through PROMETHEUS_MULTIPROC_DIR on the served engine)"""
    monkeypatch.setenv("GLM53_ADMIT_HEADROOM_EVERY_S", "0")
    g = _gauges(monkeypatch, tmp_path)
    s = _sched()
    _resident(s, ["L1", "L2", "L3"])
    s.add_request(_request("M", 4000, max_tokens=64))
    _commit(s, s.schedule())
    val = {k: g[k]._value.get() for k in ("pool", "reserved", "headroom", "queued")}
    assert val == {"pool": 255 * 64, "reserved": 192 * 64, "headroom": 63 * 64, "queued": 64 * 64}, val


def test_the_gauges_follow_a_drain(monkeypatch, tmp_path):
    """v3b6 (program 2026-10-29T01:17:33Z): a drained engine schedules nothing, so gauges set from schedule() kept
    their last BUSY value -- router v6-v6c read an empty block as full. The step that empties the engine and an abort
    that empties it publish at once (here the throttle is an hour: only the drain can publish)"""
    from vllm.v1.request import RequestStatus

    monkeypatch.setenv("GLM53_ADMIT_HEADROOM_EVERY_S", "0")
    g = _gauges(monkeypatch, tmp_path)
    val = lambda: {k: g[k]._value.get() for k in ("reserved", "headroom")}  # noqa: E731
    empty = {"reserved": 0, "headroom": 255 * 64}
    s = _sched()
    # (1) an abort empties the engine
    _resident(s, ["L1", "L2"])
    assert val() == {"reserved": 128 * 64, "headroom": 127 * 64}, val()
    monkeypatch.setenv("GLM53_ADMIT_HEADROOM_EVERY_S", "3600")
    s.finish_requests("L1", RequestStatus.FINISHED_ABORTED)
    assert val() == {"reserved": 128 * 64, "headroom": 127 * 64}, val()  # L2 still runs: the throttle holds
    s.finish_requests("L2", RequestStatus.FINISHED_ABORTED)
    assert val() == empty, val()
    # (2) the last request finishes in a step
    monkeypatch.setenv("GLM53_ADMIT_HEADROOM_EVERY_S", "0")
    _resident(s, ["S1"], prompt=512, max_tokens=4)
    assert val()["reserved"] == 9 * 64, val()
    monkeypatch.setenv("GLM53_ADMIT_HEADROOM_EVERY_S", "3600")
    for _ in range(12):
        if not s.get_num_unfinished_requests():
            break
        _commit(s, s.schedule())
    assert not s.get_num_unfinished_requests() and val() == empty, val()


class _PendingLookupConnector:
    """a KV connector whose lookup stays 'not yet' (None) for the request ids in `pending`: vLLM parks those in
    skipped_waiting and scans that queue FIRST (the Galaxy blk3 shape of 2026-10-28T18:11Z)"""

    def __init__(self, pending):
        self.pending = set(pending)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return (None, False) if request.request_id in self.pending else (0, False)

    def update_state_after_alloc(self, *a, **k):
        return None

    def build_connector_meta(self, *a, **k):
        return None

    def get_kv_connector_stats(self):
        return None

    def take_events(self):
        return []

    def has_pending_push_work(self):
        return False

    def request_finished(self, *a, **k):
        return False, None

    def request_finished_all_groups(self, *a, **k):
        return False, None

    def update_connector_output(self, *a, **k):
        return None

    def on_new_request(self, *a, **k):
        return None


@POLICIES
def test_v3b_a_short_parked_in_skipped_waiting_does_not_stop_the_overtake(policy):
    """v3 stopped its walk at a fitting short in skipped_waiting (scanned first) with nothing hidden, so the deferred
    head of waiting kept the FIFO stop while the hold held the long: stuck. v3b hides order-free over both queues."""
    s = _sched(policy)
    s.kv_cache_manager.block_pool.num_gpu_blocks = 70  # the gate's pool: the 4,000-token long leaves 6 blocks (a 1M on 1M)
    s.connector = _PendingLookupConnector({"P"})
    s.add_request(_request("LONG", 4000, max_tokens=8))
    for _ in range(3):
        _commit(s, s.schedule())
    s.add_request(_request("A1", T, max_tokens=8))  # 17 blocks: deferred while the long is resident
    s.add_request(_request("P", 256, max_tokens=8))  # its lookup never resolves: parked in skipped_waiting
    _commit(s, s.schedule())
    s.add_request(_request("S", 256, max_tokens=8))  # behind A1 in waiting
    k = _first_step_of(s, "S", steps=6)
    assert k is not None and k <= 3, (k, s.prefill_short_first_stuck, getattr(s, "_kv_hide_why", None))
    # one stuck step at most: the step held for P itself (its lookup is pending); P is then retried later, not re-held
    assert getattr(s, "prefill_short_first_stuck", 0) <= 1 and s.prefill_short_first_kv_overtakes >= 1
    assert s.requests["A1"].num_computed_tokens == 0  # A1 still waits for the long


@POLICIES
def test_v3b_one_predicate_for_the_gate_and_the_hold(policy):
    s = _sched(policy)
    s.kv_cache_manager.block_pool.num_gpu_blocks = 70
    s.add_request(_request("LONG", 4000, max_tokens=8))
    for _ in range(3):
        _commit(s, s.schedule())
    a1, sh = _request("A1", T, max_tokens=8), _request("S", 256, max_tokens=8)
    for r in (a1, sh):
        s.add_request(r)
    assert s._kv_fits(sh) and not s._kv_fits(a1)
    assert s._kv_budget_defers(a1) and not s._kv_budget_defers(sh)  # the gate = not _kv_fits


def _priority(s):
    """the served fleet runs --scheduling-policy priority (2026-10-28 server_cmd): the waiting queues are heaps"""
    from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

    s.policy = SchedulingPolicy.PRIORITY
    s.waiting = create_request_queue(s.policy)
    s.skipped_waiting = create_request_queue(s.policy)
    return s


def test_v3b2_the_overtake_runs_under_the_served_priority_policy():
    """THE Galaxy cause (18:11:24Z, holds 23 / stuck 23 / kv_overtakes 0): v3 / v3b's overtake returned at once unless the
    policy was FCFS, and the fleet serves with --scheduling-policy priority; the hold (no such guard) still held the long"""
    s = _priority(_sched())
    s.kv_cache_manager.block_pool.num_gpu_blocks = 70  # the 4,000-token long leaves 6 blocks (a 1M on 1M)
    s.add_request(_request("LONG", 4000, max_tokens=8))
    for _ in range(3):
        _commit(s, s.schedule())
    s.add_request(_request("A1", T, max_tokens=8))  # deferred beside the long
    s.add_request(_request("A2", 3000, max_tokens=8))
    s.add_request(_request("S", 256, max_tokens=8))
    k = _first_step_of(s, "S", steps=6)
    assert k is not None and k <= 2, (k, getattr(s, "prefill_short_first_stuck", 0), getattr(s, "_kv_hide_why", None))
    assert getattr(s, "prefill_short_first_stuck", 0) == 0 and s.prefill_short_first_kv_overtakes >= 1
    assert s.requests["A1"].num_computed_tokens == 0 and s.requests["A2"].num_computed_tokens == 0
