"""Stage 12 M1 (program 2026-10-26T17:08:39Z, a k3 MUST): GLM53_KV_LEAVE_DRAIN=1 -- a finished request's held page
stores no longer flush ALL at once on the engine thread as it leaves (the post-1M ~50-s stall: 128 held 8,192-token
stores). The scheduler-side request_finished delays the request's block free; the host drains its held steps at
GLM53_KV_LEAVE_FLUSH_STEPS per engine step while the engine keeps stepping, and reports it in finished_sending (its
blocks freed) only once none of its steps is held, pending or queued. A preemption, HOLD_MAX and a lookup still flush
everything (their blocks are not delay-freed / a hit must see every store)."""

import threading
from types import SimpleNamespace

import pytest

from vllm_tt_plugin import kv_connector as K

CHUNK = 4


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(K, "_SAVE_HOSTS", set())
    monkeypatch.setattr(K, "_LEAVE_PENDING", {})
    monkeypatch.setenv("GLM53_KV_STORE_HOLD", "1")
    monkeypatch.setenv("GLM53_KV_LEAVE_DRAIN", "1")
    for k in ("GLM53_KV_STORE_PRIORITY", "GLM53_KV_SAVE_DEFER", "GLM53_KV_STORE_HOLD_MAX", "GLM53_KV_STORE_HOLD_OVER",
              "GLM53_KV_STORE_FT_GRACE", "GLM53_KV_LOOKUP_FLUSH", "GLM53_KV_LEAVE_FLUSH_STEPS"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)


def _host(histories):
    host = K.TTKVConnectorHost.__new__(K.TTKVConnectorHost)
    host._deferred_save, host.saves_deferred = None, 0
    pages = []
    host._save_state = lambda mi, out=None: None
    host._save_pages = lambda so, mi: pages.append(mi.tag)
    host._tokens_of = lambda rid: histories.get(rid)
    host._head_of_fn = lambda rid, n: None if rid not in histories else histories[rid][:n]
    host._lmcache_chunk = lambda: CHUNK
    host._prefill_rows = lambda mi: ([(i, rid, s0, e) for i, (rid, s0, e) in enumerate(mi.rows)]
                                     if mi.prompt_lens is not None else [])  # fmt: skip
    host.connector = SimpleNamespace(bind_connector_metadata=lambda m: None, clear_connector_metadata=lambda: None,
                                     handle_preemptions=lambda m: None)  # fmt: skip
    host._metadata, host._lock = (lambda so: None), threading.Lock()
    host.store_worker = SimpleNamespace(drain=lambda: None, busy=lambda: False)
    return host, pages


def _step(tag, *rows):
    so = SimpleNamespace(finished_req_ids=set())
    mi = SimpleNamespace(prompt_lens=[e for _r, _s, e in rows], row_req_ids=[r for r, _s, _e in rows], rows=list(rows),
                         tag=tag)  # fmt: skip
    return so, mi


A = list(range(100, 1100))
NXT = SimpleNamespace(finished_req_ids=set(), preempted_req_ids=set())


def _hold_a(host):
    for tag, s0, e in (("a1", 0, 128), ("a2", 128, 256), ("a3", 256, 384)):
        host.save(*_step(tag, ("A", s0, e)))
        host.begin_step(NXT)
    assert [mi.tag for _so, mi in host._held_pages] == ["a1", "a2", "a3"]


def test_request_finished_delays_the_block_free_and_registers_the_drain(monkeypatch):
    class Inner:
        def request_finished(self, request, block_ids):
            return False, {"x": 1}

    c = K.TTPageAlignedConnector(Inner(), 128, None)
    r = SimpleNamespace(request_id="A")
    assert c.request_finished(r, [3, 4]) == (True, {"x": 1})
    assert K._LEAVE_PENDING == {"A": {"inner": False, "inner_done": False}}
    monkeypatch.setenv("GLM53_KV_LEAVE_DRAIN", "0")
    assert c.request_finished(SimpleNamespace(request_id="B"), [5]) == (False, {"x": 1}) and "B" not in K._LEAVE_PENDING


def test_a_leaving_request_drains_one_held_step_per_engine_step_and_is_reported_last():
    host, pages = _host({"A": A})
    _hold_a(host)
    K._LEAVE_PENDING["A"] = {"inner": False, "inner_done": False}  # (the scheduler-side request_finished)
    host.begin_step(SimpleNamespace(finished_req_ids={"A"}, preempted_req_ids=set()))
    assert pages == ["a1"]  # NOT all three on this engine step
    assert host._leave_report(None) is None  # a2, a3 still held: its blocks stay allocated
    host.begin_step(NXT)
    assert pages == ["a1", "a2"] and host._leave_report(set()) is None
    host.begin_step(NXT)
    assert pages == ["a1", "a2", "a3"]
    assert host._leave_report({"other"}) == {"other", "A"} and K._LEAVE_PENDING == {}


def test_a_queued_store_job_keeps_the_request_until_the_worker_is_done():
    host, _pages = _host({"A": A})
    K._LEAVE_PENDING["A"] = {"inner": False, "inner_done": False}
    host._held_pages = []
    host._inflight_rids({"A"}, +1)
    assert host._leave_report(None) is None
    host._job_done({"rids": {"A"}})
    assert host._leave_report(None) == {"A"}


def test_a_preemption_or_a_lookup_still_flushes_everything():
    host, pages = _host({"A": A})
    _hold_a(host)
    K._LEAVE_PENDING["A"] = {"inner": False, "inner_done": False}
    host.begin_step(SimpleNamespace(finished_req_ids=set(), preempted_req_ids={"A"}))
    assert pages == ["a1", "a2", "a3"]
    K._LEAVE_PENDING.clear()
    host2, pages2 = _host({"A": A})
    _hold_a(host2)
    host2.flush_deferred_save()  # a lookup / no_forward / shutdown flush (no next step)
    assert pages2 == ["a1", "a2", "a3"]


def test_an_inner_connector_delay_is_reported_only_when_both_landed():
    host, _pages = _host({"A": A})
    host._held_pages = []
    K._LEAVE_PENDING["A"] = {"inner": True, "inner_done": False}
    assert host._leave_report(None) is None  # the inner connector has not reported it yet
    assert host._leave_report({"A"}) == {"A"}  # it did, and nothing of this host holds it


def test_off_by_default_the_leaving_flush_is_unchanged(monkeypatch):
    monkeypatch.setenv("GLM53_KV_LEAVE_DRAIN", "0")
    host, pages = _host({"A": A})
    _hold_a(host)
    host.begin_step(SimpleNamespace(finished_req_ids={"A"}, preempted_req_ids=set()))
    assert pages == ["a1", "a2", "a3"]


# --- program 2026-10-26T22:40:55Z (b): LEAVE_DRAIN must not hold back admitting a waiting request (the k3 ladder's
# cold rung after a stored one: +0.22 / +0.51 s between its arrival and its LMCache lookup) -- admit first, drain between
# steps.

OTHER = list(range(5000, 5300))  # a prompt sharing no first LMCache chunk with A (each ladder rung has its own head)
LEFT = {"inner": False, "inner_done": False}


def _gone_a(hist, host):
    _hold_a(host)
    K._LEAVE_PENDING["A"] = dict(LEFT)
    hist.pop("A")  # A finished: the runner forgot it, its held steps did not


def test_a_gone_request_keeps_its_head_so_an_unrelated_lookup_does_not_flush_it(monkeypatch):
    monkeypatch.setenv("GLM53_KV_LOOKUP_FLUSH", "prefix")
    hist = {"A": A}
    host, pages = _host(hist)
    K._SAVE_HOSTS.add(host)
    _gone_a(hist, host)
    # before: the unknown head matched every prompt, so the next lookup flushed + drained A's stores in front of it
    assert not host.serves_prefix(OTHER)
    K.flush_deferred_saves(SimpleNamespace(request_id="B", prompt_token_ids=OTHER))
    assert pages == [] and host.lookup_flushes_skipped == 1
    # a prompt that shares A's first chunk still sees every store of its prefix (a hit must)
    assert host.serves_prefix(A[:300])
    K.flush_deferred_saves(SimpleNamespace(request_id="C", prompt_token_ids=A[:300]))
    assert pages == ["a1", "a2", "a3"]
    assert "A" not in host.__dict__.get("_rid_heads", {})  # pruned once nothing refers to it


def test_a_store_job_queued_after_its_request_left_records_the_real_head():
    hist = {"A": A}
    host, _pages = _host(hist)
    _gone_a(hist, host)
    heads = host._queue_heads(SimpleNamespace(requests=[SimpleNamespace(req_id="A")]), +1)
    assert heads == [tuple(A[:CHUNK])]  # (was None: an unknown head, a match for every prompt)
    host._held_pages = []
    assert not host.serves_prefix(OTHER) and host.serves_prefix(A[:CHUNK])
    host._queue_heads(None, -1, heads)
    assert not host.serves_prefix(A[:CHUNK])


def _admit(*finished):
    return SimpleNamespace(finished_req_ids=set(finished), preempted_req_ids=set(), scheduled_new_reqs=[object()],
                           total_num_scheduled_tokens=2048)  # fmt: skip


def test_the_drain_yields_to_a_step_that_admits_a_new_request(monkeypatch):
    monkeypatch.setenv("GLM53_KV_LEAVE_YIELD_STEPS", "2")
    hist = {"A": A}
    host, pages = _host(hist)
    _gone_a(hist, host)
    host.begin_step(_admit("A"))
    assert pages == []  # the new request's first chunk runs without A's store in front of it
    host.begin_step(_admit())
    assert pages == []
    host.begin_step(_admit())
    assert pages == ["a1"]  # bounded: past GLM53_KV_LEAVE_YIELD_STEPS in a row the drain goes on
    host.begin_step(NXT)
    assert pages == ["a1", "a2"] and host.leave_yields_admit == 2
    host.begin_step(NXT)
    assert pages == ["a1", "a2", "a3"] and host._leave_report(None) == {"A"}


def test_an_empty_step_right_after_the_leave_waits_the_grace(monkeypatch):
    empty = SimpleNamespace(finished_req_ids=set(), preempted_req_ids=set(), scheduled_new_reqs=[],
                            total_num_scheduled_tokens=0)  # fmt: skip
    monkeypatch.setenv("GLM53_KV_LEAVE_GRACE_MS", "60000")
    hist = {"A": A}
    host, pages = _host(hist)
    _gone_a(hist, host)
    host.begin_step(SimpleNamespace(**{**empty.__dict__, "finished_req_ids": {"A"}}))
    host.begin_step(empty)
    assert pages == [] and host.leave_yields_grace == 2  # the engine thread stays free for a back-to-back request
    monkeypatch.setenv("GLM53_KV_LEAVE_GRACE_MS", "0")
    host.begin_step(empty)
    assert pages == ["a1"]
    # a step that runs a forward (decode rows, no new request) drains as before
    host.begin_step(SimpleNamespace(**{**empty.__dict__, "total_num_scheduled_tokens": 3}))
    assert pages == ["a1", "a2"]


def test_a_slow_lookup_logs_its_split(monkeypatch):
    import time as _time

    monkeypatch.setenv("GLM53_KV_LOOKUP_LOG_MS", "20")
    monkeypatch.setattr(K, "flush_deferred_saves", lambda request=None: None)
    lines = []
    monkeypatch.setattr(K.logger, "info", lambda msg, *args: lines.append(msg % args))

    class Inner:
        def get_num_new_matched_tokens(self, request, n):
            _time.sleep(0.03)
            return 0, False

    c = K.TTPageAlignedConnector(Inner(), 128, None)
    r = SimpleNamespace(request_id="slow", prompt_token_ids=[1, 2, 3], num_prompt_tokens=3,
                        sampling_params=SimpleNamespace(extra_args=None), kv_transfer_params=None)  # fmt: skip
    assert c.get_num_new_matched_tokens(r, 0) == (0, False)
    got = [x for x in lines if x.startswith("TT KV connector lookup (slow)")]
    assert got and "lmcache 3" in got[0] and "flush" in got[0], lines
    lines.clear()
    monkeypatch.setenv("GLM53_KV_LOOKUP_LOG_MS", "0")  # off
    c.get_num_new_matched_tokens(r, 0)
    assert not [x for x in lines if x.startswith("TT KV connector lookup")]
