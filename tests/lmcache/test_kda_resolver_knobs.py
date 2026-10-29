"""Program 2026-10-28T08:07:52Z (3) + (2): on the k4d8 line the deferred KDA exports resolve 6-13x slower than on
k4d5p, and a 32,000-token repeat's load waited 3.35 s for its pending 31,872 checkpoint.
- GLM53_KDA_CKPT_RESOLVERS=N: N resolver threads (no queue behind an unrelated export).
- GLM53_KV_LOOKUP_DEFER_KDA=1: a lookup whose prefix has a deferred export still being produced is undetermined (None),
  so the engine keeps stepping instead of blocking inside the load; bounded by GLM53_KV_LOOKUP_DEFER_MAX_S.
- GLM53_KDA_CKPT_RESOLVER_SAMPLE_S: the producer's Python stack sampled in-process (no ptrace) and logged."""

import logging
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from vllm_tt_plugin import kv_connector as K


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("GLM53_KDA_CKPT_RESOLVERS", "GLM53_KV_LOOKUP_DEFER_KDA", "GLM53_KDA_CKPT_RESOLVER_SAMPLE_S",
              "GLM53_KDA_CKPT_RESOLVER_SAMPLE_AFTER_S", "GLM53_KDA_CKPT_LOCAL_DIR", "GLM53_KDA_CKPT_RECENT_N",
              "GLM53_KV_LOOKUP_DEFER", "GLM53_KV_LOOKUP_DEFER_MAX_S", "GLM53_KDA_CKPT_COLLECT_SERIAL"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GLM53_KDA_CKPT_MIN_FREE_GB", "0")
    monkeypatch.setattr(K, "_KDA_LOOKUP_DEFERRED", {})
    monkeypatch.setattr(K, "_SAVE_HOSTS", set())


def _states(v):
    return [{"L": 0, "S": torch.full((2, 2), float(v))}]


def _slow(v, dt):
    def produce_states_slowly():
        time.sleep(dt)
        return _states(v)

    return produce_states_slowly


def test_two_resolvers_run_two_slow_exports_side_by_side(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_KDA_CKPT_COLLECT_SERIAL", "0")  # (the gate off: the collects may overlap)
    toks = list(range(1000))
    for n_res, bound in ((1, 0.55), (2, 0.45)):
        monkeypatch.setenv("GLM53_KDA_CKPT_RESOLVERS", str(n_res))
        st = K.KDACheckpointStore(str(tmp_path / f"r{n_res}"))
        t0 = time.perf_counter()
        st.save_deferred(toks, 128, _slow(1, 0.3))
        st.save_deferred(toks, 256, _slow(2, 0.3))
        st._rqueue.join()
        dt = time.perf_counter() - t0
        assert len(st._resolvers) == n_res
        assert (dt >= bound) if n_res == 1 else (dt < bound), (n_res, dt)
        st.flush()
        assert torch.equal(st.load(toks, 256)[0]["S"], _states(2)[0]["S"])


def test_pending_prefix_names_an_unproduced_export_of_the_prompt(tmp_path):
    st = K.KDACheckpointStore(str(tmp_path / "s"))
    toks = list(range(600))
    go = threading.Event()

    def blocked():
        go.wait(5)
        return _states(3)

    st.save_deferred(toks, 384, blocked)
    assert st.pending_prefix(toks + [1, 2]) == (384, st.path(toks, 384))
    assert st.pending_prefix(toks[:100]) is None and st.pending_prefix([7] + toks) is None
    go.set()
    st._rqueue.join()
    assert st.pending_prefix(toks) is None  # produced: a hit takes it from the RAM tier
    st.flush()


class _Inner:
    def __init__(self):
        self.calls = 0

    def get_num_new_matched_tokens(self, request, n):
        self.calls += 1
        return 0, False


def _req(tokens, rid="B"):
    return SimpleNamespace(request_id=rid, prompt_token_ids=tokens, num_prompt_tokens=len(tokens),
                           sampling_params=SimpleNamespace(extra_args=None), kv_transfer_params=None)  # fmt: skip


def test_the_lookup_waits_for_the_pending_export_then_goes_ahead(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_KV_LOOKUP_DEFER_KDA", "1")
    st = K.KDACheckpointStore(str(tmp_path / "s"))
    toks = list(range(700))
    go = threading.Event()
    st.save_deferred(toks, 512, lambda: (go.wait(5), _states(4))[1])
    inner = _Inner()
    c = K.TTPageAlignedConnector(inner, 128, st)
    r = _req(toks + [9])
    assert c.get_num_new_matched_tokens(r, 0) == (None, False) and inner.calls == 0
    assert c.get_num_new_matched_tokens(r, 0) == (None, False) and c.lookups_deferred_kda == 2
    go.set()
    st._rqueue.join()
    c.get_num_new_matched_tokens(r, 0)
    assert inner.calls == 1 and "B" not in K._KDA_LOOKUP_DEFERRED
    st.flush()


def test_off_by_default_and_bounded(tmp_path, monkeypatch):
    st = K.KDACheckpointStore(str(tmp_path / "s"))
    toks = list(range(700))
    go = threading.Event()
    st.save_deferred(toks, 512, lambda: (go.wait(5), _states(5))[1])
    inner = _Inner()
    c = K.TTPageAlignedConnector(inner, 128, st)
    c.get_num_new_matched_tokens(_req(toks + [9], "C"), 0)
    assert inner.calls == 1  # the knob off: no KDA deferral
    monkeypatch.setenv("GLM53_KV_LOOKUP_DEFER_KDA", "1")
    monkeypatch.setenv("GLM53_KV_LOOKUP_DEFER_MAX_S", "0")
    K._KDA_LOOKUP_DEFERRED["D"] = time.monotonic() - 1.0  # deferred past the bound
    c.get_num_new_matched_tokens(_req(toks + [9], "D"), 0)
    assert inner.calls == 2 and "D" not in K._KDA_LOOKUP_DEFERRED
    go.set()
    st.flush()


def test_the_sampler_logs_the_producers_stack(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_KDA_CKPT_RESOLVER_SAMPLE_S", "0.02")
    monkeypatch.setenv("GLM53_KDA_CKPT_RESOLVER_SAMPLE_AFTER_S", "0")
    lines = []
    h = logging.Handler()
    h.emit = lambda rec: lines.append(rec.getMessage())
    K.logger.addHandler(h)
    old = K.logger.level
    K.logger.setLevel(logging.INFO)
    try:
        st = K.KDACheckpointStore(str(tmp_path / "s"))
        st.save_deferred(list(range(300)), 128, _slow(6, 0.3))
        st._rqueue.join()
        st.flush()
    finally:
        K.logger.removeHandler(h)
        K.logger.setLevel(old)
    got = [x for x in lines if "producer:" in x]
    assert got and "produce_states_slowly" in got[0] and "stack samples every 0.02 s" in got[0], got


def test_one_collect_at_a_time_and_a_hit_goes_first(tmp_path, monkeypatch):
    """GLM53_KDA_CKPT_COLLECT_SERIAL=1 (opt-in since 2026-10-28T11:22Z): two resolvers never collect at once; a load()
    needing a pending checkpoint collects before the queued ones"""
    monkeypatch.setenv("GLM53_KDA_CKPT_RESOLVERS", "2")
    monkeypatch.setenv("GLM53_KDA_CKPT_COLLECT_SERIAL", "1")
    order, live, peak = [], [0], [0]
    lock = threading.Lock()

    def prod(tag, dt):
        def f():
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            time.sleep(dt)
            with lock:
                live[0] -= 1
                order.append(tag)
            return _states(1)
        return f

    st = K.KDACheckpointStore(str(tmp_path / "s"))
    toks = list(range(900))
    t0 = time.perf_counter()
    st.save_deferred(toks, 128, prod("a", 0.25))
    st.save_deferred(toks, 256, prod("b", 0.25))
    st._rqueue.join()
    assert peak[0] == 1 and time.perf_counter() - t0 >= 0.45  # serialized
    # a hit's collect goes before a queued one: c is being collected, d and e wait; the load of e comes first
    order.clear()
    st.save_deferred(toks, 384, prod("c", 0.3))
    time.sleep(0.05)
    st.save_deferred(toks, 512, prod("d", 0.05))
    d_e = K._Deferred(prod("e", 0.05), "e")  # a pending checkpoint a load needs (not on the resolver queue)
    th = threading.Thread(target=lambda: d_e.result(prio=True))
    time.sleep(0.05)
    th.start()
    st._rqueue.join()
    th.join()
    assert order[0] == "c" and order.index("e") < order.index("d"), order
    st.flush()


def test_the_sampler_splits_dma_wait_from_host_work(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_KDA_CKPT_RESOLVER_SAMPLE_S", "0.02")
    monkeypatch.setenv("GLM53_KDA_CKPT_RESOLVER_SAMPLE_AFTER_S", "0")
    lines = []
    h = logging.Handler()
    h.emit = lambda rec: lines.append(rec.getMessage())
    K.logger.addHandler(h)
    old = K.logger.level
    K.logger.setLevel(logging.INFO)

    def fetch_like():  # a poll sleep (the DMA wait) then host copies
        t = time.time()
        while time.time() - t < 0.2:
            time.sleep(0.0005)
        a = [list(range(2000)) for _ in range(200)]
        return a

    try:
        st = K.KDACheckpointStore(str(tmp_path / "s"))
        st.save_deferred(list(range(300)), 128, lambda: (fetch_like(), _states(2))[1])
        st._rqueue.join()
        st.flush()
    finally:
        K.logger.removeHandler(h)
        K.logger.setLevel(old)
    got = [x for x in lines if "producer:" in x]
    assert got and "dma_wait" in got[0] and " pct" in got[0], got


def test_the_store_worker_yields_to_an_in_flight_state_save(tmp_path, monkeypatch):
    """GLM53_KV_STORE_YIELD_TO_STATE_SAVE (program 2026-10-28T08:20:18Z): no new page-store job while a deferred
    state save is on the rings, bounded by GLM53_KV_STORE_YIELD_MAX_S; off by default"""
    monkeypatch.setattr(K, "_STATE_SAVES", {"n": 0, "cv": threading.Condition()})
    st = K.KDACheckpointStore(str(tmp_path / "s"))
    toks = list(range(400))
    assert K.yield_to_state_saves() == 0.0  # off
    monkeypatch.setenv("GLM53_KV_STORE_YIELD_TO_STATE_SAVE", "1")
    st.save_deferred(toks, 128, _slow(1, 0.3))
    host = SimpleNamespace()
    waited = K.yield_to_state_saves(host)
    assert 0.15 < waited < 2.0 and host.store_yields == 1 and K._STATE_SAVES["n"] == 0
    monkeypatch.setenv("GLM53_KV_STORE_YIELD_MAX_S", "0.05")
    st.save_deferred(toks, 256, _slow(2, 0.5))
    assert K.yield_to_state_saves() < 0.2  # bounded
    st._rqueue.join()
    st.flush()
    assert K._STATE_SAVES["n"] == 0


def test_the_sampler_never_holds_another_threads_frames(tmp_path, monkeypatch):
    """2026-10-28T09:49Z (the k4d8t4 L1-floor trip): the v1 sampler held sys._current_frames() -- every thread's frame,
    the engine thread's included -- across its sleep, so an engine function's locals (live ttnn L1 tensors) outlived
    its return. The in-thread sampler references no other thread's frame: a local is freed when its function returns."""
    import weakref

    monkeypatch.setenv("GLM53_KDA_CKPT_RESOLVER_SAMPLE_S", "0.005")
    monkeypatch.setenv("GLM53_KDA_CKPT_RESOLVER_SAMPLE_AFTER_S", "0")

    class Big:
        pass

    def engine_step(refs):
        x = Big()  # an engine-thread local (a ttnn tensor in the served engine)
        refs.append(weakref.ref(x))
        time.sleep(0.02)  # the sampler would pass while this frame is current

    st = K.KDACheckpointStore(str(tmp_path / "s"))
    st.save_deferred(list(range(300)), 128, _slow(1, 0.6))  # a producer sampled for ~0.6 s
    refs = []
    for _ in range(20):
        engine_step(refs)
        assert refs[-1]() is None  # freed at return, never pinned by the sampler
    st._rqueue.join()
    st.flush()


def test_the_collect_gate_is_off_by_default_and_counted_when_on(tmp_path, monkeypatch):
    """k4d8t4wp8 (2026-10-28T11:22Z): COLLECT_SERIAL=1 queued a 32K prompt's second collect 1.701 s behind the first and
    the stored-hit repeat waited for the chain -> default 0; the gate's waits and the collects are counters"""
    monkeypatch.setattr(K, "_COLLECT_STATS", {"kda_collects": 0, "kda_collect_s": 0.0, "kda_collect_gate_waits": 0,
                                              "kda_collect_gate_wait_s": 0.0})  # fmt: skip
    monkeypatch.setenv("GLM53_KDA_CKPT_RESOLVERS", "2")
    toks = list(range(800))
    st = K.KDACheckpointStore(str(tmp_path / "off"))
    t0 = time.perf_counter()
    st.save_deferred(toks, 128, _slow(1, 0.3))
    st.save_deferred(toks, 256, _slow(2, 0.3))
    st._rqueue.join()
    assert time.perf_counter() - t0 < 0.5  # default: the two collects overlap
    assert K._COLLECT_STATS["kda_collects"] == 2 and K._COLLECT_STATS["kda_collect_gate_waits"] == 0
    st.flush()
    monkeypatch.setenv("GLM53_KDA_CKPT_COLLECT_SERIAL", "1")
    st2 = K.KDACheckpointStore(str(tmp_path / "on"))
    st2.save_deferred(toks, 384, _slow(3, 0.3))
    st2.save_deferred(toks, 512, _slow(4, 0.3))
    st2._rqueue.join()
    assert K._COLLECT_STATS["kda_collect_gate_waits"] == 1 and K._COLLECT_STATS["kda_collect_gate_wait_s"] > 0.2
    snap = K.counters_snapshot()
    assert snap["kda_collects"] == 4 and snap["kda_collect_gate_waits"] == 1, snap
    st2.flush()
