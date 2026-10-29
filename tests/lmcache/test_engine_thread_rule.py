"""The engine-thread rule (program 2026-10-28T09:22:52Z FIX ORDER, the k4d8t4 L1-floor trip): no plugin thread but
the engine thread issues ttnn / device work. A deferred KDA producer that calls ttnn on the resolver / writer is refused
before the call runs, handed back to the engine thread (drain_engine_todo at the next _save_state / save_deferred, or a
load() there), and the writer then writes its file. GLM53_KDA_CKPT_HOST_ONLY=0 turns the refusal off;
GLM53_KV_DEVICE_THREAD_AUDIT=1 only logs the store worker's ttnn call sites."""

import logging
import threading
import types

import pytest
import torch

from vllm_tt_plugin import kv_connector as K

# a stand-in for a ttnn function: a Python function whose module is under the ttnn package (the guard's 'call' rule)
_FAKE = types.ModuleType("ttnn.fake_device")
exec(  # noqa: S102
    "import threading\nCALLS = []\ndef device_op(v):\n    CALLS.append(threading.current_thread().name)\n    return v\n",
    _FAKE.__dict__,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("GLM53_KDA_CKPT_HOST_ONLY", "GLM53_KV_DEVICE_THREAD_AUDIT", "GLM53_KDA_CKPT_RESOLVERS",
              "GLM53_KDA_CKPT_COLLECT_SERIAL", "GLM53_KDA_CKPT_LOCAL_DIR", "GLM53_KDA_CKPT_ENGINE_WAIT_S"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GLM53_KDA_CKPT_MIN_FREE_GB", "0")
    monkeypatch.setattr(K, "_ENGINE", {"ident": None})
    monkeypatch.setattr(K, "_ENGINE_TODO", [])
    monkeypatch.setattr(K, "_OFF_ENGINE_SEEN", set())
    monkeypatch.setattr(K, "_ENGINE_STATS", {"refused": 0, "handbacks": 0, "drained": 0, "audited": 0})
    _FAKE.CALLS.clear()


def _states(v):
    return [{"L": 0, "S": torch.full((2, 2), float(v))}]


def _device_producer(v):
    def produce():
        _FAKE.device_op(v)  # "device work" inside the producer
        return _states(v)

    return produce


def test_a_ttnn_call_on_the_resolver_is_refused_and_runs_on_the_engine_thread(tmp_path):
    K.mark_engine_thread()  # this test's thread steps the "model"
    st = K.KDACheckpointStore(str(tmp_path / "s"))
    toks = list(range(500))
    st.save_deferred(toks, 128, _device_producer(3))
    st._rqueue.join()
    assert _FAKE.CALLS == [] and K._ENGINE_STATS["refused"] >= 1 and len(K._ENGINE_TODO) == 1  # never ran there
    assert st.stats.get("engine_handbacks") == 1
    assert K.drain_engine_todo() == 1  # the engine thread's next step
    assert _FAKE.CALLS == [threading.current_thread().name]
    st.flush()  # the writer waited for the engine thread, then wrote the file
    assert torch.equal(st.load(toks, 128)[0]["S"], _states(3)[0]["S"])
    st2 = K.KDACheckpointStore(str(tmp_path / "s"))  # from the file
    assert torch.equal(st2.load(toks, 128)[0]["S"], _states(3)[0]["S"])


def test_a_load_on_the_engine_thread_runs_a_handed_back_producer_inline(tmp_path):
    K.mark_engine_thread()
    st = K.KDACheckpointStore(str(tmp_path / "s"))
    toks = list(range(500))
    st.save_deferred(toks, 256, _device_producer(4))
    st._rqueue.join()
    assert _FAKE.CALLS == []
    assert torch.equal(st.load(toks, 256)[0]["S"], _states(4)[0]["S"])  # the hit: produced here, on the engine
    assert _FAKE.CALLS == [threading.current_thread().name]
    st.flush()


def test_host_only_producers_still_resolve_ahead(tmp_path):
    K.mark_engine_thread()
    st = K.KDACheckpointStore(str(tmp_path / "s"))
    toks = list(range(500))
    st.save_deferred(toks, 128, lambda: _states(5))
    st._rqueue.join()
    assert st.stats.get("resolved_ahead") == 1 and not K._ENGINE_TODO and K._ENGINE_STATS["refused"] == 0
    st.flush()


def test_the_refusal_is_a_knob(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_KDA_CKPT_HOST_ONLY", "0")
    K.mark_engine_thread()
    st = K.KDACheckpointStore(str(tmp_path / "s"))
    st.save_deferred(list(range(500)), 128, _device_producer(6))
    st._rqueue.join()
    assert _FAKE.CALLS and _FAKE.CALLS[0].startswith("tt-kda-ckpt-")  # the old behaviour: the resolver or the writer
    st.flush()


def test_the_writer_full_queue_put_drains_on_the_engine_thread(tmp_path, monkeypatch):
    """the writer waits for a handed-back producer while the engine thread's put waits for queue space: the put
    drains (no deadlock)"""
    monkeypatch.setenv("GLM53_KDA_CKPT_QUEUE", "1")
    K.mark_engine_thread()
    st = K.KDACheckpointStore(str(tmp_path / "s"))
    toks = list(range(900))
    for n in (128, 256, 384, 512):
        st.save_deferred(toks, n, _device_producer(n))
    st.flush()  # (flush from the engine thread: the queue drained by the puts' drains and the writer)
    K.drain_engine_todo()
    st.flush()
    for n in (128, 256, 384, 512):
        assert torch.equal(st.load(toks, n)[0]["S"], _states(n)[0]["S"])
    assert set(_FAKE.CALLS) == {threading.current_thread().name}


def test_the_callee_names():
    assert K._ttnn_callee(None, "c_call", types.SimpleNamespace(__module__="ttnn._ttnn.device", __qualname__="synchronize_device")) \
        == "ttnn._ttnn.device.synchronize_device"
    assert K._ttnn_callee(None, "c_call", types.SimpleNamespace(__module__="numpy", __qualname__="copy")) is None
    assert K._ttnn_callee(None, "return", None) is None


def test_the_store_worker_audit_logs_and_never_refuses():
    lines = []
    h = logging.Handler()
    h.emit = lambda rec: lines.append(rec.getMessage())
    K.logger.addHandler(h)
    out = []

    def job():
        with K._host_only("store job", refuse=False):
            out.append(_FAKE.device_op(7))
            out.append(_FAKE.device_op(8))

    try:
        th = threading.Thread(target=job, name="tt-kv-store-worker")
        th.start()
        th.join()
    finally:
        K.logger.removeHandler(h)
    assert out == [7, 8] and K._ENGINE_STATS["audited"] == 2
    logged = [x for x in lines if "engine-thread rule" in x]
    assert len(logged) == 1 and "ttnn.fake_device.device_op" in logged[0] and "tt-kv-store-worker" in logged[0], logged
