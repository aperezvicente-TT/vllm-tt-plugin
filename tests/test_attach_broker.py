# SPDX-License-Identifier: Apache-2.0
"""The standing engine's attach broker (vllm_tt_plugin.attach_broker; doc/ttnn/12-fast-start/standing_engine/design.md):
host tests over vLLM 0.26's REAL protocol code -- the engine side runs EngineCoreProc.startup_handshake and the
engine's input / output socket conventions, the frontend side runs vllm.v1.engine.utils.wait_for_engine_startup and
MPClient's ready-payload wait. A frontend attaches, a request is relayed, the frontend goes away (its requests are
aborted on the engine), a second frontend re-attaches to the same engine (the ready payload replayed), and the
engine's death is forwarded. No device, no vLLM engine."""

from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace

import msgspec
import pytest
import zmq

from vllm_tt_plugin.attach_broker import ENGINE_CORE_DEAD, AttachBroker

pytest.importorskip("vllm.v1.engine.core")
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType  # noqa: E402
from vllm.v1.engine.core import EngineCoreProc  # noqa: E402
from vllm.v1.engine.utils import CoreEngine, EngineZmqAddresses, wait_for_engine_startup  # noqa: E402
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder  # noqa: E402

IDENT = (0).to_bytes(2, "little")


def _port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _Engine:
    """the headless engine's side of the protocol (vLLM's startup_handshake + its socket conventions)"""

    def __init__(self, ctx, port):
        self.hs = ctx.socket(zmq.DEALER)
        self.hs.setsockopt(zmq.IDENTITY, IDENT)
        self.hs.setsockopt(zmq.LINGER, 0)
        self.hs.connect(f"tcp://127.0.0.1:{port}")
        self.addresses = EngineCoreProc.startup_handshake(self.hs, local_client=False, headless=True)
        self.hs.send(msgspec.msgpack.encode({"status": "READY", "local": False, "headless": True}))
        self.inp = ctx.socket(zmq.DEALER)
        self.inp.setsockopt(zmq.IDENTITY, IDENT)
        self.inp.setsockopt(zmq.LINGER, 0)
        self.inp.connect(self.addresses.inputs[0])
        self.ready = msgspec.msgpack.encode({"max_model_len": 4096, "num_gpu_blocks": 77, "block_size": 64})
        self.inp.send(self.ready)  # (EngineCoreReadyResponse's bytes: opaque to the broker)
        self.out = ctx.socket(zmq.PUSH)
        self.out.setsockopt(zmq.LINGER, 0)
        self.out.connect(self.addresses.outputs[0])

    def recv(self, timeout=5000):
        assert self.inp.poll(timeout), "the engine received nothing"
        return self.inp.recv_multipart()


class _Frontend:
    """an API-only frontend's side (wait_for_engine_startup over its handshake ROUTER, then MPClient's ready wait)"""

    def __init__(self, ctx, port):
        self.ctx = ctx
        self.inp = ctx.socket(zmq.ROUTER)
        self.inp.setsockopt(zmq.LINGER, 0)
        self.inp_addr = "ipc:///tmp/tt_attach_test_in_%d_%d" % (port, id(self))
        self.inp.bind(self.inp_addr)
        self.out = ctx.socket(zmq.PULL)
        self.out.setsockopt(zmq.LINGER, 0)
        self.out_addr = "ipc:///tmp/tt_attach_test_out_%d_%d" % (port, id(self))
        self.out.bind(self.out_addr)
        self.hs = ctx.socket(zmq.ROUTER)
        self.hs.setsockopt(zmq.LINGER, 0)
        self.hs.bind(f"tcp://127.0.0.1:{port}")

    def attach(self):
        pc = SimpleNamespace(data_parallel_size_local=0, data_parallel_hybrid_lb=False, data_parallel_external_lb=False)
        wait_for_engine_startup(self.hs, EngineZmqAddresses(inputs=[self.inp_addr], outputs=[self.out_addr]),
                                [CoreEngine(index=0, local=False)], pc, False, None, None, None)  # fmt: skip
        assert self.inp.poll(5000), "no ready payload replayed"
        ident, payload = self.inp.recv_multipart()  # MPClient's wait
        assert ident == IDENT
        return payload

    def send(self, rtype, obj):
        self.inp.send_multipart([IDENT, rtype.value, *MsgpackEncoder().encode(obj)])

    def recv_out(self, timeout=5000):
        assert self.out.poll(timeout), "the frontend received no output"
        return self.out.recv_multipart()

    def close(self):
        for s in (self.inp, self.out, self.hs):
            s.close(linger=0)


@pytest.fixture
def broker():
    ep, fp = _port(), _port()
    b = AttachBroker(ep, fp, poll_ms=10)
    rc = {}
    t = threading.Thread(target=lambda: rc.setdefault("rc", b.run()), daemon=True)
    t.start()
    ctx = zmq.Context()
    yield b, ep, fp, ctx, rc, t
    b.stop()
    t.join(5)
    ctx.destroy(linger=0)


def _req(rid):
    return EngineCoreRequest(request_id=rid, prompt_token_ids=[1, 2, 3], mm_features=None, sampling_params=None,
                             pooling_params=None, arrival_time=0.0, lora_request=None, cache_salt=None,
                             data_parallel_rank=None)  # fmt: skip


def test_attach_relay_detach_reattach_and_engine_death(broker):
    b, ep, fp, ctx, rc, t = broker
    eng = _Engine(ctx, ep)  # the engine's one handshake, with the broker
    fe1 = _Frontend(ctx, fp)
    assert fe1.attach() == eng.ready  # the frontend sees a starting engine
    # a request through the broker, an output back
    fe1.send(EngineCoreRequestType.ADD, _req("r1"))
    rtype, *data = eng.recv()
    assert rtype == EngineCoreRequestType.ADD.value and MsgpackDecoder(EngineCoreRequest).decode(data).request_id == "r1"
    eng.out.send_multipart([b"outputs-1", b"aux"])
    assert fe1.recv_out() == [b"outputs-1", b"aux"]
    # the frontend goes away: its request is aborted on the engine
    fe1.close()
    rtype, *data = eng.recv()
    assert rtype == EngineCoreRequestType.ABORT.value and MsgpackDecoder().decode(data) == ["r1"]
    assert b.stats["detaches"] == 1 and b.stats["aborted"] == 1
    # a second frontend, the same port: re-attaches to the same engine (no new engine handshake)
    fe2 = _Frontend(ctx, fp)
    t0 = time.monotonic()
    assert fe2.attach() == eng.ready
    assert time.monotonic() - t0 < 3.0 and b.stats["attaches"] == 2
    fe2.send(EngineCoreRequestType.ADD, _req("r2"))
    rtype, *data = eng.recv()
    assert MsgpackDecoder(EngineCoreRequest).decode(data).request_id == "r2"
    eng.out.send_multipart([b"outputs-2"])
    assert fe2.recv_out() == [b"outputs-2"]
    # the engine dies: forwarded, the broker exits
    eng.out.send(ENGINE_CORE_DEAD)
    assert fe2.recv_out() == [ENGINE_CORE_DEAD]
    t.join(5)
    assert rc.get("rc") == 1
    fe2.close()


def test_outputs_without_a_frontend_are_dropped(broker):
    b, ep, fp, ctx, rc, t = broker
    eng = _Engine(ctx, ep)
    deadline = time.monotonic() + 5
    while b.ready_payload is None and time.monotonic() < deadline:
        time.sleep(0.01)
    eng.out.send_multipart([b"late output of an aborted request"])
    deadline = time.monotonic() + 5
    while b.stats["dropped_out"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert b.stats["dropped_out"] == 1
    fe = _Frontend(ctx, fp)
    assert fe.attach() == eng.ready  # a frontend that comes later still attaches
    fe.close()


def test_finished_and_frontend_aborted_ids_leave_the_attach(broker):
    """a request that finished (an output with a finish reason) or that the frontend aborted itself is not aborted again
    on a loss: only the still-running ones are (else a long-lived frontend's loss would abort every id it ever served)"""
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason

    b, ep, fp, ctx, rc, t = broker
    eng = _Engine(ctx, ep)
    fe = _Frontend(ctx, fp)
    assert fe.attach() == eng.ready
    for rid in ("done", "cancelled", "running"):
        fe.send(EngineCoreRequestType.ADD, _req(rid))
        eng.recv()
    outs = EngineCoreOutputs(outputs=[EngineCoreOutput(request_id="done", new_token_ids=[5], finish_reason=FinishReason.LENGTH),
                                      EngineCoreOutput(request_id="running", new_token_ids=[6])])  # fmt: skip
    eng.out.send_multipart(MsgpackEncoder().encode(outs))
    fe.recv_out()
    fe.send(EngineCoreRequestType.ABORT, ["cancelled"])
    rtype, *data = eng.recv()
    assert rtype == EngineCoreRequestType.ABORT.value and MsgpackDecoder().decode(data) == ["cancelled"]
    assert b.stats["finished"] == 1
    fe.close()
    rtype, *data = eng.recv()
    assert rtype == EngineCoreRequestType.ABORT.value and MsgpackDecoder().decode(data) == ["running"]
    assert b.stats["aborted"] == 1
