# SPDX-License-Identifier: Apache-2.0
"""The standing engine's attach broker (Stage 12, program 2026-10-27T05:33:29Z; doc/ttnn/12-fast-start/
standing_engine/design.md): a small long-lived, host-only process between a HEADLESS vLLM engine core and whichever
API-server frontend is current, so a frontend restart (~15 s of its own init) re-attaches to the running engine
instead of a ~4-5-min full restart. vLLM 0.26's engine handshake is one-shot (core.py _perform_handshakes), so the
broker speaks vLLM's own ZMQ protocol on both faces, with no vLLM patch:

  engine face (the broker is the engine's frontend, once, for good):
    - binds the handshake ROUTER at tcp://<host>:<engine_port> (the engine runs `vllm serve --headless
      --data-parallel-address <host> --data-parallel-rpc-port <engine_port>`); answers its HELLO with
      EngineHandshakeMetadata(addresses = the broker's own input ROUTER / output PULL, parallel_config = {}: DP 1);
    - stores the engine's first input-socket message: the msgpack EngineCoreReadyResponse;
  frontend face (the broker is the engine, once per frontend):
    - a DEALER with the engine's identity connects to the frontend's handshake ROUTER at tcp://<host>:<frontend_port>
      (the frontend runs `vllm serve --data-parallel-size-local 0 --data-parallel-address <host>
      --data-parallel-rpc-port <frontend_port>`): HELLO, the frontend's addresses, READY;
    - a DEALER (the same identity) to the frontend's input ROUTER replays the stored ready payload (MPClient waits
      for exactly that), and a PUSH to its output PULL carries the engine's outputs;
  steady state: frontend requests [type, *data] -> the engine as [identity, type, *data]; engine outputs unchanged ->
    the frontend; every ADDed request id is kept for the attach until it finishes (an output with a finish reason,
    or the frontend's own ABORT of it);
  a frontend's loss (zmq monitor: its input socket disconnected): ABORT of the attach's request ids to the engine
    (idempotent in the scheduler), the frontend sockets closed, a HELLO re-armed for the next frontend;
  ENGINE_CORE_DEAD from the engine: forwarded, then the broker exits.

Scope: DP 1, one frontend at a time. usage: python -m vllm_tt_plugin.attach_broker --engine-port P --frontend-port Q
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

import msgspec
import zmq
from zmq.utils.monitor import recv_monitor_message

ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"
log = logging.getLogger("tt_attach_broker")


def _vllm():
    """vLLM's own protocol types (imported late: the broker's host tests stub nothing, they use the real ones)"""
    from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest, EngineCoreRequestType  # noqa: PLC0415
    from vllm.v1.engine.utils import EngineHandshakeMetadata, EngineZmqAddresses  # noqa: PLC0415
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder  # noqa: PLC0415

    return EngineCoreRequest, EngineCoreRequestType, EngineHandshakeMetadata, EngineZmqAddresses, MsgpackDecoder, \
        MsgpackEncoder, EngineCoreOutputs  # fmt: skip


class AttachBroker:
    """see the module docstring; run() blocks until the engine dies or stop() is called"""

    def __init__(self, engine_port: int, frontend_port: int, host: str = "127.0.0.1", ipc_dir: str | None = None,
                 poll_ms: int = 50):  # fmt: skip
        (self._Req, self._Type, self._Meta, self._Addrs, MsgpackDecoder, MsgpackEncoder, Outs) = _vllm()
        self._dec_add = MsgpackDecoder(self._Req)
        self._dec_out = MsgpackDecoder(Outs)
        self._dec_any = MsgpackDecoder()
        self._enc = MsgpackEncoder()
        self.host, self.frontend_port, self.poll_ms = host, int(frontend_port), int(poll_ms)
        self.ctx = zmq.Context()
        d = ipc_dir or os.environ.get("TMPDIR", "/tmp")
        tag = f"tt_attach_{os.getpid()}"
        self.eng_in_addr, self.eng_out_addr = f"ipc://{d}/{tag}_in", f"ipc://{d}/{tag}_out"
        self.eng_hs = self._sock(zmq.ROUTER, bind=f"tcp://{host}:{int(engine_port)}")
        self.eng_in = self._sock(zmq.ROUTER, bind=self.eng_in_addr)
        self.eng_out = self._sock(zmq.PULL, bind=self.eng_out_addr)
        self.identity: bytes | None = None
        self.ready_payload: bytes | None = None
        self.fe: dict | None = None  # the current frontend attach
        self.stats = dict(attaches=0, detaches=0, aborted=0, relayed_in=0, relayed_out=0, dropped_out=0, finished=0,
                          untracked_out=0)  # fmt: skip
        self._stop = False

    def _sock(self, kind, bind=None, connect=None, identity=None):
        s = self.ctx.socket(kind)
        s.setsockopt(zmq.LINGER, 0)
        if identity is not None:
            s.setsockopt(zmq.IDENTITY, identity)
        if bind:
            s.bind(bind)
        if connect:
            s.connect(connect)
        return s

    # -- the engine face ------------------------------------------------------------------------------------------
    def _on_engine_handshake(self) -> None:
        ident, raw = self.eng_hs.recv_multipart()
        msg = msgspec.msgpack.decode(raw)
        if msg.get("status") == "HELLO":
            self.identity = ident
            meta = self._Meta(addresses=self._Addrs(inputs=[self.eng_in_addr], outputs=[self.eng_out_addr]),
                              parallel_config={})  # fmt: skip
            self.eng_hs.send_multipart((ident, msgspec.msgpack.encode(meta)))
            log.info("engine %s: HELLO answered (inputs %s, outputs %s)", ident.hex(), self.eng_in_addr, self.eng_out_addr)
        elif msg.get("status") == "READY":
            log.info("engine %s: READY", ident.hex())

    def _on_engine_input(self) -> None:
        ident, *frames = self.eng_in.recv_multipart()
        if self.ready_payload is None:  # the engine's first input-socket message: its EngineCoreReadyResponse
            self.ready_payload = frames[0] if frames else b""
            self.identity = ident
            log.info("engine ready payload stored (%d B): arming the frontend face", len(self.ready_payload))
            self._arm_frontend()

    def _on_engine_output(self) -> bool:
        frames = self.eng_out.recv_multipart(copy=False)
        dead = len(frames) == 1 and bytes(frames[0].buffer) == ENGINE_CORE_DEAD
        fe = self.fe
        if fe is not None and fe.get("out") is not None:
            fe["out"].send_multipart(frames, copy=False)
            self.stats["relayed_out"] += 1
            if fe["ids"] and not dead:  # after the forward: the tracking adds no latency to this message
                self._drop_finished(fe["ids"], frames)
        else:
            self.stats["dropped_out"] += 1  # (no frontend attached: e.g. outputs of an aborted attach's requests)
        return dead

    def _drop_finished(self, ids: set, frames) -> None:
        """a finished request leaves the attach's id set (else every request ever served would be aborted on a loss)"""
        try:
            outs = self._dec_out.decode([f.buffer for f in frames])
        except Exception:  # noqa: BLE001 - tracking only: an undecodable output keeps its ids (aborted on a loss)
            self.stats["untracked_out"] += 1
            return
        for o in outs.outputs or ():
            if o.finish_reason is not None and o.request_id in ids:
                ids.discard(o.request_id)
                self.stats["finished"] += 1
        for rid in outs.finished_requests or ():
            if rid in ids:
                ids.discard(rid)
                self.stats["finished"] += 1

    # -- the frontend face -----------------------------------------------------------------------------------------
    def _arm_frontend(self) -> None:
        """a fresh handshake DEALER to the frontend's port with a HELLO queued: the next frontend that binds it gets it"""
        hs = self._sock(zmq.DEALER, connect=f"tcp://{self.host}:{self.frontend_port}", identity=self.identity)
        hs.send(msgspec.msgpack.encode({"status": "HELLO", "local": False, "headless": True}))
        self.fe = {"hs": hs, "in": None, "out": None, "mon": None, "ids": set(), "t0": time.monotonic()}

    def _on_frontend_handshake(self) -> None:
        fe = self.fe
        meta = msgspec.msgpack.decode(fe["hs"].recv(), type=self._Meta)
        fe["hs"].send(msgspec.msgpack.encode({"status": "READY", "local": False, "headless": True}))
        fe["in"] = self._sock(zmq.DEALER, connect=meta.addresses.inputs[0], identity=self.identity)
        fe["mon"] = fe["in"].get_monitor_socket(zmq.EVENT_DISCONNECTED)
        fe["out"] = self._sock(zmq.PUSH, connect=meta.addresses.outputs[0])
        fe["in"].send(self.ready_payload)  # the replay: the frontend's MPClient applies it as from a starting engine
        self.stats["attaches"] += 1
        log.info("frontend %d attached in %.2f s (inputs %s, outputs %s)", self.stats["attaches"],
                 time.monotonic() - fe["t0"], meta.addresses.inputs[0], meta.addresses.outputs[0])  # fmt: skip

    def _on_frontend_input(self) -> None:
        fe = self.fe
        frames = fe["in"].recv_multipart(copy=False)
        self.eng_in.send_multipart([self.identity, *frames], copy=False)  # first: no latency added by the tracking
        self.stats["relayed_in"] += 1
        rtype = bytes(frames[0].buffer) if frames else b""
        if rtype == self._Type.ADD.value:
            try:
                fe["ids"].add(self._dec_add.decode(frames[1:]).request_id)
            except Exception:  # noqa: BLE001 - tracking only: a request we cannot read is not aborted on a loss
                log.exception("ADD not decodable for tracking")
        elif rtype == self._Type.ABORT.value and fe["ids"]:  # the frontend's own aborts: those ids are done
            try:
                fe["ids"].difference_update(self._dec_any.decode(frames[1:]))
            except Exception:  # noqa: BLE001
                pass

    def _detach(self, why: str) -> None:
        fe, self.fe = self.fe, None
        ids = sorted(fe["ids"]) if fe else []
        if ids:  # the gone frontend's requests: nobody reads their outputs
            self.eng_in.send_multipart([self.identity, self._Type.ABORT.value, *self._enc.encode(ids)])
            self.stats["aborted"] += len(ids)
        for k in ("mon", "in", "out", "hs"):
            if fe and fe.get(k) is not None:
                if k == "in":
                    fe["in"].disable_monitor()
                fe[k].close(linger=0)
        self.stats["detaches"] += 1
        log.info("frontend detached (%s): %d request(s) aborted; re-armed for the next frontend", why, len(ids))
        self._arm_frontend()

    # -- the loop --------------------------------------------------------------------------------------------------
    def stop(self) -> None:
        self._stop = True

    def run(self) -> int:
        poller = zmq.Poller()
        for s in (self.eng_hs, self.eng_in, self.eng_out):
            poller.register(s, zmq.POLLIN)
        registered: set = set()
        try:
            while not self._stop:
                fe = self.fe
                want = set()
                if fe is not None:
                    want = {fe[k] for k in ("hs", "in", "mon") if fe.get(k) is not None}
                for s in registered - want:
                    poller.unregister(s)
                for s in want - registered:
                    poller.register(s, zmq.POLLIN)
                registered = want
                for s, _ in poller.poll(self.poll_ms):
                    if s is self.eng_hs:
                        self._on_engine_handshake()
                    elif s is self.eng_in:
                        self._on_engine_input()
                    elif s is self.eng_out:
                        if self._on_engine_output():
                            log.info("ENGINE_CORE_DEAD forwarded: the broker exits")
                            return 1
                    elif fe is not None and s is fe.get("hs"):
                        self._on_frontend_handshake()
                    elif fe is not None and s is fe.get("in"):
                        self._on_frontend_input()
                    elif fe is not None and s is fe.get("mon"):
                        ev = recv_monitor_message(s)
                        if ev.get("event") == zmq.EVENT_DISCONNECTED:
                            for r in list(registered):
                                poller.unregister(r)
                            registered = set()
                            self._detach("input socket disconnected")
                            break
            return 0
        finally:
            for r in list(registered):
                try:
                    poller.unregister(r)
                except KeyError:
                    pass
            self.close()

    def close(self) -> None:
        fe, self.fe = self.fe, None
        for s in [self.eng_hs, self.eng_in, self.eng_out] + ([fe[k] for k in ("mon", "in", "out", "hs") if fe and fe.get(k)]
                                                             if fe else []):  # fmt: skip
            try:
                s.close(linger=0)
            except Exception:  # noqa: BLE001
                pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine-port", type=int, required=True, help="the headless engine's --data-parallel-rpc-port")
    ap.add_argument("--frontend-port", type=int, required=True, help="the frontends' --data-parallel-rpc-port")
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s attach-broker %(levelname)s %(message)s",
                        stream=sys.stdout)  # fmt: skip
    b = AttachBroker(a.engine_port, a.frontend_port, host=a.host)
    # explicit handlers: started as a background job its SIGINT is ignored on entry (no KeyboardInterrupt), and the
    # default SIGTERM would end it without its stats line; both stop the loop within one poll
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: b.stop())
    log.info("attach broker up: engine port %d, frontend port %d (pid %d)", a.engine_port, a.frontend_port, os.getpid())
    try:
        return b.run()
    finally:
        log.info("attach broker stats: %s", b.stats)


if __name__ == "__main__":
    sys.exit(main())
