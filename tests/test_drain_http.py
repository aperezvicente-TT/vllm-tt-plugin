# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Drain close over HTTP: `vllm serve` with the TT plugin over Stage 9's CPU mock generator (tt-metal tree on
PYTHONPATH: tests/vllm/engine_harness.py), the engine core in its own process, a fake mesh. SIGINT to the API
server (what serve_window.sh does on STOP) while a streamed generation is in flight. Never opens a TT device.

1  one in-flight 2,000-token generation: it completes (finish "length", 2,000 tokens), the engine exits 0, the
   mesh is closed exactly once by the engine process, the API server exits 0 (not 70), nothing force-killed
2  a generation longer than the drain budget (--shutdown-timeout 30, slow decode): aborted through the engine's
   abort path at the budget (finish "abort"), orderly close before the process manager's kill timer, mesh closed
3  the engine process dies silently during the drain (_exit(7), tt-umd's SIGBUS exit): the open stream ends with
   an error at once (not at the client timeout), the exit status is logged, the API server exits 70
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")
H = pytest.importorskip("models.autoports.zai_org_glm_5_3_flash.tests.vllm.engine_harness")

MODEL = "zai-org/GLM-5.3-Flash"
SUPPORT = Path(__file__).resolve().parent / "drain_support"
if not os.path.isdir(H.SNAPSHOT):
    pytest.skip("GLM-5.3-Flash snapshot not available", allow_module_level=True)


def _launch(tmp, *, shutdown_timeout, env_extra=None):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    marker, log = tmp / "marker.jsonl", tmp / "server.log"
    args = [H.SNAPSHOT, "--served-model-name", MODEL, "--host", "127.0.0.1", "--port", str(port), "--block-size",
            "64", "--max-num-seqs", "8", "--max-num-batched-tokens", "512", "--no-enable-prefix-caching",
            "--shutdown-timeout", str(shutdown_timeout), "--max-model-len", "2051",
            "--additional-config", json.dumps({"tt": {"sample_on_device_mode": "all", "trace_mode": "all"}})]
    env = dict(os.environ, GLM53_MOCK_MARKER=str(marker), HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
               GLM53_SELFTEST="off", TT_DRAIN_TEST_FACTORY="faults:build_drain_generator",
               PYTHONPATH=f"{SUPPORT}:{os.environ.get('PYTHONPATH', '')}", **(env_extra or {}))
    env.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
    f = log.open("w")
    proc = subprocess.Popen([sys.executable, str(SUPPORT / "mock_server.py"), *args], stdout=f,
                            stderr=subprocess.STDOUT, env=env, start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"mock server exited rc={proc.returncode}:\n{log.read_text()[-3000:]}")
        try:
            if requests.get(base + "/health", timeout=2).status_code == 200:
                break
        except Exception:
            time.sleep(1)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
        pytest.fail("mock server not healthy in 300 s")
    return dict(base=base, proc=proc, marker=marker, log=log, file=f)


def _stream_in_thread(base, n_tokens):
    res = dict(tokens=0, finish=None, error=None, t_end=None)

    def run():
        body = {"model": MODEL, "prompt": list(range(1000, 1024)), "max_tokens": n_tokens, "temperature": 0.0,
                "ignore_eos": True, "return_token_ids": True, "stream": True}
        try:
            with requests.post(f"{base}/v1/completions", json=body, stream=True, timeout=(10, 600)) as r:
                for line in r.iter_lines():
                    if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
                        continue
                    msg = json.loads(line[6:])
                    if "error" in msg:
                        res["error"] = msg["error"]
                    for c in msg.get("choices", []):
                        res["tokens"] += len(c.get("token_ids") or [])
                        res["finish"] = c.get("finish_reason") or res["finish"]
        except Exception as exc:
            res["error"] = f"{type(exc).__name__}: {exc}"
        res["t_end"] = time.time()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, res


def _sigint_after(srv, res, n):
    deadline = time.time() + 120
    while res["tokens"] < n and time.time() < deadline:
        time.sleep(0.05)
    assert res["tokens"] >= n, res
    t_stop = time.time()
    srv["proc"].send_signal(signal.SIGINT)
    return t_stop


def _finish(srv, timeout):
    try:
        rc = srv["proc"].wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(srv["proc"].pid, signal.SIGKILL)
        pytest.fail(f"API server still up {timeout} s after SIGINT:\n{srv['log'].read_text()[-3000:]}")
    srv["file"].close()
    events = [json.loads(x) for x in srv["marker"].read_text().splitlines()]
    return rc, events, srv["log"].read_text()


def _mesh(events):
    built = [e for e in events if "max_batch" in e]
    closed = [e for e in events if e.get("event") == "mesh_closed"]
    assert len(built) == 1, events
    return built[0]["pid"], [e["pid"] for e in closed]


def test_drain_one_inflight_generation_completes_then_orderly_close(tmp_path, evidence=None):
    srv = _launch(tmp_path, shutdown_timeout=900)
    th, res = _stream_in_thread(srv["base"], 2000)
    _sigint_after(srv, res, 50)
    rc, events, log = _finish(srv, 300)
    th.join(30)
    assert res["finish"] == "length" and res["tokens"] == 2000 and res["error"] is None, res
    pid, closed = _mesh(events)
    assert closed == [pid], f"mesh not closed exactly once by the engine: {events}"
    assert f"EngineCore pid {pid} exit 0 (orderly)" in log
    assert "force killing" not in log and "TT engine core died" not in log
    assert rc == 0, (rc, log[-2000:])


def test_drain_budget_aborts_a_long_generation_before_the_kill_timer(tmp_path):
    srv = _launch(tmp_path, shutdown_timeout=30, env_extra={"TT_DRAIN_TEST_DECODE_SLEEP_S": "0.05"})
    th, res = _stream_in_thread(srv["base"], 2000)
    t_stop = _sigint_after(srv, res, 20)
    rc, events, log = _finish(srv, 120)
    th.join(30)
    assert res["finish"] == "abort" and 20 <= res["tokens"] < 2000 and res["error"] is None, res
    assert res["t_end"] - t_stop < 30, "the stream outlived the kill timer"
    pid, closed = _mesh(events)
    assert closed == [pid], events
    assert "TT drain budget 22 s reached: aborting 1 in-flight request(s)" in log
    assert "force killing" not in log, "the process manager force-killed the engine (mesh left open)"
    assert f"EngineCore pid {pid} exit 0 (orderly)" in log
    assert rc == 0, (rc, log[-2000:])


def test_engine_death_during_drain_ends_the_stream_and_is_reported(tmp_path):
    flag = tmp_path / "EXIT7"
    srv = _launch(tmp_path, shutdown_timeout=900,
                  env_extra={"TT_DRAIN_TEST_DECODE_SLEEP_S": "0.02", "TT_DRAIN_TEST_EXIT_FLAG": str(flag)})
    th, res = _stream_in_thread(srv["base"], 2000)
    _sigint_after(srv, res, 50)
    time.sleep(3)
    t_exit = time.time()
    flag.touch()  # the next decode step _exit(7)s in the engine process
    rc, events, log = _finish(srv, 120)
    th.join(30)
    assert res["error"] is not None and res["finish"] is None, res
    assert res["t_end"] - t_exit < 60, "the stream was left to the client timeout"
    pid, closed = _mesh(events)
    assert closed == [], "a dead engine cannot have closed its mesh"
    assert f"EngineCore pid {pid} exit status 7: tt-umd's SIGBUS handler" in log
    assert "exiting with status 70" in log and rc == 70, rc
