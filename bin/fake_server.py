#!/usr/bin/env python3
"""A fake vLLM API server for the supervisor's host-only harness. Never touches a device.

It behaves like the real server as the supervisor sees it: prints the UMD open line, serves
/health, /v1/models and a vLLM-shaped /metrics once "started", writes the plugin's step
status file through the REAL ``vllm_tt_plugin.stepmeter`` module (so the file contract is
the one the hook writes), prints ``SELFTEST PASS|FAIL`` and reacts to a control file:

    exit0     orderly close line, exit 0           (clean exit -> restart)
    crash     Traceback + orderly close, exit 1    (exception exit, mesh closed -> restart, no reset)
    watcher   'Watcher detected tripped assert and stopped device' then SIGABRT, no close (-> reset)
    timeout   TT_THROW 'Device timeout ... is unrecoverable', exit 1, no close (-> reset)
    stall     steps stop completing (in_flight stays set)              (-> SIGINT by the supervisor)
    hang      ignore SIGINT from now on (until 'exit0')                 (-> HUNG, never killed)
    nohealth  /health stops answering                                  (-> stall by health)

Like vLLM, SIGINT is ignored until the server is up; afterwards it closes the mesh (prints the
orderly close line) and exits 0.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from vllm_tt_plugin.stepmeter import get_step_meter  # noqa: E402

STATE = {
    "up": False,
    "stalled": False,
    "hang": False,
    "nohealth": False,
    "ttft": [0.4, 0.6, 0.5, 3.2],  # seconds, for the histogram
    "tpot": [0.024, 0.025, 0.03],
    "accepted": 0,
    "drafts": 0,
}


def log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _hist(name: str, values: list[float], bounds: list[float]) -> str:
    lines = [f"# TYPE {name} histogram"]
    cum = 0
    for b in bounds:
        cum = sum(1 for v in values if v <= b)
        lines.append(f'{name}_bucket{{le="{b}"}} {cum}')
    lines.append(f'{name}_bucket{{le="+Inf"}} {len(values)}')
    lines.append(f"{name}_sum {sum(values)}")
    lines.append(f"{name}_count {len(values)}")
    return "\n".join(lines)


def metrics_text() -> str:
    parts = [
        _hist("vllm:time_to_first_token_seconds", STATE["ttft"], [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]),
        _hist("vllm:time_per_output_token_seconds", STATE["tpot"], [0.01, 0.02, 0.025, 0.05, 0.1]),
        "vllm:num_requests_waiting 3",
        "vllm:num_requests_running 2",
        f"vllm:spec_decode_num_accepted_tokens_total {STATE['accepted']}",
        f"vllm:spec_decode_num_draft_tokens_total {STATE['drafts']}",
        'glm53_startup_seconds{phase="restore"} 36.0',
    ]
    return "\n".join(parts) + "\n"


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        return

    def do_GET(self):
        if not STATE["up"] or (self.path == "/health" and STATE["nohealth"]):
            if STATE["nohealth"] and STATE["up"]:
                time.sleep(30)  # a wedged front end: the probe times out
            self.send_response(503)
            self.end_headers()
            return
        if self.path == "/health":
            body = b""
        elif self.path == "/metrics":
            body = metrics_text().encode()
        elif self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "zai-org/GLM-5.3-Flash"}]}).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def step_loop(meter):
    while True:
        if STATE["up"] and not STATE["stalled"]:
            meter.begin("decode")
            time.sleep(0.02)
            meter.spec_accepted(2)
            STATE["accepted"] += 2
            STATE["drafts"] += 3
            meter.end()
            time.sleep(0.08)
        elif STATE["up"] and STATE["stalled"]:
            if not meter.in_flight:
                meter.begin("decode")  # a step that never ends
            meter.flush()
            time.sleep(0.5)
        else:
            time.sleep(0.1)


def orderly_exit(rc: int = 0):
    meter = get_step_meter()
    meter.mark_shutting_down()
    log("Closing devices in cluster completed")
    sys.stdout.flush()
    os._exit(rc)


def on_sigint(signum, frame):
    if not STATE["up"]:
        log("fake: SIGINT ignored during startup (vLLM installs its handler once it serves)")
        return
    if STATE["hang"]:
        log("fake: SIGINT received but hanging (hang mode)")
        return
    log("fake: SIGINT -> shutting down the engine")
    threading.Thread(target=lambda: (time.sleep(0.3), orderly_exit(0)), daemon=True).start()


def ctl_loop(path: str):
    seen = ""
    while True:
        try:
            with open(path) as f:
                cmd = f.read().strip()
        except OSError:
            cmd = ""
        if cmd and cmd != seen:
            seen = cmd
            try:
                os.unlink(path)
            except OSError:
                pass
            log(f"fake: control command {cmd}")
            if cmd == "exit0":
                STATE["hang"] = False
                orderly_exit(0)
            elif cmd == "crash":
                log("Traceback (most recent call last):\n  File \"fake\", line 1\nRuntimeError: boom (host-side exception)")
                orderly_exit(1)
            elif cmd == "watcher":
                log("Always | FATAL | Watcher detected tripped assert and stopped device: core (x=1,y=1) ...")
                sys.stdout.flush()
                os.abort()
            elif cmd == "timeout":
                log("TT_THROW @ device.cpp:1 Device timeout detected: chip 3 is unrecoverable (Timeout detected)")
                sys.stdout.flush()
                os._exit(1)
            elif cmd == "stall":
                STATE["stalled"] = True
            elif cmd == "hang":
                STATE["hang"] = True
            elif cmd == "nohealth":
                STATE["nohealth"] = True
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--ctl", required=True, help="control file polled for a command")
    ap.add_argument("--startup-s", type=float, default=2.0)
    ap.add_argument("--selftest", choices=["pass", "fail", "none"], default="pass")
    ap.add_argument("--pidfile", default=os.environ.get("GLM53_SUPERVISOR_PIDFILE"))
    ap.add_argument("--refuse", action="store_true", help="exit 1 at once before any device open (lock held)")
    a = ap.parse_args()
    if a.refuse:
        log("group all is held by someone@host:1 — refusing")
        sys.exit(1)
    if a.pidfile:
        with open(a.pidfile, "w") as f:
            f.write(str(os.getpid()))
    signal.signal(signal.SIGINT, on_sigint)
    log(f"fake vLLM server pid {os.getpid()} starting on port {a.port}")
    meter = get_step_meter()
    threading.Thread(target=step_loop, args=(meter,), daemon=True).start()
    threading.Thread(target=ctl_loop, args=(a.ctl,), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.2)
    log("Opening user mode device driver")
    log("Opening local chip ids/PCIe ids: {0, 1, 2, 3, 4, 5, 6, 7}/[0, 1, 2, 3, 4, 5, 6, 7]")
    time.sleep(a.startup_s)
    meter.mark_ready()
    STATE["up"] = True
    log("INFO: Application startup complete.")
    if a.selftest != "none":
        time.sleep(0.3)
        log("SELFTEST PASS" if a.selftest == "pass" else "SELFTEST FAIL: strict TF top1 0.62 < 0.85")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
