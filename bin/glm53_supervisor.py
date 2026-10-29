#!/usr/bin/env python3
"""glm53-supervisor: standing-server daemon for the GLM-5.3-Flash vLLM server (host-8chip TP8).

One long-lived process: starts the server command (pinned env / image come from the command
file, serve_cmd.host-8chip.sh), waits for /health and the startup self-test, serves
/healthz, /readyz, /metrics and /status on its own port, watches the server log, the
plugin's step status file and the kernel log, and applies supervisor_policy.Policy:

  clean exit                       -> restart (image restore ~85 s to ready)
  watcher abort / device timeout / -> write RESET_PENDING.<group> (+ optional reset command),
  kernel fault / unclean close        wait for RESET DONE (marker removed or a journal line),
                                      then restart
  stall (step in flight > stall_s, /health dead > health_fail_s) -> SIGINT, then as above
  no exit after SIGINT             -> HUNG + escalation, never SIGKILL
  2 failures in 30 min             -> ESCALATED: marker file + "ESCALATION:" log line; the
                                      coordinator removes the marker to resume

Stdlib only. Config: a JSON file (--config) with CLI overrides; see supervisor.host-8chip.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import promparse  # noqa: E402
from supervisor_policy import (  # noqa: E402
    DEVICE_OPEN_RE,
    FAULT_RE,
    KERNEL_FAULT_RE,
    KERNEL_SUBJECT_RE,
    ORDERLY_CLOSE_RE,
    SELFTEST_RE,
    ExitInfo,
    Policy,
    PolicyConfig,
    State,
)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Config:
    server_cmd: list[str] | str = ""  # the server launch (a shell string or argv)
    run_root: str = "/opt/glm53/work-s9/supervisor"
    host: str = "127.0.0.1"
    port: int = 8000
    listen: str = "127.0.0.1:9108"
    api_marker: str = "vllm.entrypoints.openai.api_server"  # cmdline marker of the API server
    status_file: str = ""  # GLM53_STEP_STATUS; default <run_root>/step_status.json
    pidfile: str = ""  # GLM53_SUPERVISOR_PIDFILE; default <run_root>/api_server.pid
    selftest_cmd: list[str] | str | None = None  # exit 0 = PASS; runs after /health
    selftest_from_log: bool = True  # 'SELFTEST PASS|FAIL' line in the server log
    selftest_timeout_s: float = 600.0
    reset_group: str = "all"
    reset_marker: str = "/opt/glm53/RESET_PENDING.all"
    reset_cmd: list[str] | str | None = None  # optional: the chain script, run detached
    reset_journal: str = "/opt/glm53/home/.tt-locks/reset.journal"
    reset_done_files: list[str] = field(default_factory=list)  # more files whose new lines may say RESET DONE
    reset_done_re: str = r"reset (all|{group}) done rc=0|RESET (all|{group}) DONE rc=0"
    lock_dir: str = "/opt/glm53/home/.tt-locks"
    escalation_marker: str = ""  # default <run_root>/SUPERVISOR_ESCALATION
    coordinator_request: str = ""  # optional: work-s9/COORDINATOR_REQUEST.txt (one line per event)
    stop_file: str = ""  # default <run_root>/STOP
    kernel_cmd: list[str] | None = field(
        default_factory=lambda: ["journalctl", "-k", "--since", "@{since}", "--no-pager", "-o", "short-iso"]
    )
    kernel_scan_s: float = 30.0
    scrape_s: float = 10.0
    health_probe_s: float = 5.0
    health_timeout_s: float = 5.0
    tick_s: float = 1.0
    env: dict[str, str] = field(default_factory=dict)
    policy: dict = field(default_factory=dict)

    def finalize(self) -> None:
        os.makedirs(self.run_root, exist_ok=True)
        self.status_file = self.status_file or os.path.join(self.run_root, "step_status.json")
        self.pidfile = self.pidfile or os.path.join(self.run_root, "api_server.pid")
        self.escalation_marker = self.escalation_marker or os.path.join(self.run_root, "SUPERVISOR_ESCALATION")
        self.stop_file = self.stop_file or os.path.join(self.run_root, "STOP")


class TailReader:
    """Incremental line reader of a growing log file."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.pos = 0
        self.partial = ""

    def lines(self) -> list[str]:
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read()
                self.pos = f.tell()
        except OSError:
            return []
        if not data:
            return []
        text = self.partial + data.decode("utf-8", "replace")
        parts = text.split("\n")
        self.partial = parts.pop()
        return parts


class AppendWatcher:
    """New text appended to a file since the watch began."""

    def __init__(self, path: str) -> None:
        self.path = path
        try:
            self.pos = os.path.getsize(path)
        except OSError:
            self.pos = 0

    def new_text(self) -> str:
        try:
            size = os.path.getsize(self.path)
            if size < self.pos:
                self.pos = 0  # truncated / rotated
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read()
                self.pos = f.tell()
            return data.decode("utf-8", "replace")
        except OSError:
            return ""


def proc_session_members(sid: int) -> list[int]:
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat", "rb") as f:
                st = f.read().decode("latin-1")
            fields = st[st.rindex(")") + 2 :].split()
            if int(fields[3]) == sid:  # field 6 of stat = session id
                out.append(int(d))
        except (OSError, ValueError, IndexError):
            continue
    return out


def pid_holds_tt_nodes(pid: int) -> bool:
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                if os.readlink(f"/proc/{pid}/fd/{fd}").startswith("/dev/tenstorrent"):
                    return True
            except OSError:
                continue
    except OSError:
        pass
    return False


def pid_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Supervisor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.policy = Policy(PolicyConfig(**cfg.policy), selftest_configured=bool(cfg.selftest_cmd) or cfg.selftest_from_log)
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.run_dir = ""
        self.server_log = ""
        self.tail: TailReader | None = None
        self.launch_t = 0.0
        self.device_opened = False
        self.orderly_close = False
        self.fault_line = ""
        self.kernel_fault_line = ""
        self.kernel_seen = 0
        self.selftest_log_result: bool | None = None
        self.selftest_proc: subprocess.Popen | None = None
        self.selftest_started = 0.0
        self.health_ok_at = 0.0
        self.health_fail_since = 0.0
        self.last_probe = 0.0
        self.last_scrape = 0.0
        self.last_kernel = 0.0
        self.step: dict = {}
        self.scraped: list = []
        self.derived: dict = {}
        self.reset_watchers: list[AppendWatcher] = []
        self.reset_requested_at = 0.0
        self.stopping = False
        self.sup_log = open(os.path.join(cfg.run_root, "supervisor.log"), "a", buffering=1)
        self.run_no = 0
        self.escalation_reason = ""
        self._done = False
        self.health_last_ok = 0.0  # probe thread results
        self.health_last_fail = 0.0
        self.health_seen = 0.0
        self.hung_pending = False  # exited, but its session still holds processes
        self.exit_handled = False

    # -- logging ---------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"{utc()} [supervisor] {msg}"
        self.sup_log.write(line + "\n")
        sys.stderr.write(line + "\n")
        sys.stderr.flush()

    def coordinator(self, msg: str) -> None:
        if self.cfg.coordinator_request:
            try:
                with open(self.cfg.coordinator_request, "a") as f:
                    f.write(f"{utc()} stage9: SUPERVISOR — {msg}\n")
            except OSError as e:
                self.log(f"coordinator note failed: {e}")

    # -- process ---------------------------------------------------------
    def spawn(self) -> None:
        self.run_no += 1
        self.run_dir = os.path.join(self.cfg.run_root, "runs", f"{self.run_no:03d}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.server_log = os.path.join(self.run_dir, "server.log")
        for p in (self.cfg.status_file, self.cfg.pidfile):
            try:
                os.unlink(p)
            except OSError:
                pass
        prom = os.path.join(self.run_dir, "prometheus")
        os.makedirs(prom, exist_ok=True)
        env = dict(os.environ)
        env.update(self.cfg.env)
        env.update(
            {
                "GLM53_STEP_STATUS": self.cfg.status_file,
                "GLM53_SUPERVISOR_PIDFILE": self.cfg.pidfile,
                "GLM53_SUPERVISOR_RUN_DIR": self.run_dir,
                "PROMETHEUS_MULTIPROC_DIR": prom,
                "GLM53_STARTUP_TIMELINE": os.path.join(self.run_dir, "startup_timeline.json"),
                "GLM53_TIMELINE_PATH": os.path.join(self.run_dir, "fast_start_timeline.json"),
            }
        )
        cmd = self.cfg.server_cmd
        if isinstance(cmd, str):
            cmd = ["bash", "-c", cmd]
        logf = open(self.server_log, "ab")
        self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, start_new_session=True, cwd=self.run_dir)
        logf.close()
        self.tail = TailReader(self.server_log)
        self.launch_t = time.time()
        self.device_opened = self.orderly_close = False
        self.fault_line = self.kernel_fault_line = ""
        self.kernel_seen = 0
        self.selftest_log_result = None
        self.selftest_proc = None
        self.health_ok_at = self.health_fail_since = 0.0
        self.health_last_ok = self.health_last_fail = self.health_seen = 0.0
        self.hung_pending = self.exit_handled = False
        self.step = {}
        self.scraped = []
        self.derived = {}
        with open(os.path.join(self.run_dir, "launch.json"), "w") as f:
            json.dump({"cmd": cmd, "pid": self.proc.pid, "utc": utc(), "env_extra": self.cfg.env}, f, indent=1)
        self.log(f"spawned pid {self.proc.pid} (session leader) run_dir {self.run_dir}")

    def api_pid(self) -> int | None:
        try:
            with open(self.cfg.pidfile) as f:
                pid = int(f.read().strip())
            if pid_alive(pid):
                return pid
        except (OSError, ValueError):
            pass
        if self.proc is None:
            return None
        for pid in proc_session_members(self.proc.pid):
            if self.cfg.api_marker in pid_cmdline(pid):
                return pid
        return self.proc.pid if self.proc.poll() is None else None

    def sigint(self, reason: str) -> None:
        pid = self.api_pid()
        if pid is None:
            self.log(f"SIGINT wanted ({reason}) but no live API server pid")
            return
        self.log(f"SIGINT -> pid {pid} ({reason})")
        try:
            os.kill(pid, signal.SIGINT)
        except OSError as e:
            self.log(f"SIGINT failed: {e}")

    def session_leftovers(self) -> tuple[bool, bool]:
        if self.proc is None:
            return False, False
        pids = [p for p in proc_session_members(self.proc.pid) if p != self.proc.pid or self.proc.poll() is None]
        return bool(pids), any(pid_holds_tt_nodes(p) for p in pids)

    # -- observation -----------------------------------------------------
    def read_log(self, now: float) -> None:
        if self.tail is None:
            return
        for line in self.tail.lines():
            if not self.device_opened and DEVICE_OPEN_RE.search(line):
                self.device_opened = True
            if ORDERLY_CLOSE_RE.search(line):
                self.orderly_close = True
            if not self.fault_line and FAULT_RE.search(line):
                self.fault_line = line.strip()[:300]
                for a in self.policy.on_fault_signature(now, self.fault_line):
                    self.act(a, now)
            m = SELFTEST_RE.search(line)
            if m and self.cfg.selftest_from_log and self.selftest_log_result is None:
                self.selftest_log_result = m.group(1) == "PASS"
                self.log(f"server log: {line.strip()[:200]}")

    def read_step(self) -> None:
        try:
            with open(self.cfg.status_file) as f:
                self.step = json.load(f)
        except (OSError, ValueError):
            pass

    def probe_health(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://{self.cfg.host}:{self.cfg.port}/health", timeout=self.cfg.health_timeout_s) as r:
                return r.status == 200
        except Exception:
            return False

    def probe_thread(self) -> None:
        """Health probes and metric scrapes off the tick: a wedged front end blocks only this thread."""
        last_scrape = 0.0
        while not self._done:
            if self.proc is not None and self.proc.poll() is None and self.policy.state in (State.STARTING, State.SELFTEST, State.READY, State.STOPPING):
                launch = self.launch_t
                ok = self.probe_health()
                now = time.time()
                if launch == self.launch_t:  # still the same server
                    if ok:
                        self.health_last_ok = now
                    else:
                        self.health_last_fail = now
                    self.health_seen = now
                    if ok and self.policy.state == State.READY and now - last_scrape >= self.cfg.scrape_s:
                        last_scrape = now
                        self.scrape()
            time.sleep(self.cfg.health_probe_s)

    def scrape(self) -> None:
        try:
            with urllib.request.urlopen(f"http://{self.cfg.host}:{self.cfg.port}/metrics", timeout=self.cfg.health_timeout_s) as r:
                text = r.read().decode("utf-8", "replace")
        except Exception:
            return
        s = promparse.parse(text)
        d = {
            "ttft_p50": promparse.histogram_quantile(s, "vllm:time_to_first_token_seconds", 0.5),
            "ttft_p99": promparse.histogram_quantile(s, "vllm:time_to_first_token_seconds", 0.99),
            "tpot_p50": promparse.histogram_quantile(s, "vllm:time_per_output_token_seconds", 0.5),
            "tpot_p99": promparse.histogram_quantile(s, "vllm:time_per_output_token_seconds", 0.99),
            "tpot_mean": promparse.histogram_mean(s, "vllm:time_per_output_token_seconds"),
            "queue_depth": promparse.total(s, "vllm:num_requests_waiting"),
            "live_slots": promparse.total(s, "vllm:num_requests_running"),
            "spec_accepted": promparse.total(s, "vllm:spec_decode_num_accepted_tokens_total"),
            "spec_drafts": promparse.total(s, "vllm:spec_decode_num_draft_tokens_total"),
            "kv_cache_usage": promparse.total(s, "vllm:kv_cache_usage_perc"),
        }
        acc, dr = d["spec_accepted"], d["spec_drafts"]
        d["mtp_acceptance_rate"] = (acc / dr) if acc is not None and dr else None
        with self.lock:
            self.scraped = s
            self.derived = d

    def scan_kernel(self) -> None:
        if not self.cfg.kernel_cmd or not self.launch_t:
            return
        cmd = [a.replace("{since}", str(int(self.launch_t))) for a in self.cfg.kernel_cmd]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
        except Exception as e:
            self.log(f"kernel scan failed: {e}")
            self.cfg.kernel_cmd = None
            return
        lines = out.splitlines()
        new = lines[self.kernel_seen :]
        self.kernel_seen = len(lines)
        for line in new:
            if "Configured ACS" in line:
                continue
            if KERNEL_FAULT_RE.search(line) and KERNEL_SUBJECT_RE.search(line):
                self.log(f"KERNEL: {line.strip()[:240]}")
                with open(os.path.join(self.run_dir, "kernel_anomalies.txt"), "a") as f:
                    f.write(line + "\n")
                if not self.kernel_fault_line:
                    self.kernel_fault_line = line.strip()[:300]
                    for a in self.policy.on_fault_signature(time.time(), "kernel: " + self.kernel_fault_line):
                        self.act(a, time.time())

    # -- self-test -------------------------------------------------------
    def selftest_step(self, now: float) -> None:
        if self.policy.state != State.SELFTEST:
            return
        if self.cfg.selftest_cmd:
            if self.selftest_proc is None:
                cmd = self.cfg.selftest_cmd
                if isinstance(cmd, str):
                    cmd = ["bash", "-c", cmd]
                env = dict(os.environ, SERVER_URL=f"http://{self.cfg.host}:{self.cfg.port}", GLM53_SUPERVISOR_RUN_DIR=self.run_dir)
                self.selftest_proc = subprocess.Popen(cmd, stdout=open(os.path.join(self.run_dir, "selftest.log"), "ab"), stderr=subprocess.STDOUT, env=env)
                self.selftest_started = now
                self.log(f"self-test started: {cmd}")
                return
            rc = self.selftest_proc.poll()
            if rc is None:
                if now - self.selftest_started > self.cfg.selftest_timeout_s:
                    self.selftest_proc.terminate()  # host-only client: TERM is allowed
                    for a in self.policy.on_selftest(False, now, "self-test timed out"):
                        self.act(a, now)
                return
            for a in self.policy.on_selftest(rc == 0, now, f"(self-test command rc={rc})"):
                self.act(a, now)
            return
        if self.selftest_log_result is not None:
            for a in self.policy.on_selftest(self.selftest_log_result, now, "(server log line)"):
                self.act(a, now)
        elif now - self.health_ok_at > self.cfg.selftest_timeout_s:
            for a in self.policy.on_selftest(False, now, "no SELFTEST line in the server log"):
                self.act(a, now)

    # -- reset -----------------------------------------------------------
    def request_reset(self, reason: str) -> None:
        self.reset_requested_at = time.time()
        try:
            with open(self.cfg.reset_marker, "w") as f:
                f.write(f"{utc()} supervisor: {reason}\n")
            self.log(f"RESET requested: wrote {self.cfg.reset_marker} ({reason[:160]})")
        except OSError as e:
            self.log(f"could not write the reset marker {self.cfg.reset_marker}: {e}")
        files = [self.cfg.reset_journal] + list(self.cfg.reset_done_files)
        self.reset_watchers = [AppendWatcher(p) for p in files if p]
        self.coordinator(f"RESET REQUESTED for {self.cfg.reset_group} — {reason[:200]}; marker {self.cfg.reset_marker}; the server restarts on RESET DONE")
        if self.cfg.reset_cmd:
            cmd = self.cfg.reset_cmd
            if isinstance(cmd, str):
                cmd = ["bash", "-c", cmd]
            subprocess.Popen(cmd, stdout=open(os.path.join(self.run_dir, "reset_cmd.log"), "ab"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            self.log(f"reset command started: {cmd}")

    def reset_done(self) -> bool:
        rx = re.compile(self.cfg.reset_done_re.replace("{group}", re.escape(self.cfg.reset_group)))
        for w in self.reset_watchers:
            if rx.search(w.new_text()):
                self.log(f"RESET DONE seen in {w.path}")
                return True
        if self.reset_requested_at and not os.path.exists(self.cfg.reset_marker):
            self.log("RESET DONE: the marker was removed")
            return True
        return False

    def box_resetting(self) -> bool:
        return any(os.path.exists(os.path.join(self.cfg.lock_dir, m)) for m in ("RESETTING", "QUIESCE"))

    # -- actions ---------------------------------------------------------
    def act(self, a, now: float) -> None:
        if a.kind == "log":
            self.log(a.reason)
        elif a.kind == "spawn":
            if self.box_resetting():
                self.log("spawn deferred: the box is RESETTING / QUIESCED")
                self.policy.next_spawn_at = now + 10
                return
            self.spawn()
            for b in self.policy.on_spawned(now):
                self.act(b, now)
        elif a.kind == "sigint":
            self.sigint(a.reason)
        elif a.kind == "request_reset":
            self.request_reset(a.reason)
        elif a.kind == "escalate":
            self.escalation_reason = a.reason
            try:
                with open(self.cfg.escalation_marker, "w") as f:
                    f.write(f"{utc()} {a.reason}\n")
            except OSError as e:
                self.log(f"escalation marker failed: {e}")
            self.log(f"ESCALATION: {a.reason} -> {self.cfg.escalation_marker} (remove it to resume)")
            self.coordinator(f"ESCALATION: {a.reason}; automatic restarts paused; remove {self.cfg.escalation_marker} to resume")
        elif a.kind == "stopped":
            self.log(f"stopped ({a.reason})")
            self._done = True

    # -- main loop -------------------------------------------------------
    def tick(self) -> None:
        now = time.time()
        p = self.policy
        if os.path.exists(self.cfg.stop_file) and not p.stop_requested:
            self.log(f"stop file {self.cfg.stop_file} seen")
            for a in p.on_stop_requested(now):
                self.act(a, now)
        if p.escalated and not os.path.exists(self.cfg.escalation_marker):
            for a in p.on_escalation_cleared(now):
                self.act(a, now)
        alive = self.proc is not None and self.proc.poll() is None
        if self.proc is not None:
            self.read_log(now)
            if now - self.last_kernel >= self.cfg.kernel_scan_s:
                self.last_kernel = now
                self.scan_kernel()
        if alive:
            self.read_step()
            if self.health_seen and self.health_seen > self.last_probe:  # a new probe result
                self.last_probe = self.health_seen
                if self.health_last_ok >= self.health_last_fail:
                    self.health_fail_since = 0.0
                    if not self.health_ok_at:
                        self.health_ok_at = self.health_last_ok
                    for a in p.on_health_ok(now):
                        self.act(a, now)
                elif p.state == State.READY:
                    self.health_fail_since = self.health_fail_since or self.health_last_fail
                    if now - self.health_fail_since > p.cfg.health_fail_s:
                        for a in p.on_stall(now, f"/health failing for {now - self.health_fail_since:.0f} s"):
                            self.act(a, now)
            if p.state == State.READY and self.step.get("in_flight") and now - float(self.step.get("step_begin") or now) > p.cfg.stall_s:
                for a in p.on_stall(now, f"step ({self.step.get('kind')}) in flight for {now - float(self.step['step_begin']):.0f} s"):
                    self.act(a, now)
            self.selftest_step(now)
        elif self.proc is not None and not self.exit_handled and (
            p.state in (State.STARTING, State.SELFTEST, State.READY, State.STOPPING)
            or (p.state == State.HUNG and self.hung_pending and not self.session_leftovers()[0])
        ):
            self.read_log(now)
            self.scan_kernel()
            leftovers, held = self.session_leftovers()
            info = ExitInfo(
                rc=self.proc.returncode,
                uptime_s=now - self.launch_t,
                device_opened=self.device_opened,
                orderly_close=self.orderly_close,
                fault=self.fault_line or None,
                kernel_fault=self.kernel_fault_line or None,
                descendants_alive=leftovers,
                nodes_held=held,
            )
            with open(os.path.join(self.run_dir, "exit.json"), "w") as f:
                json.dump(asdict(info), f, indent=1)
            self.coordinator(f"server exited rc={info.rc} after {info.uptime_s:.0f} s (fault={info.fault or info.kernel_fault or '-'}; close={info.orderly_close}); {self.run_dir}")
            self.hung_pending = leftovers  # a HUNG verdict is re-examined once the session is gone
            self.exit_handled = not leftovers
            for a in p.on_exit(info, now):
                self.act(a, now)
        if p.state == State.RESET_WAIT and self.reset_done():
            for a in p.on_reset_done(now):
                self.act(a, now)
        for a in p.on_tick(now, alive):
            self.act(a, now)
        self.write_state(now, alive)

    def write_state(self, now: float, alive: bool) -> None:
        snap = self.snapshot(now, alive)
        tmp = os.path.join(self.cfg.run_root, "state.json.tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(snap, f, indent=1)
            os.replace(tmp, os.path.join(self.cfg.run_root, "state.json"))
        except OSError:
            pass

    def snapshot(self, now: float | None = None, alive: bool | None = None) -> dict:
        now = now or time.time()
        if alive is None:
            alive = self.proc is not None and self.proc.poll() is None
        st = self.step or {}
        last_step = float(st.get("last_step") or 0)
        in_flight = bool(st.get("in_flight"))
        step_begin = float(st.get("step_begin") or 0)
        last_step_age = (now - last_step) if last_step else None
        device_ok = alive and (not in_flight or now - step_begin < self.policy.cfg.stall_s)
        with self.lock:
            derived = dict(self.derived)
        return {
            "utc": utc(),
            "policy": self.policy.snapshot(),
            "server": {"alive": alive, "pid": self.proc.pid if self.proc else None, "api_pid": self.api_pid() if alive else None, "run_dir": self.run_dir, "uptime_s": (now - self.launch_t) if self.launch_t and alive else None, "device_opened": self.device_opened, "orderly_close": self.orderly_close, "fault_line": self.fault_line, "kernel_fault_line": self.kernel_fault_line, "health_ok_at": self.health_ok_at},
            "device": {"responsive": device_ok, "last_step_age_s": last_step_age, "in_flight": in_flight, "in_flight_s": (now - step_begin) if in_flight and step_begin else 0.0},
            "step": st,
            "metrics": derived,
            "escalation": self.escalation_reason if self.policy.escalated else "",
        }

    def healthz(self) -> tuple[int, dict]:
        s = self.snapshot()
        ok = s["server"]["alive"] and s["device"]["responsive"] and self.policy.state in (State.READY, State.SELFTEST)
        return (200 if ok else 503), {"ok": ok, "state": self.policy.state.value, "device": s["device"]}

    def readyz(self) -> tuple[int, dict]:
        ok = self.policy.state == State.READY and self.policy.selftest_passed and self.proc is not None and self.proc.poll() is None
        return (200 if ok else 503), {"ready": ok, "state": self.policy.state.value, "selftest_passed": self.policy.selftest_passed}

    def metrics_text(self) -> str:
        s = self.snapshot()
        p = s["policy"]
        d = s["metrics"]
        st = s["step"]
        L = []

        def g(name, value, help_, labels=""):
            if value is None:
                return
            L.append(f"# HELP {name} {help_}")
            L.append(f"# TYPE {name} gauge")
            L.append(f"{name}{labels} {float(value)}")

        for state in State:
            L.append(f'glm53_supervisor_state{{state="{state.value}"}} {1 if self.policy.state == state else 0}')
        g("glm53_supervisor_server_up", 1 if s["server"]["alive"] else 0, "server process alive")
        g("glm53_supervisor_ready", 1 if self.readyz()[0] == 200 else 0, "server READY (health + self-test)")
        g("glm53_supervisor_device_responsive", 1 if s["device"]["responsive"] else 0, "no step in flight beyond stall_s")
        g("glm53_supervisor_last_step_age_seconds", s["device"]["last_step_age_s"], "age of the last completed engine step")
        g("glm53_supervisor_restarts_total", p["restarts"], "server starts by this supervisor")
        g("glm53_supervisor_failures_in_window", len(p["failures"]), "failures inside the escalation window")
        g("glm53_supervisor_resets_requested_total", p["resets_requested"], "idle-box resets requested")
        g("glm53_supervisor_escalations_total", p["escalations"], "escalations raised")
        g("glm53_supervisor_escalated", 1 if p["escalated"] else 0, "escalation marker active")
        g("glm53_supervisor_server_uptime_seconds", s["server"]["uptime_s"], "current server uptime")
        g("glm53_ttft_seconds", d.get("ttft_p50"), "time to first token (from vLLM histogram)", '{quantile="0.5"}')
        g("glm53_ttft_seconds", d.get("ttft_p99"), "time to first token (from vLLM histogram)", '{quantile="0.99"}')
        g("glm53_tpot_seconds", d.get("tpot_p50"), "time per output token (from vLLM histogram)", '{quantile="0.5"}')
        g("glm53_tpot_seconds", d.get("tpot_p99"), "time per output token (from vLLM histogram)", '{quantile="0.99"}')
        g("glm53_tpot_seconds_mean", d.get("tpot_mean"), "mean time per output token")
        g("glm53_queue_depth", d.get("queue_depth"), "requests waiting")
        g("glm53_live_slots", d.get("live_slots"), "requests running")
        g("glm53_kv_cache_usage", d.get("kv_cache_usage"), "KV cache usage fraction")
        g("glm53_mtp_acceptance_rate", d.get("mtp_acceptance_rate"), "accepted / drafted speculative tokens")
        steps = st.get("steps") or {}
        for k in ("prefill", "decode", "lane"):
            g("glm53_engine_steps_total", steps.get(k), "engine steps by kind (plugin step meter)", f'{{kind="{k}"}}')
        g("glm53_engine_step_ms_last", st.get("step_ms_last"), "last step duration ms")
        g("glm53_device_errors_total", st.get("device_errors"), "engine steps that raised")
        g("glm53_spec_accepted_tokens_total", st.get("spec_accepted_tokens"), "tokens committed by verify steps")
        g("glm53_spec_verify_rows_total", st.get("spec_verify_rows"), "rows committed by verify steps")
        return "\n".join(L) + "\n"

    def serve_http(self) -> None:
        sup = self
        host, port = self.cfg.listen.rsplit(":", 1)

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                return

            def do_GET(self):
                if self.path.startswith("/healthz"):
                    code, body = sup.healthz()
                    data, ctype = json.dumps(body).encode(), "application/json"
                elif self.path.startswith("/readyz"):
                    code, body = sup.readyz()
                    data, ctype = json.dumps(body).encode(), "application/json"
                elif self.path.startswith("/metrics"):
                    code, data, ctype = 200, sup.metrics_text().encode(), "text/plain; version=0.0.4"
                elif self.path.startswith("/status"):
                    code, data, ctype = 200, json.dumps(sup.snapshot(), indent=1).encode(), "application/json"
                else:
                    code, data, ctype = 404, b"not found\n", "text/plain"
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        srv = ThreadingHTTPServer((host, int(port)), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.log(f"listening on http://{self.cfg.listen} (/healthz /readyz /metrics /status)")

    def run(self) -> int:
        self.serve_http()
        threading.Thread(target=self.probe_thread, daemon=True).start()
        self.log(f"config: {json.dumps(asdict(self.cfg))[:1500]}")

        def on_signal(signum, frame):
            self.log(f"signal {signum}: stopping (the server gets SIGINT once it is up; never killed)")
            for a in self.policy.on_stop_requested(time.time()):
                self.act(a, time.time())

        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        for a in self.policy.start(time.time()):
            self.act(a, time.time())
        while not self._done:
            try:
                self.tick()
            except Exception as e:  # the supervisor itself never dies on a transient error
                self.log(f"tick error: {type(e).__name__}: {e}")
            time.sleep(self.cfg.tick_s)
        self.write_state(time.time(), False)
        return 0


def load_config(args) -> Config:
    d = {}
    if args.config:
        with open(args.config) as f:
            d = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    for k in ("server_cmd", "run_root", "port", "listen", "status_file", "pidfile", "selftest_cmd", "reset_marker", "reset_cmd", "reset_journal", "reset_group", "lock_dir", "coordinator_request", "escalation_marker", "stop_file", "api_marker", "host"):
        v = getattr(args, k, None)
        if v is not None:
            d[k] = v
    if args.no_kernel:
        d["kernel_cmd"] = None
    if args.no_selftest_log:
        d["selftest_from_log"] = False
    if args.policy:
        d.setdefault("policy", {}).update(json.loads(args.policy))
    if args.fast:
        d["tick_s"], d["health_probe_s"], d["scrape_s"], d["kernel_scan_s"] = 0.2, 0.5, 1.0, 5.0
    cfg = Config(**d)
    if isinstance(cfg.server_cmd, str) and not cfg.server_cmd.strip():
        raise SystemExit("server_cmd is required (config file or --server-cmd)")
    cfg.finalize()
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--server-cmd", dest="server_cmd", help="shell string that launches the server")
    ap.add_argument("--run-root", dest="run_root")
    ap.add_argument("--port", type=int)
    ap.add_argument("--host")
    ap.add_argument("--listen")
    ap.add_argument("--status-file", dest="status_file")
    ap.add_argument("--pidfile")
    ap.add_argument("--api-marker", dest="api_marker")
    ap.add_argument("--selftest-cmd", dest="selftest_cmd")
    ap.add_argument("--no-selftest-log", action="store_true", help="ignore SELFTEST lines in the server log")
    ap.add_argument("--reset-marker", dest="reset_marker")
    ap.add_argument("--reset-cmd", dest="reset_cmd")
    ap.add_argument("--reset-journal", dest="reset_journal")
    ap.add_argument("--reset-group", dest="reset_group")
    ap.add_argument("--lock-dir", dest="lock_dir")
    ap.add_argument("--coordinator-request", dest="coordinator_request")
    ap.add_argument("--escalation-marker", dest="escalation_marker")
    ap.add_argument("--stop-file", dest="stop_file")
    ap.add_argument("--no-kernel", action="store_true", help="skip the kernel-log scan")
    ap.add_argument("--policy", help='JSON overrides for PolicyConfig, e.g. {"stall_s": 5}')
    ap.add_argument("--fast", action="store_true", help="short tick / probe intervals (harness)")
    args = ap.parse_args(argv)
    cfg = load_config(args)
    return Supervisor(cfg).run()


if __name__ == "__main__":
    sys.exit(main())
