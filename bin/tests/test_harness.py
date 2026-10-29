"""End-to-end harness: the real supervisor daemon driving the fake server through every
recovery path. Host-only (no device, no vLLM). Each test is one scenario, ~5-10 s."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

BIN = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception:
        return None, ""


def wait_for(pred, timeout, what):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(0.2)
    raise AssertionError(f"timeout {timeout}s waiting for {what}")


class Sup:
    def __init__(self, tmp, policy=None, fake_args="", sup_args=()):
        self.tmp = tmp
        self.port = free_port()
        self.lport = free_port()
        self.ctl = os.path.join(tmp, "ctl")
        self.root = os.path.join(tmp, "root")
        self.marker = os.path.join(tmp, "RESET_PENDING.all")
        self.journal = os.path.join(tmp, "reset.journal")
        self.coord = os.path.join(tmp, "COORDINATOR_REQUEST.txt")
        os.makedirs(os.path.join(tmp, "locks"), exist_ok=True)
        open(self.journal, "a").close()
        cmd = f"exec {PY} {BIN}/fake_server.py --port {self.port} --ctl {self.ctl} --startup-s 1 {fake_args}"
        pol = {"restart_delay_s": 0.5, "refused_retry_s": 1, "early_exit_s": 5, "stall_s": 2, "health_fail_s": 3,
               "fault_exit_grace_s": 2, "exit_grace_s": 4, "max_failures": 2, "failure_window_s": 1800}
        pol.update(policy or {})
        args = [PY, f"{BIN}/glm53_supervisor.py", "--server-cmd", cmd, "--run-root", self.root, "--port", str(self.port),
                "--listen", f"127.0.0.1:{self.lport}", "--reset-marker", self.marker, "--reset-journal", self.journal,
                "--lock-dir", os.path.join(tmp, "locks"), "--no-kernel", "--fast", "--api-marker", "fake_server.py",
                "--coordinator-request", self.coord, "--policy", json.dumps(pol), *sup_args]
        self.log = open(os.path.join(tmp, "supervisor.stderr"), "wb")
        self.p = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT)

    def ctl_cmd(self, cmd):
        with open(self.ctl + ".tmp", "w") as f:
            f.write(cmd)
        os.replace(self.ctl + ".tmp", self.ctl)

    def status(self):
        code, body = get(f"http://127.0.0.1:{self.lport}/status")
        return json.loads(body) if code == 200 else {}

    def state(self):
        return self.status().get("policy", {}).get("state")

    def readyz(self):
        return get(f"http://127.0.0.1:{self.lport}/readyz")[0]

    def healthz(self):
        return get(f"http://127.0.0.1:{self.lport}/healthz")[0]

    def metrics(self):
        return get(f"http://127.0.0.1:{self.lport}/metrics")[1]

    def suplog(self):
        try:
            return open(os.path.join(self.root, "supervisor.log")).read()
        except OSError:
            return ""

    def wait_ready(self, timeout=20):
        wait_for(lambda: self.readyz() == 200, timeout, "readyz 200")

    def wait_restart(self, n, timeout=25):
        """The n-th start is READY (an injected exit was seen and the server came back)."""
        wait_for(lambda: self.status().get("policy", {}).get("restarts") == n and self.readyz() == 200, timeout, f"start #{n} ready")

    def api_pid(self):
        return self.status().get("server", {}).get("api_pid")

    def stop(self):
        """Orderly: STOP file -> SIGINT to the fake -> supervisor exits; then make sure nothing is left."""
        open(os.path.join(self.root, "STOP"), "w").close()
        try:
            self.p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.p.terminate()
            self.p.wait(timeout=5)
        self.log.close()
        # host-only fakes may be left in hang mode: clean the test's own processes
        subprocess.run(["pkill", "-f", f"fake_server.py --port {self.port} "], check=False)


@pytest.fixture
def sup(tmp_path):
    holder = {}

    def make(**kw):
        holder["s"] = Sup(str(tmp_path), **kw)
        return holder["s"]

    yield make
    if "s" in holder:
        holder["s"].stop()


def test_ready_metrics_and_clean_exit_restart(sup):
    s = sup()
    s.wait_ready()
    assert s.healthz() == 200
    st = s.status()
    assert st["policy"]["selftest_passed"] and st["server"]["device_opened"]
    wait_for(lambda: 'glm53_engine_steps_total{kind="decode"}' in s.metrics() and 'glm53_ttft_seconds{quantile="0.5"}' in s.metrics(), 10, "scraped metrics")
    m = s.metrics()
    assert 'glm53_ttft_seconds{quantile="0.99"}' in m and "glm53_tpot_seconds" in m
    assert "glm53_queue_depth 3.0" in m and "glm53_live_slots 2.0" in m
    assert "glm53_mtp_acceptance_rate" in m and "glm53_spec_accepted_tokens_total" in m
    assert 'glm53_supervisor_state{state="ready"} 1' in m
    assert "SELFTEST PASS" in s.suplog()
    # injected clean exit -> restarted (acceptance cell: within 120 s; here within 20 s)
    t0 = time.time()
    s.ctl_cmd("exit0")
    wait_for(lambda: s.readyz() != 200, 5, "readyz drop")
    s.wait_ready(20)
    assert time.time() - t0 < 120
    st = s.status()["policy"]
    assert st["restarts"] == 2 and st["resets_requested"] == 0 and not st["escalated"]
    assert "exit classified clean" in s.suplog()
    assert not os.path.exists(s.marker)


def test_exception_exit_with_mesh_closed_restarts_without_reset(sup):
    s = sup()
    s.wait_ready()
    s.ctl_cmd("crash")
    wait_for(lambda: "exit classified clean: rc=1" in s.suplog(), 10, "crash classified clean")
    s.wait_ready(20)
    assert not os.path.exists(s.marker)
    assert s.status()["policy"]["failures"] and len(s.status()["policy"]["failures"]) == 1


def test_watcher_abort_requests_reset_and_restarts_on_journal_done(sup):
    s = sup()
    s.wait_ready()
    s.ctl_cmd("watcher")
    wait_for(lambda: s.state() == "reset_wait", 15, "reset_wait")
    assert os.path.exists(s.marker) and "Watcher" in open(s.marker).read()
    assert "exit classified fault" in s.suplog() and s.readyz() == 503
    assert "RESET REQUESTED" in open(s.coord).read()
    time.sleep(1.5)
    assert s.state() == "reset_wait"  # nothing restarts before RESET DONE
    with open(s.journal, "a") as f:
        f.write("2026-10-22T10:00:00Z reset all done rc=0\n")
    s.wait_ready(20)
    assert "RESET DONE seen in" in s.suplog()
    assert s.status()["policy"]["resets_requested"] == 1


def test_device_timeout_requests_reset_and_restarts_when_marker_removed(sup):
    s = sup()
    s.wait_ready()
    s.ctl_cmd("timeout")
    wait_for(lambda: s.state() == "reset_wait", 15, "reset_wait")
    assert "unrecoverable" in open(s.marker).read()
    os.unlink(s.marker)  # chain_reset.sh removes the marker on rc=0
    s.wait_ready(20)
    assert "RESET DONE: the marker was removed" in s.suplog()


def test_stall_gets_sigint_then_restarts(sup):
    s = sup()
    s.wait_ready()
    s.ctl_cmd("stall")
    wait_for(lambda: "SIGINT -> pid" in s.suplog() and "stall: step" in s.suplog(), 15, "stall SIGINT")
    wait_for(lambda: s.healthz() == 503, 5, "healthz 503 during the stall")  # device not responsive
    s.wait_ready(20)  # the fake closed orderly on SIGINT -> clean restart
    assert s.status()["policy"]["restarts"] == 2
    assert not os.path.exists(s.marker)


def test_health_outage_counts_as_stall(sup):
    s = sup(policy={"health_fail_s": 2})
    s.wait_ready()
    s.ctl_cmd("nohealth")
    wait_for(lambda: "/health failing" in s.suplog(), 20, "health stall")
    s.wait_ready(25)


def test_hung_server_is_never_killed_and_escalates(sup):
    s = sup(policy={"exit_grace_s": 3})
    s.wait_ready()
    pid = s.api_pid()
    s.ctl_cmd("hang")
    time.sleep(0.5)
    s.ctl_cmd("stall")  # -> SIGINT, which the hung fake ignores
    wait_for(lambda: s.state() == "hung", 20, "hung")
    assert os.path.exists(s.root + "/SUPERVISOR_ESCALATION")
    assert "NOT killed" in s.suplog() and "ESCALATION:" in s.suplog()
    os.kill(pid, 0)  # still alive: nothing SIGKILLed it
    time.sleep(1)
    os.kill(pid, 0)
    s.ctl_cmd("exit0")  # the "operator" resolves it
    wait_for(lambda: s.status()["server"]["alive"] is False, 10, "fake gone")
    os.unlink(s.root + "/SUPERVISOR_ESCALATION")
    s.wait_ready(20)
    assert "escalation cleared" in s.suplog()


def test_two_failures_in_window_escalate_and_coordinator_clears(sup):
    s = sup()
    s.wait_ready()
    s.ctl_cmd("crash")
    s.wait_restart(2)
    s.ctl_cmd("crash")
    wait_for(lambda: s.state() == "escalated", 15, "escalated")
    marker = s.root + "/SUPERVISOR_ESCALATION"
    assert os.path.exists(marker) and "2 failures" in open(marker).read()
    assert "ESCALATION:" in s.suplog() and "ESCALATION" in open(s.coord).read()
    time.sleep(2)
    assert s.status()["policy"]["restarts"] == 2  # no third start while escalated
    os.unlink(marker)
    s.wait_ready(20)
    assert s.status()["policy"]["restarts"] == 3 and s.status()["policy"]["failures"] == []


def test_selftest_fail_in_log_stops_the_server(sup):
    s = sup(fake_args="--selftest fail")
    wait_for(lambda: "SELFTEST FAIL" in s.suplog() and "SIGINT -> pid" in s.suplog(), 20, "selftest fail -> SIGINT")
    assert s.readyz() == 503
    wait_for(lambda: s.state() == "escalated", 30, "escalated after the second self-test failure")
    assert "self-test failed 2x" in s.suplog()


def test_selftest_command_exit_code(sup):
    s = sup(fake_args="--selftest none", sup_args=("--selftest-cmd", "test -n \"$SERVER_URL\" && curl -sf $SERVER_URL/v1/models >/dev/null", "--no-selftest-log"))
    s.wait_ready(20)
    assert "self-test command rc=0" in s.suplog()
    assert json.loads(get(f"http://127.0.0.1:{s.lport}/readyz")[1])["selftest_passed"] is True


def test_refused_launch_retries_without_failures(sup):
    s = sup(fake_args="--refuse", policy={"refused_retry_s": 0.5})
    wait_for(lambda: s.status().get("policy", {}).get("restarts", 0) >= 3, 15, "3 retries")
    st = s.status()["policy"]
    assert st["failures"] == [] and not st["escalated"]
    assert "launch refused before any device open" in s.suplog()


def test_stop_file_shuts_down_orderly(sup):
    s = sup()
    s.wait_ready()
    pid = s.api_pid()
    open(s.root + "/STOP", "w").close()
    s.p.wait(timeout=20)
    assert s.p.returncode == 0
    assert "SIGINT -> pid" in s.suplog() and "stopped (stop requested)" in s.suplog()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    st = json.load(open(s.root + "/state.json"))
    assert st["policy"]["state"] == "stopped"
    assert "Closing devices in cluster completed" in open(os.path.join(st["server"]["run_dir"], "server.log")).read()
