"""Pure state-machine tests for supervisor_policy.Policy (no processes, no I/O)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from supervisor_policy import ExitInfo, Policy, PolicyConfig, State  # noqa: E402


def kinds(acts):
    return [a.kind for a in acts]


def boot(p: Policy, t=0.0):
    p.start(t)
    assert kinds(p.on_tick(t, False)) == ["spawn"]
    p.on_spawned(t)
    assert p.state == State.STARTING


def test_startup_health_selftest_ready():
    p = Policy(PolicyConfig(), selftest_configured=True)
    boot(p)
    p.on_health_ok(10)
    assert p.state == State.SELFTEST
    p.on_selftest(True, 20)
    assert p.state == State.READY and p.selftest_passed and p.ready_at == 20


def test_no_selftest_means_ready_on_health():
    p = Policy(PolicyConfig(), selftest_configured=False)
    boot(p)
    p.on_health_ok(10)
    assert p.state == State.READY and p.selftest_passed


def test_clean_exit_restarts_after_delay_no_reset():
    p = Policy(PolicyConfig(restart_delay_s=5))
    boot(p)
    p.on_health_ok(10)
    p.on_selftest(True, 11)
    acts = p.on_exit(ExitInfo(rc=0, uptime_s=100, device_opened=True, orderly_close=True), 111)
    assert "request_reset" not in kinds(acts) and p.state == State.RESTART_WAIT
    assert kinds(p.on_tick(113, False)) == []  # delay not elapsed
    assert kinds(p.on_tick(116, False)) == ["spawn"]
    assert p.resets_requested == 0


def test_exception_exit_with_mesh_closed_is_clean():
    p = Policy(PolicyConfig())
    boot(p)
    info = ExitInfo(rc=1, uptime_s=300, device_opened=True, orderly_close=True)
    assert p.classify_exit(info) == "clean"


def test_watcher_abort_requests_reset_then_restarts_on_done():
    p = Policy(PolicyConfig(reset_settle_s=0))
    boot(p)
    p.on_health_ok(10)
    p.on_selftest(True, 11)
    info = ExitInfo(rc=-6, uptime_s=500, device_opened=True, orderly_close=False,
                    fault="Watcher detected tripped assert and stopped device")
    acts = p.on_exit(info, 511)
    assert kinds(acts)[-1] == "request_reset" and p.state == State.RESET_WAIT
    assert kinds(p.on_tick(600, False)) == []  # nothing while the reset is pending
    p.on_reset_done(700)
    assert p.state == State.RESTART_WAIT
    assert kinds(p.on_tick(700, False)) == ["spawn"]
    assert p.resets_requested == 1


def test_timeout_signature_requests_reset():
    p = Policy(PolicyConfig())
    boot(p)
    info = ExitInfo(rc=1, uptime_s=500, device_opened=True, orderly_close=True,
                    fault="TT_THROW Device timeout detected ... is unrecoverable")
    assert p.classify_exit(info) == "fault"


def test_kernel_fault_requests_reset():
    p = Policy(PolicyConfig())
    info = ExitInfo(rc=0, uptime_s=500, device_opened=True, orderly_close=True,
                    kernel_fault="pcieport 0000:54:03.3: AER: Completion Timeout")
    assert p.classify_exit(info) == "fault"


def test_unclean_close_is_fault_unless_disabled():
    info = ExitInfo(rc=137, uptime_s=500, device_opened=True, orderly_close=False)
    assert Policy(PolicyConfig()).classify_exit(info) == "fault"
    assert Policy(PolicyConfig(unclean_close_is_fault=False)).classify_exit(info) == "clean"


def test_refused_launch_retries_without_counting():
    p = Policy(PolicyConfig(refused_retry_s=30, max_failures=2))
    boot(p)
    for i in range(5):
        info = ExitInfo(rc=1, uptime_s=2, device_opened=False, orderly_close=False)
        p.on_exit(info, 10 + i * 40)
        assert p.state == State.RESTART_WAIT and not p.escalated
        assert kinds(p.on_tick(10 + i * 40 + 31, False)) == ["spawn"]
        p.on_spawned(10 + i * 40 + 31)
    assert p.failures == []


def test_escalation_after_two_failures_in_window():
    p = Policy(PolicyConfig(max_failures=2, failure_window_s=1800, restart_delay_s=0))
    boot(p)
    p.on_exit(ExitInfo(rc=1, uptime_s=200, device_opened=True, orderly_close=True), 200)
    assert not p.escalated and p.state == State.RESTART_WAIT
    assert kinds(p.on_tick(200, False)) == ["spawn"]
    p.on_spawned(200)
    acts = p.on_exit(ExitInfo(rc=1, uptime_s=300, device_opened=True, orderly_close=True), 500)
    assert "escalate" in kinds(acts) and p.state == State.ESCALATED
    assert kinds(p.on_tick(600, False)) == []
    # the coordinator removes the marker -> restart, failures cleared
    assert kinds(p.on_escalation_cleared(700)) == ["log"]
    assert kinds(p.on_tick(700, False)) == ["spawn"] and p.failures == []


def test_failures_outside_window_do_not_escalate():
    p = Policy(PolicyConfig(max_failures=2, failure_window_s=1800, restart_delay_s=0))
    boot(p)
    p.on_exit(ExitInfo(rc=1, uptime_s=200, device_opened=True, orderly_close=True), 200)
    p.on_tick(200, False)
    p.on_spawned(200)
    p.on_exit(ExitInfo(rc=1, uptime_s=2000, device_opened=True, orderly_close=True), 2201)
    assert not p.escalated and p.state == State.RESTART_WAIT


def test_fault_exit_escalates_but_still_requests_reset():
    p = Policy(PolicyConfig(max_failures=2, restart_delay_s=0))
    boot(p)
    p.on_exit(ExitInfo(rc=1, uptime_s=200, device_opened=True, orderly_close=True), 200)
    p.on_tick(200, False)
    p.on_spawned(200)
    acts = p.on_exit(ExitInfo(rc=-6, uptime_s=50, device_opened=True, orderly_close=False, fault="watcher stopped"), 250)
    assert "escalate" in kinds(acts) and "request_reset" in kinds(acts) and p.state == State.RESET_WAIT
    p.on_reset_done(400)
    assert p.state == State.ESCALATED  # reset done, but no restart until cleared
    assert kinds(p.on_tick(401, False)) == []
    p.on_escalation_cleared(500)
    assert kinds(p.on_tick(500, False)) == ["spawn"]


def test_stall_sends_sigint_once_then_hung_after_grace_never_kill():
    p = Policy(PolicyConfig(exit_grace_s=100))
    boot(p)
    p.on_health_ok(10)
    p.on_selftest(True, 11)
    acts = p.on_stall(300, "step in flight 200 s")
    assert kinds(acts) == ["sigint"] and p.state == State.STOPPING
    assert kinds(p.on_stall(301, "again")) == []  # only once
    assert kinds(p.on_tick(350, True)) == []
    acts = p.on_tick(401, True)
    assert kinds(acts) == ["escalate"] and p.state == State.HUNG
    assert all(a.kind != "kill" for a in acts)


def test_fault_signature_while_alive_sigints_after_grace():
    p = Policy(PolicyConfig(fault_exit_grace_s=120))
    boot(p)
    p.on_health_ok(10)
    p.on_selftest(True, 11)
    p.on_fault_signature(100, "Timeout detected on device 3")
    assert kinds(p.on_tick(150, True)) == []
    assert kinds(p.on_tick(221, True)) == ["sigint"] and p.state == State.STOPPING


def test_selftest_fail_stops_and_counts():
    p = Policy(PolicyConfig(max_failures=2))
    boot(p)
    p.on_health_ok(10)
    acts = p.on_selftest(False, 11, "top1 0.62")
    assert kinds(acts) == ["sigint"] and p.state == State.STOPPING and len(p.failures) == 1
    p.on_exit(ExitInfo(rc=0, uptime_s=20, device_opened=True, orderly_close=True), 30)
    assert p.state == State.RESTART_WAIT and len(p.failures) == 1  # one incident


def test_hung_exit_with_leftover_session_escalates_without_restart():
    p = Policy(PolicyConfig())
    boot(p)
    acts = p.on_exit(ExitInfo(rc=0, uptime_s=100, device_opened=True, orderly_close=False, descendants_alive=True, nodes_held=True), 100)
    assert "escalate" in kinds(acts) and p.state == State.HUNG
    assert kinds(p.on_tick(200, False)) == []


def test_stop_requested_paths():
    p = Policy(PolicyConfig())
    boot(p)
    # during startup: no signal until health, then SIGINT
    assert kinds(p.on_stop_requested(5)) == ["log"]
    assert kinds(p.on_health_ok(10)) == ["sigint"] and p.state == State.STOPPING
    acts = p.on_exit(ExitInfo(rc=0, uptime_s=12, device_opened=True, orderly_close=True), 12)
    assert kinds(acts)[-1] == "stopped" and p.state == State.STOPPED
    # idle states stop at once
    q = Policy(PolicyConfig())
    q.start(0)
    assert kinds(q.on_stop_requested(0)) == ["stopped"]


def test_startup_alert_only_logs():
    p = Policy(PolicyConfig(startup_timeout_s=100))
    boot(p)
    acts = p.on_tick(150, True)
    assert kinds(acts) == ["log"] and p.state == State.STARTING
    assert kinds(p.on_tick(160, True)) == []


def test_stall_then_exit_counts_one_failure():
    p = Policy(PolicyConfig(max_failures=2, restart_delay_s=0))
    boot(p)
    p.on_health_ok(10)
    p.on_selftest(True, 11)
    p.on_stall(300, "step in flight")
    p.on_exit(ExitInfo(rc=0, uptime_s=300, device_opened=True, orderly_close=True), 310)
    assert len(p.failures) == 1 and not p.escalated and p.state == State.RESTART_WAIT
    assert kinds(p.on_tick(310, False)) == ["spawn"]


def test_selftest_fail_then_exit_counts_one_failure():
    p = Policy(PolicyConfig(max_failures=2, restart_delay_s=0))
    boot(p)
    p.on_health_ok(10)
    p.on_selftest(False, 11, "top1 low")
    p.on_exit(ExitInfo(rc=0, uptime_s=20, device_opened=True, orderly_close=True), 30)
    assert len(p.failures) == 1 and not p.escalated
    p.on_tick(30, False); p.on_spawned(30); p.on_health_ok(40)
    p.on_selftest(False, 41, "top1 low again")
    assert p.escalated  # the second incident
