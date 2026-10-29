#!/usr/bin/env python3
"""Recovery policy of the GLM-5.3-Flash serving supervisor: a pure state machine, no I/O.

The daemon (glm53_supervisor.py) feeds it events and executes the actions it returns.
Rules it encodes (doc/hardware.md 9b, knowledge/LEARNED_failures.md override + the
2026-10-22 watcher-abort clarification, serve_window.sh / dirty_check.sh):

* a device process that raised and EXITED with its mesh closed needs no reset: restart;
* a watcher-stopped abort, a device timeout / "unrecoverable", a device-failure signature
  in its log, a Completion Timeout / all-1s kernel line, or an exit without the orderly
  mesh close -> the box needs a reset on an idle box: request it, wait for RESET DONE;
* the device process tree only ever gets SIGINT (once up), never SIGKILL: a process that
  does not exit is reported (HUNG + escalation) and waited for;
* a launch refused before any device open (lock held, QUIESCE, pre-flight) is retried,
  not counted;
* max_failures failures inside failure_window_s -> ESCALATED: no more automatic
  restarts until the coordinator clears the marker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    INIT = "init"
    STARTING = "starting"  # spawned, waiting for /health
    SELFTEST = "selftest"  # /health ok, startup self-test running
    READY = "ready"  # serving
    STOPPING = "stopping"  # SIGINT sent, waiting for the exit
    HUNG = "hung"  # did not exit / descendants still hold devices: operator
    RESET_WAIT = "reset_wait"  # reset requested, waiting for RESET DONE
    RESTART_WAIT = "restart_wait"  # exited, waiting for the restart delay
    ESCALATED = "escalated"  # too many failures: waiting for the coordinator
    STOPPED = "stopped"  # supervisor asked to stop and the server is gone


# Device-failure signatures (dirty_check.sh SIG + the watcher / TT_THROW timeout forms).
FAULT_RE = re.compile(
    r"Try resetting the board|failed to initialize FW|Firmware startup error|Failed to open device"
    r"|Timeout detected|device timeout|potential hang|is unrecoverable|waiting for physical cores"
    r"|the board should be reset|should reset the board|appears hung| is hung|SIGBUS signal detected"
    r"|Device access failed|MMIO per-op timeout|Memory barrier timed out|DMA timeout"
    r"|training.* timed out|training failed|Watcher detected|watcher.*(stopped|assert)"
    r"|detected tripped assert|all-1s|all ones|Completion Timeout",
    re.IGNORECASE,
)
# Kernel-log lines that mark a program chip or its root port as misbehaving.
KERNEL_FAULT_RE = re.compile(
    r"Completion Timeout|CmpltTO|all-1s|0xffffffff|\[Hardware Error\]|UnsupReq|MalfTLP|device gone",
    re.IGNORECASE,
)
KERNEL_SUBJECT_RE = re.compile(r"tenstorrent|pcieport|Hardware Error|ghes|AER", re.IGNORECASE)
SELFTEST_RE = re.compile(r"SELFTEST (PASS|FAIL)")
DEVICE_OPEN_RE = re.compile(r"Opening user mode device driver|Opening local chip ids")
ORDERLY_CLOSE_RE = re.compile(r"Closing devices in cluster completed")


@dataclass
class PolicyConfig:
    max_failures: int = 2
    failure_window_s: float = 1800.0
    restart_delay_s: float = 5.0
    refused_retry_s: float = 30.0
    early_exit_s: float = 60.0  # an exit this early without a device open = a refusal
    stall_s: float = 180.0  # a step in flight longer than this = a stall
    health_fail_s: float = 60.0  # /health failing this long while READY = a stall
    fault_exit_grace_s: float = 120.0  # a fault line without an exit: SIGINT after this
    exit_grace_s: float = 900.0  # after SIGINT: HUNG + escalation after this (no kill)
    startup_timeout_s: float = 1800.0  # alert (log) only: SIGINT is ignored during startup
    reset_settle_s: float = 0.0  # the reset chain holds 150 s itself; extra wait here
    unclean_close_is_fault: bool = True  # exit with the mesh open and no orderly close


@dataclass
class ExitInfo:
    rc: int | None
    uptime_s: float
    device_opened: bool
    orderly_close: bool
    fault: str | None = None  # the first device-failure line (server log)
    kernel_fault: str | None = None  # the first kernel fault line
    descendants_alive: bool = False  # processes of the server's session still alive
    nodes_held: bool = False  # ... and some hold /dev/tenstorrent


@dataclass
class Action:
    kind: str  # spawn | sigint | request_reset | escalate | log | stopped
    reason: str = ""


@dataclass
class Policy:
    cfg: PolicyConfig = field(default_factory=PolicyConfig)
    selftest_configured: bool = True
    state: State = State.INIT
    failures: list[tuple[float, str]] = field(default_factory=list)
    escalated: bool = False
    stop_requested: bool = False
    started_at: float = 0.0
    ready_at: float = 0.0
    stopping_since: float = 0.0
    next_spawn_at: float = 0.0
    fault_seen: str = ""
    fault_seen_at: float = 0.0
    sigint_sent: bool = False
    startup_alerted: bool = False
    restarts: int = 0
    failure_recorded: bool = False  # this run's incident already counted (stall / self-test)
    resets_requested: int = 0
    escalations: int = 0
    last_exit: str = ""
    selftest_passed: bool = False

    # -- helpers -----------------------------------------------------------
    def _record_failure(self, now: float, reason: str) -> bool:
        """Append a failure; True when the window now holds max_failures."""
        self.failures = [(t, r) for t, r in self.failures if now - t < self.cfg.failure_window_s]
        self.failures.append((now, reason))
        return len(self.failures) >= self.cfg.max_failures

    def _escalate(self, now: float, reason: str) -> list[Action]:
        self.escalated = True
        self.escalations += 1
        return [Action("escalate", reason)]

    def _schedule_restart(self, now: float, delay: float) -> list[Action]:
        if self.stop_requested:
            self.state = State.STOPPED
            return [Action("stopped", "stop requested")]
        if self.escalated:
            self.state = State.ESCALATED
            return [Action("log", "escalated: no automatic restart until the marker is cleared")]
        self.state = State.RESTART_WAIT
        self.next_spawn_at = now + delay
        return []

    def classify_exit(self, info: ExitInfo) -> str:
        if info.descendants_alive:
            return "hung"
        if info.fault or info.kernel_fault:
            return "fault"
        if not info.device_opened and info.uptime_s < self.cfg.early_exit_s:
            return "refused"
        if info.device_opened and not info.orderly_close and self.cfg.unclean_close_is_fault:
            return "fault"
        return "clean"

    # -- events ------------------------------------------------------------
    def start(self, now: float) -> list[Action]:
        self.state = State.RESTART_WAIT
        self.next_spawn_at = now
        return []

    def on_spawned(self, now: float) -> list[Action]:
        self.state = State.STARTING
        self.started_at = now
        self.ready_at = 0.0
        self.fault_seen = ""
        self.fault_seen_at = 0.0
        self.sigint_sent = False
        self.startup_alerted = False
        self.selftest_passed = False
        self.failure_recorded = False
        self.restarts += 1
        return [Action("log", f"spawned (start #{self.restarts})")]

    def on_health_ok(self, now: float) -> list[Action]:
        if self.state != State.STARTING:
            return []
        if self.stop_requested:
            # vLLM ignores SIGINT until it serves: stop it the moment it is up
            self.state = State.STOPPING
            self.stopping_since = now
            self.sigint_sent = True
            return [Action("sigint", "stop requested during startup")]
        if self.selftest_configured:
            self.state = State.SELFTEST
            return [Action("log", f"health ok after {now - self.started_at:.0f} s; running the self-test")]
        self.state = State.READY
        self.ready_at = now
        self.selftest_passed = True
        return [Action("log", f"READY after {now - self.started_at:.0f} s (no self-test configured)")]

    def on_selftest(self, passed: bool, now: float, detail: str = "") -> list[Action]:
        if self.state != State.SELFTEST:
            return []
        if passed:
            self.state = State.READY
            self.ready_at = now
            self.selftest_passed = True
            return [Action("log", f"SELFTEST PASS: READY after {now - self.started_at:.0f} s {detail}".rstrip())]
        self.state = State.STOPPING
        self.stopping_since = now
        self.sigint_sent = True
        acts = [Action("sigint", f"SELFTEST FAIL {detail}".rstrip())]
        self.failure_recorded = True
        if self._record_failure(now, "selftest fail"):
            acts += self._escalate(now, f"self-test failed {len(self.failures)}x within the window: {detail}")
        return acts

    def on_stall(self, now: float, why: str) -> list[Action]:
        if self.state not in (State.READY, State.SELFTEST):
            return []
        self.state = State.STOPPING
        self.stopping_since = now
        self.sigint_sent = True
        self.failure_recorded = True
        self._record_failure(now, f"stall: {why}")
        return [Action("sigint", f"stall: {why}")]

    def on_fault_signature(self, now: float, line: str) -> list[Action]:
        """A device-failure line while the process is still alive."""
        if self.fault_seen:
            return []
        self.fault_seen = line
        self.fault_seen_at = now
        return [Action("log", f"device-failure signature while alive: {line[:200]}")]

    def on_exit(self, info: ExitInfo, now: float) -> list[Action]:
        kind = self.classify_exit(info)
        self.last_exit = f"{kind} rc={info.rc} after {info.uptime_s:.0f} s"
        desc = (
            f"rc={info.rc} uptime={info.uptime_s:.0f}s opened={info.device_opened} "
            f"closed={info.orderly_close} fault={bool(info.fault or info.kernel_fault)}"
        )
        acts = [Action("log", f"exit classified {kind}: {desc}")]
        if self.stop_requested and kind != "hung":
            self.state = State.STOPPED
            return acts + [Action("stopped", "stop requested")]
        if kind == "hung":
            self.state = State.HUNG
            if not self.failure_recorded:
                self.failure_recorded = True
                self._record_failure(now, "hung after exit")
            return acts + self._escalate(
                now,
                "server exited but its session still holds processes"
                + (" with /dev/tenstorrent open" if info.nodes_held else "")
                + ": never killed; operator needed",
            )
        if kind == "refused":
            self.state = State.RESTART_WAIT
            self.next_spawn_at = now + self.cfg.refused_retry_s
            return acts + [Action("log", f"launch refused before any device open: retry in {self.cfg.refused_retry_s:.0f} s")]
        # one incident, one failure: a stall / self-test failure already counted this run
        escalate = False if self.failure_recorded else self._record_failure(now, f"{kind} exit")
        if escalate:
            acts += self._escalate(now, f"{len(self.failures)} failures within {self.cfg.failure_window_s / 60:.0f} min: "
                                        + "; ".join(r for _, r in self.failures))
        if kind == "fault":
            self.state = State.RESET_WAIT
            self.resets_requested += 1
            reason = info.fault or info.kernel_fault or "exit without the orderly mesh close"
            return acts + [Action("request_reset", reason[:300])]
        return acts + self._schedule_restart(now, self.cfg.restart_delay_s)

    def on_reset_done(self, now: float) -> list[Action]:
        if self.state != State.RESET_WAIT:
            return []
        acts = [Action("log", f"RESET DONE; restart in {self.cfg.reset_settle_s:.0f} s")]
        return acts + self._schedule_restart(now, self.cfg.reset_settle_s)

    def on_escalation_cleared(self, now: float) -> list[Action]:
        if not self.escalated:
            return []
        self.escalated = False
        self.failures = []
        if self.state in (State.ESCALATED, State.HUNG):
            self.state = State.RESTART_WAIT
            self.next_spawn_at = now
            return [Action("log", "escalation cleared by the coordinator: restarting")]
        return [Action("log", "escalation cleared by the coordinator")]

    def on_stop_requested(self, now: float) -> list[Action]:
        self.stop_requested = True
        if self.state in (State.READY, State.SELFTEST):
            self.state = State.STOPPING
            self.stopping_since = now
            self.sigint_sent = True
            return [Action("sigint", "stop requested")]
        if self.state in (State.RESTART_WAIT, State.RESET_WAIT, State.ESCALATED, State.INIT):
            self.state = State.STOPPED
            return [Action("stopped", "stop requested")]
        return [Action("log", f"stop requested in state {self.state.value}: waiting for the server")]

    def on_tick(self, now: float, process_alive: bool) -> list[Action]:
        acts: list[Action] = []
        if self.state == State.RESTART_WAIT and now >= self.next_spawn_at:
            if self.escalated:
                self.state = State.ESCALATED
            elif self.stop_requested:
                self.state = State.STOPPED
                acts.append(Action("stopped", "stop requested"))
            else:
                acts.append(Action("spawn"))
        elif self.state == State.STARTING and not self.startup_alerted and now - self.started_at > self.cfg.startup_timeout_s:
            self.startup_alerted = True
            acts.append(Action("log", f"ALERT: no /health after {now - self.started_at:.0f} s (SIGINT is ignored during startup: waiting)"))
        elif self.state == State.STOPPING and now - self.stopping_since > self.cfg.exit_grace_s:
            self.state = State.HUNG
            acts += self._escalate(now, f"no exit {self.cfg.exit_grace_s:.0f} s after SIGINT: NOT killed (device process); operator needed")
        if (
            self.fault_seen
            and process_alive
            and not self.sigint_sent
            and self.state in (State.READY, State.SELFTEST, State.STARTING)
            and now - self.fault_seen_at > self.cfg.fault_exit_grace_s
        ):
            self.sigint_sent = True
            self.state = State.STOPPING
            self.stopping_since = now
            acts.append(Action("sigint", f"device-failure signature {self.cfg.fault_exit_grace_s:.0f} s ago without an exit"))
        return acts

    def snapshot(self) -> dict:
        return {
            "state": self.state.value,
            "escalated": self.escalated,
            "stop_requested": self.stop_requested,
            "failures": [{"t": t, "reason": r} for t, r in self.failures],
            "restarts": self.restarts,
            "resets_requested": self.resets_requested,
            "escalations": self.escalations,
            "last_exit": self.last_exit,
            "selftest_passed": self.selftest_passed,
            "started_at": self.started_at,
            "ready_at": self.ready_at,
            "fault_seen": self.fault_seen[:200],
        }
