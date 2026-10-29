# SPDX-License-Identifier: Apache-2.0
"""GLM53_TTFT_PHASES=1 (default 0; program 2026-10-28T09:57:04Z ttft2k (c)): the host span between a cold prompt's
LMCache lookup and the backend's prefill step, split on the engine thread. The 2,048-token cold TTFT (421 ms) has
~9.9 ms there that no counter times (the scheduler step, the connector, the runner's input build).

For every step that runs a prompt's FIRST chunk (no computed tokens), one INFO line
  TT TTFT phases <req ids>: {"arrival_to_sched_ms", "sched_ms", "sched_to_runner_ms", "begin_step_ms", "drain_ms",
                             "build_ms", "segment_ms", "load_ms", "forward_ms", "forward_to_finish_ms", "finish_ms"}
- arrival_to_sched_ms: the request's arrival (vLLM Request.arrival_time, the API process' clock) -> the end of the
  schedule() that admitted it (the API -> engine hop, the queue, the lookup inside schedule());
- sched_ms: that schedule() call; sched_to_runner_ms: its end -> execute_model's entry (the engine core in between);
- begin_step .. load: _execute_model_impl's phases (connector begin_step, the async-decode drain, build_model_input,
  _segment_prepare, the connector load); forward_ms: the backend prefill (incl. its device wait);
- forward_to_finish_ms: the forward's end -> sample_tokens' finish call (the engine's work in between: async
  scheduling builds the next step first); finish_ms: the sample + the connector's saves.
Host-only timestamps (time.perf_counter); off = no cost."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_MARKS: dict = {}  # req id -> {"arrival_to_sched_ms", "sched_ms", "t_sched_end"} (the scheduler's stamp)


def on() -> bool:
    return os.environ.get("GLM53_TTFT_PHASES", "0") == "1"


def mark_scheduled(scheduler: Any, out: Any, t_start: float) -> None:
    """after schedule(): stamp every newly scheduled request that has no computed tokens (a cold first chunk)"""
    t_end, now = time.perf_counter(), time.time()
    for nr in getattr(out, "scheduled_new_reqs", None) or ():
        if int(getattr(nr, "num_computed_tokens", 0) or 0):
            continue
        req = getattr(scheduler, "requests", {}).get(nr.req_id)
        arrival = getattr(req, "arrival_time", None)
        _MARKS[nr.req_id] = {
            "arrival_to_sched_ms": round((now - arrival) * 1e3, 2) if arrival else None,
            "sched_ms": round((t_end - t_start) * 1e3, 2),
            "t_sched_end": t_end,
        }
    if len(_MARKS) > 4096:  # requests that never reached a runner (aborted): drop the oldest
        for k in list(_MARKS)[:2048]:
            _MARKS.pop(k, None)


class StepPhases:
    """the runner's phase stamps for one step: stamp(name) after each phase; finish() logs the first chunks' line"""

    def __init__(self, scheduler_output: Any):
        self.t = [("enter", time.perf_counter())]
        ids = [nr.req_id for nr in getattr(scheduler_output, "scheduled_new_reqs", None) or ()]
        self.marks = {i: _MARKS.pop(i) for i in ids if i in _MARKS}

    def stamp(self, name: str) -> None:
        self.t.append((name, time.perf_counter()))

    def report(self) -> dict | None:
        if not self.marks:
            return None
        out: dict = {}
        first = next(iter(self.marks.values()))
        out["arrival_to_sched_ms"] = first["arrival_to_sched_ms"]
        out["sched_ms"] = first["sched_ms"]
        out["sched_to_runner_ms"] = round((self.t[0][1] - first["t_sched_end"]) * 1e3, 2)
        for (_a, ta), (b, tb) in zip(self.t, self.t[1:]):
            out[f"{b}_ms"] = round((tb - ta) * 1e3, 2)
        return out

    def finish(self) -> None:
        rep = self.report()
        if rep is not None:
            logger.info("TT TTFT phases %s: %s", ",".join(self.marks), json.dumps(rep))
