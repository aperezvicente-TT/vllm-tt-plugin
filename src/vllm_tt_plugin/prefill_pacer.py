# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""v3b4 SHORT_FIRST pacing at the device (program 2026-10-28T20:11:59Z, the mesh f2i3b3 phase-2 RED).

The served step loop is vLLM's EngineCore.step_with_batch_queue over the uniproc executor: execute_model and
sample_tokens run inline, and the batch queue's futures complete when the runner RETURNS. A head-less intermediate
prefill chunk run returns as soon as its replays are enqueued (no read-back), so a lone 131K's 16 runs were submitted
0.04 s apart and the scheduler never ran while the shorts waited; a scheduler-side in-flight count (v3b3) was released
at the same host pace and never throttled anything.

Here the runner itself waits: after a head-less chunk run is enqueued it records a CQ0 event, and it waits for the run
GLM53_PREFILL_SHORT_FIRST_INFLIGHT - 1 behind it (default N = 2: the one before). execute_model then returns at the
device pace with one run still queued (the device never idles on the host), and the scheduler runs once per device run:
a short arriving mid-prefill is scheduled behind at most N runs. Only with GLM53_PREFILL_SHORT_FIRST = T; a step with a
head is not paced (its read-back waits for the queue anyway). The replays stay traced; the events are CQ commands."""

from __future__ import annotations

import collections
import time
from typing import Any

from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin.scheduler import prefill_short_first_inflight, prefill_short_first_tokens

logger = init_tt_logger(__name__)


class PrefillPacer:
    """the head-less chunk runs this runner has enqueued and not yet seen finish (their CQ0 events), oldest first"""

    def __init__(self, device: Any):
        self.device = device
        self.pending: collections.deque = collections.deque()
        # the counters line (kv_connector._COUNTER_KEYS["runner"])
        self.prefill_short_first_paced = 0  # head-less runs after which the runner waited for an earlier run
        self.prefill_short_first_paced_wait_s = 0.0  # host time spent in those waits (the device was busy)

    def after_submit(self, cap: int) -> None:
        import ttnn  # noqa: PLC0415

        self.pending.append(ttnn.record_event(self.device, 0))
        waited, popped = 0.0, 0
        while len(self.pending) > max(0, cap - 1):
            t0 = time.perf_counter()
            ttnn.event_synchronize(self.pending.popleft())
            waited += time.perf_counter() - t0
            popped += 1
        if not popped:
            return  # the first run of a stretch: nothing earlier to wait for
        n = self.prefill_short_first_paced = self.prefill_short_first_paced + 1
        self.prefill_short_first_paced_wait_s = round(self.prefill_short_first_paced_wait_s + waited, 3)
        if n <= 5 or n % 500 == 0:
            logger.info("TT runner: SHORT_FIRST pacing (%d): waited %.3f s for the chunk run %d behind (INFLIGHT=%d)",
                        n, waited, max(1, cap - 1), cap)  # fmt: skip


def _headless(model_input: Any) -> bool:
    """a prefill step every row of which is an intermediate chunk (no head, no read-back)"""
    if getattr(model_input, "prompt_lens", None) is None:
        return False  # a decode step
    mask = getattr(model_input, "intermediate_prefill_mask", None)
    return mask is not None and len(mask) > 0 and bool(mask.all())


def after_prefill_submit(runner: Any, model_input: Any) -> None:
    """the runner's prefill path, right after the step's forward was enqueued (before execute_model returns)"""
    if prefill_short_first_tokens() <= 0:
        return
    cap = prefill_short_first_inflight()
    if cap <= 0 or not _headless(model_input):
        return
    pacer = runner.__dict__.get("_prefill_pacer")
    if pacer is None:
        from vllm_tt_plugin.kv_connector import _COUNTER_SOURCES  # noqa: PLC0415  (the counters line)

        pacer = runner.__dict__["_prefill_pacer"] = PrefillPacer(runner.mesh_device)
        _COUNTER_SOURCES["runner"].add(pacer)
    pacer.after_submit(cap)
