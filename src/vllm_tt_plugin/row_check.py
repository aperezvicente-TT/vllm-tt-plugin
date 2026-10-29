# SPDX-License-Identifier: Apache-2.0
"""A prefill whose persistent-batch page-table row is SHORTER than the positions it writes (Stage 17, the Galaxy blk2
engine death 2026-10-29T01:53Z: a request resumed after a failed preempt-save recomputed from 0, and its third chunk
reached the model with a 65-block row against the 96 blocks vLLM had allocated; the model's own guard raised, and
_latch_device_faults took the whole engine down with 8 requests in flight).

The runner checks every prefill row BEFORE any device work. When every row of the step is short it returns an empty
output for them (no token, as an intermediate chunk) and records the requests here. The scheduler finishes them ABORTED
at its next schedule (stop_reason 'page row: ...', the client notified at the next update_from_output, as
plan_check's refusals), and every other request keeps stepping. A step that mixes short and sound rows still raises:
part of a prefill cannot be refused. One process (uniproc): a module-level registry, as plan_check / kv_connector."""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_REFUSED: dict[str, str] = {}


def refuse(req_id: str, why: str) -> None:
    """the runner's record: this request's prefill was refused (the first reason is kept)"""
    with _LOCK:
        _REFUSED.setdefault(str(req_id), str(why))


def take() -> dict[str, str]:
    """the scheduler's read: every refusal recorded since the last read (cleared)"""
    with _LOCK:
        out = dict(_REFUSED)
        _REFUSED.clear()
    return out
