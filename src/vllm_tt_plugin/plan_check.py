# SPDX-License-Identifier: Apache-2.0
"""The model's host-only prefill PLAN check, seen from the scheduler side (program 2026-10-26T17:09:24Z, a k3 MUST):
a prefill whose planning raises (TPPrefill.plan / clamp_lengths: 'chunk [s, s+c) past the cache') used to raise inside
the runner step, where _latch_device_faults latched it as a device fault and took the engine down. The model runner
registers the model's ``prefill_plan_error(start_pos, n) -> str | None`` (no device work) here at load; the scheduler
refuses a request whose remaining prefill cannot plan (finished aborted, the engine keeps stepping) and the connector
steps a prefix hit down to a checkpoint whose continuation plans. Absent (another model, or none registered): every
check passes, as before."""

from __future__ import annotations

from typing import Callable

_CHECK: list[Callable[[int, int], "str | None"]] = []


def register(fn: Callable[[int, int], "str | None"] | None) -> None:
    """the runner's hook: the model's host-only plan check (None: clears it)"""
    _CHECK.clear()
    if fn is not None:
        _CHECK.append(fn)


def plan_error(start_pos: int, n: int) -> str | None:
    """the plan error of a prefill of n tokens at start_pos, or None (also when no model registered a check, n <= 0,
    or the check itself failed: a broken check never refuses a request)"""
    if not _CHECK or int(n) <= 0:
        return None
    try:
        err = _CHECK[0](int(start_pos), int(n))
    except Exception as exc:  # noqa: BLE001 -- the check is advisory; the step's own planning still raises
        return None if not isinstance(exc, ValueError) else str(exc)
    return None if err is None else str(err)
