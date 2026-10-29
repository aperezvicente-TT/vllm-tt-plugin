# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side view of a model's prefix cache (GLM-5.3-Flash srv/prefixcache).

vLLM's prefix cache shares PAGES of a common prefix. A hybrid model with per-request state that is not paged (the
KDA recurrent state, the DSA indexer rows) can only continue from a prefix boundary where it SNAPSHOTTED that
state. The model registers an oracle here (the TT executor is in-process, so the scheduler and the model runner
share this module); ``TTScheduler`` truncates every local page hit to ``oracle.longest_cached(prompt, hit)`` and
ends the final prefill chunk of a prompt on its last ``oracle.boundary`` so the snapshot exists for the next hit.

Protocol (duck-typed):
    boundary: int                                   snapshot granularity in tokens (128 for GLM-5.3-Flash)
    longest_cached(token_ids, max_tokens) -> int    the longest snapshotted boundary <= max_tokens of the prompt, 0 if none
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class PrefixOracle(Protocol):
    boundary: int

    def longest_cached(self, token_ids: Sequence[int], max_tokens: int | None = None) -> int: ...


_ORACLE: Any = None


def register(oracle: Any) -> None:
    """The model (or a test) installs its oracle; ``None`` removes it."""
    global _ORACLE
    if oracle is not None and (not hasattr(oracle, "longest_cached") or not hasattr(oracle, "boundary")):
        raise TypeError("a prefix oracle has `boundary` and `longest_cached(token_ids, max_tokens)`")
    _ORACLE = oracle


def get() -> Any:
    return _ORACLE


def aligned_hit(oracle: Any, token_ids: Sequence[int], hit_tokens: int, block_size: int) -> int:
    """``hit_tokens`` page-cached tokens of ``token_ids`` -> the boundary the model can resume at (a multiple of both
    the page and the oracle's boundary that has a snapshot), 0 when none."""
    n = min(int(hit_tokens), len(token_ids))
    n = n // block_size * block_size
    if n <= 0:
        return 0
    cached = int(oracle.longest_cached(token_ids, n))
    return cached // block_size * block_size
