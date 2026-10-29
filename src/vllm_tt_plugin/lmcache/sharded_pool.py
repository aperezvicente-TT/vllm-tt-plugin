# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""LMCache pages for the sequence-sharded DSA pool (Stage 12 x 14, ``GLM53_DSA_SHARD=1``).

Under the sharded pool (doc/ttnn/14-production-tp8-megakernel/dsa/served_1m_memory_plan.md lever B,
shard_tp8_design.md 6.1-6.3) each chip holds ONLY its own units' latent pages: 128-token unit ``u`` of a request lives on
chip ``u % C``. vLLM allocates LOGICAL blocks; the generator's OwnerBlockMap (tt-metal ``tt/owner_blocks.py``) maps a
logical block ``b`` listed at column ``j`` (= position // 64) to a PHYSICAL global block ``g`` with ``g % C == (j // bpu)
% C`` and local block ``g // C`` on that chip. The model exposes every chip's LOCAL pool as one mesh tensor
``[nb_local, 1, 64, 512]`` (page index = local block) plus the map (``kv_cache_connector_shard()``).

What this module adds, knob-gated by the presence of that hook (the replicated path is untouched without it):

* ``OwnerRule``: the placement arithmetic (column -> owner, physical <-> (owner, local)).
* ``ShardedPagePool``: logical page -> (owner chip, local block) through the block map; a STORE needs the block mapped
  (``physical(b)``: no device content = refused, never a silent garbage store); a LOAD ``admit``s the hit's blocks at
  their columns first (the map takes a physical block on the column's owner before the request's row exists), and
  records one T2 page-in request per block (``PageInRequest``: unit, owner chip, local block -- what the device-initiated
  page-in path of doc/ttnn/12-fast-start/page_dma/t2_pagein_design.md consumes), optionally appended as JSONL to
  ``$GLM53_T2_PAGEIN_LOG``.
* ``SimpleOwnerBlockMap``: a minimal host implementation of the block-map contract (``physical`` / ``admit`` /
  ``release``) for the plugin's own tests and mocks (the production map is the generator's OwnerBlockMap).
* ``FakeMeshLatentPool``: C "chips" as host tensors ``[nb_local, 1, 64, 512]`` per layer, the ``read_local`` /
  ``write_local`` interface the sharded handle drives; the device implementation lives in ``kv_connector``
  (``_TtnnMeshPageIO``: slice into the persistent page buffer + the owner shard read, ``paged_fill_cache`` with a
  per-chip page table that points the non-owners at their null block 0).

Blob layout: an LMCache memory object stays ``[1, num_layers, n_tokens, 544]`` in TOKEN order per layer (latent 512 +
32 reserved) -- unit order, chip-independent: the owner is a placement rule, not part of the format, so a hit can land
on another physical block, another owner mapping or another C (served_1m_memory_plan.md E). A 256-token chunk = 2 units
spans two owner chips; a 128-token chunk = 1 unit = 2 pages of one owner.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch

from vllm_tt_plugin.lmcache import page_contract as pc

UNIT_TOKENS = 128  # the shard unit = one pooled-key page = 2 vLLM blocks of 64 (dsa_shard.py UNIT)


class BlockMapLike(Protocol):
    """the generator's OwnerBlockMap as the connector uses it"""

    C: int
    bpu: int
    nb_local: int

    def physical(self, b: int) -> int | None: ...

    def admit(self, pairs: Sequence[tuple[int, int]]) -> list[int]: ...


@dataclass(frozen=True)
class OwnerRule:
    """placement arithmetic of the sharded pool: C owner chips, bpu blocks (of ``tokens_per_page``) per unit"""

    C: int
    bpu: int = UNIT_TOKENS // pc.TOKENS_PER_PAGE
    tokens_per_page: int = pc.TOKENS_PER_PAGE

    def owner_of_column(self, j: int) -> int:
        return (int(j) // self.bpu) % self.C

    def unit_of_column(self, j: int) -> int:
        return int(j) // self.bpu

    def column_of_position(self, pos: int) -> int:
        return int(pos) // self.tokens_per_page

    def split(self, g: int) -> tuple[int, int]:
        """physical global block -> (owner chip, local block)"""
        return int(g) % self.C, int(g) // self.C

    def join(self, owner: int, local: int) -> int:
        return int(local) * self.C + int(owner)


@dataclass
class PageInRequest:
    """One T2 page-in request record (t2_pagein_design.md section 3: the host's role is to place a user's units in pinned
    memory and name them; the service core pages the selected pools of a unit in by (owner chip, local block)). Written
    by every connector LOAD of a page into the sharded pool; the device-initiated path consumes the same fields."""

    unit: int  # the request's 128-token unit id (column // bpu)
    owner: int  # owner chip of the unit (unit % C)
    local_block: int  # the chip's local block the page lands in (physical // C)
    logical_block: int  # vLLM's block id
    physical_block: int  # the global physical id (owner + C x local)
    column: int  # position // tokens_per_page in the request
    layers: int  # layers written (one page per DSA layer)
    source: str = "lmcache_hit"
    ts: float = 0.0


class ShardedPagePool:
    """logical page ids -> (owner chip, local block) through the generator's block map; the T2 page-in records"""

    def __init__(self, block_map: BlockMapLike, num_blocks: int, *, layers: int = pc.NUM_DSA_LAYERS,
                 record_path: str | None = None):
        self.map = block_map
        self.rule = OwnerRule(int(block_map.C), int(block_map.bpu))
        self.num_blocks = int(num_blocks)  # vLLM's logical pool
        self.nb_local = int(block_map.nb_local)
        self.layers = int(layers)
        self.pagein_log: list[PageInRequest] = []
        self.record_path = record_path if record_path is not None else os.environ.get("GLM53_T2_PAGEIN_LOG")
        self.stats = {"located": 0, "admitted": 0, "moved": 0}

    @property
    def C(self) -> int:
        return self.rule.C

    def locate(self, b: int) -> tuple[int, int]:
        """(owner chip, local block) of logical block ``b``'s device content; refused when the map has none (never
        written, evicted on its owner, or a LOAD that was not admitted)"""
        b = int(b)
        if not 0 <= b < self.num_blocks:
            raise IndexError(f"logical block {b} outside [0, {self.num_blocks})")
        g = self.map.physical(b)
        if g is None:
            raise ValueError(
                f"logical block {b} has no device content on the sharded pool (not written, evicted on its owner, or a "
                "LOAD not admitted with its column: ShardedPagePool.admit)"
            )
        self.stats["located"] += 1
        return self.rule.split(g)

    def admit(self, pairs: Sequence[tuple[int, int]], *, source: str = "lmcache_hit") -> list[tuple[int, int]]:
        """a LOAD of logical blocks at their request columns: the map places each on its column's owner (a fresh
        physical block, the block's own if already there, a move if cached elsewhere) and one page-in record per block
        is logged. Returns [(owner, local)] in order."""
        pairs = [(int(b), int(j)) for b, j in pairs]
        for b, _j in pairs:
            if not 0 <= b < self.num_blocks:
                raise IndexError(f"logical block {b} outside [0, {self.num_blocks})")
        before = {b: self.map.physical(b) for b, _j in pairs}
        phys = self.map.admit(pairs)
        out = []
        now = time.time()
        for (b, j), g in zip(pairs, phys, strict=True):
            owner, local = self.rule.split(g)
            if owner != self.rule.owner_of_column(j):
                raise AssertionError(f"block map put logical block {b} (column {j}) on chip {owner}")
            if before[b] is not None and before[b] != g:
                self.stats["moved"] += 1
            rec = PageInRequest(unit=self.rule.unit_of_column(j), owner=owner, local_block=local, logical_block=b,
                                physical_block=g, column=j, layers=self.layers, source=source, ts=now)
            self.pagein_log.append(rec)
            out.append((owner, local))
        self.stats["admitted"] += len(pairs)
        if self.record_path:
            with open(self.record_path, "a") as f:
                for rec in self.pagein_log[-len(pairs):]:
                    f.write(json.dumps(asdict(rec)) + "\n")
        return out

    def columns_of(self, start: int, n_pages: int) -> list[int]:
        """the request columns of the ``n_pages`` whole pages that cover token positions from ``start`` (page-aligned)"""
        c0 = self.rule.column_of_position(start)
        return [c0 + k for k in range(n_pages)]


class SimpleOwnerBlockMap:
    """The block-map contract on the host (tests / mocks): logical -> physical with ``g % C == owner(column)``, per-chip
    free lists (local 0 = the null block, never handed out), no eviction (a dry chip raises). Mirrors tt-metal's
    OwnerBlockMap.physical / admit / release semantics the connector relies on."""

    def __init__(self, C: int, nb_local: int, bpu: int = UNIT_TOKENS // pc.TOKENS_PER_PAGE):
        self.C, self.nb_local, self.bpu = int(C), int(nb_local), int(bpu)
        self.free = [list(range(self.nb_local - 1, 0, -1)) for _ in range(self.C)]
        self.phys: dict[int, int] = {}
        self.live: dict[int, int] = {}

    def owner(self, j: int) -> int:
        return (int(j) // self.bpu) % self.C

    def physical(self, b: int) -> int | None:
        return self.phys.get(int(b))

    def _take(self, r: int) -> int:
        if not self.free[r]:
            raise RuntimeError(f"owner chip {r} has no free physical block")
        return self.free[r].pop() * self.C + r

    def admit(self, pairs: Sequence[tuple[int, int]]) -> list[int]:
        out = []
        for b, j in pairs:
            b, r = int(b), self.owner(j)
            g = self.phys.get(b)
            if g is not None and g % self.C != r:
                if self.live.get(b):
                    raise ValueError(f"logical block {b} is live on chip {g % self.C} but loaded at column {j} of chip {r}")
                self.free[g % self.C].append(g // self.C)
                g = None
            if g is None:
                g = self._take(r)
                self.phys[b] = g
            out.append(g)
        return out

    def set_row(self, row: Sequence[int], keep: int = 0) -> list[int]:
        """a request's row (column j = row[j]); keep columns must be mapped already (a hit's loaded pages)"""
        out = []
        for j, b in enumerate(row):
            b = int(b)
            if j < keep and self.phys.get(b) is None:
                raise ValueError(f"logical block {b} at computed column {j} has no device content")
            g = self.admit([(b, j)])[0]
            self.live[b] = self.live.get(b, 0) + 1
            out.append(g)
        return out

    def release(self, row: Sequence[int]) -> None:
        for b in row:
            n = self.live.get(int(b), 0) - 1
            if n > 0:
                self.live[int(b)] = n
            else:
                self.live.pop(int(b), None)

    def forget(self, b: int) -> None:
        """drop a cached block's content (its physical block is free again): a later STORE of it is refused"""
        g = self.phys.pop(int(b), None)
        if g is not None:
            self.free[g % self.C].append(g // self.C)


class FakeMeshLatentPool:
    """C 'chips' as host tensors: per layer ``chips[c]`` = ``[nb_local, 1, 64, 512]`` bf16; ``read_local`` /
    ``write_local`` are what the sharded page handle drives, with a per-chip access log for the tests."""

    def __init__(self, C: int, nb_local: int, layer_names: Sequence[str] | None = None, *,
                 tokens_per_page: int = pc.TOKENS_PER_PAGE, latent_dim: int = pc.LATENT_DIM):
        self.C, self.nb_local = int(C), int(nb_local)
        self.layer_names = list(layer_names) if layer_names is not None else pc.dsa_layer_names()
        self.shape = (self.nb_local, 1, tokens_per_page, latent_dim)
        self.chips: dict[str, list[torch.Tensor]] = {
            n: [torch.zeros(self.shape, dtype=pc.LATENT_DTYPE) for _ in range(self.C)] for n in self.layer_names
        }
        self.reads: list[tuple[str, int, int]] = []  # (layer, owner, local)
        self.writes: list[tuple[str, int, int]] = []

    def layer_io(self, name: str) -> "_FakeLayerIO":
        return _FakeLayerIO(self, name)


class _FakeLayerIO:
    def __init__(self, pool: FakeMeshLatentPool, name: str):
        self.pool, self.name = pool, name
        self.page_shape = (1,) + pool.shape[1:]
        self.shape = pool.shape  # the local pool's shape, as a ttnn tensor's: the handle sizes its pages from it
        self.dtype = pc.LATENT_DTYPE

    def read_local(self, owner: int, local: int) -> torch.Tensor:
        self.pool.reads.append((self.name, int(owner), int(local)))
        return self.pool.chips[self.name][owner][local : local + 1].clone()

    def write_local(self, owner: int, local: int, page: torch.Tensor) -> None:
        self.pool.writes.append((self.name, int(owner), int(local)))
        self.pool.chips[self.name][owner][local] = page.reshape(self.page_shape[1:]).to(pc.LATENT_DTYPE)

    def warm(self) -> None:  # the device path's pre-capture program warm; nothing to do on the host
        pass


def shard_record_from_model(model: Any) -> Mapping[str, Any] | None:
    """the model's ``kv_cache_connector_shard()`` record ({"C", "nb_local", "bpu", "num_blocks", "map"}) or None"""
    hook = getattr(model, "kv_cache_connector_shard", None)
    if not callable(hook):
        return None
    rec = hook()
    if rec is None:
        return None
    missing = {"C", "nb_local", "bpu", "num_blocks", "map"} - set(rec)
    if missing:
        raise ValueError(f"kv_cache_connector_shard() record lacks {sorted(missing)}")
    return rec
