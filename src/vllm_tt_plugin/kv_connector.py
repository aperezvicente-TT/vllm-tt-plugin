# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

"""KV-connector host layer: the TT runner as a vLLM ``KVConnectorBase_V1`` host.

vLLM's KV connectors (``LMCacheConnectorV1``, ``ExampleConnector``, ...) are
split into a scheduler-side half, which the TT scheduler inherits unchanged from
upstream (``TTScheduler`` extends ``AsyncScheduler``), and a worker-side half
that a model runner must drive. The GPU runner drives it from inside the
attention layers; the TT runner executes a whole forward as one device program,
so the calls are hoisted to the step boundary:

* ``begin_step``  -- ``handle_preemptions`` before the persistent batch is
  mutated (evicted pages are still intact);
* ``load``        -- ``start_load_kv`` and ``wait_for_layer_load`` for every
  layer **before** the device forward is submitted, so a prefix hit is resident
  when the program reads it;
* ``save`` / ``finish`` -- ``save_kv_layer`` for every layer and
  ``wait_for_save`` **after** the forward has completed (sync: at sample time;
  async decode: when the deferred output resolves), then ``get_finished`` and
  the ``KVConnectorOutput`` attached to the step's ``ModelRunnerOutput``;
* ``no_forward``  -- the load/finish pair for a step that schedules nothing
  (``kv_connector_no_forward`` upstream).

Page contract (doc/ttnn/15-lmcache/page_contract.md, rev 1): a page is one vLLM
block of ``TT_KV_PAGE_TOKENS`` tokens; a layer's paged cache is one tensor in
the native TT layout ``[num_blocks, num_kv_heads, block_size, head_size]``
(``[num_blocks, 1, 64, 512]`` bf16 for the GLM-5.3-Flash latent). Connectors are
handed vLLM's convention for that layout:

* MLA / latent layers (``MLAAttentionSpec``, one kv head): the
  ``[num_blocks, block_size, head_size]`` view -- what LMCache's MLA GPU
  connector and vLLM's ``ExampleConnector`` (``MLACommonMetadata`` branch,
  ``cache.reshape(num_blocks * block_size, -1)[slot_mapping]``) index. The
  attention metadata for these layers is a real ``MLACommonMetadata``.
* other attention layers: the native ``[num_blocks, heads, block_size, head]``
  tensor, which vLLM's generic ``cache[block, :, offset]`` indexing addresses,
  with a ``TTKVAttentionMetadata``.

Speculative decoding (Stage 17, doc/ttnn/17-mtp-spec-decode/lmcache_spec_contract.md):
a decode step is a verify of ``1 + K`` rows at positions ``c .. c + K`` that commits
``n + 1`` tokens (``P = c + n + 1``); the latent rows past ``P`` are dead until the
next step overwrites them, and the committed KDA state sits in candidate row
``n`` of the shared state, not in the slot's own row, until the generator leaves
speculation. What the host guarantees under speculation:

* chunk stores are decided by the connector's scheduler side from vLLM's
  ``num_computed_tokens`` (rolled back by the rejected count in
  ``update_from_output``) and ``request.all_token_ids`` (committed tokens, never
  drafts), and executed in ``wait_for_save`` after the step's forward completed:
  a chunk reaches the store only once every one of its tokens is committed and
  its latent row final. The forward context's slot mapping does describe the
  verify rows (the forward wrote them), but neither LMCache nor
  ``ExampleConnector`` stores from it (``connector_spec_decode_mode``).
* KDA / DSA checkpoints at prefill page boundaries (``_export_kda_checkpoints``)
  are unchanged: the generator leaves speculation before any prefill.
* a page boundary ``B`` crossed by a verify is checkpointed only once
  ``B <= P`` (``step_boundary_checkpoints``), from the candidate row
  ``B - c - 1``, and only through the model's ``export_spec_boundary_state``
  hook: the plain ``export_kda_state`` reads the slot row, which is stale under
  speculation. Without the hook (the int28 generator) decode-time boundaries are
  not checkpointed at all (the prefill boundaries and the chunk stores remain),
  which is counted in ``decode_boundaries_skipped``.

``slot_mapping = block_id * block_size + offset`` in both cases. On the CPU mock
path the connector-facing object is a ``torch.Tensor`` view that writes
through to the model's cache; on a device it is a ``TTPagedCacheHandle`` that
wraps the ``ttnn.Tensor`` and exposes the same ``shape`` / ``dtype`` plus page
reads and writes (page_contract.md §6). ``tt_copy_kv_blocks`` (the
``CopyBlocksOp`` handed to ``set_host_xfer_buffer_ops``) moves page ranges
between a host buffer of that shape and either kind. The ttnn branch
(``ttnn.slice`` / ``ttnn.experimental.paged_fill_cache``) has not been
exercised on a device by this change; the torch branch is unit tested.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import queue
import sys
import threading
import weakref
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

from vllm_tt_plugin.lmcache.sharded_pool import ShardedPagePool, shard_record_from_model
from vllm_tt_plugin.logger import init_tt_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_tt_logger(__name__)

# Tokens per page (= vLLM block). doc/ttnn/15-lmcache/page_contract.md rev 1 §2:
# the GLM-5.3-Flash latent page is 64 tokens and the adapter refuses any other
# --block-size. Models with another page size declare it through their
# KVCacheSpec; the constant is the contract's value and the sanity default.
TT_KV_PAGE_TOKENS = 64

# ``CopyBlocksOp``: (src caches, dst caches, src block ids, dst block ids, direction)
CopyDirection = Literal["h2d", "d2h"]


# ---------------------------------------------------------------------------
# Connector-facing layer views
# ---------------------------------------------------------------------------


def _host_zeros(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """A zero host transfer buffer whose pages are committed only when written (Stage 12): an anonymous mmap (zero
    pages on demand) instead of torch.zeros (which writes, so commits, all of it). The buffers are vLLM's LOGICAL pool,
    11 KiB of host RAM per pool token over the 11 DSA layers (2M tokens: 23.6 GB, 4M: 47 GB); with the page-DMA store
    and load (GLM53_KV_T2) whole pages bypass them, so they stay almost entirely uncommitted.
    GLM53_KV_HOST_XFER_LAZY=0: torch.zeros."""
    n = int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size()
    if os.environ.get("GLM53_KV_HOST_XFER_LAZY", "1") != "1" or n == 0:
        return torch.zeros(shape, dtype=dtype)
    import mmap

    buf = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    t = torch.frombuffer(buf, dtype=dtype).view(shape)
    t._glm53_mmap = buf  # the tensor keeps its mapping alive  # noqa: SLF001
    return t


class TTPagedCacheHandle:
    """A device (``ttnn.Tensor``) paged cache as a connector-facing object.

    Exposes the connector-facing ``shape`` (``[num_blocks, block_size, head]``
    for a latent cache, the native 4-D shape otherwise) and ``dtype`` as torch
    values, so a connector can size its host buffers from it like from a
    ``torch.Tensor``, plus ``read_pages`` / ``write_pages`` over the wrapped
    ``ttnn.Tensor`` (page_contract.md §6). ``device_tensor`` is the model's
    tensor itself; nothing is copied at construction.
    """

    def __init__(self, device_tensor: Any, shape: tuple[int, ...], dtype: torch.dtype):
        self.device_tensor = device_tensor
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.device = "tt"

    def __repr__(self) -> str:
        return f"TTPagedCacheHandle(shape={tuple(self.shape)}, dtype={self.dtype})"

    def read_pages(self, ids: list[int]) -> torch.Tensor:
        pages = _ttnn_read_pages(self.device_tensor, ids)
        return pages.reshape(len(ids), *self.shape[1:])

    def write_pages(self, ids: list[int], pages: torch.Tensor) -> None:
        native = _ttnn_shape(self.device_tensor)[1:]
        # On a device the write is in place; a HOST ttnn tensor has no in-place
        # page write, so the tensor is rebuilt and the handle follows it.
        self.device_tensor = _ttnn_write_pages(
            self.device_tensor, ids, pages.reshape(len(ids), *native)
        )


class _SqueezedPagedHandle(TTPagedCacheHandle):
    """A 4-D ``[N, 1, bs, D]`` device handle presented as ``[N, bs, D]``.

    The GLM adapter keeps the native 4-D view per DSA layer (``paged_fill_cache``
    takes 4-D); connectors are handed the 3-D MLA view, and page reads / writes
    reshape at this boundary.
    """

    def __init__(self, inner: TTPagedCacheHandle):
        n, _one, bs, d = (int(s) for s in inner.shape)
        super().__init__(inner.device_tensor, (n, bs, d), inner.dtype)
        self.inner = inner

    def read_pages(self, ids: list[int]) -> torch.Tensor:
        return self.inner.read_pages(ids).reshape(len(ids), *self.shape[1:])

    def write_pages(self, ids: list[int], pages: torch.Tensor) -> None:
        self.inner.write_pages(ids, pages.reshape(len(ids), *self.inner.shape[1:]))
        self.device_tensor = self.inner.device_tensor


class TTShardedPagedCacheHandle(TTPagedCacheHandle):
    """A layer's latent pages on the sequence-sharded DSA pool (GLM53_DSA_SHARD; lmcache/sharded_pool.py).

    ``device_tensor`` is every chip's LOCAL pool as one mesh tensor ``[nb_local, 1, 64, 512]`` (per-chip content, page
    index = local block); the handle presents vLLM's LOGICAL pool ``[num_blocks, 1, 64, 512]`` and translates each
    logical block through the generator's block map (``pool``): a STORE reads block b from its owner chip only
    (``physical(b)`` -> chip g % C, local g // C; refused without device content), a LOAD writes it to its owner's local
    block only, after the GPU connector admitted the hit's blocks at their columns (``pool.admit``). ``io`` moves one
    local page per call: ``_TtnnMeshPageIO`` on a device, ``FakeMeshLatentPool`` in the host tests.
    """

    def __init__(self, device_tensor: Any, pool: Any, io: Any, num_blocks: int, dtype: torch.dtype):
        native = _ttnn_shape(device_tensor) if not isinstance(device_tensor, torch.Tensor) else list(device_tensor.shape)
        super().__init__(device_tensor, (int(num_blocks), *native[1:]), dtype)
        self.pool = pool
        self.io = io
        self.page_shape = tuple(native[1:])

    def read_pages(self, ids: list[int]) -> torch.Tensor:
        bulk = getattr(self.io, "read_locals", None)
        if callable(bulk) and len(ids) > 1:  # Stage 12: owner-grouped batches, one host read per batch
            return bulk([self.pool.locate(i) for i in ids]).reshape(len(ids), *self.page_shape)
        chunks = [self.io.read_local(*self.pool.locate(i)).reshape(1, *self.page_shape) for i in ids]
        return torch.cat(chunks, dim=0) if chunks else torch.empty(0, *self.page_shape, dtype=self.dtype)

    def read_pages_async(self, ids: list[int]) -> Any:
        """read_pages enqueued (Stage 12 cf3h): a _PendingPages the store worker completes, or the pages read now"""
        start = getattr(self.io, "read_locals_async", None)
        if callable(start) and len(ids) > 1:
            return start([self.pool.locate(i) for i in ids])
        return self.read_pages(ids)

    def write_pages(self, ids: list[int], pages: torch.Tensor) -> None:
        pages = pages.reshape(len(ids), *self.page_shape)
        for k, i in enumerate(ids):
            owner, local = self.pool.locate(i)  # admitted (or already mapped) on its column's owner
            self.io.write_local(owner, local, pages[k : k + 1])

    def warm_page_paths(self) -> None:
        self.io.warm()


def _ttnn_dtype_to_torch(dtype: Any) -> torch.dtype:
    name = str(dtype).split(".")[-1].lower()
    return {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "bfloat8_b": torch.bfloat16,  # host staging dtype; the device keeps bfp8
        "bfloat4_b": torch.bfloat16,
        "uint8": torch.uint8,
        "int32": torch.int32,
        "uint32": torch.int32,
    }.get(name, torch.bfloat16)


@dataclass(frozen=True)
class TTKVLayerView:
    """One attention layer's paged cache as a connector sees it.

    ``device_tensor`` is the model's per-layer paged tensor in the TT layout
    ``[num_blocks, num_kv_heads, block_size, head_size]`` (``torch.Tensor`` on
    the mock path, ``ttnn.Tensor`` on a device); ``tensor`` is the
    connector-facing object derived from it (see the module docstring).
    ``group_index`` selects the vLLM kv-cache group whose block table addresses
    this layer.
    """

    name: str
    layer_index: int | None
    group_index: int
    spec: KVCacheSpec
    device_tensor: Any
    tensor: Any

    @property
    def block_size(self) -> int:
        return int(self.spec.block_size)

    @property
    def num_blocks(self) -> int:
        return int(self.tensor.shape[0])

    @property
    def is_mla(self) -> bool:
        return isinstance(self.spec, MLAAttentionSpec)


def connector_facing_tensor(tensor: Any, spec: KVCacheSpec) -> Any:
    """The object a connector gets for one layer (module docstring)."""
    if isinstance(tensor, TTPagedCacheHandle):
        # A model hook may hand the handle itself; a 4-D latent handle is
        # presented squeezed like a torch view would be.
        hshape = tuple(int(s) for s in tensor.shape)
        if isinstance(spec, MLAAttentionSpec) and len(hshape) == 4 and hshape[1] == 1:
            return _SqueezedPagedHandle(tensor)
        return tensor
    shape = tuple(int(s) for s in tensor.shape)
    squeeze = isinstance(spec, MLAAttentionSpec) and len(shape) == 4 and shape[1] == 1
    facing_shape = (shape[0], shape[2], shape[3]) if squeeze else shape
    if isinstance(tensor, torch.Tensor):
        return tensor.view(*facing_shape) if squeeze else tensor
    dtype = getattr(spec, "dtype", None)
    if not isinstance(dtype, torch.dtype):
        dtype = _ttnn_dtype_to_torch(getattr(tensor, "dtype", None))
    return TTPagedCacheHandle(tensor, facing_shape, dtype)


def _parse_layer_index(layer_name: str) -> int | None:
    """``model.layers.<i>.self_attn`` -> ``i``; ``None`` when no index is present."""
    for part in layer_name.split("."):
        if part.isdigit():
            return int(part)
    return None


def _is_paged_tensor(obj: Any) -> bool:
    if isinstance(obj, torch.Tensor):
        return obj.dim() >= 2
    if isinstance(obj, TTPagedCacheHandle):
        return True
    # ttnn.Tensor without importing ttnn here (host tests run without a device).
    return hasattr(obj, "shape") and hasattr(obj, "buffer_address")


def _per_layer_specs(
    kv_cache_config: KVCacheConfig,
) -> dict[str, tuple[int, KVCacheSpec]]:
    """layer name -> (group index, spec) for every layer vLLM knows pages for."""
    out: dict[str, tuple[int, KVCacheSpec]] = {}
    for g_idx, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            for name, layer_spec in spec.kv_cache_specs.items():
                out[name] = (g_idx, layer_spec)
            continue
        for name in group.layer_names:
            out[name] = (g_idx, spec)
    return out


def resolve_connector_kv_views(
    kv_caches: Any,
    kv_cache_config: KVCacheConfig,
    model: Any = None,
) -> dict[str, TTKVLayerView]:
    """Map vLLM's per-layer specs onto the model's paged tensors.

    Resolution order:

    1. ``model.kv_cache_connector_views()`` -> ``{layer_name: paged tensor}``.
       A generator whose cache object is opaque (the GLM-5.3-Flash stack keeps
       the latent pages inside its own cache binding) exposes them here, keyed
       by the names it returned from ``get_kv_cache_spec``.
    2. ``kv_caches`` as a ``dict`` keyed by layer name.
    3. ``kv_caches`` as a sequence with one paged tensor per attention layer,
       addressed by the index parsed from the spec's layer name
       (``model.layers.<i>...``); the legacy single ``"foo"`` spec maps onto the
       whole sequence as ``model.layers.<i>.attn``.

    A layer entry that is not a single paged tensor (e.g. a ``(K, V)`` pair) is
    refused: the connector API is one tensor per layer, so such a model must
    provide the hook.
    """
    specs = _per_layer_specs(kv_cache_config)
    if not specs:
        raise ValueError("kv_cache_config names no attention layers")

    hook = getattr(model, "kv_cache_connector_views", None)
    shard = shard_record_from_model(model) if callable(hook) else None
    if callable(hook):
        tensors = dict(hook())
        source = "model.kv_cache_connector_views()"
        if shard is not None:  # GLM53_DSA_SHARD: local pools -> logical handles through the block map
            pool = ShardedPagePool(shard["map"], int(shard["num_blocks"]), layers=len(tensors))
            wrapped = {}
            for name, t in tensors.items():
                if isinstance(t, TTPagedCacheHandle):
                    raise TypeError(f"KV connector layer {name!r}: the sharded pool hook must hand the local pool tensor")
                if hasattr(t, "read_local") and hasattr(t, "write_local"):  # a host mesh stand-in (tests, mocks)
                    io = t
                elif isinstance(t, torch.Tensor):
                    io = _FakeLayerIO_for(t)
                else:  # a ttnn mesh tensor: every chip's local pool
                    io = _TtnnMeshPageIO(t, pool.C)
                dt = t.dtype if isinstance(getattr(t, "dtype", None), torch.dtype) else _ttnn_dtype_to_torch(t.dtype)
                wrapped[name] = TTShardedPagedCacheHandle(t, pool, io, int(shard["num_blocks"]), dt)
            tensors = wrapped
            source += " (sequence-sharded pool: per-owner pages)"
    elif isinstance(kv_caches, dict):
        tensors = dict(kv_caches)
        source = "kv_caches dict"
    else:
        try:
            seq = list(kv_caches)
        except TypeError as exc:
            raise TypeError(
                "KV connectors need the model's paged caches as a dict, a "
                "per-layer sequence, or through kv_cache_connector_views(); got "
                f"{type(kv_caches).__name__}"
            ) from exc
        indexed = {name: _parse_layer_index(name) for name in specs}
        if all(i is not None for i in indexed.values()):
            tensors = {}
            for name, idx in indexed.items():
                if idx >= len(seq):  # type: ignore[operator]
                    raise ValueError(
                        f"KV spec layer {name!r} needs cache index {idx}, but "
                        f"the model allocated {len(seq)} per-layer caches"
                    )
                tensors[name] = seq[idx]  # type: ignore[index]
        else:
            # Legacy single-spec path: one spec for every layer of the sequence.
            if len(specs) != 1:
                raise ValueError(
                    "KV spec layer names carry no layer index and there is more "
                    f"than one spec: {sorted(specs)}"
                )
            ((_, (g_idx, spec)),) = specs.items()
            tensors = {f"model.layers.{i}.attn": t for i, t in enumerate(seq)}
            specs = {name: (g_idx, spec) for name in tensors}
        source = "kv_caches sequence"

    views: dict[str, TTKVLayerView] = {}
    for name, (g_idx, spec) in specs.items():
        if name not in tensors:
            raise ValueError(
                f"KV spec layer {name!r} has no paged tensor in {source} "
                f"(available: {sorted(tensors)})"
            )
        tensor = tensors[name]
        if not _is_paged_tensor(tensor):
            raise TypeError(
                f"KV connector layer {name!r}: expected one paged tensor "
                "[num_blocks, num_kv_heads, block_size, head_size], got "
                f"{type(tensor).__name__}; models that keep K and V apart must "
                "implement kv_cache_connector_views()"
            )
        shape = tuple(int(s) for s in tensor.shape)
        if isinstance(tensor, TTShardedPagedCacheHandle) and isinstance(spec, AttentionSpec):
            if shape[1:] != (shape[1], spec.block_size, spec.head_size):
                raise ValueError(f"KV connector layer {name!r}: sharded pool page {shape[1:]} does not match its spec")
        elif isinstance(spec, AttentionSpec) and len(shape) == 4:
            expected = (shape[0], shape[1], spec.block_size, spec.head_size)
            if shape != expected:
                raise ValueError(
                    f"KV connector layer {name!r}: tensor shape {shape} does not "
                    f"match its spec [num_blocks, heads, {spec.block_size}, "
                    f"{spec.head_size}]"
                )
        views[name] = TTKVLayerView(
            name=name,
            layer_index=_parse_layer_index(name),
            group_index=g_idx,
            spec=spec,
            device_tensor=tensor,
            tensor=connector_facing_tensor(tensor, spec),
        )
    return views


# ---------------------------------------------------------------------------
# Host transfer buffer copy op
# ---------------------------------------------------------------------------


def _contiguous_runs(ids: list[int]) -> Iterable[tuple[int, int, int]]:
    """Yield ``(start_offset, first_id, count)`` runs of consecutive ids."""
    if not ids:
        return
    run_start, first = 0, ids[0]
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[i - 1] + 1:
            yield run_start, first, i - run_start
            if i < len(ids):
                run_start, first = i, ids[i]


def _read_pages(tensor: Any, ids: list[int]) -> torch.Tensor:
    """Pages ``ids`` of a paged tensor as a host tensor ``[len(ids), ...]``."""
    if isinstance(tensor, torch.Tensor):
        index = torch.as_tensor(ids, dtype=torch.int64, device=tensor.device)
        return tensor.index_select(0, index).cpu()
    if isinstance(tensor, TTPagedCacheHandle):
        return tensor.read_pages(ids)
    return _ttnn_read_pages(tensor, ids)


def _read_pages_async(tensor: Any, ids: list[int]) -> Any:
    """Stage 12 cf3h: a store's page read ENQUEUED where the handle can (the sharded pool: _PendingPages, completed by
    the store worker with ``_pages_now``), else ``_read_pages`` now"""
    inner = tensor.inner if isinstance(tensor, _SqueezedPagedHandle) else tensor
    start = getattr(inner, "read_pages_async", None) if isinstance(inner, TTPagedCacheHandle) else None
    return start(ids) if callable(start) else _read_pages(tensor, ids)


def _write_pages(tensor: Any, ids: list[int], pages: torch.Tensor) -> None:
    """Write ``pages[k]`` into page ``ids[k]`` of a paged tensor.

    ``pages`` may carry the source's page shape; it is reshaped to the
    destination's (the element count per page must match), which is what lets
    a ``[n, 64, 512]`` host page land in a ``[N, 1, 64, 512]`` device tensor.
    """
    dst_page_shape = tuple(int(s) for s in tensor.shape[1:])
    if pages.shape[1:] != dst_page_shape:
        pages = pages.reshape(pages.shape[0], *dst_page_shape)
    if isinstance(tensor, torch.Tensor):
        index = torch.as_tensor(ids, dtype=torch.int64, device=tensor.device)
        tensor.index_copy_(0, index, pages.to(device=tensor.device, dtype=tensor.dtype))
        return
    if isinstance(tensor, TTPagedCacheHandle):
        tensor.write_pages(ids, pages)
        return
    _ttnn_write_pages(tensor, ids, pages)  # a bare device tensor: in place


def _ttnn_shape(tensor: Any) -> list[int]:
    """``list(tensor.shape)``: a ttnn ``Shape`` takes no slice (finding J10)."""
    return [int(s) for s in list(tensor.shape)]


def _ttnn_is_host(tensor: Any) -> bool:
    import ttnn  # noqa: PLC0415

    storage = getattr(tensor, "storage_type", None)
    return callable(storage) and storage() == ttnn.StorageType.HOST


# Persistent per-page device buffers: a connector page read / write at request time must not ALLOCATE a device buffer.
# The async-decode hook stores a step's pages while the next step may already be submitted, so any buffer allocated
# after the trace capture can be alive when a decode replay is issued -> the trace-safety guard ("device buffer(s) still
# alive before trace replay") kills the engine. The buffers are created by the first use, which is
# TTKVConnectorHost._warm_page_paths (before the first capture); later creation raises.
_PAGE_BUFS: dict = {}
_PAGE_BATCH: dict = {}  # same key -> the multi-page read buffer [GLM53_KV_PAGE_READ_BATCH, heads, block, dim]
_PAGE_BUFS_FROZEN = [False]


def _mesh_kwargs(device: Any) -> dict:
    import ttnn  # noqa: PLC0415  (device path only)

    replicate = getattr(ttnn, "ReplicateTensorToMesh", None)
    if replicate is not None and getattr(device, "get_num_devices", lambda: 1)() > 1:
        return {"mesh_mapper": replicate(device)}
    return {}


def _page_bufs(tensor: Any) -> tuple:
    """(read page, write stage, page table) device buffers for ``tensor``'s page shape, created once"""
    import ttnn  # noqa: PLC0415  (device path only)

    device = tensor.device()
    page = [1] + [int(v) for v in _ttnn_shape(tensor)[1:]]
    key = (id(device), str(tensor.dtype), str(tensor.layout), tuple(page))
    bufs = _PAGE_BUFS.get(key)
    if bufs is None:
        if _PAGE_BUFS_FROZEN[0]:
            raise RuntimeError(
                f"KV connector page buffers for {key[1:]} first needed after the warm-up: they would be allocated after "
                "the trace capture (TTKVConnectorHost._warm_page_paths must cover every cache shape)"
            )
        mk = _mesh_kwargs(device)
        dt = _ttnn_dtype_to_torch(tensor.dtype)
        heads, bs, hd = page[1], page[2], page[3]
        bufs = (
            ttnn.from_torch(torch.zeros(page, dtype=dt), dtype=tensor.dtype, layout=tensor.layout, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, **mk),
            ttnn.from_torch(torch.zeros(1, heads, bs, hd, dtype=dt), dtype=tensor.dtype, layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, **mk),
            ttnn.from_torch(torch.zeros(1, 1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, **mk),
        )
        _PAGE_BUFS[key] = bufs
        nb = int(os.environ.get("GLM53_KV_PAGE_READ_BATCH", "64"))
        t2_store = ("1" in (os.environ.get("GLM53_KV_T2", "0"), os.environ.get("GLM53_LMCACHE_PAGE_DMA", "0"))
                    and os.environ.get("GLM53_KV_T2_STORE", "1") == "1")  # fmt: skip
        if nb > 1 and not t2_store:  # Stage 12 (n29h5e cold TTFT +0.18 ms/token): the store's pages read nb at a time
            # (the T2 store reads through the page-DMA: no 4 MiB-per-chip batch buffer then, program 10:56:22Z)
            _PAGE_BATCH[key] = ttnn.from_torch(torch.zeros([nb] + page[1:], dtype=dt), dtype=tensor.dtype,
                                               layout=tensor.layout, device=device,
                                               memory_config=ttnn.DRAM_MEMORY_CONFIG, **mk)  # fmt: skip
    return bufs


def _page_batch(tensor: Any) -> Any:
    """the multi-page read buffer of tensor's page shape (created with _page_bufs, before the capture), or None"""
    device = tensor.device()
    page = [1] + [int(v) for v in _ttnn_shape(tensor)[1:]]
    return _PAGE_BATCH.get((id(device), str(tensor.dtype), str(tensor.layout), tuple(page)))


_SLOT_OPS: dict = {}


def _slot_ops(device: Any) -> Any:
    """The model's SlotOps (tt/stack_model.py, kernels/head/slot_op.cpp): interleaved page copies by runtime-arg
    addresses, so ONE program serves every page. ttnn.slice keys its program on slice_start: a page index first read
    after the trace capture compiles a new slice program, and that cached program's buffers are alive at the next replay
    (d0 card1, 2026-10-23 00:28Z: store of blocks 1-2 -> program cache 8 -> 10, the guard's two SliceDeviceOperation
    buffers; clean after clearing the program cache). Created by the first use = the pre-capture warm; None without the
    model tree (host tests: the slice path)."""
    so = _SLOT_OPS.get(id(device))
    if so is None:
        if os.environ.get("GLM53_KV_PAGE_SLOTOPS", "1") != "1":
            return None
        try:
            from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import SlotOps  # noqa: PLC0415
        except ImportError:
            return None
        if _PAGE_BUFS_FROZEN[0]:
            raise RuntimeError(
                "KV connector SlotOps first needed after the warm-up: its program and tensor would be created after "
                "the trace capture (TTKVConnectorHost._warm_page_paths must read a page first)"
            )
        so = _SLOT_OPS[id(device)] = SlotOps(device)
    return so


def _page_copy_geometry(tensor: Any) -> tuple[int, int] | None:
    """(tiles per cache page, bytes per tile) of a DRAM-interleaved bf16 / fp32 TILE paged cache; None otherwise"""
    import ttnn  # noqa: PLC0415

    try:
        mc = tensor.memory_config()
        if mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED or mc.buffer_type != ttnn.BufferType.DRAM:
            return None
        if tensor.layout != ttnn.TILE_LAYOUT or tensor.dtype not in (ttnn.bfloat16, ttnn.float32):
            return None
    except Exception:  # noqa: BLE001
        return None
    page = _ttnn_shape(tensor)[1:]
    if page[-1] % 32 or page[-2] % 32:
        return None
    n = 1
    for v in page:
        n *= int(v)
    return n // 1024, 1024 * (2 if tensor.dtype == ttnn.bfloat16 else 4)


def freeze_page_buffers() -> None:
    """after the connector's warm-up (before the first capture): no page buffer may be created any more"""
    _PAGE_BUFS_FROZEN[0] = True


_WARMING = [False]  # the pre-capture warm's FIRST pass: programs compile (a cached program owns a kernel-binary DRAM
# buffer: +2,048 B/bank for the SlotOps copy on d0 card2, 2026-10-23 03:45Z), so the probe is suspended; the warm's
# second pass is probed and must show zero delta


def _alloc_probe(tensor: Any) -> tuple[int, int] | None:
    """(DRAM bytes allocated per bank, program-cache entries) before a page op -- measured only on the image mock
    (GLM53_MOCK=1) or with GLM53_KV_PAGE_ALLOC_CHECK=1 (the program's mock rule, 23:47:38Z: zero live device
    buffers after a STORE / HIT). The program count matters as much as the bytes: a program first compiled after the
    capture owns buffers the replay guard reports (ttnn.slice keyed on the page index, d0 silicon 2026-10-23); the mock
    compiles programs, so it sees the count grow even where its allocator shows no byte left behind. Suspended during
    the warm's first pass (_WARMING: the programs' own compile-time buffers are not request-time allocations)."""
    if _WARMING[0]:
        return None
    if os.environ.get("GLM53_MOCK") != "1" and os.environ.get("GLM53_KV_PAGE_ALLOC_CHECK", "0") != "1":
        return None
    import ttnn  # noqa: PLC0415

    device = tensor.device()
    count = getattr(device, "num_program_cache_entries", None)
    return (int(ttnn.get_memory_view(device, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank),
            int(count()) if callable(count) else -1)  # fmt: skip


def _alloc_check(tensor: Any, before: tuple[int, int] | None, what: str) -> None:
    if before is None:
        return
    after = _alloc_probe(tensor)
    if after[0] != before[0]:
        raise RuntimeError(f"KV connector {what} changed the allocated DRAM by {after[0] - before[0]} B/bank: a "
                           "request-time device allocation (it would be alive across a decode replay)")
    if _PAGE_BUFS_FROZEN[0] and before[1] >= 0 and after[1] != before[1]:
        raise RuntimeError(f"KV connector {what} compiled {after[1] - before[1]} new program(s) after the warm-up: a "
                           "program first built after the trace capture owns buffers alive at the next replay (a "
                           "page-dependent program key, e.g. ttnn.slice on the page index)")


def _ttnn_read_pages(tensor: Any, ids: list[int]) -> torch.Tensor:
    """Device -> host page gather (page_contract.md §6).

    A HOST ttnn tensor (the mock cluster, host-side tests) is read whole and
    indexed; a device tensor is sliced per contiguous run and one chip's copy
    read back. The device branch has not run on a chip.
    """
    import ttnn  # noqa: PLC0415  (device path only)

    trailing = _ttnn_shape(tensor)[1:]
    if _ttnn_is_host(tensor):
        return ttnn.to_torch(tensor)[torch.as_tensor(ids, dtype=torch.int64)]
    chunks: list[torch.Tensor] = []
    # One page per copy into the PERSISTENT page buffer (_page_bufs) through the model's SlotOps (_slot_ops: one program
    # for every page, warmed before the capture by TTKVConnectorHost._warm_page_paths; ttnn.slice, the fallback, keys
    # its program on the page index) and the read allocates no device buffer at request time (a per-page slice output
    # alive while the next decode replay is issued trips the trace-safety guard).
    read_buf, _stage, _table = _page_bufs(tensor)
    so = _slot_ops(tensor.device())  # created (with its tensor) by the warm, like the page buffers: before the probe
    geo = _page_copy_geometry(tensor) if so is not None else None
    before = _alloc_probe(tensor)
    get_device_tensors = getattr(ttnn, "get_device_tensors", None)
    batch = _page_batch(tensor) if geo is not None else None
    if batch is not None and len(ids) > 1:
        # Stage 12 (program 2026-10-23T07:51Z, n29h5e: cold TTFT +0.18 ms/token over int27c = the store's page
        # reads, one SlotOps launch + one blocking to_torch per page per layer): up to nb pages per SlotOps op list into
        # the persistent multi-page buffer, then ONE host read of it -- the same program, no device allocation
        n, tile_bytes = geo
        nb = int(batch.padded_shape[0]) if hasattr(batch, "padded_shape") else int(_ttnn_shape(batch)[0])
        src_addr, dst_addr = int(tensor.buffer_address()), int(batch.buffer_address())
        for s0 in range(0, len(ids), nb):
            part = ids[s0 : s0 + nb]
            so.run([(1, src_addr, int(i) * n, dst_addr, k * n, n, tile_bytes, 0) for k, i in enumerate(part)])
            src = batch
            if get_device_tensors is not None:  # the latent cache is replicated across the mesh: read one chip
                shards = get_device_tensors(batch)
                if shards:
                    src = shards[0]
            chunks.append(ttnn.to_torch(src)[: len(part)])
        _alloc_check(tensor, before, "page read")
        return torch.cat(chunks, dim=0)
    for i in ids:
        if geo is not None:  # page i's tiles -> the persistent read page: one SlotOps program for every page
            n, tile_bytes = geo
            so.run([(1, int(tensor.buffer_address()), i * n, int(read_buf.buffer_address()), 0, n, tile_bytes, 0)])
        else:
            ttnn.slice(tensor, [i] + [0] * len(trailing), [i + 1] + trailing, output_tensor=read_buf)
        src = read_buf
        if get_device_tensors is not None:  # the latent cache is replicated across the mesh: read one chip
            shards = get_device_tensors(read_buf)
            if shards:
                src = shards[0]
        chunks.append(ttnn.to_torch(src))
    _alloc_check(tensor, before, "page read")
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0)


def _ttnn_write_pages(tensor: Any, ids: list[int], pages: torch.Tensor) -> Any:
    """Host -> device page scatter through ``paged_fill_cache``.

    ``pages`` is ``[n, heads, block_size, head_size]``; the op takes the rows as
    one ``[1, heads, n * block_size, head_size]`` input and a ``[1, n]`` page
    table naming the destination blocks. Returns the tensor that now holds the
    pages: the same device tensor (in-place), or, for a HOST ttnn tensor, a
    rebuilt host tensor (no in-place host write exists). The device branch has
    not run on a chip.
    """
    import ttnn  # noqa: PLC0415  (device path only)

    n = len(ids)
    if n == 0:
        return tensor
    if _ttnn_is_host(tensor):
        full = ttnn.to_torch(tensor)
        full[torch.as_tensor(ids, dtype=torch.int64)] = pages.to(full.dtype)
        return ttnn.from_torch(full, dtype=tensor.dtype, layout=tensor.layout)
    # One page per fill through the PERSISTENT stage + page table (_page_bufs, created before the capture): the host
    # rows go in with copy_host_to_device_tensor, so a write allocates no device buffer at request time.
    _read_buf, stage, table = _page_bufs(tensor)
    before = _alloc_probe(tensor)
    device = tensor.device()
    mesh_kwargs = _mesh_kwargs(device)
    for k, i in enumerate(ids):
        rows = pages[k : k + 1].permute(1, 0, 2, 3).reshape(1, *(int(v) for v in pages.shape[1:2]), -1, int(pages.shape[-1]))
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(rows.contiguous(), dtype=tensor.dtype, layout=ttnn.TILE_LAYOUT, **mesh_kwargs), stage
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([[int(i)]], dtype=torch.int32), dtype=ttnn.int32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, **mesh_kwargs),
            table,
        )
        ttnn.experimental.paged_fill_cache(tensor, stage, table, batch_idx=0)
    ttnn.synchronize_device(device)
    _alloc_check(tensor, before, "page write")
    return tensor


def _owner_batches(pairs: Sequence[tuple[int, int]], nb: int) -> list[tuple[int, list[tuple[int, int]]]]:
    """[(owner, local)] -> [(owner, [(position in pairs, local)] of at most nb)], owners in first-seen order"""
    by: dict[int, list[tuple[int, int]]] = {}
    for k, (owner, local) in enumerate(pairs):
        by.setdefault(int(owner), []).append((k, int(local)))
    return [(o, items[s0 : s0 + nb]) for o, items in by.items() for s0 in range(0, len(items), nb)]


class _PendingPages:
    """Stage 12 cf3h (9's profile of mock n29h7p2gprof: 94 % of the engine thread's store time was read_locals'
    to_torch waiting for the device to drain the steps ahead of the read): a bulk page read ENQUEUED -- per owner batch,
    the SlotOps copy and a non-blocking read of the owner's shard, then one event -- whose host shards fill
    asynchronously (the completion-queue reader). ``result()`` waits for the event (GIL released) and assembles the pages;
    the store worker calls it, so the engine thread never waits for the device in a store."""

    def __init__(self, reads: list, n: int, page_shape: tuple, event: Any, nbytes: int):
        self.reads, self.n, self.page_shape, self.event, self.nbytes = reads, n, page_shape, event, nbytes

    def result(self) -> torch.Tensor:
        if self.event is not None:
            import ttnn  # noqa: PLC0415

            ttnn.event_synchronize(self.event)
            self.event = None
        out: list[torch.Tensor | None] = [None] * self.n
        for ks, host in self.reads:
            got = host.to_torch()  # host only (the read landed with the event): what ttnn.to_torch does after from_device
            for j, k in enumerate(ks):
                out[k] = got[j : j + 1]
        self.reads = []
        return torch.cat(out, dim=0).reshape(self.n, *self.page_shape)


def _pages_now(pages: _PendingPages | torch.Tensor) -> torch.Tensor:
    """a store read's pages: an enqueued read's result (waits for its event) or the pages already read"""
    return pages.result() if isinstance(pages, _PendingPages) else pages


class _TtnnMeshPageIO:
    """One local page per call on a mesh tensor (the sharded pool's per-chip local pools), through the PERSISTENT page
    buffers of ``_page_bufs`` (created by the pre-capture warm; no request-time device allocation):

    * ``read_local(owner, local)``: local block ``local``'s tiles copied on EVERY chip into the persistent read page by
      the model's SlotOps (``_slot_ops``: runtime-arg addresses, ONE program for every page; the same args on every
      chip of the mesh -- lock-step allocation -- so the non-owners copy their own, never-read block), then the OWNER's
      shard of that page to host. ``ttnn.slice`` is only the fallback without the model tree: it keys its PROGRAM on
      the page index, and a program first compiled after the trace capture owns buffers alive at the next replay
      (gx2c2 / gx2c3, d0 card1 2026-10-23 00:28Z; review_shard_pool.md B1). ``GLM53_KV_PAGE_ALLOC_CHECK`` asserts the
      DRAM allocation AND the program-cache count are unchanged by every read.
    * ``write_local(owner, local, page)``: the page rows into the persistent stage (replicated), the persistent
      ``[1, 1]`` page table written PER CHIP (``ShardTensorToMesh``: the owner's row = ``local``, every other chip's row
      = 0, its null block, never a request's page), then ``paged_fill_cache`` -- the page lands on its owner only.
    """

    def __init__(self, tensor: Any, C: int):
        self.tensor = tensor
        self.C = int(C)
        self.page_shape = tuple(_ttnn_shape(tensor)[1:])

    def _device_order(self) -> Any:
        import ttnn  # noqa: PLC0415

        return getattr(ttnn, "get_device_tensors", None)

    def read_local(self, owner: int, local: int) -> torch.Tensor:
        import ttnn  # noqa: PLC0415

        if _ttnn_is_host(self.tensor):  # host ttnn tensors (tests): one shard, the local index addresses it
            return ttnn.to_torch(self.tensor)[int(local) : int(local) + 1]
        trailing = list(self.page_shape)
        read_buf, _stage, _table = _page_bufs(self.tensor)
        so = _slot_ops(self.tensor.device())  # created (with its tensor) by the warm, before the freeze and the probe
        geo = _page_copy_geometry(self.tensor) if so is not None else None
        before = _alloc_probe(self.tensor)
        if geo is not None:  # the local block's tiles -> the persistent read page: one SlotOps program for every page
            n, tile_bytes = geo
            so.run([(1, int(self.tensor.buffer_address()), int(local) * n, int(read_buf.buffer_address()), 0, n,
                     tile_bytes, 0)])
            out = None
        else:  # fallback (no model tree / an unsupported cache layout): page-keyed program, refused by the guard
            out = ttnn.slice(self.tensor, [int(local)] + [0] * len(trailing), [int(local) + 1] + trailing,
                             output_tensor=read_buf)
        try:
            src = read_buf
            get_device_tensors = self._device_order()
            if get_device_tensors is not None:
                shards = get_device_tensors(read_buf)
                if shards:
                    if int(owner) >= len(shards):
                        raise IndexError(f"owner chip {owner} outside the {len(shards)}-chip mesh")
                    src = shards[int(owner)]
            page = ttnn.to_torch(src)
        finally:
            _release_slice_output(out, read_buf)
        _alloc_check(self.tensor, before, "sharded page read")
        return page.reshape(1, *self.page_shape)

    def read_locals(self, pairs: list[tuple[int, int]]) -> torch.Tensor:
        """Stage 12 (n29h7 M1, program 2026-10-23T19:26:26Z: a 131,072-token store = 2,048 pages x 11 layers of
        read_local, ~0.01 GB/s on the engine thread): pages [(owner, local)] in bulk -- per owner chip, up to nb local
        blocks per SlotOps op list into the persistent multi-page buffer (_page_batch; lock-step args, so every chip
        copies ITS local block of slot k and only the owner's shard is read), then ONE to_torch of that shard per batch.
        [len(pairs), *page_shape] in the given order; read_local per page without the buffer / SlotOps."""
        import ttnn  # noqa: PLC0415

        so = None if _ttnn_is_host(self.tensor) else _slot_ops(self.tensor.device())
        geo = _page_copy_geometry(self.tensor) if so is not None else None
        batch = _page_batch(self.tensor) if geo is not None else None
        get_device_tensors = self._device_order()
        if batch is None or get_device_tensors is None:
            return torch.cat([self.read_local(o, lo).reshape(1, *self.page_shape) for o, lo in pairs], dim=0)
        n, tile_bytes = geo
        nb = int(batch.padded_shape[0]) if hasattr(batch, "padded_shape") else int(_ttnn_shape(batch)[0])
        shards = get_device_tensors(batch)
        src_addr, dst_addr = int(self.tensor.buffer_address()), int(batch.buffer_address())
        before = _alloc_probe(self.tensor)
        out: list[torch.Tensor | None] = [None] * len(pairs)
        for owner, items in _owner_batches(pairs, nb):
            if int(owner) >= len(shards):
                raise IndexError(f"owner chip {owner} outside the {len(shards)}-chip mesh")
            so.run([(1, src_addr, int(local) * n, dst_addr, j * n, n, tile_bytes, 0) for j, (_k, local) in enumerate(items)])
            got = ttnn.to_torch(shards[int(owner)])
            for j, (k, _local) in enumerate(items):
                out[k] = got[j : j + 1]
        _alloc_check(self.tensor, before, "sharded page read (bulk)")
        return torch.cat(out, dim=0).reshape(len(pairs), *self.page_shape)

    def read_locals_async(self, pairs: list[tuple[int, int]]) -> _PendingPages | torch.Tensor:
        """read_locals ENQUEUED (Stage 12 cf3h): the same SlotOps batches, each owner shard read non-blocking right
        behind its copy (the queue is in order: a batch is read before the next copy overwrites the buffer, and the
        pages are copied before any later step writes them), then an event -> _PendingPages. Without the batch
        buffer / SlotOps: read_locals' pages, read now."""
        import ttnn  # noqa: PLC0415

        so = None if _ttnn_is_host(self.tensor) else _slot_ops(self.tensor.device())
        geo = _page_copy_geometry(self.tensor) if so is not None else None
        batch = _page_batch(self.tensor) if geo is not None else None
        get_device_tensors = self._device_order()
        if batch is None or get_device_tensors is None or not hasattr(ttnn, "record_event"):
            return self.read_locals(pairs)
        n, tile_bytes = geo
        nb = int(batch.padded_shape[0]) if hasattr(batch, "padded_shape") else int(_ttnn_shape(batch)[0])
        shards = get_device_tensors(batch)
        src_addr, dst_addr = int(self.tensor.buffer_address()), int(batch.buffer_address())
        before = _alloc_probe(self.tensor)
        reads = []
        for owner, items in _owner_batches(pairs, nb):
            if int(owner) >= len(shards):
                raise IndexError(f"owner chip {owner} outside the {len(shards)}-chip mesh")
            so.run([(1, src_addr, int(local) * n, dst_addr, j * n, n, tile_bytes, 0) for j, (_k, local) in enumerate(items)])
            reads.append(([k for k, _local in items], ttnn.from_device(shards[int(owner)], blocking=False)))
        event = ttnn.record_event(self.tensor.device(), 0)
        _alloc_check(self.tensor, before, "sharded page read (enqueued)")
        page_elems = 1
        for d in self.page_shape:
            page_elems *= int(d)
        nbytes = len(pairs) * page_elems * _ttnn_dtype_to_torch(self.tensor.dtype).itemsize
        return _PendingPages(reads, len(pairs), self.page_shape, event, nbytes)

    def write_local(self, owner: int, local: int, page: torch.Tensor) -> None:
        import ttnn  # noqa: PLC0415

        page = page.reshape(1, *self.page_shape)
        if _ttnn_is_host(self.tensor):
            full = ttnn.to_torch(self.tensor)
            full[int(local)] = page[0].to(full.dtype)
            self.tensor = ttnn.from_torch(full, dtype=self.tensor.dtype, layout=self.tensor.layout)
            return
        _read_buf, stage, table = _page_bufs(self.tensor)
        before = _alloc_probe(self.tensor)
        device = self.tensor.device()
        heads = int(self.page_shape[0])
        rows = page.permute(1, 0, 2, 3).reshape(1, heads, -1, int(self.page_shape[-1]))
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(rows.contiguous(), dtype=self.tensor.dtype, layout=ttnn.TILE_LAYOUT,
                            **_mesh_kwargs(device)),
            stage,
        )
        per_chip = torch.zeros(self.C, 1, dtype=torch.int32)  # every chip's row: its null block 0 ...
        per_chip[int(owner), 0] = int(local)  # ... except the owner's: the page's local block
        shard = getattr(ttnn, "ShardTensorToMesh", None)
        if shard is not None and getattr(device, "get_num_devices", lambda: 1)() > 1:
            host_table = ttnn.from_torch(per_chip, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                         mesh_mapper=shard(device, dim=0))
        else:
            host_table = ttnn.from_torch(per_chip[int(owner) : int(owner) + 1], dtype=ttnn.int32,
                                         layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_table, table)
        ttnn.experimental.paged_fill_cache(self.tensor, stage, table, batch_idx=0)
        ttnn.synchronize_device(device)
        _alloc_check(self.tensor, before, "sharded page write")

    def warm(self) -> None:
        """Before the capture, on the null block (local 0 of chip 0: never a request's page, read and written back
        unchanged): pass 1 (probe suspended) creates the persistent page buffers and the SlotOps tensor and COMPILES
        the read / write programs (their kernel-binary DRAM buffers belong to the cached programs: d0 card2 03:45Z
        measured +2,048 B/bank for the SlotOps copy); pass 2 (probed) must allocate nothing and compile nothing --
        a per-request allocation or a page-keyed program fails here, at init, not at a decode replay."""
        if _ttnn_is_host(self.tensor):
            return
        _WARMING[0] = True
        try:
            page = self.read_local(0, 0)
            self.read_locals([(0, 0), (0, 0)])  # the bulk store read (Stage 12): same SlotOps program, the batch buffer
            _pages_now(self.read_locals_async([(0, 0), (0, 0)]))  # the enqueued form (cf3h): no program of its own
            self.write_local(0, 0, page)
        finally:
            _WARMING[0] = False
        check = os.environ.get("GLM53_KV_PAGE_ALLOC_CHECK", "0")
        os.environ["GLM53_KV_PAGE_ALLOC_CHECK"] = "1"  # the second pass is always probed (bytes + programs)
        try:
            before = _alloc_probe(self.tensor)
            page = self.read_local(0, 0)
            self.read_locals([(0, 0), (0, 0)])
            _pages_now(self.read_locals_async([(0, 0), (0, 0)]))
            self.write_local(0, 0, page)
            after = _alloc_probe(self.tensor)
        finally:
            os.environ["GLM53_KV_PAGE_ALLOC_CHECK"] = check
        if after != before:
            raise RuntimeError(
                f"KV connector sharded page warm: the probed second pass changed the device state ({before} -> "
                f"{after}: DRAM B/bank, programs): a per-request allocation or program in the read / write path"
            )


def _release_slice_output(out: Any, read_buf: Any) -> None:
    """drop whatever ``ttnn.slice(..., output_tensor=read_buf)`` returned: a handle of the persistent page (nothing to
    free) or, should the op have allocated its own output, that buffer -- deallocated here so no slice output outlives
    the page read (gx2c2 / gx2c3: 'device buffer(s) still alive before trace replay')"""
    if out is None or out is read_buf:
        return
    try:
        same = int(out.buffer_address()) == int(read_buf.buffer_address())
    except Exception:  # a host tensor / no buffer: nothing on the device to free
        return
    if not same:
        import ttnn  # noqa: PLC0415

        ttnn.deallocate(out)


def _FakeLayerIO_for(tensor: torch.Tensor) -> Any:
    """a torch 'local pool' handed by a mock model hook: one chip's pool (owner ignored; tests use FakeMeshLatentPool)"""

    class _Io:
        def read_local(self, owner, local):
            return tensor[int(local) : int(local) + 1].clone()

        def write_local(self, owner, local, page):
            tensor[int(local)] = page.reshape(tensor.shape[1:]).to(tensor.dtype)

        def warm(self):
            pass

    return _Io()


def tt_copy_kv_blocks(
    src_kv_caches: dict[str, Any],
    dst_kv_caches: dict[str, Any],
    src_block_ids: list[int],
    dst_block_ids: list[int],
    direction: CopyDirection,
) -> None:
    """``CopyBlocksOp`` for ``KVConnectorBase_V1.set_host_xfer_buffer_ops``.

    Copies page ``src_block_ids[k]`` of every layer in ``src_kv_caches`` to
    page ``dst_block_ids[k]`` of the same layer in ``dst_kv_caches``. The host
    buffer (``h2d`` source / ``d2h`` destination) is a ``torch.Tensor`` whose
    trailing dims equal the device tensor's; the device side is a
    ``torch.Tensor`` (mock) or ``ttnn.Tensor``. The two id lists must be the
    same length; nothing is copied when either side or list is empty, which is
    upstream ``copy_kv_blocks``' contract too.
    """
    if direction not in ("h2d", "d2h"):
        raise ValueError(f"direction must be 'h2d' or 'd2h', got {direction!r}")
    if not src_kv_caches or not dst_kv_caches or not src_block_ids or not dst_block_ids:
        return
    if len(src_block_ids) != len(dst_block_ids):
        raise ValueError(
            f"{len(src_block_ids)} source blocks but {len(dst_block_ids)} "
            "destination blocks"
        )
    src_ids = [int(i) for i in src_block_ids]
    dst_ids = [int(i) for i in dst_block_ids]
    for layer_name, src_tensor in src_kv_caches.items():
        dst_tensor = dst_kv_caches.get(layer_name)
        if dst_tensor is None:
            raise KeyError(f"no destination cache for layer {layer_name!r}")
        for side, tensor, ids in (
            ("src", src_tensor, src_ids),
            ("dst", dst_tensor, dst_ids),
        ):
            num_blocks = int(tensor.shape[0])
            bad = [i for i in ids if not 0 <= i < num_blocks]
            if bad:
                raise IndexError(
                    f"layer {layer_name!r} {side}: block ids {bad} outside "
                    f"[0, {num_blocks})"
                )
        _write_pages(dst_tensor, dst_ids, _read_pages(src_tensor, src_ids))


# ---------------------------------------------------------------------------
# Step metadata: slot mappings and a forward context for the connector
# ---------------------------------------------------------------------------


@dataclass
class TTKVAttentionMetadata(AttentionMetadata):
    """What the TT runner can tell a connector about one step.

    ``slot_mapping[t] = block_id * block_size + offset`` for every token the
    step writes, in forward-row order; ``query_start_loc`` delimits the rows.
    Deliberately not an ``MLACommonMetadata`` (see the module docstring).
    """

    num_reqs: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    is_prefill: bool


def _layer_attn_metadata(
    view: TTKVLayerView,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    is_prefill: bool,
) -> AttentionMetadata:
    """Per-layer metadata: ``MLACommonMetadata`` for latent layers, else ours.

    vLLM's connectors select their MLA slot-mapping path on
    ``isinstance(attn_metadata, MLACommonMetadata)``; the instance is filled in
    with the step's real counts so a connector that does read them sees
    consistent values.
    """
    num_reqs = int(query_start_loc.numel() - 1)
    num_tokens = int(slot_mapping.numel())
    if view.is_mla:
        from vllm.model_executor.layers.attention.mla_attention import (
            MLACommonMetadata,
        )

        lens = (query_start_loc[1:] - query_start_loc[:-1]) if num_reqs else None
        max_query_len = int(lens.max()) if lens is not None and lens.numel() else 0
        return MLACommonMetadata(
            num_reqs=num_reqs,
            max_query_len=max_query_len,
            max_seq_len=max_query_len,
            num_actual_tokens=num_tokens,
            query_start_loc=query_start_loc,
            slot_mapping=slot_mapping,
            num_decodes=0 if is_prefill else num_reqs,
            num_decode_tokens=0 if is_prefill else num_tokens,
            num_prefills=num_reqs if is_prefill else 0,
        )
    return TTKVAttentionMetadata(
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        query_start_loc=query_start_loc,
        slot_mapping=slot_mapping,
        block_size=view.block_size,
        is_prefill=is_prefill,
    )


class _TTKVLayer:
    """Stand-in for an attention module in ``ForwardContext.no_compile_layers``."""

    __slots__ = ("kv_cache", "layer_name")

    def __init__(self, layer_name: str, kv_cache: Any):
        self.layer_name = layer_name
        self.kv_cache = kv_cache


def _row_positions(model_input: Any, row: int) -> list[int]:
    """Positions row ``row`` of a model input writes this step."""
    positions = torch.as_tensor(model_input.input_positions)
    prompt_lens = getattr(model_input, "prompt_lens", None)
    if prompt_lens is not None:
        start = int(positions[row])
        end = int(torch.as_tensor(prompt_lens)[row])
        return list(range(start, end))
    pos = positions[row].reshape(-1).tolist()
    num_valid_drafts = getattr(model_input, "num_valid_drafts", None)
    if num_valid_drafts is not None:
        pos = pos[: 1 + int(torch.as_tensor(num_valid_drafts)[row])]
    return [int(p) for p in pos if int(p) >= 0]


def build_step_slot_mapping(
    model_input: Any, block_table: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(slot_mapping, query_start_loc)`` for the rows of one forward.

    ``block_table`` is the ``[rows, max_blocks]`` table of the layer's kv-cache
    group (``TTModelInput.block_tables_per_group[g]``); row ``i`` of the input
    is request ``model_input.row_req_ids[i]``. Block 0 is vLLM's null block, so
    a position whose block table entry is 0 past the row's coverage would be a
    bug in the caller; it is reported rather than silently mapped.
    """
    row_req_ids = getattr(model_input, "row_req_ids", None)
    num_rows = (
        len(row_req_ids) if row_req_ids is not None else int(block_table.shape[0])
    )
    # one gather per row (n29h7p2 1M window, 2026-10-23: the per-position int(table[row, blk]) loop cost ~25 ms per
    # 8,192-token step and 2.6 s at the held-store flush after a 1,048,000-token prompt, on the first-token path)
    parts: list[torch.Tensor] = []
    starts = [0]
    table = torch.as_tensor(block_table)
    prompt_lens = getattr(model_input, "prompt_lens", None)
    for row in range(num_rows):
        if prompt_lens is not None:  # a prefill row: positions start..end-1 (_row_positions without the list)
            pos = torch.arange(int(torch.as_tensor(model_input.input_positions)[row]),
                               int(torch.as_tensor(prompt_lens)[row]), dtype=torch.int64)  # fmt: skip
        else:
            pos = torch.as_tensor(_row_positions(model_input, row), dtype=torch.int64)
        if pos.numel():
            blk = pos // block_size
            over = torch.nonzero(blk >= table.shape[1])
            if over.numel():
                k = int(over[0])
                raise IndexError(
                    f"row {row}: position {int(pos[k])} needs block column {int(blk[k])}, "
                    f"block table has {table.shape[1]} columns"
                )
            parts.append(table[row].to(torch.int64)[blk] * block_size + pos % block_size)
        starts.append(starts[-1] + int(pos.numel()))
    return (
        torch.cat(parts) if parts else torch.empty(0, dtype=torch.int64),
        torch.tensor(starts, dtype=torch.int32),
    )


def _make_forward_context(
    layers: dict[str, _TTKVLayer],
    attn_metadata: dict[str, AttentionMetadata],
    slot_mapping: dict[str, torch.Tensor],
) -> Any:
    try:
        from vllm.forward_context import ForwardContext

        return ForwardContext(
            no_compile_layers=layers,
            attn_metadata=attn_metadata,
            slot_mapping=slot_mapping,
        )
    except (ImportError, TypeError):  # pragma: no cover - older/newer field sets
        return SimpleNamespace(
            no_compile_layers=layers,
            attn_metadata=attn_metadata,
            slot_mapping=slot_mapping,
        )


# ---------------------------------------------------------------------------
# The host
# ---------------------------------------------------------------------------


def find_tt_gpu_connector(connector: Any) -> Any | None:
    """A2's ``VLLMPagedMemTTConnector`` inside vLLM's ``LMCacheConnectorV1``.

    Duck-typed: ``connector._lmcache_engine.lmcache_engine.gpu_connector`` with
    a ``set_host_xfer_buffer_ops`` method. ``None`` for any other connector.
    """
    impl = getattr(connector, "_lmcache_engine", None)
    engine = getattr(impl, "lmcache_engine", None)
    gpu = getattr(engine, "gpu_connector", None)
    if gpu is not None and callable(getattr(gpu, "set_host_xfer_buffer_ops", None)):
        return gpu
    return None


def step_boundary_checkpoints(
    first: int, committed: int, block_size: int, *, num_valid_drafts: int = 0
) -> tuple[list[tuple[int, int]], list[int]]:
    """Page boundaries of one decode row, split by what the step committed.

    The row's input sits at position ``first`` (= ``c``); the step committed
    ``committed`` tokens (``n + 1``: the token at ``c`` plus ``n`` accepted
    drafts), so the committed length is ``P = c + committed``; its verify rows
    wrote positions ``c .. c + num_valid_drafts``. Returns ``(committed,
    deferred)``: ``committed`` is ``(B, candidate_row)`` for every page boundary
    ``c < B <= P``, where candidate row ``B - c - 1`` holds the state after
    ``B`` tokens (contract §4); ``deferred`` lists the boundaries the verify rows
    crossed past ``P`` (``P < B <= c + num_valid_drafts + 1``): rejected, not
    committed, re-evaluated by the next step.
    """
    bs = int(block_size)
    c, m = int(first), int(committed)
    if bs <= 0 or c < 0 or m <= 0:
        return [], []
    p = c + m
    committed_boundaries = [
        (b, b - c - 1) for b in range((c // bs + 1) * bs, p + 1, bs)
    ]
    written_end = c + int(num_valid_drafts) + 1
    deferred = [b for b in range((p // bs + 1) * bs, written_end + 1, bs)]
    return committed_boundaries, deferred


def _runner_request_tokens(runner: Any, req_id: str) -> list[int] | None:
    """A request's committed token history (prompt + outputs) from the runner."""
    state = getattr(runner, "requests", {}).get(req_id)
    if state is None:
        return None
    prompt = getattr(state, "prompt_token_ids", None) or []
    return [int(t) for t in prompt] + [
        int(t) for t in getattr(state, "output_token_ids", []) or []
    ]


def _runner_request_len(runner: Any, req_id: str) -> int | None:
    """len(_runner_request_tokens(...)) without building the list (Stage 12, program 2026-10-23T19:45:01Z: per-step
    host work must be O(1) in the context -- a 1M-token history is a 1M-element list per call)"""
    state = getattr(runner, "requests", {}).get(req_id)
    if state is None:
        return None
    return len(getattr(state, "prompt_token_ids", None) or []) + len(getattr(state, "output_token_ids", []) or [])


def _runner_request_head(runner: Any, req_id: str, n: int) -> list[int] | None:
    """the first n tokens of a request's history (prompt, then outputs) without building the whole list (O(n))"""
    state = getattr(runner, "requests", {}).get(req_id)
    if state is None:
        return None
    head = list((getattr(state, "prompt_token_ids", None) or [])[:n])
    if len(head) < n:
        head += list((getattr(state, "output_token_ids", None) or [])[: n - len(head)])
    return head


def _host_history_len(host: Any, req_id: str) -> int | None:
    """a connector host's view of a request's committed history LENGTH: O(1) with the runner's len_of (every engine
    step asks while page stores are held), else len(tokens_of(...))"""
    f = getattr(host, "_len_of", None)
    if callable(f):
        return f(req_id)
    history = host._tokens_of(req_id)
    return None if history is None else len(history)


def _checkpoint_grid(runner: Any) -> int:
    """The KDA checkpoint / hit grid: max(page, tt.prefill_chunk_alignment) (plugin 630e9f7: the scheduler's grid)."""
    try:
        from vllm_tt_plugin.config import get_tt_prefill_chunk_alignment  # noqa: PLC0415

        align = int(get_tt_prefill_chunk_alignment(runner.vllm_config) or 0)
    except Exception:  # noqa: BLE001
        align = 0
    block = int(getattr(getattr(runner.vllm_config, "cache_config", None), "block_size", 0) or 0)
    return max(block, align)


_KDA_PROVENANCE: dict = {}  # request id -> full LMCache chunk tokens at admission, when they run past its KDA hit
# (provenance): LMCache and the checkpoint store are first-writer-wins per key, so a request that RECOMPUTES pages
# LMCache already holds (another request's computation, another chunk geometry) holds a KDA / DSA state that agrees
# with its own pages, not with LMCache's -- a hit on its checkpoint would pair LMCache's pages with that state (a hit's
# output then departs from the cold one). Such a request exports no checkpoint (a shorter hit later, never a mismatched
# one). Set by the scheduler-side TTPageAlignedConnector, read by the worker in the same process.
# GLM53_KDA_CKPT_PROVENANCE=restore (cf3h-e default, program 2026-10-24T20:43:55Z): the bar is lifted for a request
# that REPLACES the span it recomputes. At its first prefill step the worker removes LMCache's chunk keys for [w, m) from
# every tier (w = its load window = the KDA hit rounded up to LMCache's chunk, m = LMCache's match): LMCache's adapter
# saves the request from w (num_saved_tokens = the load window), so its own pages land there and its checkpoints pair
# with them. Guards (the served kv_load_failure_policy is "fail"): no removal, the bar stays, when another request's
# admitted LMCache load or another request's running prefill (its pages / checkpoints still being written) shares a
# chunk key past w, or when any key is still held by a tier after the removal.
_KDA_RESTORE: dict = {}  # request id -> (w, m, prompt token ids): restore candidates (scheduler -> worker)
_LMC_LOADS: dict = {}  # request id -> (token ids, load window end): admitted LMCache loads whose step has not run
_LMC_WRITERS: dict = {}  # request id -> token ids: requests in the middle of their prefill (pages being stored)
_PROVENANCE_LOCK = threading.Lock()
# Mixed provenance (program 2026-10-24T20:58:54Z, ruling A): a restore replaces [w, m), so a checkpoint DEEPER than w
# that another computation wrote BEFORE the restore (the first writer's) now pairs with the replaced pages. Per server
# process: the replaced spans (w, n, key of the prefix through the first chunk past w) and the keys of checkpoints
# exported after a restore on such a prefix (those pair with the current pages); an import of any other checkpoint
# deeper than w on a replaced prefix counts as kda_provenance_mixed_imports (a restart forgets the spans).
_RESTORED_SPANS: list = []
_POST_RESTORE_CKPTS: set = set()


def kda_ckpt_provenance_mode() -> str:
    """GLM53_KDA_CKPT_PROVENANCE: "restore" (default), "1" (the bar alone, cf3h-d) or "0" (no rule)."""
    v = os.environ.get("GLM53_KDA_CKPT_PROVENANCE", "restore").strip().lower()
    if v not in ("0", "1", "restore"):
        raise ValueError(f"GLM53_KDA_CKPT_PROVENANCE={v!r}: expected 0, 1 or restore")
    return v


def _lmcache_chunk_tokens() -> int:
    from vllm_tt_plugin.lmcache.page_contract import LMCACHE_CHUNK_TOKENS  # noqa: PLC0415

    return int(os.environ.get("GLM53_LMCACHE_CHUNK_TOKENS", LMCACHE_CHUNK_TOKENS))


def _shares_chunk_past(a: Sequence[int], b: Sequence[int], w: int, chunk: int) -> bool:
    """Do token sequences a and b share an LMCache chunk key past w? A chunk key hashes its whole prefix, so they share
    one exactly when they agree through the end of the first chunk past w (or through the shorter one's end, a partial
    chunk, when that ends inside it)."""
    n = min(int(w) + int(chunk), len(a), len(b))
    return n > int(w) and list(a[:n]) == list(b[:n])


def lmcache_remove_keys(backends: dict, keys: Iterable[Any]) -> list:
    """Remove every key from each LMCache tier that holds it (local CPU, local disk, the remote fs dir); return the keys
    some tier still holds afterwards (the removal failed for those). A tier is asked to remove only a key it holds:
    LMCache-tt's fs connector logs an ERROR for a missing file."""
    left = []
    for key in keys:
        for name, backend in backends.items():
            try:
                if backend.contains(key):
                    backend.remove(key)
            except Exception as e:  # noqa: BLE001 -- a tier that cannot remove keeps the key: the post-check catches it
                logger.warning("LMCache %s: remove of %s failed: %s", name, getattr(key, "chunk_hash", key), e)
        held = []
        for name, backend in backends.items():
            try:
                if backend.contains(key):
                    held.append(name)
            except Exception:  # noqa: BLE001
                held.append(name)
        if held:
            left.append(key)
    return left
_SAVE_HOSTS = weakref.WeakSet()  # TTKVConnectorHost instances with a deferred save (the scheduler-side flush hook)
_WORKER_HOSTS = weakref.WeakSet()  # TTKVConnectorHost instances with a store worker (drain_store_workers)
# GLM53_KV_COUNTERS_LOG_S (Stage 12, default 60; 0 = off): one INFO line of the Stage 12 counters, which otherwise
# live only as attributes on three objects: the worker's host, the scheduler-side connector and the scheduler (the
# k4 cells' pass criteria: preempt / exact hits / the ZC worker / the D-side waits)
_COUNTER_SOURCES = {"host": weakref.WeakSet(), "lookup": weakref.WeakSet(), "sched": weakref.WeakSet(),
                    "runner": weakref.WeakSet()}  # fmt: skip
_COUNTER_KEYS = {
    "host": ("preempt_saves", "preempt_saves_failed", "preempt_saves_over", "preempt_saves_short", "preempt_saves_snapped",
             "preempt_ckpts", "preempt_ckpts_spec", "preempt_snap_no_ckpt", "preempt_ckpt_skipped_provenance",
             "zc_worker_jobs", "zc_preempt_drains", "pd_exact_end_exports", "segment_saves_parked",
             "segment_saves_released", "segment_saves_dropped", "segment_slots_moved", "preempt_saves_segment_held",
             "lookups_admitted_first", "store_yields", "store_yield_s"),
    "lookup": ("preempt_exact_hits", "preempt_lookup_waits", "expect_waits", "expect_waits_ckpt", "expect_timeouts",
               "expect_timeouts_ckpt", "hits_replanned", "lookups_deferred", "cold_requests", "lookups_deferred_kda"),
    "sched": ("preempt_saves_recorded", "admissions_deferred_kv", "admit_reserve_capped", "prefill_one_request_deferred",
              "segment_chunks", "segment_steps", "segment_decode_steps", "segment_early_releases", "segment_holds_broken",
              "segment_skipped_block_edge", "segment_release_retries", "interleave_tail_exempts",
              "prefill_short_first_holds", "prefill_short_first_yields", "prefill_short_first_stuck",
              "prefill_short_first_kv_overtakes", "prefill_short_first_no_hold"),
    "runner": ("prefill_short_first_paced", "prefill_short_first_paced_wait_s",  # v3b4 (prefill_pacer.PrefillPacer)
               "gc_collections_g0", "gc_collections_g1", "gc_collections_g2", "gc_pause_s_g0", "gc_pause_s_g1",
               "gc_pause_s_g2", "gc_pause_s_max", "gc_refreezes", "gc_refreeze_pause_s", "gc_frozen_objects"),  # v3b5 (gc_tuning.GCTuning)
}
_GPU_STAT_KEYS = ("h2d_split_partial", "d2h_one_dma_saves", "d2h_one_dma_repeated_pages", "d2h_resident_saves",
                  "h2d_resident_submits")  # fmt: skip
_COUNTERS_LAST = {"t": 0.0, "snap": None}


def counters_snapshot() -> dict:
    """the Stage 12 counters summed over their live sources (+ the GPU connector's stats of every host)"""
    out: dict = {}
    for kind, keys in _COUNTER_KEYS.items():
        for obj in list(_COUNTER_SOURCES[kind]):
            d = getattr(obj, "__dict__", {})
            for k in keys:
                v = d.get(k, 0)
                if v:
                    out[k] = out.get(k, 0) + v
    for host in list(_COUNTER_SOURCES["host"]):
        st = getattr(getattr(host, "tt_gpu_connector", None), "stats", None) or {}
        for k in _GPU_STAT_KEYS:
            if st.get(k):
                out[k] = out.get(k, 0) + st[k]
    for k, v in _COLLECT_STATS.items():  # the collect gate's queueing and the deferred producers' totals (evidence)
        if v:
            out[k] = round(v, 3) if isinstance(v, float) else v
    for k, v in list(out.items()):
        if isinstance(v, float):
            out[k] = round(v, 3)
    return out


def log_counters(force: bool = False) -> None:
    """one INFO line every GLM53_KV_COUNTERS_LOG_S seconds when a counter changed (force: now, if any)"""
    every = float(os.environ.get("GLM53_KV_COUNTERS_LOG_S", "60") or 0)
    if every <= 0 and not force:
        return
    now = time.monotonic()
    if not force and now - _COUNTERS_LAST["t"] < every:
        return
    _COUNTERS_LAST["t"] = now
    snap = counters_snapshot()
    if snap and (force or snap != _COUNTERS_LAST["snap"]):
        import json  # noqa: PLC0415

        _COUNTERS_LAST["snap"] = snap
        logger.info("TT KV connector counters: %s", json.dumps(snap, sort_keys=True))


def drain_store_workers() -> None:
    """the worker's shutdown, BEFORE the runner releases the model (TTWorker.shutdown): the run's final counters line
    (GLM53_KV_COUNTERS_LOG_S), then _drain_store_workers"""
    log_counters(force=True)
    _drain_store_workers()


def _drain_store_workers() -> None:
    """every host's store worker lands
    its queued stores while the page-DMA is still open, and later zero-copy stores run inline (P3: a GLM53_KV_STORE_
    ZC_WORKER job reads through the DMA, which the model's release closes and unmaps)"""
    for host in list(_WORKER_HOSTS):
        try:
            host.__dict__["_zc_worker_decision"] = False  # (from here on: inline, as before P3)
            w = getattr(host, "store_worker", None)
            # bounded (GLM53_KV_SHUTDOWN_DRAIN_S, default 60): a wedged store must never keep the mesh from closing
            if w is not None and not w.wait_idle(float(os.environ.get("GLM53_KV_SHUTDOWN_DRAIN_S", "60") or 0)):
                logger.warning("KV connector: the store worker is still busy at shutdown; its queued stores are dropped")
        except Exception:  # noqa: BLE001 - a shutdown step never raises
            logger.exception("KV connector: draining the store worker at shutdown failed")

# GLM53_KV_LEAVE_DRAIN (Stage 12 M1, program 2026-10-26T17:08:39Z; default 0): a finished request's held LMCache page
# stores no longer flush all at once on the engine thread as it leaves (the post-1M ~50-s stall: 128 x 8,192-token
# stores). Its blocks are DELAY-FREED instead (the scheduler-side request_finished returns delay_free_blocks=True), the
# held steps drain at GLM53_KV_LEAVE_FLUSH_STEPS per engine step (default 1) while every other request keeps stepping,
# and the host reports the request in finished_sending once none of its steps is held, pending or queued on the store
# worker -- only then are its blocks freed (no reused block is read by a store). {req_id: {"inner": the inner
# connector delays it too, "inner_done": the inner reported it}}
_LEAVE_PENDING: dict = {}

# GLM53_PREEMPT_SAVE (program 2026-10-27T00:05:27Z (2); default 0; doc/ttnn/12-fast-start/preempt/design.md): no
# recompute preemption. The TT scheduler's _preempt_request records {req_id: (n computed tokens, block ids, the token
# ids [:n])} here BEFORE vLLM frees the blocks. The worker's begin_step -- which runs before that step's forward, the
# only writer of the freed blocks -- stores the request's KV [0, n) through LMCache (the decode region included:
# save_decode_cache=false never stores it) plus an EXACT KDA checkpoint at n. On resume the hit covers n (the lookup
# reads the request's whole history, not only its prompt), so nothing computed is recomputed.
_PREEMPT_SAVES: dict = {}
_PREEMPT_LOCK = threading.Lock()
# {req_id: n | None}: a preempted request's resume point (14 B1 / 9 (e), 2026-10-27T00:37-00:40Z). The resume lookup
# runs over the history cut to n -- vLLM's whole history is n + 1 tokens (the last sampled token is never computed),
# so LMCache's last chunk key, [128 k, n + 1), would miss the stored unfull chunk [128 k, n) and the hit would stop at
# floor128(n). None = recorded, the worker's save not done yet: the lookup is undetermined for that step. The worker
# sets n (its save landed) or drops the entry (it failed: recompute, today's path); request_finished drops it.
_PREEMPT_RESUME: dict = {}


def _preempt_save_on() -> bool:
    return os.environ.get("GLM53_PREEMPT_SAVE", "0") == "1"


def _preempt_save_max_tokens() -> int:
    """GLM53_PREEMPT_SAVE_MAX_TOKENS (default 131,072): a preempted request whose history is longer falls back to
    today's recompute (counted); its synchronous store would cost seconds of page DMA on the engine thread"""
    return max(0, int(os.environ.get("GLM53_PREEMPT_SAVE_MAX_TOKENS", "131072") or 0))


def _resume_as_decode() -> bool:
    """GLM53_RESUME_AS_DECODE (9's k4 model flag, c9cb202f9f4; the model reads the same key in this process): the
    model resumes a 1-token row that imports an EXACT checkpoint at any start as one decode step of its slot. Off: the
    model refuses an off-grid continuation (and its live export off the 64 grid), so a preempt save snaps its n down
    to the page grid."""
    return os.environ.get("GLM53_RESUME_AS_DECODE", "0") == "1"


def record_preempt_save(req_id: str, n: int, block_ids: Any, token_ids: Sequence[int]) -> bool:
    """the scheduler side (TTScheduler._preempt_request): one record per preemption -> True when recorded. n is the
    caller's min(num_computed_tokens, len(all_token_ids)) (14 B2: under async scheduling num_computed_tokens counts the
    in-flight step's drafts); a history shorter than n is refused (the store's token ids must cover its slots)."""
    if not _preempt_save_on() or n <= 0 or not block_ids:
        return False
    if len(token_ids) < n:
        with _PREEMPT_LOCK:
            _PREEMPT_SAVES.setdefault("__short__", []).append(req_id)
        return False
    if n > _preempt_save_max_tokens():
        with _PREEMPT_LOCK:
            _PREEMPT_SAVES.setdefault("__over__", []).append(req_id)
        return False
    groups = [list(g) for g in block_ids] if block_ids and isinstance(block_ids[0], (list, tuple)) else [list(block_ids)]
    with _PREEMPT_LOCK:
        _PREEMPT_SAVES[req_id] = {"n": int(n), "block_ids": groups, "token_ids": [int(t) for t in token_ids[:n]]}
        _PREEMPT_RESUME[req_id] = None  # the worker's save decides the resume point
    return True


# Form B v2 (P / D over Mooncake; doc/ttnn/12-fast-start/mooncake/served_config.md "the landed signal"): D's request
# carries kv_transfer_params {"tt.expect_hit": n} = the tokens P stored (its prompt; the proxy sets it). D's lookup is
# undetermined -- (None, False): vLLM retries it the next step, the engine thread never waits -- while the remote holds
# fewer than n tokens of it, for at most GLM53_KV_EXPECT_WAIT_S (default 10; then it takes what is there, counted). Its
# history is P's prompt + token 1, so the lookup runs over the history cut to n (the B1 rule: the last chunk key must
# be P's unfull [128 k, n)), and n is an exact resume point (with GLM53_RESUME_AS_DECODE token 1 runs as one decode
# step of the slot: the continuation stays bitwise). {req_id: first lookup time}
_EXPECT_T0: dict = {}
# Form B v2, the P side: {req_id: n} of a P request (kv_transfer_params "tt.pd_role": "P", under GLM53_RESUME_AS_DECODE):
# its prefill exports the EXACT prompt-end KDA checkpoint at n, off the page grid too, so D imports there and decodes
# token 1 (a bitwise continuation); recorded at its lookup, dropped at request_finished
_PD_EXACT_END: dict = {}


def _pd_role(request: Any) -> str:
    kv = getattr(request, "kv_transfer_params", None)
    if kv is None:
        sp = getattr(request, "sampling_params", None)
        kv = (getattr(sp, "extra_args", None) or {}).get("kv_transfer_params")
    return str((kv or {}).get("tt.pd_role") or "")


def _expect_hit(request: Any) -> int:
    """the tokens a D-side request expects from the remote store (kv_transfer_params "tt.expect_hit"; 0 = none)"""
    kv = getattr(request, "kv_transfer_params", None)
    if kv is None:
        sp = getattr(request, "sampling_params", None)
        kv = (getattr(sp, "extra_args", None) or {}).get("kv_transfer_params")
    try:
        return max(int((kv or {}).get("tt.expect_hit") or 0), 0)
    except (TypeError, ValueError):
        return 0


class _HistoryCut:
    """a request seen with its history cut to its resume point n (all_token_ids[:n]); every other attribute is the
    request's own (num_tokens stays n + 1: LMCache's full-prompt-hit rule does not take the last token off n)"""

    __slots__ = ("_request", "all_token_ids")

    def __init__(self, request: Any, n: int):
        object.__setattr__(self, "_request", request)
        object.__setattr__(self, "all_token_ids", list(request.all_token_ids[:n]))

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_request"), name)


def _hit_tokens(request: Any) -> list:
    """the token history a hit may cover: a request resumed from preemption hits its generated tokens too (its own
    preempt save stored them); any other request, its prompt (unchanged)"""
    if _preempt_save_on() and int(getattr(request, "num_preemptions", 0) or 0) > 0:
        allt = getattr(request, "all_token_ids", None)
        if allt is not None:
            return list(allt)
    return request.prompt_token_ids or []  # the request's own list (no O(n) copy per lookup; the provenance keeps it)
_LEAVE_LOCK = threading.Lock()


def _leave_drain() -> bool:
    return os.environ.get("GLM53_KV_LEAVE_DRAIN", "0") == "1"


def _leave_steps() -> int:
    return max(1, int(os.environ.get("GLM53_KV_LEAVE_FLUSH_STEPS", "1") or 1))


def _leave_yield_steps() -> int:
    """GLM53_KV_LEAVE_YIELD_STEPS (default 8; program 2026-10-26T22:40:55Z (b): admit first, drain between steps):
    the M1 drain takes no held step in an engine step that ADMITS a new request (its first chunk runs without a gone
    request's store in front of it), at most this many such steps in a row (the delay-freed blocks are released)"""
    return max(0, int(os.environ.get("GLM53_KV_LEAVE_YIELD_STEPS", "8") or 0))


def _leave_grace_s() -> float:
    """GLM53_KV_LEAVE_GRACE_MS (default 50): an EMPTY engine step (nothing scheduled: the engine only steps to drain a
    delay-freed request) takes no held step until this long after the drain began -- a back-to-back request arrives
    a few ms after the previous one's last token, and the engine thread must be free to admit it, not inside a store"""
    return max(0.0, float(os.environ.get("GLM53_KV_LEAVE_GRACE_MS", "50") or 0)) / 1e3


def _store_budget() -> int | None:
    """GLM53_KV_STORE_PRIORITY (default 1, Stage 12: prefill priority on the engine thread): held LMCache page stores
    flush at most one held step per engine step (a held request leaving, or past GLM53_KV_STORE_HOLD_MAX, still flushes
    everything); 0 = every held store at once (before 2026-10-23)"""
    return 1 if os.environ.get("GLM53_KV_STORE_PRIORITY", "1") == "1" else None


def flush_deferred_saves(request: Any = None) -> None:
    """Run every host's deferred prefill save now (GLM53_KV_SAVE_DEFER): called by the next step's begin_step and,
    earlier, by the scheduler's next connector lookup -- both after the first token left and before any forward can
    overwrite the pages or the slot state the save reads. Only hosts holding a deferred save are registered.

    GLM53_KV_LOOKUP_FLUSH=prefix (default all; Stage 12 W4 F2b, 2026-10-24): a lookup for `request` flushes (and drains)
    only a host whose held / deferred / queued stores belong to a request sharing the new prompt's first LMCache chunk
    -- LMCache's chunk keys chain from chunk 0, so no other store can serve the hit; the others stay held (a prompt
    arriving while a 1M prompt prefills no longer pays up to GLM53_KV_STORE_HOLD_MAX steps of page reads)."""
    tokens = None
    if request is not None and (os.environ.get("GLM53_KV_LOOKUP_FLUSH", "all") == "prefix"
                                or os.environ.get("GLM53_KV_LOOKUP_DEFER", "0") == "1"):  # fmt: skip
        tokens = getattr(request, "prompt_token_ids", None)
    t0, flushed = time.perf_counter(), 0
    for host in list(_SAVE_HOSTS):
        if getattr(host, "checkpoints_off", False):
            continue  # Stage 12: no hit of this cache can resume (every hit aligns to 0): the lookup needs no flush
        if tokens is not None and not host.serves_prefix(tokens):
            host.lookup_flushes_skipped = getattr(host, "lookup_flushes_skipped", 0) + 1
            continue
        if tokens is not None and lookup_admit_first() and not host.writes_into_prefix(tokens):
            # GLM53_KV_LOOKUP_ADMIT_FIRST: its in-flight stores write past this prompt's prefix -- a drain here would
            # wait for them on the engine thread, the very wait the admitted lookup skipped
            host.lookup_flushes_skipped = getattr(host, "lookup_flushes_skipped", 0) + 1
            continue
        host.flush_deferred_save()
        flushed += 1
    ms = (time.perf_counter() - t0) * 1000.0
    if flushed and ms >= _phase_log_ms():  # (Stage 12: the lookup's flush + drain, on the engine thread, was not in any log)
        logger.info("TT KV connector lookup flush (%s): %.1f ms (%d host(s): held page stores + the store worker's drain)",
                    getattr(request, "request_id", None), ms, flushed)  # fmt: skip


_LOOKUP_DEFERRED: dict = {}  # request id -> monotonic time of its first deferred lookup (GLM53_KV_LOOKUP_DEFER)
_LOOKUP_RELEASED: dict = {}  # request id -> seconds it was deferred (logged with its lookup's result, then dropped)
_KDA_LOOKUP_DEFERRED: dict = {}  # request id -> monotonic time of its first GLM53_KV_LOOKUP_DEFER_KDA deferral


def lookup_admit_first() -> bool:
    """GLM53_KV_LOOKUP_ADMIT_FIRST=1 (default 0; program 2026-10-28T05:07:22Z (3), the k4 fix item): under
    GLM53_KV_LOOKUP_DEFER a lookup waits only for an in-flight store that writes INSIDE its prefix. A queued store job
    records, per request, its token history and the LMCache-chunk-aligned start of the span it writes; the lookup defers
    only when some job's first written chunk lies on the common prefix of its request and the new prompt (a job
    writing past that prefix -- another ladder rung's later chunks, a different prompt sharing chunk 0 -- cannot change
    this lookup's LMCache match, so waiting for it changed nothing but the TTFT: the k4d5 warm 16,100 repeat waited
    225 ms behind such stores). Jobs without a known history or span keep the old rule (defer). Bitwise: the admitted
    lookup sees the same chunks for its prefix as after the wait."""
    return os.environ.get("GLM53_KV_LOOKUP_ADMIT_FIRST", "0") == "1"


def lookup_defer(request: Any) -> bool:
    """GLM53_KV_LOOKUP_DEFER=1 (Stage 12, program 2026-10-26T07:23:04Z; default 0): a back-to-back repeat paid
    send->lookup 0.53 / 1.57 s at 16K / 32K because the lookup flushed AND DRAINED the previous request's whole page
    store on the engine thread. Instead, for a host whose deferred / held / queued stores share this prompt's first
    LMCache chunk: its deferred and held stores are handed to the store worker as before, but nothing is drained; when
    the worker still has such stores in flight after a bounded wait (GLM53_KV_LOOKUP_DEFER_WAIT_MS, default 2), True:
    the connector reports the lookup as undetermined, vLLM skips the request this step and retries it next step, and
    every other request keeps stepping. Past GLM53_KV_LOOKUP_DEFER_MAX_S (default 30) of deferral a request takes the
    old flush + drain (progress is guaranteed). Hosts without a store worker, or whose hits cannot resume
    (checkpoints_off), never defer."""
    tokens = getattr(request, "prompt_token_ids", None)
    rid = getattr(request, "request_id", None)
    if tokens is None:
        return False
    first = _LOOKUP_DEFERRED.get(rid)
    if first is not None and time.monotonic() - first > float(os.environ.get("GLM53_KV_LOOKUP_DEFER_MAX_S", "30")):
        _LOOKUP_DEFERRED.pop(rid, None)
        logger.warning("lookup of %s deferred past GLM53_KV_LOOKUP_DEFER_MAX_S: flushing and draining", rid)
        return False
    wait_s = float(os.environ.get("GLM53_KV_LOOKUP_DEFER_WAIT_MS", "2")) / 1e3
    busy = False
    admit_first = lookup_admit_first()
    for host in list(_SAVE_HOSTS):
        worker = getattr(host, "store_worker", None)
        if getattr(host, "checkpoints_off", False) or worker is None or not host.serves_prefix(tokens):
            continue  # (every worker job records its request heads under this knob: _save_pages)
        host.flush_deferred_save(drain=False)  # the deferred save's checkpoints + its held page stores to the worker
        relevant = host.writes_into_prefix if admit_first else host.serves_prefix
        if admit_first and host.serves_prefix(tokens) and not host.writes_into_prefix(tokens):
            host.lookups_admitted_first = getattr(host, "lookups_admitted_first", 0) + 1
            continue  # the in-flight stores write past this prompt's prefix: the lookup goes ahead now
        if relevant(tokens) and not worker.wait_idle(wait_s) and relevant(tokens):
            busy = True
            host.lookups_deferred = getattr(host, "lookups_deferred", 0) + 1
    if busy:
        if rid not in _LOOKUP_DEFERRED:
            _LOOKUP_DEFERRED[rid] = time.monotonic()
            logger.info("GLM53_KV_LOOKUP_DEFER: lookup of %s deferred (its prefix's page stores are in flight)", rid)
    else:
        first = _LOOKUP_DEFERRED.pop(rid, None)
        if first is not None:
            _LOOKUP_RELEASED[rid] = time.monotonic() - first
    return busy


_KDA_PENDING: dict = {}  # root -> {path: serialized bytes} queued for the background writer (every store of the root sees them)
_KDA_LOCK = threading.Lock()
_KDA_RAM: dict = {}  # root -> OrderedDict(path -> (states, nbytes)): the RAM tier of recent checkpoints (GLM53_KDA_CKPT_RAM_GB)
_KDA_SNAP: dict = {}  # root -> (path, producer): the latest save_deferred whose producer holds a device snapshot handle
_KDA_RECENT: dict = {}  # root -> OrderedDict(path -> None): the last GLM53_KDA_CKPT_RECENT_N saves (pinned in the RAM tier)


def _kda_snap_miss(host: Any, why: str, start: int) -> None:
    """a hit that does not take the KDA snapshot fast path (lever 1 (b)): counted on the host and logged with its reason
    (the k4b window's 32,000-token repeat imported through the host path for 873 ms and the tag's logs could not say
    why)"""
    n = getattr(host, "kda_snapshot_misses", 0) + 1
    setattr(host, "kda_snapshot_misses", n)
    logger.info("KDA snapshot import not taken for the hit at %d tokens: %s (miss %d; snapshot imports so far %d)",
                int(start), why, n, getattr(host, "kda_snapshot_imports", 0))  # fmt: skip


def resolver_sample_s() -> float:
    """GLM53_KDA_CKPT_RESOLVER_SAMPLE_S (default 0 = off; program 2026-10-28T08:07:52Z (2)): the producer callstack
    without ptrace. The k4d8 line's deferred KDA exports resolve 6-13x slower than k4d5p's (5.4 / 4.2 / 1.9 s vs
    0.41 / 0.70 / 0.28 s for the same 148 MB states), and py-spy cannot attach to a served engine on host-8chip
    (yama ptrace_scope 1). While a producer runs past GLM53_KDA_CKPT_RESOLVER_SAMPLE_AFTER_S (default 1 s), its OWN
    thread samples its stack every S seconds (_SelfSampler, the producer thread's profile hook, time-weighted); the
    producer's completion logs the frame histogram (its 5 most frequent stacks, innermost frame first).

    2026-10-28T09:49Z (the k4d8t4 L1-floor trip): the v1 sampler was a separate thread reading sys._current_frames()
    and holding that dict -- EVERY thread's current frame, the engine thread's included, with their locals -- across
    its sleep. An engine function that had returned kept its live ttnn L1 tensors until the next pass (alive during
    the next decode's shard build: +8,192 / +38,912 B per bank), then freed them on the sampler thread. The in-thread
    sampler never references another thread's frame."""
    try:
        return max(0.0, float(os.environ.get("GLM53_KDA_CKPT_RESOLVER_SAMPLE_S", "0") or 0))
    except ValueError:
        return 0.0


class _SelfSampler:
    """a profile-hook sampler of the CURRENT thread's own stack (the producer's): on an event at least S seconds after
    the last sample, the stack counts once per S elapsed (time-weighted: a long C call is not under-counted)"""

    def __init__(self, ent: list, s: float, after: float):
        self.ent, self.s = ent, s
        self.last = time.perf_counter() + after  # no sample before GLM53_KDA_CKPT_RESOLVER_SAMPLE_AFTER_S

    def __call__(self, frame: Any, event: str, arg: Any) -> None:
        now = time.perf_counter()
        if now - self.last < self.s:
            return
        n = int((now - self.last) / self.s)
        self.last += n * self.s
        import traceback  # noqa: PLC0415

        stack = traceback.extract_stack(frame)[-8:]
        inner = stack[-1] if stack else None
        kind = _sample_kind(getattr(inner, "line", "") or "", getattr(inner, "name", "") or "")
        key = kind + " @@ " + " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in reversed(stack))
        self.ent[2][key] = self.ent[2].get(key, 0) + n


class _CollectGate:
    """GLM53_KDA_CKPT_COLLECT_SERIAL=1 (default 0 since 11:22Z; program 2026-10-28T08:15:11Z (1)): ONE collect at a time. 16's
    (c): the resolver and the writer collected two deferred exports at once (both in collect_snapshot beside three
    lmc-fs-write threads), each slowing the other. A producer runs only while it holds the gate; a load() that needs a
    pending checkpoint (a hit) waits with priority, so it goes ahead of the resolvers' / writer's queued collects."""

    def __init__(self) -> None:
        self.cv, self.busy, self.prio = threading.Condition(), False, 0

    def acquire(self, prio: bool = False) -> None:
        with self.cv:
            if prio:
                self.prio += 1
            try:
                while self.busy or (not prio and self.prio > 0):
                    self.cv.wait()
                self.busy = True
            finally:
                if prio:
                    self.prio -= 1

    def release(self) -> None:
        with self.cv:
            self.busy = False
            self.cv.notify_all()


_COLLECT_GATE = _CollectGate()
_STATE_SAVES = {"n": 0, "cv": threading.Condition()}  # deferred KDA state saves whose DMA is still in flight


def _state_save_begin() -> None:
    with _STATE_SAVES["cv"]:
        _STATE_SAVES["n"] += 1


def _state_save_end() -> None:
    with _STATE_SAVES["cv"]:
        _STATE_SAVES["n"] = max(0, _STATE_SAVES["n"] - 1)
        _STATE_SAVES["cv"].notify_all()


def yield_to_state_saves(host: Any = None) -> float:
    """GLM53_KV_STORE_YIELD_TO_STATE_SAVE=1 (default 0; program 2026-10-28T08:20:18Z): the store worker starts no
    NEW page-store job (its page-DMA descriptors) while a deferred KDA state save is still in flight on the rings, up to
    GLM53_KV_STORE_YIELD_MAX_S (default 5) per job -- the state save's priority over the pooled page stores as far as
    the plugin owns the ordering: descriptors already on a ring, and the ring's own order, are the model tree's
    (page_pool.py). Returns the seconds waited (counted on the host: store_yields / store_yield_s).

    HARMFUL under GLM53_KV_LOOKUP_DEFER (k4d8t4wp8, 2026-10-28T11:22Z): a state save counts as in flight until its
    PRODUCER finished (the DMA wait + the untilize), so a prompt's page stores run only after the whole export collect
    (2.07 s at 16,100; 3.49 s at 32,000 behind a COLLECT_SERIAL queue), and the stored-hit repeat's lookup, deferred
    until those stores land, waits for all of it: 0.775 -> 2.678 s and 1.659 -> 5.227 s TTFT. Keep it off."""
    if os.environ.get("GLM53_KV_STORE_YIELD_TO_STATE_SAVE", "0") != "1":
        return 0.0
    lim = float(os.environ.get("GLM53_KV_STORE_YIELD_MAX_S", "5") or 5)
    t0 = time.perf_counter()
    with _STATE_SAVES["cv"]:
        while _STATE_SAVES["n"] > 0 and time.perf_counter() - t0 < lim:
            _STATE_SAVES["cv"].wait(timeout=max(0.001, lim - (time.perf_counter() - t0)))
    dt = time.perf_counter() - t0
    if dt > 0.001 and host is not None:
        host.store_yields = getattr(host, "store_yields", 0) + 1
        host.store_yield_s = getattr(host, "store_yield_s", 0.0) + dt
    return dt


def _collect_serial() -> bool:
    """GLM53_KDA_CKPT_COLLECT_SERIAL (default 0 since 2026-10-28T11:22Z: on k4d8t4wp8 it queued a 32K prompt's second
    export collect 1.701 s behind the first, and the stored-hit repeat waited for the whole chain)"""
    return os.environ.get("GLM53_KDA_CKPT_COLLECT_SERIAL", "0") == "1"


_COLLECT_STATS = {"kda_collects": 0, "kda_collect_s": 0.0, "kda_collect_gate_waits": 0, "kda_collect_gate_wait_s": 0.0}


# -- the engine-thread rule (program 2026-10-28T09:22:52Z FIX ORDER, the k4d8t4 L1-floor trip) ---------------------
# No plugin thread but the engine thread issues ttnn / device work. The KDA checkpoint threads (resolver, writer) run a
# deferred producer under _host_only: a ttnn call there is refused BEFORE it runs (OffEngineDeviceCall), its call site
# logged once with the stack, and the producer handed back to the engine thread (drain_engine_todo at the next
# _save_state / save_deferred; a load() on the engine thread runs it inline). GLM53_KDA_CKPT_HOST_ONLY=0 turns the
# refusal off; GLM53_KV_DEVICE_THREAD_AUDIT=1 logs (never refuses) every ttnn call site of the store worker's jobs.
_ENGINE = {"ident": None}  # the thread that runs the connector's steps (_save_state)
_ENGINE_TODO: list = []  # _Deferred producers a plugin thread handed back to the engine thread
_ENGINE_STATS = {"refused": 0, "handbacks": 0, "drained": 0, "audited": 0}
_OFF_ENGINE_SEEN: set = set()  # (what, ttnn function) call sites logged
_PLUGIN_THREADS = ("tt-kda-", "tt-kv-store-worker")


class OffEngineDeviceCall(RuntimeError):
    """a ttnn call on a plugin thread other than the engine thread, refused before it ran"""


class _EngineOnly(RuntimeError):
    """a deferred producer that needs the device: handed back to the engine thread"""


def host_only_guard() -> bool:
    return os.environ.get("GLM53_KDA_CKPT_HOST_ONLY", "1") == "1"


def mark_engine_thread() -> None:
    _ENGINE["ident"] = threading.get_ident()


def off_engine_thread() -> bool:
    """not the engine thread; before the first step: one of the plugin's own threads"""
    e = _ENGINE["ident"]
    if e is not None:
        return threading.get_ident() != e
    return threading.current_thread().name.startswith(_PLUGIN_THREADS)


def _ttnn_callee(frame: Any, event: str, arg: Any) -> str | None:
    """the ttnn function a profile event enters (a Python function of a ttnn module, or a ttnn builtin / method)"""
    if event == "call":
        mod = frame.f_globals.get("__name__", "") or ""
        name = frame.f_code.co_name
    elif event == "c_call":
        mod = getattr(arg, "__module__", None) or type(getattr(arg, "__self__", None)).__module__ or ""
        name = getattr(arg, "__qualname__", None) or getattr(arg, "__name__", "?")
    else:
        return None
    return f"{mod}.{name}" if mod == "ttnn" or mod.startswith("ttnn.") else None


class _host_only:
    """`with _host_only(what):` -- every ttnn call the CURRENT thread makes inside the block is refused before it runs
    (refuse=True: OffEngineDeviceCall) or only logged (refuse=False, the audit); each call site logged once.
    sampler: a _SelfSampler run on every event first (the same hook); check=False: the sampler only"""

    def __init__(self, what: str, refuse: bool = True, sampler: Any = None, check: bool = True):
        self.what, self.refuse, self.prev, self.sampler, self.check = what, refuse, None, sampler, check

    def _profile(self, frame: Any, event: str, arg: Any) -> None:
        if self.sampler is not None:
            self.sampler(frame, event, arg)
        if not self.check:
            return
        callee = _ttnn_callee(frame, event, arg) if event in ("call", "c_call") else None
        if callee is None:
            return
        th = threading.current_thread().name
        key = (self.what.split(" ")[0], callee)
        with _KDA_LOCK:
            _ENGINE_STATS["refused" if self.refuse else "audited"] += 1
            first = key not in _OFF_ENGINE_SEEN
            _OFF_ENGINE_SEEN.add(key)
        if first:
            import traceback  # noqa: PLC0415

            site = " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}"
                               for f in reversed(traceback.extract_stack(frame, limit=10)))  # fmt: skip
            logger.warning("engine-thread rule: %s %s on thread %s (%s): %s", "refused" if self.refuse else "audit:",
                           callee, th, self.what, site)  # fmt: skip
        if self.refuse:
            raise OffEngineDeviceCall(f"{callee} on thread {th} ({self.what})")

    def __enter__(self) -> "_host_only":
        self.prev = sys.getprofile()
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *exc: Any) -> bool:
        sys.setprofile(self.prev)
        return False


def _hand_to_engine(d: "_Deferred") -> None:
    with _KDA_LOCK:
        if not any(x is d for x in _ENGINE_TODO):
            _ENGINE_TODO.append(d)
        _ENGINE_STATS["handbacks"] += 1


def drain_engine_todo() -> int:
    """on the engine thread: run the producers a plugin thread handed back; their states go to the RAM tier as the
    resolver would put them (the writer, waiting on the _Deferred, then writes the file)"""
    if not _ENGINE_TODO or off_engine_thread():
        return 0
    with _KDA_LOCK:
        items = list(_ENGINE_TODO)
        _ENGINE_TODO.clear()
    for d in items:
        try:
            states = d.result()
        except BaseException:  # noqa: BLE001  (kept in the _Deferred: the writer surfaces it)
            continue
        st, path = d.store, d.path
        if st is not None and path is not None:
            with _KDA_LOCK:
                pending = path in st._pending
            if pending:
                st._ram_put(path, states)
    with _KDA_LOCK:
        _ENGINE_STATS["drained"] += len(items)
    return len(items)


def _put_draining(q: "queue.Queue", item: Any) -> None:
    """q.put on the engine thread that keeps draining handed-back producers while the queue is full (the writer may be
    waiting for one of them: a plain blocking put would wait for the writer forever)"""
    while True:
        try:
            q.put(item, timeout=0.05)
            return
        except queue.Full:
            drain_engine_todo()


def _join_draining(q: "queue.Queue") -> None:
    """q.join() that keeps draining handed-back producers (on the engine thread) while it waits"""
    while True:
        with q.all_tasks_done:
            if not q.unfinished_tasks:
                return
            q.all_tasks_done.wait(0.05)
        drain_engine_todo()


def _sample_kind(frame_line: str, name: str) -> str:
    """the sampler's split of a producer's time: the DMA wait (StateTicket.fetch's poll sleep) vs host work"""
    t = (frame_line or "") + " " + (name or "")
    if "sleep" in t:
        return "dma_wait"
    if "untilize" in t:
        return "untilize"
    if ".copy()" in t:
        return "host_copy"
    if "stack(" in t:
        return "host_stack"
    return "other"


class _Deferred:
    """a checkpoint whose states are produced later (export_kda_state_async: a device snapshot the page-DMA saves while
    the next steps run): result() runs the producer once (the writer thread, or a load() that needs it first)"""

    def __init__(self, fn: Any, label: str = "", state_save: bool = False):
        self.fn, self.lock, self.value, self.error, self.done = fn, threading.Lock(), None, None, False
        self.label, self.state_save = label, state_save
        self.event = threading.Event()  # set once produced (the writer waits on it for a handed-back producer)
        self.store, self.path = None, None  # the KDACheckpointStore entry (drain_engine_todo's RAM tier put)
        if state_save:
            _state_save_begin()  # the export's page-DMA save is on the rings from now until the producer collected it

    def result(self, prio: bool = False) -> Any:
        handed = None
        with self.lock:
            if not self.done:
                guard = host_only_guard() and off_engine_thread()  # the engine-thread rule: host work only here
                gate = _collect_serial()
                if gate:
                    t_g = time.perf_counter()
                    _COLLECT_GATE.acquire(prio)
                    waited = time.perf_counter() - t_g
                    with _KDA_LOCK:
                        _COLLECT_STATS["kda_collect_gate_waits"] += int(waited > 0.001)
                        _COLLECT_STATS["kda_collect_gate_wait_s"] += waited
                    if waited > 0.05:
                        logger.info("KDA checkpoint %s: its collect waited %.3f s for another collect (%s)",
                                    self.label or "?", waited, "a hit: priority" if prio else "queued")  # fmt: skip
                t_c = time.perf_counter()
                s_s = resolver_sample_s()
                # GLM53_KDA_CKPT_RESOLVER_SAMPLE_S: sampled IN this thread (never another thread's frames)
                ent = [self.label, time.perf_counter(), {}] if s_s > 0 else None
                sampler = None if ent is None else _SelfSampler(
                    ent, s_s, float(os.environ.get("GLM53_KDA_CKPT_RESOLVER_SAMPLE_AFTER_S", "1") or 1))  # fmt: skip
                try:
                    if guard or sampler is not None:
                        with _host_only(f"KDA producer {self.label or '?'}", sampler=sampler, check=guard):
                            self.value = self.fn()
                    else:
                        self.value = self.fn()
                except OffEngineDeviceCall as e:  # refused before it ran: the engine thread runs the producer
                    handed = e
                except BaseException as e:  # noqa: BLE001  (re-raised to every caller)
                    self.error = e
                finally:
                    if gate:
                        _COLLECT_GATE.release()
                    if self.state_save and handed is None:
                        _state_save_end()
                    if handed is None:
                        with _KDA_LOCK:
                            _COLLECT_STATS["kda_collects"] += 1
                            _COLLECT_STATS["kda_collect_s"] += time.perf_counter() - t_c
                    if ent is not None and ent[2]:
                        top = sorted(ent[2].items(), key=lambda kv: -kv[1])[:5]
                        kinds: dict = {}
                        for k, n in ent[2].items():
                            kk = k.split(" @@ ", 1)[0]
                            kinds[kk] = kinds.get(kk, 0) + n
                        tot = sum(kinds.values())
                        split = ", ".join(f"{k} {100.0 * v / tot:.0f} pct" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
                        logger.info("KDA checkpoint %s producer: %.3f s on %s, %d stack samples every %.2f s (%s); "
                                    "top: %s", ent[0] or "?", time.perf_counter() - ent[1],
                                    threading.current_thread().name, tot, resolver_sample_s(), split,
                                    " | ".join(f"[{n}] {k.split(' @@ ', 1)[1]}" for k, n in top))  # fmt: skip
                if handed is None:
                    self.done, self.fn = True, None
                    self.event.set()
                else:
                    _hand_to_engine(self)
        if handed is not None:
            raise _EngineOnly(f"KDA checkpoint {self.label or '?'}: {handed} -> handed to the engine thread")
        if self.error is not None:
            raise self.error
        return self.value


def _resolve(states: Any, prio: bool = False) -> Any:
    return states.result(prio) if isinstance(states, _Deferred) else states


def _clone_states(obj: Any) -> Any:
    """a point-in-time copy of a checkpoint's states (tensors cloned; containers rebuilt; anything else shared)"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: _clone_states(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_clone_states(v) for v in obj)
    return obj


def _states_nbytes(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_states_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_states_nbytes(v) for v in obj)
    return 0


# GLM53_KDA_CKPT_FORMAT (Stage 12 warm-context lever 1, 2026-10-25): "raw" (default) writes a KDA checkpoint as a JSON
# skeleton + the tensors' raw bytes with file.write(memoryview) and reads it back with ONE readinto -- both syscalls drop
# the GIL. torch.save / torch.load pickle under the GIL: the writer thread's torch.save of a ~148 MB checkpoint right
# after every storing request slowed the engine thread's next hit load 3.2x (87 -> 280 ms of host work for a 32K hit,
# doc/ttnn/12-fast-start/warmctx_lever1). "torch" = the old writer. The loader reads both (the magic decides), so the
# files keep their names and old .pt files still load.
_RAW_MAGIC = b"GLM53-KDA-RAW-1\n"  # 16 bytes
_RAW_ALIGN = 64
_RAW_FALLBACK_WARNED = False


def _raw_align(n: int) -> int:
    return -(-int(n) // _RAW_ALIGN) * _RAW_ALIGN


def _raw_skeleton(obj: Any, arrays: list) -> Any:
    """a JSON-able skeleton of obj; every tensor / ndarray replaced by its index in `arrays` (TypeError: not raw-able)"""
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().contiguous()
        arrays.append(t)
        return {"__t__": len(arrays) - 1, "dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape)}
    if isinstance(obj, np.ndarray):
        if obj.dtype.hasobject:
            raise TypeError("object ndarray")
        a = np.ascontiguousarray(obj)
        arrays.append(a)
        return {"__n__": len(arrays) - 1, "dtype": a.dtype.str, "shape": list(a.shape)}
    if isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            if not (k is None or isinstance(k, (str, int, float, bool))):
                raise TypeError(f"dict key {type(k).__name__}")
            items.append([k, _raw_skeleton(v, arrays)])
        return {"__d__": items}
    if isinstance(obj, (list, tuple)):
        return {"__l__" if isinstance(obj, list) else "__u__": [_raw_skeleton(v, arrays) for v in obj]}
    if isinstance(obj, np.generic):
        obj = obj.item()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return {"__v__": obj}
    raise TypeError(f"not raw-able: {type(obj).__name__}")


def _raw_bytes(a: Any) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.reshape(-1).view(torch.uint8).numpy() if a.numel() else np.zeros(0, np.uint8)
    return a.reshape(-1).view(np.uint8)


def _raw_dump(blob: Any, path: str) -> None:
    """blob -> path in the raw format (TypeError, nothing written, when blob holds a type the skeleton cannot carry)"""
    import json  # noqa: PLC0415
    import struct  # noqa: PLC0415

    arrays: list = []
    skel = _raw_skeleton(blob, arrays)
    offs, nbytes, off = [], [], 0
    for a in arrays:
        off = _raw_align(off)
        offs.append(off)
        nbytes.append(int(_raw_bytes(a).nbytes))
        off += nbytes[-1]
    header = json.dumps({"skeleton": skel, "offsets": offs, "nbytes": nbytes, "total": off}).encode()
    data0 = _raw_align(len(_RAW_MAGIC) + 8 + len(header))
    with open(path, "wb") as f:
        f.write(_RAW_MAGIC + struct.pack("<Q", len(header)) + header + b"\0" * (data0 - len(_RAW_MAGIC) - 8 - len(header)))
        pos = 0
        for a, o in zip(arrays, offs):
            if o > pos:
                f.write(b"\0" * (o - pos))
            b = _raw_bytes(a)
            f.write(memoryview(b))  # a large write goes straight to the syscall (no GIL held during it)
            pos = o + b.nbytes


def _raw_load(path: str) -> Any:
    import json  # noqa: PLC0415
    import struct  # noqa: PLC0415

    with open(path, "rb") as f:
        if f.read(len(_RAW_MAGIC)) != _RAW_MAGIC:
            raise ValueError(f"{path} is not a raw KDA checkpoint")
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        f.seek(_raw_align(len(_RAW_MAGIC) + 8 + hlen))
        buf = np.empty(int(header["total"]), np.uint8)
        view, got = memoryview(buf), 0
        while got < buf.nbytes:  # one readinto (the syscall drops the GIL); loop only on a short read
            k = f.readinto(view[got:])
            if not k:
                raise ValueError(f"{path}: truncated at {got} of {buf.nbytes} bytes")
            got += k
    arrays = list(zip(header["offsets"], header["nbytes"]))

    def build(sk: Any) -> Any:
        if "__t__" in sk:
            o, n = arrays[sk["__t__"]]
            dt = getattr(torch, sk["dtype"])
            t = torch.from_numpy(buf[o : o + n]) if n else torch.zeros(0, dtype=torch.uint8)
            return t.view(dt).reshape(sk["shape"])
        if "__n__" in sk:
            o, n = arrays[sk["__n__"]]
            return buf[o : o + n].view(np.dtype(sk["dtype"])).reshape(sk["shape"])
        if "__d__" in sk:
            return {k: build(v) for k, v in sk["__d__"]}
        if "__l__" in sk:
            return [build(v) for v in sk["__l__"]]
        if "__u__" in sk:
            return tuple(build(v) for v in sk["__u__"])
        return sk["__v__"]

    return build(header["skeleton"])


def _blob_dump(blob: Any, path: str) -> str:
    """a checkpoint blob -> path: the raw format (GLM53_KDA_CKPT_FORMAT=raw, default) or torch.save; returns the format"""
    if os.environ.get("GLM53_KDA_CKPT_FORMAT", "raw") == "raw":
        try:
            _raw_dump(blob, path)
            return "raw"
        except TypeError as e:  # a blob the skeleton cannot carry: torch.save below (the loader reads both)
            global _RAW_FALLBACK_WARNED
            if not _RAW_FALLBACK_WARNED:
                _RAW_FALLBACK_WARNED = True
                logger.warning("KDA checkpoint: raw format cannot carry this blob (%s): torch.save (holds the GIL)", e)
    torch.save(blob, path)
    return "torch"


def _blob_load(path: str) -> Any:
    """a checkpoint file of either format (the magic decides)"""
    with open(path, "rb") as f:
        raw = f.read(len(_RAW_MAGIC)) == _RAW_MAGIC
    return _raw_load(path) if raw else torch.load(path, map_location="cpu", weights_only=False)


def _phase_log_ms() -> float:
    """GLM53_KV_PHASE_LOG_MS (default 500; 0 = off): log a connector step's per-phase host time at or above this"""
    return float(os.environ.get("GLM53_KV_PHASE_LOG_MS", "500") or 0)


def tail_split_min() -> int:
    """GLM53_PF_TAIL_SPLIT_MIN = N (default 8,192; 0 = off; program 2026-10-28T02:28:53Z (2)): prompts shorter than N
    get no KDA-checkpoint tail split in the scheduler (cold or storing: one plan), and the worker checkpoints such a
    prompt at its last chunk end instead of the last grid boundary"""
    return max(0, int(os.environ.get("GLM53_PF_TAIL_SPLIT_MIN", "8192") or 0))


def kda_ckpt_in_replay() -> bool:
    """GLM53_KDA_CKPT_IN_REPLAY = 1 (the one-replay store, option A: program 2026-10-24T06:41:05Z;
    doc/ttnn/09-vllm/perf/one_replay_store/design.md section 6): the model snapshots a prompt's KDA state at its cut
    floor((len - 1) / grid) x grid INSIDE the bucket replay (``kda_snapshot_at`` per prefill row) and exports it from
    its staging buffers (``export_kda_state(..., n_tokens=cut, source="snapshot")``). The scheduler then never splits a
    storing prompt, and GLM53_KDA_CKPT_AT_END does not apply. Needs a model with ``kda_checkpoint_in_replay = True``
    (refused at init otherwise). Default 0."""
    return os.environ.get("GLM53_KDA_CKPT_IN_REPLAY", "0") == "1"


def kda_ckpt_at_end_mode() -> str:
    """GLM53_KDA_CKPT_AT_END (cf3h-d, program 2026-10-24T06:38:54Z option C): where a STORING prompt's KDA
    checkpoint is taken.
    - "0": the last checkpoint-grid boundary STRICTLY before the prompt's end (the scheduler splits the prompt's last
      chunk there: 1,920 + 128 at 2,048 tokens), so an exact repeat of the prompt can resume.
    - "aligned" (cf3h-d's default): a prompt whose length is a multiple of the grid (128) runs its last chunk whole and is
      checkpointed at its END: a continuation (the next turn) resumes there, an exact repeat prefills from 0. Any
      other prompt keeps the "0" behaviour (a checkpoint at an unaligned end could never be resumed: hits start on a
      page).
    - "all": no split for any prompt; checkpointed at the end when it is on the grid, otherwise not at all (an
      unaligned prompt's continuation prefills from 0 too).
    - "both" (default since cf3h-e, program 2026-10-24T20:38:48Z (i)): every prompt keeps the "0" split, so an exact
      repeat resumes at the last grid boundary before the end (cf3h-d's "aligned" sent a 32,000-token repeat back to
      7,936: 10.7 s vs 1.2 s), AND a grid-aligned prompt is also checkpointed at its END (a continuation resumes there).
      Cost: an aligned prompt runs its last chunk in two replays again (1,920 + 128 at 2,048 tokens), as under "0"."""
    v = os.environ.get("GLM53_KDA_CKPT_AT_END", "both").strip().lower()
    if v not in ("0", "aligned", "all", "both"):
        raise ValueError(f"GLM53_KDA_CKPT_AT_END={v!r}: expected 0, aligned, all or both")
    return v


def lmcache_min_store_tokens() -> int:
    """GLM53_LMCACHE_MIN_STORE_TOKENS = N > 0 (served: 1,024): a prompt shorter than N tokens is neither stored in
    LMCache (the scheduler marks it lmcache.skip_save) nor KDA-checkpointed at its prefill. A short prefix gains nothing
    from a hit, and its store + checkpoint cost ~0.6 s of TTFT (a 128-token prompt: 0.91 s with them, 0.24 s without).
    Lookups are unchanged: a short prompt still hits a longer stored prefix. 0 / unset = every
    prompt (v1)."""
    return max(int(os.environ.get("GLM53_LMCACHE_MIN_STORE_TOKENS", "0") or 0), 0)


def decode_boundary_wanted(boundary: int, every: int | None = None) -> bool:
    """GLM53_KDA_CKPT_EVERY = N > 0 also bounds the DECODE-time checkpoints: a page boundary a decode step crosses is
    checkpointed only at a multiple of N (unbounded, every 64 generated tokens a synchronous 148 MB export stalls the
    decode by ~0.3-0.4 s: ITL P99 ~400 ms in the single-user benchmark). A later hit resumes from the
    prompt's checkpoint and recomputes the response tokens instead. 0 = every boundary (v1)."""
    n = int(os.environ.get("GLM53_KDA_CKPT_EVERY", "0")) if every is None else int(every)
    return n <= 0 or int(boundary) % n == 0


class KDACheckpointStore:
    """Per-prefix checkpoints of the KDA recurrent state (page_contract.md §5).

    The KDA layers keep one recurrent state per slot, not per page, so a prefix
    hit of ``N`` tokens is only correct where the state after exactly ``N``
    tokens was saved. Policy (v1): every prefill chunk that ends on a page
    boundary saves its slot's state (``model.export_kda_state``) keyed by the
    sha256 of the ``N`` token ids; a hit imports the checkpoint of its prefix
    (``model.import_kda_state``) before the continuation runs; the
    scheduler-side ``TTPageAlignedConnector`` truncates a hit to the longest
    checkpointed page boundary. Files live under
    ``kv_connector_extra_config["tt_kda_checkpoint_dir"]``: ``{"num_tokens": N, "states": [{"L", "S", "C"}, ...]}``
    in the raw format (GLM53_KDA_CKPT_FORMAT=raw, default: a JSON skeleton + the tensors' bytes, GIL-free I/O) or
    torch.save (=torch); the loader reads both.
    """

    EXTRA_CONFIG_KEY = "tt_kda_checkpoint_dir"
    # Disk bounds (2026-10-23: ~148 MB per checkpoint, 7.6 GB in 9 min of serving; a full root disk SIGBUS-killed the
    # next launch): the store keeps at most GLM53_KDA_CKPT_MAX_GB (LRU by last use; a checkpoint used in the last
    # EVICT_MIN_AGE_S is never evicted -- the scheduler counted a hit on it with has() and the worker loads it a step
    # later) and saves nothing while the disk has less than GLM53_KDA_CKPT_MIN_FREE_GB free (no checkpoint = a shorter
    # hit, never a wrong one).
    MAX_GB_ENV, MIN_FREE_GB_ENV = "GLM53_KDA_CKPT_MAX_GB", "GLM53_KDA_CKPT_MIN_FREE_GB"
    EVICT_MIN_AGE_S = 120.0

    def __init__(self, root: str, max_bytes: int | None = None, min_free_bytes: int | None = None):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        gb = 1 << 30
        self.max_bytes = int(max_bytes if max_bytes is not None else float(os.environ.get(self.MAX_GB_ENV, "16")) * gb)
        self.min_free_bytes = int(
            min_free_bytes if min_free_bytes is not None else float(os.environ.get(self.MIN_FREE_GB_ENV, "30")) * gb
        )
        self.saves_skipped_low_disk = 0
        self.evicted = 0
        # GLM53_KDA_CKPT_ASYNC (default 1; a synchronous 148 MB torch.save stalls the engine up to ~3.8 s per chunk):
        # save() serializes to memory in the step (torch.save's point-in-time snapshot, ~a
        # memcpy) and one background writer puts the bytes on disk and evicts; a full queue (GLM53_KDA_CKPT_QUEUE,
        # default 8) blocks the step = back-pressure. has() / load() see queued entries (one registry per root).
        self.async_writes = os.environ.get("GLM53_KDA_CKPT_ASYNC", "1") == "1"
        self.stats = dict(saved=0, clone_s=0.0, queue_wait_s=0.0, write_s=0.0, ram_hits=0, disk_loads=0, disk_load_s=0.0)
        # GLM53_KDA_CKPT_RAM_GB (default 8; 0 = off; Stage 12): the most recent checkpoints stay in host RAM as tensors
        # (LRU by bytes, shared by the stores of the process): a hit on one skips torch.load (gx2c6 files: 0.13 s warm,
        # 0.41 s cold per 148 MB checkpoint, as much as the KDA pack import itself). The cached states are handed to
        # import_kda_state as they are: importers read them, never write them.
        self.ram_bytes = int(float(os.environ.get("GLM53_KDA_CKPT_RAM_GB", "8")) * gb)
        with _KDA_LOCK:
            self._pending = _KDA_PENDING.setdefault(self.root, {})
            self._ram = _KDA_RAM.setdefault(self.root, __import__("collections").OrderedDict())
        self._queue = None
        self._writer = None
        self._error: BaseException | None = None
        # GLM53_KDA_CKPT_RESOLVE_AHEAD (default 1; 0 = resolve in the writer as before): a deferred checkpoint's producer
        # (export_kda_state_async: the page-DMA state save + untilize, ~0.12 s of host work for 148 MB) runs on its OWN
        # thread as soon as it is queued and the states go to the RAM tier at once, so a hit on a checkpoint that is not
        # the model's latest snapshot (the snapshot fast path covers only that one) never waits for the writer's queue.
        # The k4a / k4b 32,000-token repeats: the aligned prompt exports 31,872 (the 0 split) AND 32,000 (its end, under
        # GLM53_KDA_CKPT_AT_END=both); the end export makes the 31,872 snapshot stale, and the repeat resolved 31,872
        # inline (612 / 873 ms). The writer then only writes files
        self.resolve_ahead = os.environ.get("GLM53_KDA_CKPT_RESOLVE_AHEAD", "1") == "1"
        self._rqueue = None
        self._resolver = None
        # GLM53_KDA_CKPT_LOCAL_DIR (default unset = off; program 2026-10-28T05:07:22Z (1)): checkpoints LOCAL-FIRST.
        # The k4d5 warm 16,100 repeat read its 16,000 checkpoint from the share (kda_ckpt/stable on NFS): 372 ms of
        # kda_store_load. A checkpoint is written to <LOCAL_DIR>/<the root's hash>/ (local disk), then mirrored to the
        # root (the share) by a background thread: the share copy stays the restart truth. has() / load() prefer the
        # local copy and fall back to the share (a restarted server, a checkpoint another process wrote). The local
        # tier has its own bounds (GLM53_KDA_CKPT_LOCAL_MAX_GB, default MAX_GB; GLM53_KDA_CKPT_LOCAL_MIN_FREE_GB,
        # default MIN_FREE_GB). Same bytes in either place: bitwise.
        ldir = os.environ.get("GLM53_KDA_CKPT_LOCAL_DIR", "").strip()
        self.local_root = None
        if ldir:
            self.local_root = os.path.join(os.path.abspath(ldir), hashlib.sha1(self.root.encode()).hexdigest()[:12])
            os.makedirs(self.local_root, exist_ok=True)
        self.local_max_bytes = int(float(os.environ.get("GLM53_KDA_CKPT_LOCAL_MAX_GB", self.max_bytes / gb)) * gb)
        self.local_min_free_bytes = int(float(os.environ.get("GLM53_KDA_CKPT_LOCAL_MIN_FREE_GB",
                                                             self.min_free_bytes / gb)) * gb)  # fmt: skip
        self._mqueue = None
        self._mirror = None
        # GLM53_KDA_CKPT_RECENT_N (default 0 = off; program 2026-10-28T05:07:22Z (2)): the last N checkpoints SAVED
        # (by path = by prefix hash) keep their states in the RAM tier whatever its byte LRU does (even with
        # GLM53_KDA_CKPT_RAM_GB=0): a hit on an earlier prefix than the latest export takes the in-memory states, not a
        # store read. N x ~148 MB of host memory. The device snapshot fast path (the latest export only) is unchanged.
        self.recent_n = max(0, int(os.environ.get("GLM53_KDA_CKPT_RECENT_N", "0") or 0))
        with _KDA_LOCK:
            self._recent = _KDA_RECENT.setdefault(self.root, __import__("collections").OrderedDict())

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> KDACheckpointStore | None:
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        if kv_transfer_config is None:
            return None
        root = kv_transfer_config.get_from_extra_config(cls.EXTRA_CONFIG_KEY, None)
        return cls(root) if root else None

    @staticmethod
    def key(token_ids: Sequence[int], num_tokens: int) -> str:
        ids = np.asarray(list(token_ids)[:num_tokens], dtype=np.int32)
        if ids.shape[0] != num_tokens:
            raise ValueError(f"{ids.shape[0]} token ids for a {num_tokens}-token key")
        h = hashlib.sha256(ids.tobytes())
        h.update(int(num_tokens).to_bytes(8, "little"))
        return h.hexdigest()

    def path(self, token_ids: Sequence[int], num_tokens: int) -> str:
        return os.path.join(self.root, f"{self.key(token_ids, num_tokens)}.pt")

    def lpath(self, path: str) -> str | None:
        """the local-first copy of a checkpoint path (GLM53_KDA_CKPT_LOCAL_DIR), else None"""
        return os.path.join(self.local_root, os.path.basename(path)) if self.local_root else None

    def _mark_recent(self, path: str) -> None:
        if self.recent_n <= 0:
            return
        with _KDA_LOCK:
            self._recent[path] = None
            self._recent.move_to_end(path)
            while len(self._recent) > self.recent_n:
                self._recent.popitem(last=False)

    @staticmethod
    def _touch(path: str) -> None:
        try:
            os.utime(path)  # the LRU clock: last use
        except OSError:
            pass

    def has(self, token_ids: Sequence[int], num_tokens: int) -> bool:
        path = self.path(token_ids, num_tokens)
        with _KDA_LOCK:
            if path in self._pending:
                return True  # queued: on disk (and touched) when the writer gets to it
            if path in self._ram:
                self._ram.move_to_end(path)
                if os.path.exists(path):
                    self._touch(path)
                return True
        lp = self.lpath(path)
        if lp is not None and os.path.exists(lp):
            self._touch(lp)
            return True
        if not os.path.exists(path):
            return False
        self._touch(path)
        return True

    def _evict(self, keep: str, root: str | None = None, max_bytes: int | None = None) -> None:
        """oldest-used checkpoints out until the store fits max_bytes (never `keep`, never one used recently)"""
        root = self.root if root is None else root
        max_bytes = self.max_bytes if max_bytes is None else max_bytes
        entries = []
        for name in os.listdir(root):
            if not name.endswith(".pt"):
                continue
            full = os.path.join(root, name)
            try:
                st = os.stat(full)
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, full))
        total = sum(e[1] for e in entries)
        now = time.time()
        for mtime, size, full in sorted(entries):
            if total <= max_bytes:
                break
            if full == keep or now - mtime < self.EVICT_MIN_AGE_S:
                continue
            try:
                os.remove(full)
            except OSError:
                continue
            total -= size
            self.evicted += 1

    def can_save(self) -> bool:
        """False (counted, logged once) while the disk is below the free-space floor: callers skip the state export
        (local-first: the local disk's floor; the mirror skips a share below its floor)"""
        root, floor = (self.local_root, self.local_min_free_bytes) if self.local_root else (self.root, self.min_free_bytes)
        st = os.statvfs(root)
        if st.f_bavail * st.f_frsize >= floor:
            return True
        self.saves_skipped_low_disk += 1
        if self.saves_skipped_low_disk == 1:
            logger.warning(
                "KDA checkpoint store %s: under %.0f GB free -- checkpoints are skipped (shorter hits)",
                self.root, self.min_free_bytes / (1 << 30),
            )
        return False

    def save(self, token_ids: Sequence[int], num_tokens: int, states: Any) -> str | None:
        """the checkpoint's path, or None when the disk is below the free-space floor (nothing written)"""
        path = self.path(token_ids, num_tokens)
        if self._error is not None:
            e, self._error = self._error, None
            raise RuntimeError(f"KDA checkpoint writer failed: {e}") from e
        if not self.can_save():
            return None
        t0 = time.perf_counter()
        snap = _clone_states(states)  # the point-in-time snapshot, taken in the step (a memcpy)
        self.stats["clone_s"] += time.perf_counter() - t0
        self._mark_recent(path)
        self._ram_put(path, snap)
        if not self.async_writes:
            self._write_file(path, None, {"num_tokens": int(num_tokens), "states": snap})
            return path
        self._start_writer()
        with _KDA_LOCK:
            self._pending[path] = (int(num_tokens), snap)
        t0 = time.perf_counter()
        _put_draining(self._queue, path)  # (the engine-thread rule: drains handed-back producers while full)
        self.stats["queue_wait_s"] += time.perf_counter() - t0
        return path

    def save_deferred(self, token_ids: Sequence[int], num_tokens: int, producer: Any) -> str | None:
        """save() of a checkpoint produced later (Stage 12: export_kda_state_async's callable, the device snapshot the
        page-DMA saves while the next steps run): has() sees it at once; the writer (or a load() that comes first)
        runs the producer, then the RAM tier and the file get it like any checkpoint. Synchronous writes
        (GLM53_KDA_CKPT_ASYNC=0) run the producer here."""
        path = self.path(token_ids, num_tokens)
        if self._error is not None:
            e, self._error = self._error, None
            raise RuntimeError(f"KDA checkpoint writer failed: {e}") from e
        drain_engine_todo()  # (the engine thread: producers a plugin thread handed back run here)
        d = _Deferred(producer, os.path.basename(path), state_save=True)
        d.store, d.path = self, path
        self.stats["deferred"] = self.stats.get("deferred", 0) + 1
        self._mark_recent(path)
        with _KDA_LOCK:  # lever 1 (b): the device snapshot buffers now hold this checkpoint (the previous one is gone)
            if getattr(producer, "kda_snapshot", None) is not None:
                _KDA_SNAP[self.root] = (path, producer)
            else:
                _KDA_SNAP.pop(self.root, None)
        if not self.async_writes:
            states = d.result()
            self._ram_put(path, states)
            self._write_file(path, None, {"num_tokens": int(num_tokens), "states": states})
            return path
        self._start_writer()
        with _KDA_LOCK:
            self._pending[path] = (int(num_tokens), d)
        if self._rqueue is not None:  # resolve ahead: before the writer queue's back-pressure may block this step
            self._rqueue.put((path, d, time.perf_counter()))
        t0 = time.perf_counter()
        _put_draining(self._queue, path)  # (the engine-thread rule: drains handed-back producers while full)
        self.stats["queue_wait_s"] += time.perf_counter() - t0
        return path

    def pending_prefix(self, token_ids: Sequence[int]) -> tuple[int, str] | None:
        """GLM53_KV_LOOKUP_DEFER_KDA: (n, path) of a deferred checkpoint of a prefix of `token_ids` whose states are not
        produced yet (a hit on it would wait for its resolution inside the load), else None"""
        with _KDA_LOCK:
            items = [(p, v[0]) for p, v in self._pending.items() if isinstance(v[1], _Deferred) and not v[1].done]
        for path, n in items:
            if 0 < n <= len(token_ids) and self.path(token_ids, n) == path:
                return n, path
        return None

    def snapshot_producer(self, token_ids: Sequence[int], num_tokens: int) -> Any | None:
        """lever 1 (b): the producer of the LATEST deferred save when it is this checkpoint (its device snapshot may
        still be resident: the model's kda_snapshot_states decides), else None"""
        path = self.path(token_ids, num_tokens)
        with _KDA_LOCK:
            got = _KDA_SNAP.get(self.root)
        return got[1] if got is not None and got[0] == path else None

    def _start_writer(self) -> None:
        if self._writer is None:
            import queue  # noqa: PLC0415

            self._queue = queue.Queue(maxsize=int(os.environ.get("GLM53_KDA_CKPT_QUEUE", "8")))
            self._writer = threading.Thread(target=self._writer_loop, name="tt-kda-ckpt-writer", daemon=True)
            self._writer.start()
        if self.resolve_ahead and self._resolver is None:
            import queue  # noqa: PLC0415

            self._rqueue = queue.Queue()
            # GLM53_KDA_CKPT_RESOLVERS = N (default 1; program 2026-10-28T08:07:52Z (3)): N resolver threads on the
            # one queue, so a checkpoint never waits behind an unrelated export's resolution (k4d8t3: the 31,872 export
            # queued 0.517 s behind a 5.4-s one while a repeat's hit waited for it); a load() that needs a checkpoint
            # whose resolution has not started still runs it inline at once (_Deferred.result)
            n = max(1, int(os.environ.get("GLM53_KDA_CKPT_RESOLVERS", "1") or 1))
            self._resolvers = [threading.Thread(target=self._resolver_loop, daemon=True,
                                                name="tt-kda-ckpt-resolver" + (f"-{i}" if i else ""))
                               for i in range(n)]  # fmt: skip
            self._resolver = self._resolvers[0]
            for t in self._resolvers:
                t.start()

    def _resolver_loop(self) -> None:
        """GLM53_KDA_CKPT_RESOLVE_AHEAD: deferred checkpoints resolved in queue order, their states into the RAM tier
        while still pending (has() / load() then hit RAM). A failing producer keeps its error in the _Deferred: the
        writer's _resolve re-raises it (surfaced by the next save / flush, as before)."""
        while True:
            path, d, t_q = self._rqueue.get()
            try:
                t0 = time.perf_counter()
                states = d.result()
                dt = time.perf_counter() - t0
                self.stats["resolve_s"] = self.stats.get("resolve_s", 0.0) + dt
                if dt > 0.2 or t0 - t_q > 0.2:  # the k4a 16,000-token repeat: ~2 s, unexplained by the logs
                    logger.info("KDA checkpoint %s resolved in %.3f s after %.3f s queued (the page-DMA state save + "
                                "untilize; ~0.12 s of host work for 148 MB)", os.path.basename(path), dt, t0 - t_q)  # fmt: skip
                with _KDA_LOCK:
                    pending = path in self._pending
                if pending:
                    self._ram_put(path, states)
                    self.stats["resolved_ahead"] = self.stats.get("resolved_ahead", 0) + 1
            except _EngineOnly:  # the engine-thread rule: drain_engine_todo produces it (and puts the RAM tier entry)
                self.stats["engine_handbacks"] = self.stats.get("engine_handbacks", 0) + 1
            except BaseException:  # noqa: BLE001  (kept in the _Deferred; the writer surfaces it)
                pass
            finally:
                self._rqueue.task_done()

    def _write_file(self, path: str, data: bytes | None, blob: Any = None) -> None:
        lp = self.lpath(path)
        if lp is not None:  # local-first: the local file now, the share copy from the mirror thread
            self._write_one(lp, data, blob, self.local_root, self.local_max_bytes)
            self._start_mirror()
            self._mqueue.put((lp, path))
            return
        self._write_one(path, data, blob, self.root, self.max_bytes)

    def _start_mirror(self) -> None:
        if self._mirror is None:
            import queue  # noqa: PLC0415

            self._mqueue = queue.Queue()
            self._mirror = threading.Thread(target=self._mirror_loop, name="tt-kda-ckpt-mirror", daemon=True)
            self._mirror.start()

    def _mirror_loop(self) -> None:
        """GLM53_KDA_CKPT_LOCAL_DIR: local checkpoints copied to the share (the restart truth), in save order. A failed
        or skipped copy (a share below its floor) is counted and logged once; the local copy still serves this process."""
        import shutil  # noqa: PLC0415

        while True:
            lp, path = self._mqueue.get()
            try:
                st = os.statvfs(self.root)
                if st.f_bavail * st.f_frsize < self.min_free_bytes:
                    self.stats["mirror_skipped_low_disk"] = self.stats.get("mirror_skipped_low_disk", 0) + 1
                elif os.path.exists(lp) and not os.path.exists(path):
                    t0 = time.perf_counter()
                    tmp = f"{path}.{os.getpid()}.mtmp"
                    shutil.copyfile(lp, tmp)
                    os.replace(tmp, path)
                    self.stats["mirrored"] = self.stats.get("mirrored", 0) + 1
                    self.stats["mirror_s"] = self.stats.get("mirror_s", 0.0) + time.perf_counter() - t0
                    self._evict(keep=path)
            except Exception as e:  # noqa: BLE001 -- the local copy serves; the share copy is the restart truth only
                self.stats["mirror_errors"] = self.stats.get("mirror_errors", 0) + 1
                if self.stats["mirror_errors"] == 1:
                    logger.warning("KDA checkpoint mirror to %s failed (local copies still serve): %s", self.root, e)
            finally:
                self._mqueue.task_done()

    def _write_one(self, path: str, data: bytes | None, blob: Any, root: str, max_bytes: int) -> None:
        t0 = time.perf_counter()
        tmp = f"{path}.{os.getpid()}.tmp"
        if data is None:
            fmt = _blob_dump(blob, tmp)
            self.stats[f"written_{fmt}"] = self.stats.get(f"written_{fmt}", 0) + 1
        else:
            with open(tmp, "wb") as f:
                f.write(data)
        os.replace(tmp, path)
        self.stats["saved"] += 1
        self.stats["write_s"] += time.perf_counter() - t0
        self._evict(keep=path, root=root, max_bytes=max_bytes)

    def _writer_loop(self) -> None:
        while True:
            path = self._queue.get()
            try:
                with _KDA_LOCK:
                    item = self._pending.get(path)
                if item is not None:  # serialized here, off the engine step
                    try:
                        states = _resolve(item[1])  # a deferred checkpoint: its page-DMA bytes, untilized here
                    except _EngineOnly:  # the engine-thread rule: wait for the engine thread to produce it
                        lim = float(os.environ.get("GLM53_KDA_CKPT_ENGINE_WAIT_S", "120") or 120)
                        if not item[1].event.wait(lim):
                            self.stats["engine_wait_dropped"] = self.stats.get("engine_wait_dropped", 0) + 1
                            logger.warning("KDA checkpoint %s: its producer was handed to the engine thread, which "
                                           "did not run it within %.0f s: not written", os.path.basename(path), lim)  # fmt: skip
                            continue
                        states = _resolve(item[1])
                    if isinstance(item[1], _Deferred):
                        self._ram_put(path, states)
                    self._write_file(path, None, {"num_tokens": item[0], "states": states})
            except BaseException as e:  # noqa: BLE001  (surfaced by the next save / flush)
                self._error = e
            finally:
                with _KDA_LOCK:
                    self._pending.pop(path, None)
                self._queue.task_done()

    def flush(self) -> None:
        """wait for every queued write (tests; shutdown), and the share mirror of local-first checkpoints"""
        if self._queue is not None:
            _join_draining(self._queue)
        if self._mqueue is not None:
            self._mqueue.join()
        if self._error is not None:
            e, self._error = self._error, None
            raise RuntimeError(f"KDA checkpoint writer failed: {e}") from e

    def load(self, token_ids: Sequence[int], num_tokens: int) -> Any | None:
        path = self.path(token_ids, num_tokens)
        with _KDA_LOCK:
            hit = self._ram.get(path)
            if hit is not None:
                self._ram.move_to_end(path)
            item = self._pending.get(path)
        if hit is not None:
            self.stats["ram_hits"] += 1
            if os.path.exists(path):
                self._touch(path)
            return hit[0]
        if item is not None:
            self.stats["ram_hits"] += 1
            t0 = time.perf_counter()
            try:
                got = _resolve(item[1], prio=True)  # a hit waits: its collect goes first (GLM53_KDA_CKPT_COLLECT_SERIAL)
            except _EngineOnly:  # a load() off the engine thread whose producer needs the device: the engine runs it
                lim = float(os.environ.get("GLM53_KDA_CKPT_ENGINE_WAIT_S", "120") or 120)
                if not item[1].event.wait(lim):
                    return None
                got = _resolve(item[1])
            dt = time.perf_counter() - t0
            if dt > 0.05:  # a hit that waited for (or ran) a pending checkpoint's resolution
                self.stats["pending_load_waits"] = self.stats.get("pending_load_waits", 0) + 1
                logger.info("KDA checkpoint %s: a hit waited %.3f s for its pending resolution", os.path.basename(path),
                            dt)  # fmt: skip
            return got
        lp = self.lpath(path)
        src, kind = (lp, "local") if lp is not None and os.path.exists(lp) else (path, "disk")
        if not os.path.exists(src):
            return None
        self._touch(src)
        t0 = time.perf_counter()
        blob = _blob_load(src)
        self.stats[f"{kind}_loads"] = self.stats.get(f"{kind}_loads", 0) + 1
        self.stats[f"{kind}_load_s"] = self.stats.get(f"{kind}_load_s", 0.0) + time.perf_counter() - t0
        if int(blob.get("num_tokens", -1)) != int(num_tokens):
            raise RuntimeError(f"KDA checkpoint {src} is not a {num_tokens}-token one")
        self._ram_put(path, blob["states"])
        return blob["states"]

    def _ram_put(self, path: str, states: Any) -> None:
        with _KDA_LOCK:
            pinned = self.recent_n > 0 and path in self._recent
        if self.ram_bytes <= 0 and not pinned:
            return
        n = _states_nbytes(states)
        with _KDA_LOCK:
            self._ram[path] = (states, n)
            self._ram.move_to_end(path)
            total = sum(v[1] for v in self._ram.values())
            # GLM53_KDA_CKPT_RECENT_N: the last N saves are never evicted (oldest unpinned entries go first)
            for p in [k for k in self._ram if k != path]:
                if total <= self.ram_bytes:
                    break
                if self.recent_n > 0 and p in self._recent:
                    continue
                total -= self._ram.pop(p)[1]
            if self.recent_n > 0:  # a pinned entry that left the recent set is a plain LRU entry again
                for p in [k for k in self._ram if k not in self._recent]:
                    if total <= self.ram_bytes:
                        break
                    total -= self._ram.pop(p)[1]

    def longest_checkpointed(
        self, token_ids: Sequence[int], max_tokens: int, block_size: int
    ) -> int:
        """Longest page-aligned ``N <= max_tokens`` with a checkpoint, else 0."""
        n = (min(int(max_tokens), len(token_ids)) // block_size) * block_size
        while n > 0:
            if self.has(token_ids, n):
                return n
            n -= block_size
        return 0


class TTPageAlignedConnector:
    """Scheduler-role proxy: connector hits end on a page (finding J3).

    LMCache reports ``len - 1`` matched tokens and vLLM takes them as computed
    without page alignment, so a continuation would start mid-page and, on
    GLM-5.3-Flash, without the KDA state of that boundary. The proxy rounds
    ``local + external`` down to a multiple of the block size and, when the
    KDA checkpoint policy is on, to the longest checkpointed boundary. Every
    other call is forwarded to the wrapped connector unchanged.
    """

    def __init__(
        self,
        inner: Any,
        block_size: int,
        kda_store: KDACheckpointStore | None = None,
    ):
        self._inner = inner
        self._block_size = int(block_size)
        self._kda_store = kda_store
        _COUNTER_SOURCES["lookup"].add(self)
        # request id -> the count the wrapped connector reported. LMCache's
        # update_state_after_alloc asserts it gets exactly that number back
        # (its load spec already covers the whole hit; the pages past the
        # aligned boundary land in blocks vLLM allocated for the prompt).
        self._reported: dict[str, int] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> Any:
        reported = self._reported.pop(request.request_id, None)
        spec = self._load_spec(request.request_id)
        chunk = int(getattr(getattr(self._inner, "_lmcache_engine", None), "_lmcache_chunk_size", 0) or 0)
        if (num_external_tokens > 0 and spec is not None and chunk > 0
                and os.environ.get("GLM53_LMCACHE_LOAD_ALIGNED", "1") == "1"):  # fmt: skip
            # (with the raw match as the retrieve window, a 32K request whose LMCache match was 32,000 but whose vLLM
            # hit was the KDA checkpoint at 1,408 logged 'retrieved tokens less than expected' + double unpins):
            # LMCache's load window = the hit vLLM takes as computed, rounded up
            # to LMCache's chunk (it retrieves whole chunks; the pages past the hit land in the prompt's own blocks and
            # are recomputed, as before), never past its raw match. A shortfall inside that window is a real load
            # error (get_block_ids_with_load_errors -> vLLM's kv_load_failure_policy), not a design artifact.
            hit = int(spec.vllm_cached_tokens) + int(num_external_tokens)
            # LMCache's worker retrieves ReqMeta.token_ids = the tokens scheduled up to THIS step, floored to its chunk
            # (vllm_v1_adapter ReqMeta.from_request_tracker), not the whole match: that was the 'index vs store'
            # disagreement (05:51Z: retrieved = the first step's span, expected = the raw match). A window of exactly the
            # hit always fits that span when the hit grid is a multiple of LMCache's chunk (served: 128 / 128).
            if hit % chunk and not getattr(self, "_grid_warned", False):
                self._grid_warned = True
                logger.warning("KV connector hit %d is not a multiple of LMCache's chunk %d: its retrieve window may end "
                               "short of the hit (make the checkpoint grid a multiple of the chunk)", hit, chunk)  # fmt: skip
            window = min(int(spec.lmcache_cached_tokens), -(-hit // chunk) * chunk)
            spec.lmcache_cached_tokens = window
            with _PROVENANCE_LOCK:  # an admitted load: no provenance restore may remove its chunks until it ran
                _LMC_LOADS[request.request_id] = (getattr(request, "prompt_token_ids", None) or [], window)
            full = 1 if window == int(getattr(request, "num_tokens", -1)) else 0  # LMCache's "recalc the last token"
            self.load_windows_aligned = getattr(self, "load_windows_aligned", 0) + 1
            return self._inner.update_state_after_alloc(request, blocks, window - int(spec.vllm_cached_tokens) - full)
        if num_external_tokens == 0 and spec is not None:
            with _PROVENANCE_LOCK:
                item = _KDA_RESTORE.get(request.request_id)
            if item is not None and int(getattr(spec, "lmcache_cached_tokens", 0) or 0) > int(item[0]):
                # a restore candidate that takes no hit (KDA hit 0): LMCache's adapter saves a request from its load
                # spec's cached count (RequestTracker.num_saved_tokens), which stays the RAW match when vLLM takes
                # nothing -- the span [w, m) the worker replaces would never be stored again (cf3h-e mock r3
                # 2026-10-24T21:17Z: [0, 16000) removed, the request saved from 16,000, its repeat hit 0)
                spec.lmcache_cached_tokens = int(item[0])
                self.restore_save_windows = getattr(self, "restore_save_windows", 0) + 1
        if num_external_tokens > 0 and reported is not None:
            num_external_tokens = reported
        if num_external_tokens > 0:
            with _PROVENANCE_LOCK:
                _LMC_LOADS[request.request_id] = (getattr(request, "prompt_token_ids", None) or [],
                                                  int(getattr(spec, "vllm_cached_tokens", 0) or 0) + int(num_external_tokens))  # fmt: skip
        return self._inner.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def _load_spec(self, req_id: str) -> Any:
        """LMCache's scheduler-side LoadSpec of the request (vLLM's LMCacheConnectorV1 -> _lmcache_engine.load_specs),
        None for any other connector"""
        engine = getattr(self._inner, "_lmcache_engine", None)
        specs = getattr(engine, "load_specs", None)
        return specs.get(req_id) if isinstance(specs, dict) else None

    @property
    def inner(self) -> Any:
        return self._inner

    def _mark_provenance(self, req_id: str, raw: int, hit: int, tokens: Sequence[int] | None = None) -> None:
        """The provenance rule (_KDA_PROVENANCE): LMCache's FULL chunks of this request's match past its KDA-aligned hit are pages
        another computation wrote; the request recomputes them, so none of its checkpoints may be stored -- unless
        GLM53_KDA_CKPT_PROVENANCE=restore (the default) and the worker replaces that span first (_KDA_RESTORE: the
        request's tokens, w = the hit rounded up to LMCache's chunk = its load window, m = the full match).
        GLM53_KDA_CKPT_PROVENANCE=0 turns the rule off; GLM53_LMCACHE_CHUNK_TOKENS = LMCache's chunk (default 128)."""
        mode = kda_ckpt_provenance_mode()
        with _PROVENANCE_LOCK:
            _KDA_RESTORE.pop(req_id, None)
        if self._kda_store is None or mode == "0":
            _KDA_PROVENANCE.pop(req_id, None)
            return
        chunk = _lmcache_chunk_tokens()
        full = int(raw) // chunk * chunk
        if full > int(hit):
            _KDA_PROVENANCE[req_id] = full
            if mode == "restore" and tokens is not None:
                w = -(-int(hit) // chunk) * chunk
                if w < full:
                    with _PROVENANCE_LOCK:
                        _KDA_RESTORE[req_id] = (w, full, tokens)
            logger.debug("request %s: LMCache holds %d tokens past its KDA hit %d: no checkpoint from it%s",
                         req_id, full - int(hit), hit, " unless the worker replaces them" if req_id in _KDA_RESTORE
                         else "")  # fmt: skip
        else:
            _KDA_PROVENANCE.pop(req_id, None)

    def request_finished(self, request: Any, block_ids: Any) -> Any:
        rid = getattr(request, "request_id", None)
        _KDA_PROVENANCE.pop(rid, None)
        _PREEMPT_RESUME.pop(rid, None)
        _EXPECT_T0.pop(rid, None)
        _PD_EXACT_END.pop(rid, None)
        with _PROVENANCE_LOCK:
            _KDA_RESTORE.pop(rid, None)
            _LMC_LOADS.pop(rid, None)
            _LMC_WRITERS.pop(rid, None)
        out = self._inner.request_finished(request, block_ids)
        if _leave_drain() and rid is not None and block_ids:
            # M1: its blocks stay allocated until the host's held stores of it land (finished_sending)
            inner_delay, params = out if isinstance(out, tuple) and len(out) == 2 else (bool(out), None)
            with _LEAVE_LOCK:
                _LEAVE_PENDING[rid] = {"inner": bool(inner_delay), "inner_done": False}
            return True, params
        return out

    def _defer_on_pending_kda(self, request: Any) -> bool:
        """GLM53_KV_LOOKUP_DEFER_KDA=1 (default 0; program 2026-10-28T08:07:52Z (3)): a lookup whose prompt has a
        deferred KDA export of one of its prefixes still being produced is reported undetermined (vLLM retries next
        step, every other request keeps stepping) instead of admitting a hit whose load would block the engine thread
        for the rest of the resolution (k4d8t3's 32,000 repeat: 3.35 s inside the load step). Bounded by
        GLM53_KV_LOOKUP_DEFER_MAX_S (default 30): past it the lookup goes ahead and the load waits as before."""
        store = self._kda_store
        tokens = getattr(request, "prompt_token_ids", None)
        rid = getattr(request, "request_id", None)
        if store is None or not tokens:
            return False
        got = store.pending_prefix(tokens)
        if got is None:
            first = _KDA_LOOKUP_DEFERRED.pop(rid, None)
            if first is not None:
                logger.info("GLM53_KV_LOOKUP_DEFER_KDA: lookup of %s released after %.3f s (its prefix's KDA export landed)",
                            rid, time.monotonic() - first)  # fmt: skip
            return False
        first = _KDA_LOOKUP_DEFERRED.setdefault(rid, time.monotonic())
        if time.monotonic() - first > float(os.environ.get("GLM53_KV_LOOKUP_DEFER_MAX_S", "30")):
            _KDA_LOOKUP_DEFERRED.pop(rid, None)
            logger.warning("lookup of %s deferred past GLM53_KV_LOOKUP_DEFER_MAX_S on a pending KDA export: going ahead", rid)
            return False
        if first == _KDA_LOOKUP_DEFERRED.get(rid) and self.__dict__.get("_kda_defer_logged") != rid:
            self.__dict__["_kda_defer_logged"] = rid
            logger.info("GLM53_KV_LOOKUP_DEFER_KDA: lookup of %s deferred (the KDA export at %d of its prefix is being "
                        "produced)", rid, got[0])  # fmt: skip
        self.lookups_deferred_kda = getattr(self, "lookups_deferred_kda", 0) + 1
        return True

    def aligned_hit(self, token_ids: Sequence[int], total: int, exact: bool = False) -> int:
        """``total`` matched tokens -> the page boundary a continuation may start on. exact (GLM53_PREEMPT_SAVE, a
        request resumed from preemption): an EXACT checkpoint at ``total`` itself -- its preempt save's, off the page
        grid -- is the resume point, so not even the last partial page is recomputed."""
        total = min(int(total), len(token_ids))  # a hit never exceeds the prompt (a resumed request: its history)
        on_grid = total % self._block_size == 0
        if (exact and self._kda_store is not None and total > 0 and (on_grid or _resume_as_decode())
                and self._kda_store.has(list(token_ids[:total]), total)):  # fmt: skip
            # off the page grid only when the model resumes as a decode (9's GLM53_RESUME_AS_DECODE; 14's MUST on
            # c9cb202f9f4: the plugin plans an off-grid 1-token row only for a request carrying a resume record)
            self.__dict__["preempt_exact_hits"] = self.__dict__.get("preempt_exact_hits", 0) + 1  # (not the inner's)
            return total
        aligned = (total // self._block_size) * self._block_size
        if self._kda_store is not None and aligned > 0:
            aligned = self._kda_store.longest_checkpointed(
                token_ids, aligned, self._block_size
            )
        return aligned

    @staticmethod
    def _set_skip_save(request: Any) -> None:
        """LMCache's own per-request switch (kv_transfer_params "lmcache.skip_save"); the inner connector reads it when
        it starts tracking the request."""
        sp = getattr(request, "sampling_params", None)
        if sp is None:
            return
        extra = sp.extra_args if sp.extra_args is not None else {}
        kv = dict(extra.get("kv_transfer_params") or {})
        kv["lmcache.skip_save"] = True
        extra["kv_transfer_params"] = kv
        sp.extra_args = extra

    @classmethod
    def _skip_short_prompt_save(cls, request: Any) -> None:
        """GLM53_LMCACHE_MIN_STORE_TOKENS: a prompt shorter than N is not stored."""
        n = lmcache_min_store_tokens()
        if n and int(getattr(request, "num_prompt_tokens", 0) or 0) < n:
            cls._set_skip_save(request)

    @staticmethod
    def _cold(request: Any) -> bool:
        """kv_transfer_params {"tt.cold": true}: a probe that must prefill cold on a server whose LMCache already holds
        its prefix (cache_salt does not isolate LMCache: its keys hash the tokens)."""
        kv = getattr(request, "kv_transfer_params", None)
        if kv is None:
            sp = getattr(request, "sampling_params", None)
            kv = (getattr(sp, "extra_args", None) or {}).get("kv_transfer_params")
        return bool((kv or {}).get("tt.cold"))

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """(GLM53_KV_LOOKUP_LOG_MS, default 50; 0 = off; program 2026-10-26T22:40:55Z (b): the k3 ladder's
        0.10-0.19 s between the previous request's last store and this lookup's LMCache line was in no log): one INFO
        line when a lookup takes at least that long, split defer / flush / LMCache lookup / the rest (aligned hit,
        KDA checkpoints, plan check)"""
        tm = {"t0": time.perf_counter()}
        try:
            return self._matched_tokens(request, num_computed_tokens, tm)
        finally:
            lim = float(os.environ.get("GLM53_KV_LOOKUP_LOG_MS", "50") or 0)
            end = time.perf_counter()
            if lim > 0 and (end - tm["t0"]) * 1e3 >= lim:
                marks = [("defer", tm.get("defer")), ("flush", tm.get("flush")), ("lmcache", tm.get("lmcache"))]
                prev, parts = tm["t0"], []
                for name, t in marks:
                    if t is not None:
                        parts.append(f"{name} {1e3 * (t - prev):.1f}")
                        prev = t
                parts.append(f"rest {1e3 * (end - prev):.1f}")
                logger.info("TT KV connector lookup (%s): %.1f ms (%s ms)", getattr(request, "request_id", None),
                            1e3 * (end - tm["t0"]), ", ".join(parts))  # fmt: skip

    def _matched_tokens(self, request: Any, num_computed_tokens: int, tm: dict) -> tuple[int | None, bool]:
        if os.environ.get("GLM53_KV_LOOKUP_DEFER", "0") == "1" and lookup_defer(request):
            self.lookups_deferred = getattr(self, "lookups_deferred", 0) + 1
            tm["defer"] = time.perf_counter()
            return None, False  # undetermined: vLLM retries the request next step (no drain on the engine thread)
        if os.environ.get("GLM53_KV_LOOKUP_DEFER_KDA", "0") == "1" and self._defer_on_pending_kda(request):
            tm["defer"] = time.perf_counter()
            return None, False  # undetermined: the prefix's KDA export is still being produced; vLLM retries next step
        tm["defer"] = time.perf_counter()
        flush_deferred_saves(request)  # a deferred prefill save lands before this lookup reads LMCache / checkpoints
        tm["flush"] = time.perf_counter()
        waited = _LOOKUP_RELEASED.pop(getattr(request, "request_id", None), None)
        if waited is not None:  # (logged with its result below)
            self._deferred_waits = getattr(self, "_deferred_waits", {})
            self._deferred_waits[request.request_id] = waited
        cold = self._cold(request)
        if cold:
            self._set_skip_save(request)
        else:
            self._skip_short_prompt_save(request)
        rid = getattr(request, "request_id", None)
        if _pd_role(request).upper() == "P" and _resume_as_decode():
            _PD_EXACT_END[rid] = int(getattr(request, "num_prompt_tokens", 0) or len(request.prompt_token_ids or []))
        resume = _PREEMPT_RESUME.get(rid) or 0
        if rid in _PREEMPT_RESUME and _PREEMPT_RESUME.get(rid) is None:
            # its preempt save has not run on the worker yet (an async step): undetermined, retried next step
            self.__dict__["preempt_lookup_waits"] = self.__dict__.get("preempt_lookup_waits", 0) + 1
            tm["lmcache"] = time.perf_counter()
            return None, False
        expect = 0 if (resume or cold) else _expect_hit(request)  # Form B v2: D waits for P's stores (_EXPECT_T0)
        cut = resume or expect
        looked = request
        if cut and not cold and len(getattr(request, "all_token_ids", ()) or ()) > cut:
            looked = _HistoryCut(request, cut)  # 14 B1 / 9 (e): the hit can reach the saved unfull chunk at n
        ext, is_async = self._inner.get_num_new_matched_tokens(
            looked, num_computed_tokens
        )
        kv_short = expect and ext is not None and int(num_computed_tokens) + int(ext) < expect
        # 14's 03:41:08Z MUST: P's exact checkpoint is written by its async writer AFTER its LMCache stores can land; D
        # (another process: has() sees the file on disk) waits for it too, else the exact resume fails and D silently
        # re-prefills the whole prompt (only under resume-as-decode: without it the exact checkpoint is not used)
        ckpt_short = False
        if expect and ext is not None and not kv_short and _resume_as_decode() and self._kda_store is not None:
            src = _hit_tokens(request)
            if len(src) < expect:
                src = getattr(request, "all_token_ids", None) or src
            ckpt_short = not self._kda_store.has(list(src[:expect]), expect)
        if kv_short or ckpt_short:
            t0 = _EXPECT_T0.setdefault(rid, time.perf_counter())
            if time.perf_counter() - t0 < float(os.environ.get("GLM53_KV_EXPECT_WAIT_S", "10") or 0):
                clear = getattr(getattr(getattr(self._inner, "_lmcache_engine", None), "lookup_client", None),
                                "clear_lookup_status", None)  # fmt: skip
                if callable(clear):
                    clear(rid)  # (the retry looks up afresh, not LMCache's cached count)
                k_ = "expect_waits" if kv_short else "expect_waits_ckpt"  # (the KV landed, P's checkpoint not yet)
                self.__dict__[k_] = self.__dict__.get(k_, 0) + 1
                tm["lmcache"] = time.perf_counter()
                return None, False
            k_ = "expect_timeouts" if kv_short else "expect_timeouts_ckpt"
            self.__dict__[k_] = self.__dict__.get(k_, 0) + 1
            logger.warning("TT KV connector: %s expected %d remote tokens%s, %d after %.1f s: proceeding", rid, expect,
                           "" if kv_short else " + P's exact checkpoint (missing)", int(num_computed_tokens) + int(ext),
                           time.perf_counter() - t0)  # fmt: skip
        if expect:
            _EXPECT_T0.pop(rid, None)
        tm["lmcache"] = time.perf_counter()
        if is_async and os.environ.get("GLM53_ADMIT_KV_BUDGET", "0") == "1" and not self.__dict__.get("_async_warned"):
            # vLLM skips its split hook, where the TT scheduler's KV-budget gate lives, for an async KV load
            # (scheduler.py 'and not load_kv_async'); LMCacheConnectorV1 never reports one (is_async is False)
            self.__dict__["_async_warned"] = True
            logger.warning("TT KV connector: an async KV load (%s) bypasses the GLM53_ADMIT_KV_BUDGET gate",
                           getattr(request, "request_id", None))  # fmt: skip
        if cold and ext is not None:
            # The lookup ran (LMCache's lookup status and pins stay as for any request); the request then computes from
            # 0 as a hit rounded down to 0 does, stores nothing and exports no KDA checkpoint (the provenance mark).
            self._reported.pop(request.request_id, None)
            _KDA_PROVENANCE[request.request_id] = int(ext)
            self.cold_requests = getattr(self, "cold_requests", 0) + 1
            return 0, False
        if not ext:
            self._reported.pop(request.request_id, None)
            _KDA_PROVENANCE.pop(request.request_id, None)
            self._log_deferred(request, 0)
            return ext, is_async
        self._reported[request.request_id] = int(ext)
        hist = _hit_tokens(request)  # (GLM53_PREEMPT_SAVE: a resumed request's history; else its prompt)
        if expect and len(hist) < expect:  # (a D request's history beyond its prompt: P's prompt + token 1)
            hist = list(getattr(request, "all_token_ids", None) or hist)
        aligned = self.aligned_hit(hist, int(num_computed_tokens) + int(ext), exact=bool(resume or expect))
        aligned = self._plannable_hit(request, int(num_computed_tokens), aligned)
        self._mark_provenance(
            request.request_id, int(num_computed_tokens) + int(ext), aligned, hist
        )
        new_ext = max(aligned - int(num_computed_tokens), 0)
        if new_ext != ext:
            logger.debug(
                "connector hit for %s: %d external tokens -> %d (page %d%s)",
                request.request_id,
                ext,
                new_ext,
                self._block_size,
                ", KDA checkpoints" if self._kda_store is not None else "",
            )
        if new_ext == 0:
            self._log_deferred(request, 0)
            return 0, False
        self._log_deferred(request, new_ext)
        return new_ext, is_async

    def _plannable_hit(self, request: Any, num_computed: int, aligned: int) -> int:
        """program 2026-10-26T17:09:24Z (k3 MUST): the hit's continuation [aligned, prompt end) must PLAN on the model
        (vllm_tt_plugin.plan_check: TPPrefill.plan / clamp_lengths, host-only) -- a continuation ending near the cache
        end used to raise 'past the cache' in the runner step and take the engine down. Otherwise the hit steps down to
        the next lower checkpointed boundary (aligned_hit) until its continuation plans, or to no hit."""
        from vllm_tt_plugin import plan_check  # noqa: PLC0415

        toks = _hit_tokens(request)
        end = int(getattr(request, "num_prompt_tokens", 0) or len(toks))
        if len(toks) > end:  # a resumed request's continuation runs to its whole history (GLM53_PREEMPT_SAVE)
            end = len(toks)
        while aligned > num_computed:
            err = plan_check.plan_error(aligned, end - aligned)
            if err is None:
                break
            lower = int(self.aligned_hit(toks, aligned - 1))
            logger.info("connector hit for %s at %d: its continuation does not plan (%s); stepping down to %d",
                        request.request_id, aligned, err, lower)  # fmt: skip
            self.hits_replanned = self.__dict__.get("hits_replanned", 0) + 1  # (not the inner connector's attribute)
            if lower >= aligned:
                break
            aligned = lower
        return aligned

    def _log_deferred(self, request: Any, ext: int) -> None:
        waits = getattr(self, "_deferred_waits", None)
        if waits:
            waited = waits.pop(getattr(request, "request_id", None), None)
            if waited is not None:
                logger.info("GLM53_KV_LOOKUP_DEFER: lookup of %s released after %.3f s: %d external tokens",
                            request.request_id, waited, int(ext))  # fmt: skip


class TTKVConnectorHost:
    """Drives the worker-side connector around one TT step.

    One instance per runner, built in ``initialize_kv_cache`` once the model's
    paged caches exist. Every connector call that needs bound metadata is
    wrapped in ``bind_connector_metadata`` / ``clear_connector_metadata`` under
    one lock, because the TT runner's steps overlap (async decode resolves on
    another thread after the next step has begun) and the connector keeps a
    single metadata slot.
    """

    def __init__(
        self,
        connector: KVConnectorBase_V1,
        kv_cache_config: KVCacheConfig,
        kv_caches: Any,
        *,
        model: Any = None,
        layer_to_group_idx: list[int] | None = None,
        kda_store: KDACheckpointStore | None = None,
        slot_of: Callable[[str], int | None] | None = None,
        spec_k: int = 0,
        tokens_of: Callable[[str], Sequence[int] | None] | None = None,
        len_of: Callable[[str], int | None] | None = None,
        ckpt_grid: int | None = None,
        head_of: Callable[[str, int], Sequence[int] | None] | None = None,
    ):
        self.connector = connector
        _COUNTER_SOURCES["host"].add(self)
        self._head_of_fn = head_of  # a request's first n history tokens (O(n)); None: tokens_of(...)[:n]
        self.kv_cache_config = kv_cache_config
        # Speculative decoding: K drafts per decode step (module docstring).
        self.spec_k = int(spec_k or 0)
        # Decode-time page boundaries: checkpointed at the committed position
        # through the model's candidate-row export, deferred while past it,
        # skipped when the model has no such export.
        self.decode_boundaries_checkpointed = 0
        self.decode_boundaries_deferred = 0
        self.decode_boundaries_skipped = 0
        self.decode_boundaries_skipped_every = 0  # GLM53_KDA_CKPT_EVERY: decode boundaries not at a multiple of N
        self.kda_ckpt_skipped_short = 0  # GLM53_LMCACHE_MIN_STORE_TOKENS: prefill checkpoints of short prompts
        self.kda_ckpt_skipped_provenance = 0  # checkpoints barred by the provenance rule
        self.kda_provenance_restored = 0  # GLM53_KDA_CKPT_PROVENANCE=restore: spans replaced (the bar lifted)
        self.kda_provenance_restored_tokens = 0
        self.kda_provenance_restore_skipped_load = 0  # guard: another request's admitted load shared the span
        self.kda_provenance_restore_skipped_writer = 0  # guard: another running prefill shared the span
        self.kda_provenance_restore_failed = 0  # no engine, an exception, or a key still held after the removal
        self.kda_provenance_mixed_imports = 0  # hits on a first writer's checkpoint deeper than a replaced span
        self._skipped_boundary_logged = False
        self.views = resolve_connector_kv_views(kv_caches, kv_cache_config, model)
        self._layer_to_group_idx = layer_to_group_idx
        self._lock = threading.Lock()
        # Device-side objects per layer (torch views on the mock path,
        # TTPagedCacheHandle over ttnn tensors on a device).
        self.kv_caches: dict[str, Any] = {n: v.tensor for n, v in self.views.items()}
        self.device_path = any(
            isinstance(v.tensor, TTPagedCacheHandle) for v in self.views.values()
        )
        # On a device the connector addresses host transfer buffers of the
        # connector-facing shape; the CopyBlocksOp moves pages between them and
        # the device handles. On the mock path the views are the buffers.
        self.host_xfer_buffers: dict[str, torch.Tensor] | None = None
        if self.device_path:
            self.host_xfer_buffers = {
                n: _host_zeros(tuple(v.tensor.shape), v.tensor.dtype)
                for n, v in self.views.items()
            }
        self.connector_caches: dict[str, Any] = (
            self.host_xfer_buffers
            if self.host_xfer_buffers is not None
            else self.kv_caches
        )
        connector.register_kv_caches(self.connector_caches)
        connector.set_host_xfer_buffer_ops(tt_copy_kv_blocks)
        # Finding J4: vLLM's LMCacheConnectorV1 wrapper and LMCache's adapter do
        # not forward set_host_xfer_buffer_ops, so the TT GPU connector inside
        # LMCache is handed the copy op and the device handles directly.
        self.tt_gpu_connector = find_tt_gpu_connector(connector)
        # GLM53_DSA_SHARD (lmcache/sharded_pool.py): every layer handle shares one ShardedPagePool; the TT GPU connector
        # admits a hit's pages at their columns before the h2d copy (a LOAD lands on each page's owner chip)
        self.sharded_pool = None
        for v in self.views.values():
            h = getattr(v.tensor, "inner", v.tensor)  # a squeezed 3-D view wraps the 4-D handle
            if isinstance(h, TTShardedPagedCacheHandle):
                self.sharded_pool = h.pool
                break
        if self.tt_gpu_connector is not None:
            self.tt_gpu_connector.set_host_xfer_buffer_ops(tt_copy_kv_blocks)
            register = getattr(self.tt_gpu_connector, "register_device_kv_caches", None)
            if callable(register):
                register(self.kv_caches)
            if self.sharded_pool is not None:
                register_pool = getattr(self.tt_gpu_connector, "register_sharded_pool", None)
                if not callable(register_pool):
                    raise TypeError(
                        f"{type(self.tt_gpu_connector).__name__} cannot address the sequence-sharded pool "
                        "(no register_sharded_pool): a LOAD must be admitted per owner chip"
                    )
                register_pool(self.sharded_pool)
        # GLM53_LMCACHE_PAGE_DMA=1: hits page in over the model's page-DMA kernel (started with the connector views,
        # before any capture); without the hook (or the knob) the copy op above moves them.
        hook = getattr(model, "latent_hit_dma", None)
        self.hit_dma = hook() if callable(hook) else None
        if self.hit_dma is not None:
            set_dma = getattr(self.tt_gpu_connector, "set_hit_dma", None)
            if not callable(set_dma):
                raise RuntimeError(
                    "GLM53_LMCACHE_PAGE_DMA needs the TT GPU connector (VLLMPagedMemTTConnector) to route hits"
                )
            set_dma(self.hit_dma)
            logger.info("KV connector hits page in over the page-DMA kernel (GLM53_LMCACHE_PAGE_DMA)")
        # Trace-tracking warm-up (R1 finding / review_m6 #7): compile the page paths' programs (slice + read-back,
        # the staging upload, paged_fill_cache) before the model's trace capture. Block 0 is vLLM's null block (never a
        # request page); it is read and written back unchanged.
        if self.device_path and os.environ.get("GLM53_LMCACHE_WARM", "1") == "1":
            self._warm_page_paths()
        # GLM53_KV_POOLED_PAGES=1 (Stage 12, ruling 2026-10-23T09:54:56Z; doc/ttnn/12-fast-start/paged_pooled_keys.md):
        # every LMCache page carries its 64 tokens' DSA pooled keys in the reserved 4 KiB slice -- read from the
        # request's state slot at the store (the model's kv_read_pooled_pages), handed to the hit's checkpoint at the
        # load (the model loads them into the slot it places the request in) -- so the KDA checkpoints carry only the
        # KDA state + ring (kv_pooled_pages_enable): O(n) per checkpoint. Needs the page-DMA (GLM53_KV_T2) and FRESH
        # LMCache tiers (a page stored without its slice reads as zeros: such a hit restarts at 0).
        self.pooled_pages = False
        self.pooled_missing = 0
        self._pooled_map: dict[int, tuple[int, int]] = {}
        if os.environ.get("GLM53_KV_POOLED_PAGES", "0") == "1":
            enable = getattr(model, "kv_pooled_pages_enable", None)
            self._read_pooled = getattr(model, "kv_read_pooled_pages", None)
            conn_enable = getattr(self.tt_gpu_connector, "enable_pooled_pages", None)
            if (callable(enable) and callable(self._read_pooled) and callable(conn_enable)
                    and self.hit_dma is not None and enable()):  # fmt: skip
                plan = getattr(model, "kv_pooled_store_plan", None)
                self._read_pooled_plan = plan if callable(plan) else None
                if self._read_pooled_plan is not None and "plan" in inspect.signature(conn_enable).parameters:
                    # store path P2: the model's (targets, finish) split, so a zero-copy store reads its pooled pages
                    # in the same page-DMA wait as its latent pages (GLM53_KV_STORE_ONE_DMA)
                    conn_enable(self._pooled_source, plan=self._pooled_plan)
                else:
                    conn_enable(self._pooled_source)
                self.pooled_pages = True
                logger.info("KV connector: the DSA pooled keys ride in the LMCache pages (GLM53_KV_POOLED_PAGES); KDA "
                            "checkpoints carry the KDA state + ring only")  # fmt: skip
            else:
                logger.warning("GLM53_KV_POOLED_PAGES=1 without the page-DMA's pooled-key stores or the model hooks: "
                               "off (the KDA checkpoints keep their pooled-key rows)")  # fmt: skip
        # Stage 12 (n29h7p2 2026-10-23): a cache whose KDA checkpoints can never resume a hit (the model's
        # kv_checkpoints_resumable: a paged pooled-key store without the pooled pages above) exports none, so the
        # scheduler side aligns every hit to 0 (no checkpoint): no LMCache retrieve that the model would refuse (2-19 s
        # per hit on 1843), no prompt-end export on the next step; the LMCache stores stay write-through
        resumable = getattr(model, "kv_checkpoints_resumable", None)
        self.checkpoints_off = bool(callable(resumable) and not resumable())
        if self.checkpoints_off:
            logger.warning("KV connector: this cache's KDA checkpoints cannot resume a hit (paged pooled-key store "
                           "without the pooled pages): no checkpoint exports, every connector hit aligns to 0")  # fmt: skip
        # GLM53_KV_STORE_WORKER=1 (Stage 12 cf3h, program 2026-10-23T21:13:14Z): LMCache's page stores run on ONE
        # store worker thread. The engine thread only reads the stored pages device -> host (the bulk read; ttnn
        # stays on this thread) into a job-owned snapshot; the worker runs LMCache's own store loop for that step's
        # metadata (a thread-local override of _get_connector_metadata) with the GPU connector copying from the
        # snapshot: memory objects, hashing, puts and tiering leave the engine thread. Not with the pooled pages
        # (their read is a page-DMA call at store time) or the generic host-sync path.
        self._enable_store_worker(connector)
        # A connector that does not drive the copy op itself (ExampleConnector)
        # indexes the host buffers; on a device the host then syncs the step's
        # pages around it: loaded prefixes h2d after load, written pages d2h
        # before save. Slow but correct; LMCache's TT connector bypasses it.
        self._generic_sync = self.device_path and self.tt_gpu_connector is None
        # KDA per-slot state (page_contract.md section 5): a prefix hit is only
        # valid where a checkpoint of the recurrent state exists.
        self.kda_store = kda_store
        self._kv_cache_object = kv_caches
        self._slot_of = slot_of or (lambda req_id: None)
        self._tokens_of = tokens_of or (lambda req_id: None)
        self._len_of = len_of  # the history's LENGTH (O(1)); None: len(tokens_of(...))
        self._ckpt_grid = int(ckpt_grid or 0)
        self._deferred_save = None  # (scheduler_output, model_input, output) of a prefill step whose save waits
        self.saves_deferred = 0
        self._kda_export = getattr(model, "export_kda_state", None)
        self._kda_snapshot_ready = getattr(model, "kda_snapshot_ready", None)  # GLM53_KDA_CKPT_IN_REPLAY (12's M2)
        # GLM53_KDA_CKPT_DMA (default 1): the model's export_kda_state_async (a device snapshot the page-DMA saves while
        # the next steps run; None when it cannot) takes the prompt-end checkpoint off the first-token path
        self._kda_export_async = (getattr(model, "export_kda_state_async", None)
                                  if os.environ.get("GLM53_KDA_CKPT_DMA", "1") == "1" else None)  # fmt: skip
        self._kda_import = getattr(model, "import_kda_state", None)
        # GLM53_KDA_SNAP_IMPORT (default 1; Stage 12 lever 1 (b)): a hit resuming at the checkpoint whose device snapshot
        # is still resident (no later export rewrote it) restores the slot on device from the snapshot buffers -- no
        # wait for the deferred export's page-DMA save + untilize, no host -> device import of ~148 MB
        self._kda_snap_states = (getattr(model, "kda_snapshot_states", None)
                                 if os.environ.get("GLM53_KDA_SNAP_IMPORT", "1") == "1" else None)  # fmt: skip
        # Decode-time boundary export (module docstring): the state after
        # ``n_tokens`` read from candidate row ``candidate_row`` of a
        # speculating slot (``None``: the slot's own row, a plain decode).
        self._boundary_export = getattr(model, "export_spec_boundary_state", None)
        if callable(self._kda_export) and kda_store is None:
            raise ValueError(
                f"{type(model).__name__} keeps per-slot KDA state "
                "(export_kda_state); a KV connector needs the checkpoint policy: "
                "set kv_connector_extra_config['tt_kda_checkpoint_dir']"
            )
        if kda_store is not None and not callable(self._kda_export):
            logger.warning(
                "tt_kda_checkpoint_dir is set but %s has no export_kda_state: "
                "hits are page-aligned, no KDA checkpoints are written",
                type(model).__name__,
            )
        block_sizes = sorted({v.block_size for v in self.views.values()})
        logger.info(
            "TT KV connector host: %s over %d layer(s), page = %s token(s), "
            "device path %s, TT GPU connector %s, KDA checkpoints %s",
            type(connector).__name__,
            len(self.views),
            block_sizes[0] if len(block_sizes) == 1 else block_sizes,
            self.device_path,
            type(self.tt_gpu_connector).__name__ if self.tt_gpu_connector else "none",
            "on" if (kda_store is not None and callable(self._kda_export)) else "off",
        )
        if self.spec_k:
            logger.info(
                "KV connector with speculative decoding (K=%d): chunk stores on "
                "committed tokens only (the connector's scheduler side counts "
                "accepted tokens; stores run after the step), KDA checkpoints at "
                "prefill page boundaries, decode-time boundaries deferred to the "
                "committed position and %s",
                self.spec_k,
                "exported from the candidate row (export_spec_boundary_state)"
                if callable(self._boundary_export)
                else "not checkpointed (the model has no export_spec_boundary_state)",
            )
        # GLM53_LMCACHE_WARM_GB / GLM53_KDA_CKPT_WARM (Stage 12 lever (c) step 3, default off): after a relaunch, refill
        # the LMCache CPU tier from the cold tier and the KDA RAM tier from the checkpoint files, in the background
        # (last in the constructor: the CPU tier's allocator -- the pinned pool in tile mode -- and the store exist)
        self.warm_start_stats: dict = {}
        from vllm_tt_plugin.lmcache import warm_start as _warm  # noqa: PLC0415

        impl = getattr(self.connector, "_lmcache_engine", None)
        # B2 (program 2026-10-26T16:12:35Z): tile objects only under their own key namespace, checked before any
        # lookup / store / warm-start read touches a tier
        from vllm_tt_plugin.lmcache.connector import check_key_namespace  # noqa: PLC0415

        check_key_namespace(getattr(getattr(getattr(impl, "lmcache_engine", None), "metadata", None), "model_name", None))
        self.warm_start_thread = _warm.start(getattr(impl, "lmcache_engine", None), self.kda_store, self.warm_start_stats)

    def _warm_page_paths(self) -> None:
        if not self.host_xfer_buffers:
            return
        t0 = time.perf_counter()
        if self.sharded_pool is not None:  # the sharded pool: logical block 0 has no mapping; warm on the null blocks
            for v in self.views.values():
                h = v.tensor if isinstance(v.tensor, TTShardedPagedCacheHandle) else getattr(v.tensor, "inner", None)
                if h is not None:
                    h.warm_page_paths()
        else:
            tt_copy_kv_blocks(self.kv_caches, self.host_xfer_buffers, [0], [0], "d2h")
            tt_copy_kv_blocks(self.host_xfer_buffers, self.kv_caches, [0], [0], "h2d")
        freeze_page_buffers()  # the persistent page buffers exist now; none may be created after the capture
        logger.info(
            "KV connector page paths warmed before capture (%s, %d layers, %.2f s; persistent page buffers)",
            "null blocks of the sharded pool" if self.sharded_pool is not None else "block 0",
            len(self.kv_caches),
            time.perf_counter() - t0,
        )

    @classmethod
    def from_runner(cls, runner: Any) -> TTKVConnectorHost:
        """Build the host for a ``TTModelRunner`` whose caches are allocated."""
        if getattr(runner, "_is_lane_mode", False):
            raise NotImplementedError(
                "KV connectors are not supported in single-process lane-DP mode: "
                "TTLaneCoordinator runs no scheduler-side connector"
            )
        if kda_ckpt_in_replay() and not getattr(getattr(runner, "model", None), "kda_checkpoint_in_replay", False):
            raise RuntimeError(
                "GLM53_KDA_CKPT_IN_REPLAY=1 but the model does not snapshot KDA state inside its prefill replay "
                "(no kda_checkpoint_in_replay): a storing prompt would run unsplit and store no checkpoint"
            )
        # Speculative decoding is admitted per connector by
        # validate_tt_kv_transfer_config (connector_spec_decode_mode); the host
        # only needs K to interpret decode rows (module docstring).
        spec_k = int(getattr(runner, "_num_speculative_tokens", 0) or 0)
        # Looked up live: an accepted decode slot gather replaces the runner's
        # ``_req_state_slot`` map with a new dict (note_decode_state_slots_settled),
        # so a reference captured here would go stale at the first remap.
        def slots() -> dict:
            return getattr(runner, "_req_state_slot", {})

        return cls(
            kv_transfer_state.get_kv_transfer_group(),
            runner.kv_cache_config,
            runner.kv_caches,
            model=getattr(runner, "model", None),
            layer_to_group_idx=getattr(runner, "_layer_to_group_idx", None),
            kda_store=KDACheckpointStore.from_vllm_config(runner.vllm_config),
            slot_of=lambda req_id: slots().get(req_id),
            spec_k=spec_k,
            tokens_of=lambda req_id: _runner_request_tokens(runner, req_id),
            len_of=lambda req_id: _runner_request_len(runner, req_id),
            ckpt_grid=_checkpoint_grid(runner),
            head_of=lambda req_id, n: _runner_request_head(runner, req_id, n),
        )

    # -- metadata plumbing ---------------------------------------------------

    @staticmethod
    def _metadata(scheduler_output: SchedulerOutput) -> Any:
        meta = getattr(scheduler_output, "kv_connector_metadata", None)
        if meta is None:
            raise RuntimeError(
                "SchedulerOutput carries no kv_connector_metadata: the scheduler "
                "side of the connector did not run (lane mode, or a scheduler "
                "built without the --kv-transfer-config of this worker)"
            )
        return meta

    def _group_for(self, view: TTKVLayerView) -> int:
        if (
            self._layer_to_group_idx is not None
            and view.layer_index is not None
            and view.layer_index < len(self._layer_to_group_idx)
        ):
            return int(self._layer_to_group_idx[view.layer_index])
        return view.group_index

    def _step_context(
        self, model_input: Any | None
    ) -> tuple[Any, dict[str, AttentionMetadata]]:
        """Forward context + per-layer metadata for ``model_input`` (None: empty)."""
        layers = {n: _TTKVLayer(n, self.connector_caches[n]) for n in self.views}
        attn: dict[str, AttentionMetadata] = {}
        slot_mappings: dict[str, torch.Tensor] = {}
        per_group: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        is_prefill = bool(
            model_input is not None
            and getattr(model_input, "prompt_lens", None) is not None
        )
        for name, view in self.views.items():
            if model_input is None:
                slot_mapping = torch.empty(0, dtype=torch.int64)
                starts = torch.zeros(1, dtype=torch.int32)
            else:
                g = self._group_for(view)
                key = (g, view.block_size)
                if key not in per_group:
                    tables = getattr(model_input, "block_tables_per_group", None)
                    table = tables[g] if tables else model_input.block_tables
                    per_group[key] = build_step_slot_mapping(
                        model_input, table, view.block_size
                    )
                slot_mapping, starts = per_group[key]
            attn[name] = _layer_attn_metadata(
                view, slot_mapping, starts, is_prefill=is_prefill
            )
            slot_mappings[name] = slot_mapping
        return _make_forward_context(layers, attn, slot_mappings), attn

    # -- step hooks ----------------------------------------------------------

    def begin_step(self, scheduler_output: SchedulerOutput) -> None:
        """Before the persistent batch is updated: evicted pages are still intact."""
        self._begin_seq = getattr(self, "_begin_seq", 0) + 1  # GLM53_KV_STORE_FT_GRACE counts engine steps here
        log_counters()
        self.flush_deferred_save(scheduler_output, budget=_store_budget())  # the previous prefill step's save
        self._segment_drop(self._gone(scheduler_output))  # after the flush: a release in it saved its chunk
        # P3: after the flush (it may queue a preempted request's held steps on the worker) and before the inline
        # preempt save: a worker job still reading a preempted request's blocks lands before this step reuses them
        self._zc_drain_for(set(getattr(scheduler_output, "preempted_req_ids", None) or ()))
        self._preempt_store(scheduler_output)  # GLM53_PREEMPT_SAVE: before this step's forward reuses the blocks
        meta = self._metadata(scheduler_output)
        with self._lock:
            self.connector.bind_connector_metadata(meta)
            try:
                self.connector.handle_preemptions(meta)
            finally:
                self.connector.clear_connector_metadata()

    def _preempt_store(self, scheduler_output: SchedulerOutput) -> None:
        """GLM53_PREEMPT_SAVE (see _PREEMPT_SAVES): each request preempted in this step -> its KV [0, n) stored through
        LMCache (a save-only request: chunks LMCache already holds are skipped, the decode region and the unfull tail
        are stored) + an exact KDA checkpoint at n from its slot, which still holds its state here. Synchronous: the
        step's forward may reuse the blocks. A failure only logs -- the request then recomputes on resume, as today.
        Counters preempt_saves / _failed / _over / _s."""
        if not _preempt_save_on():
            return
        gone = set(getattr(scheduler_output, "preempted_req_ids", None) or ())
        with _PREEMPT_LOCK:
            over = _PREEMPT_SAVES.pop("__over__", [])
            short = _PREEMPT_SAVES.pop("__short__", [])
            recs = {rid: _PREEMPT_SAVES.pop(rid) for rid in list(_PREEMPT_SAVES) if rid in gone}
            for rid in [r for r in recs if r in getattr(self, "_segment_dropped", ())]:
                # preempted inside a segmented chunk's hold: its slot holds the state before the chunk and the chunk's
                # later layers never ran, so [0, n) is not saved; it recomputes on resume, today's path
                recs.pop(rid)
                _PREEMPT_RESUME.pop(rid, None)
                self.preempt_saves_segment_held = getattr(self, "preempt_saves_segment_held", 0) + 1
            for rid in [r for r in _PREEMPT_SAVES if r not in gone]:  # a record whose step never reached a worker
                st = _PREEMPT_SAVES[rid]
                st["age"] = st.get("age", 0) + 1
                if st["age"] > 2:
                    _PREEMPT_SAVES.pop(rid, None)
                    if _PREEMPT_RESUME.get(rid, 0) is None:
                        _PREEMPT_RESUME.pop(rid, None)  # its lookup no longer waits: recompute, today's path
        self.preempt_saves_over = getattr(self, "preempt_saves_over", 0) + len(over)
        self.preempt_saves_short = getattr(self, "preempt_saves_short", 0) + len(short)
        for rid, rec in recs.items():
            t0 = time.perf_counter()
            try:
                n = self._preempt_store_one(scheduler_output, rid, rec)
                if n > 0:
                    _PREEMPT_RESUME[rid] = n  # the resume lookup runs over the history cut to n
                else:
                    _PREEMPT_RESUME.pop(rid, None)
                self.preempt_saves = getattr(self, "preempt_saves", 0) + 1
            except Exception as e:  # noqa: BLE001 - a failed save degrades to today's recompute, never to a crash
                _PREEMPT_RESUME.pop(rid, None)
                self.preempt_saves_failed = getattr(self, "preempt_saves_failed", 0) + 1
                logger.warning("GLM53_PREEMPT_SAVE: the save of preempted %s failed (%s: %s); it recomputes on resume",
                               rid, type(e).__name__, e)  # fmt: skip
            self.preempt_save_s = getattr(self, "preempt_save_s", 0.0) + time.perf_counter() - t0

    def _preempt_store_one(self, scheduler_output: SchedulerOutput, rid: str, rec: dict) -> int:
        """one preempted request's save -> its resume point (0: nothing saved). With GLM53_RESUME_AS_DECODE (the model
        resumes an exact off-grid checkpoint as one decode step): KV [0, n) + an exact KDA checkpoint at n. Without it
        the model refuses any continuation off the 64-token page grid (9 / 14, 00:53Z): the save snaps to the page
        floor m of n, KV [0, m) only -- the slot holds the state after n, not m, so the resume needs the checkpoint the
        decode-boundary export already took at m (counted preempt_snap_no_ckpt when there is none: the hit then walks
        down). Ordering (14 M1): begin_step runs before vLLM drains a pending async step, so the previous step's
        replay may still be queued; the page reads (synchronize_device) and the KDA export both run on CQ0 behind it,
        and the slot is read before the batch update. A CQ1 or host-mapped read here would race that step."""
        import copy  # noqa: PLC0415

        from lmcache.integration.vllm.vllm_v1_adapter import ReqMeta, SaveSpec  # noqa: PLC0415

        n, toks, blocks = int(rec["n"]), list(rec["token_ids"]), list(rec["block_ids"][0])
        bs = next(iter(self.views.values())).block_size
        exact = _resume_as_decode() or n % bs == 0
        if not exact:
            n = (n // bs) * bs
            self.preempt_saves_snapped = getattr(self, "preempt_saves_snapped", 0) + 1
            if n == 0:
                return 0
            toks = toks[:n]
        slots = torch.tensor([b * bs + i for b in blocks for i in range(bs)][:n], dtype=torch.long)
        if slots.numel() < n or len(toks) < n:
            raise RuntimeError(f"{len(blocks)} blocks of {bs} / {len(toks)} token ids hold fewer than its {n} tokens")
        meta = copy.copy(self._metadata(scheduler_output))
        meta.requests = [ReqMeta(req_id=rid, token_ids=toks, slot_mapping=slots, is_last_prefill=True,
                                 save_spec=SaveSpec(0, True))]  # fmt: skip
        _, attn = self._step_context(None)
        # GLM53_KV_POOLED_PAGES (Stage 12 v3b6, the blk2 kill of 2026-10-29T01:53Z): a store reads each page's pooled
        # keys through a page -> (state slot, column) map, built from the STORED STEP's block-table rows
        # (_page_columns). A preempted request is in no row of this step, so every preempt save failed on its first
        # page ("pooled keys of page 1525: no row of the stored step covers it") and the request recomputed from 0.
        # Its own map: column j = its block j, at its slot (which still holds its state here: the KDA export below
        # reads it too).
        own = None
        if getattr(self, "pooled_pages", False):
            pslot = self._slot_of(rid)
            if pslot is None:
                raise RuntimeError(f"no device state slot known for preempted {rid!r} (its pooled keys)")
            own = {int(b): (int(pslot), j) for j, b in enumerate(blocks[: -(-n // bs)])}
        tl = getattr(self, "_store_meta_tl", None)
        prev = (self.__dict__.get("_pooled_map"), self.__dict__.get("_pooled_map_input"),
                getattr(tl, "pooled_map", None) if tl is not None else None)  # fmt: skip
        with self._lock:
            if own is not None:
                self._pooled_map, self._pooled_map_input = own, None
                if tl is not None:
                    tl.pooled_map = own
            self.connector.bind_connector_metadata(meta)
            try:
                for name in self.views:
                    self.connector.save_kv_layer(name, self.connector_caches[name], attn[name])
                self.connector.wait_for_save()  # (the pages are on the host when it returns: the step may reuse them)
            finally:
                self.connector.clear_connector_metadata()
                if own is not None:
                    self._pooled_map, self._pooled_map_input = prev[0], prev[1]
                    if tl is not None:
                        tl.pooled_map = prev[2]
        # the KDA checkpoint at n (the provenance rule as for decode boundaries: a request whose state started from a
        # foreign hit stores none)
        if self.kda_store is None or getattr(self, "checkpoints_off", False) or not callable(self._kda_export):
            return n
        if rid in _KDA_PROVENANCE:
            self.preempt_ckpt_skipped_provenance = getattr(self, "preempt_ckpt_skipped_provenance", 0) + 1
            return n
        if self.kda_store.has(toks, n):
            return n
        if n != int(rec["n"]):  # snapped: the slot's state is past m; only an earlier export could serve the resume
            self.preempt_snap_no_ckpt = getattr(self, "preempt_snap_no_ckpt", 0) + 1
            return n
        slot = self._slot_of(rid)
        if slot is None:
            raise RuntimeError(f"no device state slot known for preempted {rid!r}")
        last = self.__dict__.get("_last_decode", {}).get(rid)
        if last is not None and last[1]:
            # 14 B2: the slot was verifying -- its committed state after n sits in candidate row n - c - 1 of the
            # shared state (c = that verify's first position), not in the slot's own row (module docstring)
            row = n - int(last[0]) - 1
            if not callable(self._boundary_export) or not 0 <= row <= max(self.spec_k, 0):
                raise RuntimeError(f"verifying slot {slot}: no candidate row for {n} tokens (first position {last[0]})")
            states = self._boundary_export(self._kv_cache_object, int(slot), n_tokens=n, candidate_row=row)
            self.preempt_ckpts_spec = getattr(self, "preempt_ckpts_spec", 0) + 1
        else:
            if last is not None and n != int(last[0]) + 1:  # a plain decode's own row holds the state after c + 1
                raise RuntimeError(f"slot {slot} decoded position {last[0]} last, not the state after {n} tokens")
            states = self._kda_export(self._kv_cache_object, int(slot))
        self.kda_store.save(toks, n, states)
        self.preempt_ckpts = getattr(self, "preempt_ckpts", 0) + 1
        return n

    def _note_decode_rows(self, model_input: Any) -> None:
        """GLM53_PREEMPT_SAVE: each decode row's first position and whether it verifies drafts, kept per request until
        the next step (a preempt save of a verifying slot exports from its candidate row: _preempt_store_one). Taken at
        load, before the step's forward, so the next step's begin_step sees this step's rows even when its save runs
        later (async scheduling)."""
        if not _preempt_save_on():
            return
        last = self.__dict__.setdefault("_last_decode", {})
        rows = getattr(model_input, "row_req_ids", None) or []
        if getattr(model_input, "prompt_lens", None) is not None:  # a prefill build: the slot row holds the state
            for rid in rows:
                last.pop(rid, None)
            return
        if not rows:
            return
        positions = torch.as_tensor(model_input.input_positions)
        speculative = getattr(model_input, "num_valid_drafts", None) is not None
        for row, rid in enumerate(rows):
            if rid is None:
                continue
            first = int(positions[row].reshape(-1)[0])
            if first >= 0:
                last[rid] = (first, speculative)
        if len(last) > 4 * max(len(rows), 64):
            live = set(rows)
            for rid in [r for r in last if r not in live]:
                last.pop(rid, None)

    def load(self, scheduler_output: SchedulerOutput, model_input: Any) -> Any:
        """Before the forward is submitted: start every load and wait for all layers. Returns the model input the
        forward and the step's save must use: ``model_input`` itself, or a copy carrying the step's
        ``kda_snapshot_at`` (GLM53_KDA_CKPT_IN_REPLAY; ``TTModelInput`` is frozen, so it is replaced, never set)."""
        ph = self._phase_begin("load")
        self._note_decode_rows(model_input)
        meta = self._metadata(scheduler_output)
        forward_context, _ = self._step_context(model_input)
        if getattr(self, "pooled_pages", False):
            self._release_loaded_pooled()
        with self._lock:
            self.connector.bind_connector_metadata(meta)
            try:
                self._phase(ph, "prepare")
                self.connector.start_load_kv(forward_context)
                self._phase(ph, "start_load_kv")
                for name in self.views:
                    self.connector.wait_for_layer_load(name)
                self._phase(ph, "wait_layers")
            finally:
                self.connector.clear_connector_metadata()
        if self._generic_sync:
            self._sync_pages(model_input, prefix=True, direction="h2d")
        self._import_kda_checkpoints(scheduler_output, model_input)
        self._phase(ph, "kda_imports")
        self._provenance_step(model_input)  # first: a restore lifts the bar the snapshot marking checks
        model_input = self._mark_kda_snapshots(model_input)
        if getattr(self, "pooled_pages", False):
            self._release_loaded_pooled()  # slices of pages no hit row took (+ the objects their views held)
        self._phase(ph, "provenance_mark")
        if self.hit_dma is not None:
            self.hit_dma.wait()  # the step's forward reads the loaded pages: they must have landed
        self._phase(ph, "hit_dma_wait")
        self._phase_end(ph, model_input)
        return model_input

    # GLM53_KV_PHASE_LOG_MS (default 500; 0 = off; Stage 12, program 2026-10-25T23:33:00Z: tier16 wave 2's one
    # 134-s hit step had ~127 s the event counters do not see): the connector's host time per step by phase -- load
    # (prepare, LMCache start_load_kv = the retrieve + its page-DMA submits, the layer waits, the KDA checkpoint imports
    # split into store load / pooled attach / model import, provenance + snapshot marking, the hit DMA wait) and save
    # (the deferred flush, the store) -- one INFO line when a step's total reaches the threshold; totals in phase_s.
    def _phase_begin(self, kind: str) -> dict | None:
        if not _phase_log_ms():
            return None
        return {"_kind": kind, "_t": time.perf_counter(), "_t0": time.perf_counter()}

    def _phase(self, ph: dict | None, name: str) -> None:
        if ph is None:
            return
        now = time.perf_counter()
        ph[name] = ph.get(name, 0.0) + now - ph["_t"]
        ph["_t"] = now

    def _phase_end(self, ph: dict | None, model_input: Any = None) -> None:
        if ph is None:
            return
        total = time.perf_counter() - ph["_t0"]
        tot = getattr(self, "phase_s", None)
        if tot is None:
            tot = self.phase_s = {}
        parts = {k: v for k, v in ph.items() if not k.startswith("_")}
        sub = getattr(self, "_kda_import_phase", None)
        if sub:
            parts.update({f"kda_{k}": v for k, v in sub.items()})
            self._kda_import_phase = None
        for k, v in parts.items():
            key = f"{ph['_kind']}.{k}"
            tot[key] = tot.get(key, 0.0) + v
        if total * 1000.0 >= _phase_log_ms():
            import json  # noqa: PLC0415

            rows = len(self._prefill_rows(model_input)) if model_input is not None else None
            logger.info("TT KV connector step phases (%s, %.1f ms, prefill rows %s): %s", ph["_kind"], total * 1000.0,
                        rows, json.dumps({k: round(v * 1000.0, 1) for k, v in parts.items()}))  # fmt: skip

    def save(
        self,
        scheduler_output: SchedulerOutput,
        model_input: Any,
        output: ModelRunnerOutput | None = None,
    ) -> None:
        """After the forward completed: save every layer and wait for the saves.

        ``output`` is the step's ``ModelRunnerOutput`` when the caller has it
        (``finish``): a decode row's committed count is the length of its
        sampled ids, which is what decides the decode-time boundary
        checkpoints (module docstring). Chunk stores need nothing from it.
        """
        ph = self._phase_begin("save")
        try:
            self._save_timed(scheduler_output, model_input, output, ph)
        finally:
            self._phase_end(ph, model_input)

    def _save_timed(self, scheduler_output: SchedulerOutput, model_input: Any, output: Any, ph: dict | None) -> None:
        deferring = hasattr(self, "_deferred_save")  # a host built by __init__ (test stubs save at once)
        if (deferring and os.environ.get("GLM53_KV_SAVE_DEFER", "1") == "1"
                and getattr(model_input, "prompt_lens", None) is not None):  # fmt: skip
            # Stage 12 (program 2026-10-23T07:51Z: n29h5e cold TTFT +0.18 ms/token over int27c): a prefill step's
            # LMCache store and KDA checkpoint leave the first-token path -- they run at the next step's begin_step or
            # the scheduler's next lookup (flush_deferred_saves), after this step's tokens were returned and before any
            # forward rewrites what they read (one step deep; the page stores longer with GLM53_KV_STORE_HOLD=1).
            self.flush_deferred_save(scheduler_output)  # a previous pending save (begin_step not called): in order
            self._phase(ph, "flush")
            if self._segment_park(scheduler_output, model_input, output):
                return
            self._deferred_save = (scheduler_output, model_input, output)
            self._remember_heads(model_input)
            self.saves_deferred += 1
            _SAVE_HOSTS.add(self)
            return
        if deferring:
            if os.environ.get("GLM53_KV_STORE_PRIORITY", "1") == "1":
                # Stage 12 (n29h7 M1, program 2026-10-23T19:24:42Z: every interleaved decode step flushed EVERY held
                # store on the engine thread): prefill priority -- a decode step flushes at most one held step, and none
                # while the last held prompt is still mid-prefill; begin_step still flushes everything before a held
                # request finishes or is preempted (its blocks are reused only after that)
                self.flush_deferred_save(scheduler_output, budget=_store_budget())
            else:
                self.flush_deferred_save()  # step order: every pending / held save before this step's
            self._phase(ph, "flush")
        if self._segment_park(scheduler_output, model_input, output):
            return
        for so, mi, out in self._segment_unpark((scheduler_output, model_input, output)):
            self._save_now(so, mi, out)
        self._phase(ph, "save_now")

    def flush_deferred_save(self, next_step: Any = None, budget: int | None = None, drain: bool = True) -> None:
        """the deferred prefill step's save. Its KDA / DSA checkpoints always run now (the next forward changes the slot
        state). GLM53_KV_STORE_HOLD=1 (default 0): its LMCache page store is HELD while the step had a row of a prompt
        that has not emitted its first token (an intermediate chunk), so a chunked prompt's earlier stores leave its
        first-token path too; the held stores run, in step order, with the first step after which every held prompt
        has emitted its first token, before a held request finishes / is preempted (next_step, the coming step's
        scheduler output), at the next connector lookup, at no_forward / shutdown, or past GLM53_KV_STORE_HOLD_MAX
        held steps (default 64). The pages of a live request are not rewritten while it prefills.

        Stage 12 W4 (2026-10-24, doc/ttnn/12-fast-start/w4_host): vLLM's batch queue runs step N + 1 (this begin_step,
        then its save) BEFORE it pops step N's output, so a flush here sits on the first token of a prompt whose last
        chunk ran in N (the stored 1M needle: 24 s). GLM53_KV_STORE_FT_GRACE=1 (default 0): no page store flushes from
        the begin_step that takes such a step until the next begin_step (a held request leaving, past HOLD_MAX and a
        lookup still flush). GLM53_KV_STORE_HOLD_OVER=excess (default all): past HOLD_MAX only the oldest steps above
        it flush (all of them at once took 11.55 s of the needle's engine thread at 526K)."""
        pending, self._deferred_save = getattr(self, "_deferred_save", None), None
        held = getattr(self, "_held_pages", None)
        if held is None:
            held = self._held_pages = []
        for so, mi, out in self._segment_unpark(pending) if pending is not None else ():
            self._remember_heads(mi)
            self._save_state(mi, out)
            held.append((so, mi))
            if os.environ.get("GLM53_KV_STORE_FT_GRACE", "0") == "1" and self._ends_a_prompt(mi):
                self._ft_grace_until = getattr(self, "_begin_seq", 0) + 1
        if held:
            hold_max = int(os.environ.get("GLM53_KV_STORE_HOLD_MAX", "64"))
            trim = 0  # past HOLD_MAX under GLM53_KV_STORE_HOLD_OVER=excess: the oldest steps above it, the rest as usual
            if next_step is not None and os.environ.get("GLM53_KV_STORE_HOLD_OVER", "all") == "excess":
                trim = max(0, len(held) - hold_max)
            view, k = held[trim:], 0
            ids = self._held_ids(view)
            draining = set()  # (not `drain`: the parameter -- a lookup's flush drains the store worker)
            if _leave_drain():
                with _LEAVE_LOCK:
                    draining = ids & set(_LEAVE_PENDING)
            if not self._may_hold(view, next_step) or draining:
                gone = self._gone(next_step) if next_step is not None else set()
                leaving_ids = ids & gone
                leaving = bool(leaving_ids)
                preempted = bool(ids & set(getattr(next_step, "preempted_req_ids", None) or ()))
                over = len(view) > hold_max
                grace = next_step is not None and getattr(self, "_ft_grace_until", 0) > getattr(self, "_begin_seq", 0)
                delayed = not preempted and leaving_ids <= draining  # every leaving held request's blocks are delay-freed
                if next_step is not None and (draining or leaving) and delayed and not over and _leave_drain():
                    # M1: the oldest held steps, a few per engine step (a lookup / no_forward / shutdown flush, next_step
                    # None, still runs everything: a hit must see every store of its prefix)
                    k = min(len(view), _leave_steps())
                    k = self._leave_yield(next_step, k, draining | leaving_ids)
                # a held request leaving (its blocks are reused next) or too many held steps: everything now
                elif leaving or over:
                    k = len(view)
                elif grace:
                    k = 0  # a first token is still in vLLM's batch queue
                elif budget is not None and not self._store_room():
                    k = 0  # GLM53_KV_STORE_NONBLOCK: the worker is at its cap -- a later step, never a blocking submit
                else:
                    k = len(view) if budget is None else max(0, min(int(budget), len(view)))
            k += trim
            self._held_pages = held[k:]  # budget: the oldest k held steps now, the rest at the next steps
            for so, mi in held[:k]:
                self._save_pages(so, mi)
        if drain and next_step is None and getattr(self, "store_worker", None) is not None and not getattr(
                self, "checkpoints_off", False):  # fmt: skip
            self.store_worker.drain()  # a lookup that may hit sees every queued store
        if not getattr(self, "_held_pages", None):
            self.__dict__.pop("_leave_drain_t0", None)  # (the M1 grace clock: the next leave starts its own)
        self._prune_heads()
        if not getattr(self, "_held_pages", None) and not self._stores_in_flight():
            _SAVE_HOSTS.discard(self)

    def _inflight_rids(self, rids, delta: int) -> None:
        """M1: per-request count of store-worker jobs not finished (+1 at submit, -1 when the worker is done)"""
        with _LEAVE_LOCK:
            c = self.__dict__.setdefault("_inflight_rid_counts", {})
            for r in rids:
                n = c.get(r, 0) + delta
                if n > 0:
                    c[r] = n
                else:
                    c.pop(r, None)

    def _job_done(self, job: dict) -> None:
        rids = job.get("rids")
        if rids:
            self._inflight_rids(rids, -1)

    def _leave_report(self, finished_sending):
        """M1: finished_sending + the delay-freed finished requests with no step held, pending or queued any more (an
        inner-connector delay is reported only once the inner connector reported it as well)"""
        with _LEAVE_LOCK:
            if not _LEAVE_PENDING:
                return finished_sending
            fs = set(finished_sending or ())
            held = self._held_ids(getattr(self, "_held_pages", None) or [])
            pend = getattr(self, "_deferred_save", None)
            if pend is not None:
                held |= set(getattr(pend[1], "row_req_ids", None) or ())
            busy = set(self.__dict__.get("_inflight_rid_counts", {}))
            for rid, st in list(_LEAVE_PENDING.items()):
                if st.get("inner"):
                    if rid in fs:
                        st["inner_done"] = True
                        fs.discard(rid)  # held back until this host's part has landed too
                    if not st.get("inner_done"):
                        continue
                if rid in held or rid in busy:
                    continue
                fs.add(rid)
                del _LEAVE_PENDING[rid]
            return fs or None

    def _stores_in_flight(self) -> bool:
        """page stores handed to the store worker and not finished: the host stays in _SAVE_HOSTS until they land, so
        the next connector lookup still drains them (Stage 12 F2c, 2026-10-25: a held request's 'leaving' flush hands
        its remaining steps to the worker WITHOUT a drain; the host dropped out of the registry with them in flight
        and a repeat of the prompt looked up LMCache before they landed -- cf3h-i mock, cached 0 instead of 16,000)"""
        busy = getattr(getattr(self, "store_worker", None), "busy", None)
        return bool(busy()) if callable(busy) else False

    def _store_room(self) -> bool:
        """GLM53_KV_STORE_NONBLOCK=1 (Stage 12 W4 F2c, default 0): a budgeted flush (one held step per engine step) waits
        for a later step while the store worker is at its in-flight cap instead of blocking the engine thread in submit
        (each such step cost a decode ~0.3 s while LMCache drained at ~0.3 GB/s); True when off or there is room"""
        worker = getattr(self, "store_worker", None)
        if os.environ.get("GLM53_KV_STORE_NONBLOCK", "0") != "1" or worker is None:
            return True
        return worker.has_room()

    def _segment_park(self, scheduler_output: SchedulerOutput, model_input: Any, output: Any) -> bool:
        """Segmented prefill traces (doc/ttnn/14b-persistent-stack/segtrace/runner_design.md section 4, item 12). The
        first step of a segmented chunk (the scheduler's set_tt_segmented stamp, {req_id: S > 1}) ran only segment 0 of
        the chunk's layers: the chunk's DSA pages past it and its KDA state (stored at segment S - 1's state_out) are
        not written yet. So the step's save (its page store, KDA checkpoint and in-replay snapshot) is PARKED until the
        request's next prefill row, the release step (the chunk's 1-token tail, segment S - 1), and runs right before
        that step's own save (_segment_unpark). A request that leaves or is preempted during the hold drops it: an
        unfinished chunk is never stored. Returns True when parked. The stamped step's prefill part must be that one
        chunk: another prefill row's checkpoint would be exported after it decoded, so that is refused."""
        from vllm_tt_plugin.scheduler import get_tt_segmented  # noqa: PLC0415  (the scheduler imports this module)

        seg = get_tt_segmented(scheduler_output)
        if not seg:
            return False
        rows = self._prefill_rows(model_input)
        held = [rid for _row, rid, _s, _e in rows if int(seg.get(rid, 1)) > 1]
        if not held:
            return False
        parked = self.__dict__.setdefault("_segment_parked", {})
        if len(rows) != 1 or held[0] in parked:
            raise RuntimeError(
                f"a segmented chunk's step must carry that chunk as its only prefill row, once per release: rows "
                f"{[rid for _r, rid, _s, _e in rows]}, segmented {held}, already parked {sorted(parked)}")
        parked[held[0]] = (scheduler_output, model_input, output)
        self.segment_saves_parked = getattr(self, "segment_saves_parked", 0) + 1
        return True

    def _segment_unpark(self, item: tuple) -> list:
        """[the parked save of the segmented chunk this prefill save releases, if any] + [item]: in step order"""
        parked = getattr(self, "_segment_parked", None)
        if not parked:
            return [item]
        out = []
        for _row, rid, _s, _e in self._prefill_rows(item[1]):
            p = parked.pop(rid, None)
            if p is not None:
                out.append(self._segment_slot_now(p, item[1], rid))
                self.segment_saves_released = getattr(self, "segment_saves_released", 0) + 1
        return out + [item]

    def _segment_slot_now(self, parked: tuple, release_input: Any, rid: str) -> tuple:
        """the parked save reads the request's state slot of the RELEASE step: the runner's mixed step names each
        prefill row's slot as of that step (a decode step's gather may move slots), and the hold pins the slot, so a
        difference is a failed pin -- counted segment_slots_moved, logged, and the save follows the release's slot"""
        so, mi, out = parked
        now, then = getattr(release_input, "prefill_empty_slots", None), getattr(mi, "prefill_empty_slots", None)
        if now is None or then is None:
            return parked
        i, j = list(release_input.row_req_ids).index(rid), list(mi.row_req_ids).index(rid)
        if int(now[i]) == int(then[j]):
            return parked
        self.segment_slots_moved = getattr(self, "segment_slots_moved", 0) + 1
        logger.warning("segmented chunk of %s: its state slot moved from %d to %d inside the hold (a failed pin); "
                       "the chunk's save reads slot %d", rid, int(then[j]), int(now[i]), int(now[i]))  # fmt: skip
        slots = [int(x) for x in then]
        slots[j] = int(now[i])
        return so, replace(mi, prefill_empty_slots=slots), out

    def _segment_drop(self, ids: set) -> None:
        """a request leaving / preempted with its segmented chunk parked (before the release): the save is dropped"""
        parked = getattr(self, "_segment_parked", None)
        self._segment_dropped = dropped = {rid for rid in ids if parked and parked.pop(rid, None) is not None}
        if dropped:
            self.segment_saves_dropped = getattr(self, "segment_saves_dropped", 0) + len(dropped)
            logger.warning("segmented chunk saves dropped (the request left inside its hold): %s", sorted(dropped))

    @staticmethod
    def _held_ids(held: list) -> set:
        return {rid for _so, mi in held for rid in (getattr(mi, "row_req_ids", None) or [])}

    @staticmethod
    def _gone(next_step: Any) -> set:
        return set(getattr(next_step, "finished_req_ids", None) or ()) | set(
            getattr(next_step, "preempted_req_ids", None) or ())

    def _may_hold(self, held: list, next_step: Any) -> bool:
        if os.environ.get("GLM53_KV_STORE_HOLD", "0") != "1" or next_step is None:
            return False  # off, or a lookup / no_forward / shutdown / step-order flush: everything now
        if len(held) > int(os.environ.get("GLM53_KV_STORE_HOLD_MAX", "64")):
            return False
        gone = set(getattr(next_step, "finished_req_ids", None) or ()) | set(
            getattr(next_step, "preempted_req_ids", None) or ())
        ids = {rid for _so, mi in held for rid in (getattr(mi, "row_req_ids", None) or [])}
        if gone & ids:
            return False  # a held request is leaving: its blocks may be reused by the next forward
        # hold while the LAST held step still had an intermediate chunk (a prompt without its first token yet)
        _so, last = held[-1]
        for _row, req_id, _start, end in self._prefill_rows(last):
            n = _host_history_len(self, req_id)
            if n is not None and end < n:
                return True
        return False

    def _history_len(self, req_id: str) -> int | None:
        return _host_history_len(self, req_id)

    def _lmcache_chunk(self) -> int:
        impl = getattr(self.connector, "_lmcache_engine", None)
        n = int(getattr(impl, "_lmcache_chunk_size", 0) or 0)
        if n:
            return n
        views = getattr(self, "views", None) or {}
        return int(getattr(next(iter(views.values()), None), "block_size", 0) or 256)

    def _runner_head(self, req_id: str, n: int) -> list[int] | None:
        f = getattr(self, "_head_of_fn", None)
        if callable(f):
            h = f(req_id, n)
            return None if h is None else list(h)
        history = self._tokens_of(req_id)
        return None if history is None else list(history[:n])

    def _head_of(self, req_id: str, n: int) -> list[int] | None:
        """the request's first n history tokens: the runner's, else the head kept from its stored step
        (_remember_heads: a held step / a queued job outliving its request under GLM53_KV_LEAVE_DRAIN)"""
        h = self._runner_head(req_id, n)
        if h is None:
            kept = self.__dict__.get("_rid_heads", {}).get(req_id)
            if kept is not None:
                h = list(kept[:n])
        return h

    def _remember_heads(self, model_input: Any) -> None:
        """a stored step's request heads, taken while the runner still knows its requests. Under GLM53_KV_LEAVE_DRAIN a
        held step (and a store-worker job submitted after its request left) outlives the request in the runner; with
        an unknown head serves_prefix matched it against EVERY prompt, so the next request's scheduler lookup flushed
        and drained the gone request's stores before LMCache's lookup (k3 ladder, 2026-10-26 22:39Z: +0.22 / +0.51 s
        on the cold rung after a stored one, whose prompts share no first chunk). Kept only while a held step, the
        pending save or a live request refers to it (pruned at every flush)."""
        if not (_leave_drain() or os.environ.get("GLM53_KV_LOOKUP_FLUSH", "all") == "prefix"
                or os.environ.get("GLM53_KV_LOOKUP_DEFER", "0") == "1"):  # fmt: skip
            return
        n = self._lmcache_chunk()
        kept = self.__dict__.setdefault("_rid_heads", {})
        for rid in getattr(model_input, "row_req_ids", None) or ():
            if rid is not None and rid not in kept:
                h = self._runner_head(rid, n)
                if h is not None:
                    kept[rid] = tuple(h)

    def _prune_heads(self) -> None:
        kept = self.__dict__.get("_rid_heads")
        if not kept:
            return
        refs = self._held_ids(getattr(self, "_held_pages", None) or [])
        pending = getattr(self, "_deferred_save", None)
        if pending is not None:
            refs |= set(getattr(pending[1], "row_req_ids", None) or ())
        for rid in [r for r in kept if r not in refs and self._runner_head(r, 1) is None]:
            kept.pop(rid, None)

    def _leave_yield(self, next_step: Any, k: int, ids: set) -> int:
        """the M1 drain's held steps for this engine step: k, or 0 while it yields to admission (_leave_yield_steps,
        _leave_grace_s: the clock starts when a request first drains); counters leave_yields_admit / _grace"""
        if not k:
            return k
        now = time.monotonic()
        clock = self.__dict__.get("_leave_drain_t0")
        if clock is None or not set(ids) <= clock[0]:  # a newly leaving request: its own grace
            clock = self._leave_drain_t0 = (frozenset(ids) | (clock[0] if clock else frozenset()), now)
        t0 = clock[1]
        if getattr(next_step, "scheduled_new_reqs", None):
            n = self.__dict__.get("_leave_admit_yields", 0)
            if n < _leave_yield_steps():
                self._leave_admit_yields = n + 1
                self.leave_yields_admit = getattr(self, "leave_yields_admit", 0) + 1
                return 0
        elif getattr(next_step, "total_num_scheduled_tokens", None) == 0 and now - t0 < _leave_grace_s():
            self.leave_yields_grace = getattr(self, "leave_yields_grace", 0) + 1
            return 0
        self._leave_admit_yields = 0
        return k

    def serves_prefix(self, tokens: Sequence[int]) -> bool:
        """GLM53_KV_LOOKUP_FLUSH=prefix: True when a held, deferred or queued store of this host belongs to a request
        whose first LMCache chunk agrees with `tokens` (or whose history is unknown: flush, as before)"""
        n = self._lmcache_chunk()
        want = list(tokens[:n])
        ids = set(self._held_ids(getattr(self, "_held_pages", None) or []))
        pending = getattr(self, "_deferred_save", None)
        if pending is not None:
            ids |= set(getattr(pending[1], "row_req_ids", None) or [])
        heads = [self._head_of(rid, n) for rid in ids if rid is not None]
        with self._queued_lock():
            heads += [None if h is None else list(h) for h in getattr(self, "_queued_heads", {})]
        for h in heads:
            if h is None:
                return True
            m = min(len(h), len(want))
            if m and h[:m] == want[:m]:
                return True
        return False

    def writes_into_prefix(self, tokens: Sequence[int]) -> bool:
        """GLM53_KV_LOOKUP_ADMIT_FIRST: True when a queued store job writes a chunk inside the common prefix of its
        request and `tokens` (or a job's history / span is unknown, or a held / deferred store is still on this host:
        the old rule). See lookup_admit_first."""
        if getattr(self, "_held_pages", None) or getattr(self, "_deferred_save", None) is not None:
            return self.serves_prefix(tokens)  # (not handed to the worker: no span recorded)
        n = len(tokens)
        with self._queued_lock():
            spans = [e for job in (getattr(self, "_queued_spans", None) or {}).values() for e in job]
        for hist, start in spans:
            if hist is None or start is None:
                return True
            end = min(n, len(hist), start + self._lmcache_chunk())
            if end > start and list(hist[:end]) == list(tokens[:end]):
                return True
        return False

    def _queue_spans(self, meta: Any, delta: int, job: dict) -> None:
        """GLM53_KV_LOOKUP_ADMIT_FIRST: a queued job's (request history, first written chunk) entries; +1 at submit,
        -1 when it is done"""
        if not lookup_admit_first():
            return
        with self._queued_lock():
            q = self.__dict__.setdefault("_queued_spans", {})
            if delta < 0:
                q.pop(id(job), None)
                return
            chunk = self._lmcache_chunk()
            ents = []
            for req in getattr(meta, "requests", None) or []:
                spec = getattr(req, "save_spec", None)
                if spec is not None and not getattr(spec, "can_save", True):
                    continue
                rid = getattr(req, "req_id", None)
                hist = self._tokens_of(rid) if rid is not None else None
                if hist is None:
                    hist = getattr(req, "token_ids", None)
                    hist = list(hist.tolist() if hasattr(hist, "tolist") else hist) if hist is not None else None
                start = int(getattr(spec, "skip_leading_tokens", 0)) // chunk * chunk if spec is not None else None
                ents.append((hist, start))
            q[id(job)] = ents

    def _queued_lock(self) -> threading.Lock:
        lock = self.__dict__.get("_queued_heads_lock")
        if lock is None:
            lock = self.__dict__.setdefault("_queued_heads_lock", threading.Lock())
        return lock

    def _queue_heads(self, meta: Any, delta: int, heads: list | None = None) -> list:
        """the store worker's queued jobs by request head (GLM53_KV_LOOKUP_FLUSH=prefix): +1 at submit, -1 when done"""
        if heads is None:
            n = self._lmcache_chunk()
            heads = []
            for req in getattr(meta, "requests", None) or []:
                rid = getattr(req, "req_id", None)
                h = self._head_of(rid, n) if rid is not None else None
                heads.append(None if h is None else tuple(h))
        with self._queued_lock():
            q = self.__dict__.setdefault("_queued_heads", {})
            for h in heads:  # None: an unknown history (serves_prefix treats it as a match)
                q[h] = q.get(h, 0) + delta
                if q[h] <= 0:
                    q.pop(h, None)
        return heads

    def _ends_a_prompt(self, model_input: Any) -> bool:
        """a prefill step with a row that ran a prompt's last chunk (its first token is sampled in that step); the
        history length is O(1) (_may_hold's rule: a row ending before the history is an intermediate chunk)"""
        for _row, req_id, _start, end in self._prefill_rows(model_input):
            n = _host_history_len(self, req_id)
            if n is None or end >= n:
                return True
        return False

    def _save_now(
        self,
        scheduler_output: SchedulerOutput,
        model_input: Any,
        output: ModelRunnerOutput | None = None,
    ) -> None:
        self._save_state(model_input, output)
        self._save_pages(scheduler_output, model_input)
        if _LMC_WRITERS:  # a prefill whose last chunk's pages and state are now saved stops being a writer
            for _row, req_id, _start, end in self._prefill_rows(model_input):
                n = _host_history_len(self, req_id)
                if n is not None and end >= n:
                    with _PROVENANCE_LOCK:
                        _LMC_WRITERS.pop(req_id, None)

    # -- the provenance restore (GLM53_KDA_CKPT_PROVENANCE=restore, see _KDA_RESTORE) ------------------------------

    def _provenance_step(self, model_input: Any) -> None:
        """After this step's LMCache loads, before its forward and its saves: (1) the loads of this step's rows ran, so
        no guard keeps them; (2) this step's prefill rows are writers until their last chunk is saved; (3) a restore
        candidate starting its prefill here replaces its span [w, m), or keeps its bar."""
        if not (_KDA_RESTORE or _LMC_LOADS or kda_ckpt_provenance_mode() == "restore"):
            return
        rows = self._prefill_rows(model_input)
        if not rows:
            return
        with _PROVENANCE_LOCK:
            for _row, req_id, _start, _end in rows:
                _LMC_LOADS.pop(req_id, None)
                if req_id not in _LMC_WRITERS:  # the tokens are read only when a restore needs them (O(1) per step)
                    _LMC_WRITERS[req_id] = (lambda rid=req_id: self._tokens_of(rid))
            todo = [(req_id, _KDA_RESTORE.pop(req_id)) for _row, req_id, _s, _e in rows if req_id in _KDA_RESTORE]
        for req_id, (w, m, tokens) in todo:
            self._restore_span(req_id, int(w), int(m), tokens)

    def _restore_span(self, req_id: str, w: int, m: int, tokens: Sequence[int]) -> bool:
        """Remove LMCache's chunk keys for [w, m) of ``tokens`` from every tier, unless a guard holds; True = the
        request's bar is lifted (its own pages replace the span, its checkpoints are exported as a first-time
        request's)."""
        chunk = _lmcache_chunk_tokens()
        impl = getattr(self.connector, "_lmcache_engine", None)
        engine = getattr(impl, "lmcache_engine", None)
        backends = getattr(getattr(engine, "storage_manager", None), "storage_backends", None)
        db = getattr(engine, "token_database", None)
        with _PROVENANCE_LOCK:
            loads = [s for s, (t, win) in _LMC_LOADS.items() if s != req_id and int(win) > w
                     and _shares_chunk_past(tokens, t, w, chunk)]  # fmt: skip
            writers = [(s, f) for s, f in _LMC_WRITERS.items() if s != req_id]
        busy = []
        for s, f in writers:
            t = f() if callable(f) else f
            if t is None or _shares_chunk_past(tokens, t, w, chunk):  # unknown tokens: assume it shares the span
                busy.append(s)
        if loads:
            self.kda_provenance_restore_skipped_load += 1
            return self._provenance_event(req_id, "skipped_load", w, m, loads)
        if busy:
            self.kda_provenance_restore_skipped_writer += 1
            return self._provenance_event(req_id, "skipped_writer", w, m, busy)
        if not isinstance(backends, dict) or db is None:
            self.kda_provenance_restore_failed += 1
            return self._provenance_event(req_id, "failed", w, m, ["no LMCache engine / token database"])
        try:
            keys = [key for start, _end, key in db.process_tokens(tokens=list(tokens[:m])) if int(start) >= w]
            left = lmcache_remove_keys(backends, keys)
        except Exception as e:  # noqa: BLE001 -- any failure keeps the bar
            self.kda_provenance_restore_failed += 1
            return self._provenance_event(req_id, "failed", w, m, [f"{type(e).__name__}: {e}"[:200]])
        if left:
            self.kda_provenance_restore_failed += 1
            return self._provenance_event(req_id, "failed", w, m, [f"{len(left)} of {len(keys)} keys still held"])
        _KDA_PROVENANCE.pop(req_id, None)
        n = min(m, w + chunk)
        with _PROVENANCE_LOCK:
            _RESTORED_SPANS.append((w, n, KDACheckpointStore.key(list(tokens[:n]), n)))
        self.kda_provenance_restored += 1
        self.kda_provenance_restored_tokens += m - w
        return self._provenance_event(req_id, "restored", w, m, [f"{len(keys)} chunk keys removed"], ok=True)

    @staticmethod
    def _on_replaced_span(tokens: Sequence[int], end: int) -> tuple[int, int] | None:
        """the replaced span (w, n) whose prefix ``tokens`` shares, with ``end`` past its w; None when there is none"""
        with _PROVENANCE_LOCK:
            spans = list(_RESTORED_SPANS)
        for w, n, key in spans:
            if end > w and len(tokens) >= n and KDACheckpointStore.key(list(tokens[:n]), n) == key:
                return w, n
        return None

    def _provenance_event(self, req_id: str, what: str, w: int, m: int, detail: list, ok: bool = False) -> bool:
        """One INFO line per restore decision with every counter (the window row reads the last one)."""
        import json  # noqa: PLC0415

        counters = {k: getattr(self, k, 0) for k in ("kda_provenance_restored", "kda_provenance_restored_tokens",
                    "kda_provenance_restore_skipped_load", "kda_provenance_restore_skipped_writer",
                    "kda_provenance_restore_failed", "kda_ckpt_skipped_provenance",
                    "kda_provenance_mixed_imports")}  # fmt: skip
        logger.info("TT KV connector provenance %s: request %s span [%d, %d) %s; counters %s", what, req_id, w, m,
                    "; ".join(str(d) for d in detail)[:300], json.dumps(counters))  # fmt: skip
        return ok

    def _save_state(self, model_input: Any, output: ModelRunnerOutput | None = None) -> None:
        mark_engine_thread()  # the engine-thread rule: the thread that steps the model is the only device thread
        drain_engine_todo()
        if self.kda_store is not None and callable(self._kda_export) and not getattr(self, "checkpoints_off", False):
            self._export_kda_snapshots(model_input)
        self._export_kda_checkpoints(model_input)
        self._export_decode_boundary_checkpoints(model_input, output)

    def _save_pages(self, scheduler_output: SchedulerOutput, model_input: Any) -> None:
        meta = self._metadata(scheduler_output)
        worker = getattr(self, "store_worker", None)
        if worker is not None and getattr(self.tt_gpu_connector, "zero_copy_store", lambda: False)():
            if self._zc_worker_ok():
                pages = self._store_page_ids(meta)
                if pages:  # store path P3: the whole zero-copy store (its page-DMA wait included) on the worker
                    self._submit_zc_job(meta, model_input, len(pages))
                    return
            worker = None  # GLM53_KV_STORE_ZEROCOPY: the page-DMA SAVEs into the objects here, while the pages are valid
        if worker is not None and not self._generic_sync:
            pages = self._store_page_ids(meta)
            if pages:  # the engine thread reads the pages (device -> host); the worker does the rest of the store
                if getattr(self, "pooled_pages", False):  # the pooled source of this stored step (as the inline path)
                    self._pooled_map, self._pooled_map_input = None, model_input
                if getattr(self.tt_gpu_connector, "kvcaches", False) is None:  # a connector whose host buffers are unbound
                    # the process's first store: LMCache binds the host buffers only inside its own to_gpu / from_gpu,
                    # which this path reaches after the read; bind the buffers LMCache was registered with
                    self.tt_gpu_connector.initialize_kvcaches_ptr(kvcaches=self.connector_caches)
                job = self.tt_gpu_connector.prefetch_store_pages(pages)
                if (os.environ.get("GLM53_KV_LOOKUP_FLUSH", "all") == "prefix"
                        or os.environ.get("GLM53_KV_LOOKUP_DEFER", "0") == "1"):  # fmt: skip
                    # the queued job's request heads: the prefix filter / the lookup deferral (2026-10-26 j3cngx2 mock:
                    # under LOOKUP_FLUSH=all the deferral saw no queued heads, did not wait, a 32K repeat looked up a
                    # 10,240-token partial store and aligned it to a cold prefill)
                    job["heads"] = self._queue_heads(meta, +1)
                    self._queue_spans(meta, +1, job)
                if _leave_drain():  # M1: the requests whose blocks wait for this job
                    job["rids"] = set(getattr(model_input, "row_req_ids", None) or ())
                    self._inflight_rids(job["rids"], +1)
                worker.submit(meta, job)
                return
        _, attn = self._step_context(model_input)
        if self._generic_sync:
            self._sync_pages(model_input, prefix=False, direction="d2h")
        if getattr(self, "pooled_pages", False):
            # built on the first store's demand (_pooled_source): a step that stores nothing (a decode step without
            # save_decode_cache) skips the O(columns) walk of every row (Stage 12, O(1) per step in the context)
            self._pooled_map, self._pooled_map_input = None, model_input
        with self._lock:
            self.connector.bind_connector_metadata(meta)
            try:
                for name in self.views:
                    self.connector.save_kv_layer(
                        name, self.connector_caches[name], attn[name]
                    )
                self.connector.wait_for_save()
            finally:
                self.connector.clear_connector_metadata()
                self._pooled_map, self._pooled_map_input = {}, None

    def _zc_worker_ok(self) -> bool:
        """GLM53_KV_STORE_ZC_WORKER=1 (store path P3, program 2026-10-26T23:48:49Z; default 0): zero-copy stores run
        whole on the store worker -- the engine thread no longer waits for their page-DMA round trip. Only where the
        stored pages stay valid until the worker's DMA has read them: GLM53_KV_LEAVE_DRAIN (a finished request's blocks
        are freed only after its in-flight jobs; a preempted one's are drained in begin_step), a thread-safe page-DMA
        (LatentHitDma.thread_safe: one transaction at a time), and per-block pooled keys (GLM53_DSA_PAGED_POOLS, held
        with the blocks; a slot's unit pages could be reassigned before the worker reads them). Else inline, logged
        once."""
        ok = self.__dict__.get("_zc_worker_decision")
        if ok is None:
            why = []
            if os.environ.get("GLM53_KV_STORE_ZC_WORKER", "0") != "1":
                why.append("off")
            if not _leave_drain():
                why.append("GLM53_KV_LEAVE_DRAIN=0")
            if not getattr(getattr(self, "hit_dma", None), "thread_safe", False):
                why.append("the page-DMA is not thread-safe")
            if getattr(self, "pooled_pages", False) and os.environ.get("GLM53_DSA_PAGED_POOLS", "0") != "1":
                why.append("pooled keys in slot units (no GLM53_DSA_PAGED_POOLS)")
            ok = not why
            self.__dict__["_zc_worker_decision"] = ok
            if os.environ.get("GLM53_KV_STORE_ZC_WORKER", "0") == "1":
                logger.info("KV connector: zero-copy stores on the store worker (GLM53_KV_STORE_ZC_WORKER): %s",
                            "on" if ok else "OFF (" + ", ".join(why) + ")")  # fmt: skip
        return ok

    def _submit_zc_job(self, meta: Any, model_input: Any, n_pages: int) -> None:
        """the engine thread's share of a P3 store: the pooled map of the stored step (its rows' pages -> (slot,
        column), the step's own) and the M1 / lookup bookkeeping; the worker runs LMCache's store and its page-DMA"""
        job = {"zc": True, "nbytes": int(n_pages) * len(self.views) * 69632}
        if getattr(self, "pooled_pages", False):
            job["pooled_map"] = self._page_columns(model_input)
        if (os.environ.get("GLM53_KV_LOOKUP_FLUSH", "all") == "prefix"
                or os.environ.get("GLM53_KV_LOOKUP_DEFER", "0") == "1"):  # fmt: skip
            job["heads"] = self._queue_heads(meta, +1)
            self._queue_spans(meta, +1, job)
        job["rids"] = set(getattr(model_input, "row_req_ids", None) or ())
        self._inflight_rids(job["rids"], +1)
        self.zc_worker_jobs = getattr(self, "zc_worker_jobs", 0) + 1
        self.store_worker.submit(meta, job)

    def _zc_drain_for(self, rids) -> None:
        """P3: a request preempted this step frees its blocks before the step's forward; a worker job still reading
        them drains first (its DMA completes before anything rewrites the pages)"""
        if not rids or getattr(self, "store_worker", None) is None or not self.__dict__.get("_zc_worker_decision"):
            return
        with _LEAVE_LOCK:
            c = self.__dict__.get("_inflight_rid_counts") or {}
            busy = any(c.get(r, 0) > 0 for r in rids)
        if busy:
            t0 = time.perf_counter()
            self.store_worker.drain()
            self.zc_preempt_drains = getattr(self, "zc_preempt_drains", 0) + 1
            self.zc_preempt_drain_s = getattr(self, "zc_preempt_drain_s", 0.0) + time.perf_counter() - t0

    def _enable_store_worker(self, connector: Any) -> None:
        """GLM53_KV_STORE_WORKER=1: see __init__ (the store worker + the thread-local metadata override)"""
        self.store_worker = None
        self._store_meta_tl = threading.local()
        # (GLM53_KV_POOLED_PAGES too since Stage 12 cf3h-c, program 2026-10-24T02:20:57Z: prefetch_store_pages reads
        # the pages' pooled keys through the pooled source on the engine thread; before, the worker was refused there)
        if (os.environ.get("GLM53_KV_STORE_WORKER", "0") == "1" and self.tt_gpu_connector is not None
                and callable(getattr(self.tt_gpu_connector, "prefetch_store_pages", None))):  # fmt: skip
            impl = getattr(connector, "_lmcache_engine", None)
            parent = getattr(impl, "_parent", None) or connector
            orig = getattr(parent, "_get_connector_metadata", None)
            if callable(orig):
                tl = self._store_meta_tl

                def _metadata_for_this_thread():
                    m = getattr(tl, "meta", None)
                    return m if m is not None else orig()

                parent._get_connector_metadata = _metadata_for_this_thread
                cap = int(float(os.environ.get("GLM53_KV_STORE_WORKER_MAX_GB", "8")) * 2**30)
                self.store_worker = _StoreWorker(self, cap)
                _WORKER_HOSTS.add(self)
                logger.info("KV connector: LMCache page stores on a store worker thread (GLM53_KV_STORE_WORKER), "
                            "%.1f GB in flight at most", cap / 2**30)  # fmt: skip

    def _store_page_ids(self, meta: Any) -> list[int]:
        """the pages LMCache's store of this step may read: every saving request's slot mapping past its saved prefix
        (aligned down to the LMCache chunk, as wait_for_save aligns it; a superset is harmless)"""
        bs = next(iter(self.views.values())).block_size
        impl = getattr(self.connector, "_lmcache_engine", None)
        chunk = int(getattr(impl, "_lmcache_chunk_size", 0) or bs)
        out: list[int] = []
        for req in getattr(meta, "requests", None) or []:
            spec = getattr(req, "save_spec", None)
            sm = getattr(req, "slot_mapping", None)
            if spec is None or not getattr(spec, "can_save", False) or sm is None:
                continue
            skip = int(getattr(spec, "skip_leading_tokens", 0)) // chunk * chunk
            sm = torch.as_tensor(sm)[skip:]
            sm = sm[sm >= 0]
            if sm.numel():
                out.extend(torch.unique(sm // bs).tolist())
        return list(dict.fromkeys(int(p) for p in out))

    def _run_store_job(self, meta: Any, job: dict) -> None:
        """on the store worker: LMCache's store loop for one step's metadata, its GPU connector reading the snapshot
        (a P3 zero-copy job: reading the pages itself through the thread-safe page-DMA, the step's pooled map).
        GLM53_KV_DEVICE_THREAD_AUDIT=1 (default 0; the engine-thread rule): every ttnn call site the job makes off the
        engine thread is logged once with its stack (never refused: the async-read completion's event wait is one)."""
        if os.environ.get("GLM53_KV_DEVICE_THREAD_AUDIT", "0") == "1" and off_engine_thread():
            with _host_only("store job", refuse=False):
                return self._run_store_job_body(meta, job)
        return self._run_store_job_body(meta, job)

    def _run_store_job_body(self, meta: Any, job: dict) -> None:
        yield_to_state_saves(self)  # GLM53_KV_STORE_YIELD_TO_STATE_SAVE: a state save in flight goes first
        if job.get("zc"):
            tl = self._store_meta_tl
            tl.meta, tl.pooled_map = meta, job.get("pooled_map")
            try:
                self.connector.wait_for_save()
            finally:
                tl.meta, tl.pooled_map = None, None
                if job.get("heads") is not None:
                    self._queue_heads(meta, -1, job["heads"])
                    self._queue_spans(meta, -1, job)
            return
        complete = getattr(self.tt_gpu_connector, "complete_store_job", None)
        if callable(complete):  # the page reads the engine thread only enqueued (GLM53_KV_STORE_ASYNC_READ)
            complete(job)
        tl = self._store_meta_tl
        tl.meta = meta
        self.tt_gpu_connector.set_store_job(job)
        try:
            self.connector.wait_for_save()
        finally:
            self.tt_gpu_connector.set_store_job(None)
            tl.meta = None
            if job.get("heads") is not None:
                self._queue_heads(meta, -1, job["heads"])
                self._queue_spans(meta, -1, job)

    # -- GLM53_KV_POOLED_PAGES: the pooled keys in the LMCache pages -----------

    def _page_columns(self, model_input: Any) -> dict[int, tuple[int, int]]:
        """{block id: (vLLM state slot, column)} of every page a row of this step covers (columns [0, ceil(end / page))
        of its block-table row: the pages its stores may hold)"""
        row_req_ids = getattr(model_input, "row_req_ids", None) or []
        tables = getattr(model_input, "block_tables_per_group", None)
        view = next(iter(self.views.values()))
        table = torch.as_tensor(tables[self._group_for(view)] if tables else model_input.block_tables)
        bs = view.block_size
        out: dict[int, tuple[int, int]] = {}
        for row, req_id in enumerate(row_req_ids):
            positions = _row_positions(model_input, row)
            if req_id is None or not positions:
                continue
            try:
                slot = self._slot_for(req_id, row, model_input)
            except RuntimeError:
                continue
            ncols = min(-(-(max(positions) + 1) // bs), int(table.shape[1]))
            for j, b in enumerate(table[row, :ncols].tolist()):
                if int(b) > 0:
                    out.setdefault(int(b), (int(slot), j))
        return out

    def _pooled_pairs(self, page_ids: list[int]) -> list:
        """the stored step's pages -> their (slot, column) (on the store worker: the map its P3 job carries)"""
        tl = getattr(self, "_store_meta_tl", None)
        pmap = getattr(tl, "pooled_map", None) if tl is not None else None
        if pmap is None:
            if getattr(self, "_pooled_map", None) is None:
                mi = getattr(self, "_pooled_map_input", None)
                self._pooled_map = self._page_columns(mi) if mi is not None else {}
            pmap = self._pooled_map
        pairs = []
        for p in page_ids:
            sc = pmap.get(int(p))
            if sc is None:
                raise RuntimeError(f"pooled keys of page {p}: no row of the stored step covers it")
            pairs.append(sc)
        return pairs

    def _pooled_source(self, page_ids: list[int]) -> Any:
        """the TT GPU connector's pooled source at a store: the pages' (slot, column) -> the model's page-DMA read"""
        pairs = TTKVConnectorHost._pooled_pairs(self, page_ids)  # (host tests call this on stand-ins of the host)
        if getattr(self, "_read_pooled_takes_pages", None) is None:
            try:
                self._read_pooled_takes_pages = "pages" in inspect.signature(self._read_pooled).parameters
            except (TypeError, ValueError):
                self._read_pooled_takes_pages = False
        if self._read_pooled_takes_pages:  # GLM53_DSA_PAGED_POOLS addresses the pages' blocks
            return self._read_pooled(pairs, pages=[int(p) for p in page_ids])
        return self._read_pooled(pairs)

    def _pooled_plan(self, page_ids: list[int]) -> Any:
        """store path P2: the pages' pooled reads as the model's (targets, finish) (kv_pooled_store_plan), for the TT
        GPU connector's one-wait zero-copy store"""
        return self._read_pooled_plan(TTKVConnectorHost._pooled_pairs(self, page_ids), pages=[int(p) for p in page_ids])

    def _release_loaded_pooled(self) -> None:
        release = getattr(self.tt_gpu_connector, "release_loaded_pooled", None)
        if callable(release):
            release()
        else:
            self.tt_gpu_connector.loaded_pooled.clear()

    def _with_pooled(self, states: Any, row: int, start: int, model_input: Any) -> Any:
        """a hit's checkpoint + the pooled keys of its loaded pages [0, start): {"units": U, "data": per layer [U, 8,192]
        uint8 device-native unit pages} (units 0..U-1; a unit's two 64-token halves side by side per tile). Unchanged
        when a page came without its slice (or a zero one: a page stored before GLM53_KV_POOLED_PAGES): the model then
        restarts the row at 0."""
        tables = getattr(model_input, "block_tables_per_group", None)
        view = next(iter(self.views.values()))
        table = torch.as_tensor(tables[self._group_for(view)] if tables else model_input.block_tables)
        bs = view.block_size
        ids = [int(b) for b in table[row, : -(-start // bs)].tolist()]
        halves = self.tt_gpu_connector.take_loaded_pooled(ids)
        if halves is None or any(bool((h.reshape(len(ids), -1) == 0).all(dim=1).any()) for h in halves[:1]):
            self.pooled_missing += 1
            logger.warning("prefix hit of %d tokens: its pages carry no pooled keys; the row restarts at 0", start)
            return states
        units = -(-len(ids) // 2)
        data = []
        for h in halves:  # [pages, 4,096] -> pad to whole units -> [U, 2 halves, 4 tiles, 1,024] -> tiles of both halves
            h = h.reshape(len(ids), -1).numpy()  # numpy: one single-threaded copy per layer (no intra-op thread overhead)
            if len(ids) % 2:
                h = np.concatenate([h, np.zeros((1, h.shape[1]), dtype=h.dtype)])
            data.append(np.ascontiguousarray(h.reshape(units, 2, 4, 1024).transpose(0, 2, 1, 3)).reshape(units, -1))
        out = dict(states)
        out["pooled"] = {"units": units, "pages": len(ids), "data": data}
        return out

    # -- device-path page sync for connectors that index the host buffers -----

    def _sync_pages(self, model_input: Any, *, prefix: bool, direction: str) -> None:
        """Copy the step rows' pages: loaded prefix (h2d) or written span (d2h)."""
        if self.host_xfer_buffers is None:
            return
        row_req_ids = getattr(model_input, "row_req_ids", None) or []
        tables = getattr(model_input, "block_tables_per_group", None)
        per_group: dict[int, set[int]] = {}
        for g in {self._group_for(v) for v in self.views.values()}:
            table = torch.as_tensor(tables[g] if tables else model_input.block_tables)
            ids: set[int] = set()
            for row in range(len(row_req_ids)):
                positions = _row_positions(model_input, row)
                if not positions:
                    continue
                bs = next(iter(self.views.values())).block_size
                if prefix:
                    blocks = range(0, positions[0] // bs)
                else:
                    blocks = range(positions[0] // bs, positions[-1] // bs + 1)
                ids.update(int(table[row, b]) for b in blocks if int(table[row, b]) > 0)
            per_group[g] = ids
        if direction == "h2d" and self.sharded_pool is not None:  # the prefix's blocks at their request columns
            pairs: dict[int, int] = {}
            for row in range(len(row_req_ids)):
                positions = _row_positions(model_input, row)
                if not positions:
                    continue
                for g in per_group:
                    table = torch.as_tensor(tables[g] if tables else model_input.block_tables)
                    bs = next(iter(self.views.values())).block_size
                    for j in range(0, positions[0] // bs):
                        if int(table[row, j]) > 0:
                            pairs.setdefault(int(table[row, j]), j)
            if pairs:
                self.sharded_pool.admit(sorted(pairs.items()), source="connector_sync")
        for name, view in self.views.items():
            ids = sorted(per_group[self._group_for(view)])
            if not ids:
                continue
            src, dst = (
                ({name: self.host_xfer_buffers[name]}, {name: view.tensor})
                if direction == "h2d"
                else ({name: view.tensor}, {name: self.host_xfer_buffers[name]})
            )
            tt_copy_kv_blocks(src, dst, ids, ids, direction)  # type: ignore[arg-type]

    # -- KDA checkpoints (page_contract.md section 5) ------------------------

    def _prefill_rows(self, model_input: Any) -> list[tuple[int, str, int, int]]:
        """``(row, req_id, start, end)`` for every real row of a prefill build."""
        prompt_lens = getattr(model_input, "prompt_lens", None)
        if prompt_lens is None:
            return []
        rows = []
        for row, req_id in enumerate(model_input.row_req_ids or []):
            start = int(torch.as_tensor(model_input.input_positions)[row])
            end = int(torch.as_tensor(prompt_lens)[row])
            rows.append((row, req_id, start, end))
        return rows

    def _slot_for(self, req_id: str, row: int, model_input: Any) -> int:
        slots = getattr(model_input, "prefill_empty_slots", None)
        if slots is not None and row < len(slots):
            return int(slots[row])
        slot = self._slot_of(req_id)
        if slot is None:
            raise RuntimeError(f"no device state slot known for request {req_id!r}")
        return int(slot)

    def _import_kda_checkpoints(self, scheduler_output: Any, model_input: Any) -> None:
        """A row that starts past 0 on a connector hit continues from its checkpoint."""
        if self.kda_store is None or not callable(self._kda_import):
            return
        hits = {
            r.req_id
            for r in getattr(scheduler_output, "scheduled_new_reqs", [])
            if r.num_computed_tokens > 0
        }
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is not None:
            for i, req_id in enumerate(cached.req_ids):
                if (
                    req_id in cached.resumed_req_ids
                    and cached.num_computed_tokens[i] > 0
                ):
                    hits.add(req_id)
        for row, req_id, start, _end in self._prefill_rows(model_input):
            if req_id not in hits or start <= 0:
                continue
            t_ = time.perf_counter()
            tokens = torch.as_tensor(model_input.input_tokens)[row, :start].tolist()
            states = self._kda_snapshot_import(tokens, start)
            if states is None:
                states = self.kda_store.load(tokens, start)
            sub_ = self._kda_import_phase = getattr(self, "_kda_import_phase", None) or {}
            sub_["store_load"] = sub_.get("store_load", 0.0) + time.perf_counter() - t_
            if states is None:
                raise RuntimeError(
                    f"request {req_id!r}: prefix hit of {start} tokens has no KDA "
                    "checkpoint; the scheduler-side TTPageAlignedConnector must "
                    "truncate hits to a checkpointed boundary"
                )
            if _RESTORED_SPANS:  # ruling A: a first writer's checkpoint deeper than a replaced span = mixed provenance
                span = self._on_replaced_span(tokens, start)
                if span is not None and KDACheckpointStore.key(tokens, start) not in _POST_RESTORE_CKPTS:
                    self.kda_provenance_mixed_imports = getattr(self, "kda_provenance_mixed_imports", 0) + 1
                    self._provenance_event(req_id, "mixed_import", span[0], start,
                                           [f"checkpoint {start} written before the restore of [{span[0]}, ...)"])
            slot = self._slot_for(req_id, row, model_input)
            t_ = time.perf_counter()
            if getattr(self, "pooled_pages", False):
                states = self._with_pooled(states, row, start, model_input)
            sub_["with_pooled"] = sub_.get("with_pooled", 0.0) + time.perf_counter() - t_
            t_ = time.perf_counter()
            self._kda_import(self._kv_cache_object, slot, states)
            sub_["model_import"] = sub_.get("model_import", 0.0) + time.perf_counter() - t_
            logger.debug(
                "KDA checkpoint of %d tokens imported into slot %d", start, slot
            )

    def _kda_snapshot_import(self, tokens: list, start: int) -> Any | None:
        """lever 1 (b): import_kda_state's input restoring the hit from the device snapshot (the model's
        kda_snapshot_states), or None (off / not the latest export / stale): the host checkpoint path"""
        fn = getattr(self, "_kda_snap_states", None)
        if not callable(fn) or self.kda_store is None:
            _kda_snap_miss(self, "no model hook" if not callable(fn) else "no KDA store", start)
            return None
        producer = self.kda_store.snapshot_producer(tokens, start)
        if producer is None:
            _kda_snap_miss(self, "not the latest export", start)
            return None
        store = self.kda_store

        def fallback() -> Any:
            got = store.load(tokens, start)
            if got is None:
                raise RuntimeError(f"KDA checkpoint of {start} tokens: its snapshot went stale and the host copy is gone")
            return got

        states = fn(producer, fallback)
        if states is not None:
            self.kda_snapshot_imports = getattr(self, "kda_snapshot_imports", 0) + 1
        else:
            _kda_snap_miss(self, "the model's handle is stale or does not match", start)
        return states


    def _mark_kda_snapshots(self, model_input: Any) -> Any:
        """GLM53_KDA_CKPT_IN_REPLAY: model_input.kda_snapshot_at[row] = the prompt's cut when this prefill chunk covers
        it strictly inside (start < cut < end; a chunk that ENDS on the cut exports the slot's live state as before) and
        the prompt is checkpointed at all (the store floor, the provenance rule); 0 = no snapshot. None when no row
        has one. The model writes the state at the cut into its staging buffers inside the replay. Returns the model
        input carrying it: a ``dataclasses.replace`` copy (``TTModelInput`` is frozen; the 2026-10-25 cf3h-g mock died
        on an assignment here), or ``model_input`` itself when nothing changes."""
        if not kda_ckpt_in_replay() or self.kda_store is None or getattr(self, "checkpoints_off", False):
            return model_input
        rows = self._prefill_rows(model_input)
        if not rows:
            return model_input
        bs = next(iter(self.views.values())).block_size
        grid = max(bs, int(getattr(self, "_ckpt_grid", 0) or 0))
        min_store = lmcache_min_store_tokens()
        snap = [0] * len(rows)
        for i, (row, req_id, start, end) in enumerate(rows):
            n_hist = _host_history_len(self, req_id)
            if n_hist is None or (min_store and n_hist < min_store) or req_id in _KDA_PROVENANCE:
                continue
            cut = max(int(n_hist) - 1, 0) // grid * grid
            if start < cut < end:
                snap[i] = cut
        snap = snap if any(snap) else None
        if getattr(model_input, "kda_snapshot_at", None) == snap:
            return model_input
        return replace(model_input, kda_snapshot_at=snap)

    def _export_kda_snapshots(self, model_input: Any) -> None:
        """the checkpoints the replay snapshotted (_mark_kda_snapshots), exported from the model's staging buffers with
    12's M2 contract (doc/ttnn/09-vllm/perf/one_replay_store/design.md 7.1): export_kda_state(kv, slot, at=cut) /
    export_kda_state_async(kv, slot, at=cut), after model.kda_snapshot_ready(kv, slot, cut); an async None falls back to
    the sync export (the snapshot tag is consumed only by a successful export)"""
        snap = getattr(model_input, "kda_snapshot_at", None)
        if not snap:
            return
        for i, (row, req_id, _start, _end) in enumerate(self._prefill_rows(model_input)):
            cut = int(snap[i]) if i < len(snap) else 0
            if cut <= 0:
                continue
            tokens = torch.as_tensor(model_input.input_tokens)[row, :cut].tolist()
            if self.kda_store.has(tokens, cut) or not self.kda_store.can_save():
                continue
            if _RESTORED_SPANS and self._on_replaced_span(tokens, cut) is not None:
                with _PROVENANCE_LOCK:  # written after the restore: it pairs with the current pages
                    _POST_RESTORE_CKPTS.add(KDACheckpointStore.key(tokens, cut))
            slot = self._slot_for(req_id, row, model_input)
            ready = getattr(self, "_kda_snapshot_ready", None)
            if callable(ready) and not ready(self._kv_cache_object, slot, cut):
                self.kda_snapshots_not_ready = getattr(self, "kda_snapshots_not_ready", 0) + 1
                logger.warning("KDA snapshot at %d of slot %d not ready: no checkpoint for this prompt", cut, slot)
                continue
            export_async = getattr(self, "_kda_export_async", None)
            if callable(export_async):
                producer = export_async(self._kv_cache_object, slot, at=cut)
                if producer is not None:
                    self.kda_store.save_deferred(tokens, cut, producer)
                    continue
            self.kda_store.save(tokens, cut, self._kda_export(self._kv_cache_object, slot, at=cut))
            logger.debug("KDA checkpoint of %d tokens saved from slot %d's in-replay snapshot", cut, slot)

    def _export_kda_checkpoints(self, model_input: Any) -> None:
        """A prefill chunk that ends on a page boundary checkpoints its slot's state.

        Prefill rows only: ``end`` is the row's committed position (a prefill
        commits every token it computes), and the slot holds no speculation
        state (the generator leaves speculation before any prefill), so the
        plain ``export_kda_state`` reads the right row. Decode rows go through
        ``_export_decode_boundary_checkpoints``.
        """
        if self.kda_store is None or not callable(self._kda_export) or getattr(self, "checkpoints_off", False):
            return
        # GLM53_KDA_CKPT_EVERY = N > 0 (Stage 12): a boundary is checkpointed only every N tokens and at the prompt's
        # LAST grid boundary (where the next turn's hit resumes); 0 = every page-aligned chunk end (v1). A checkpoint is
        # the slot's whole state (148 MB at TP8), so every-boundary saves dominate a long prompt's prefill.
        every = int(os.environ.get("GLM53_KDA_CKPT_EVERY", "0"))
        min_store = lmcache_min_store_tokens()
        for row, req_id, _start, end in self._prefill_rows(model_input):
            bs = next(iter(self.views.values())).block_size
            exact_end = _PD_EXACT_END.get(req_id) == end and end > 0  # a P request's prompt end: exact (Form B v2)
            if end <= 0 or (end % bs and not exact_end):
                continue
            if min_store:
                n_hist = _host_history_len(self, req_id)
                if n_hist is not None and n_hist < min_store:
                    self.kda_ckpt_skipped_short += 1
                    continue  # GLM53_LMCACHE_MIN_STORE_TOKENS: a short prompt is not checkpointed
            if every > 0 and end % every and not exact_end:
                n_hist = _host_history_len(self, req_id)
                # the resume point = the LAST grid boundary STRICTLY before the prompt's end: a repeat of the prompt
                # must compute >= 1 token, so a checkpoint at the end itself never serves it (a 4,096-token prompt
                # checkpointed only at 4,096 gives its exact repeat no hit); a longer next turn resumes there too
                grid = max(bs, self._ckpt_grid)
                # GLM53_PF_TAIL_SPLIT_MIN: a prompt below it runs its last chunk whole (no split at the last grid
                # boundary), so its resume point is its LAST CHUNK END: the chunk end whose next chunk (at most one
                # model chunk, GLM53_PF_MODEL_CHUNK) reaches the prompt's end
                tmin = tail_split_min()
                if tmin and n_hist is not None and n_hist < tmin:
                    grid = max(grid, int(os.environ.get("GLM53_PF_MODEL_CHUNK", "2048") or 2048))
                at_end = (n_hist is not None and end == n_hist and not end % grid and kda_ckpt_at_end_mode() != "0"
                          and not kda_ckpt_in_replay())  # fmt: skip
                if not at_end and (n_hist is None or end >= n_hist or end + grid < n_hist):
                    continue  # not the prompt's resume point (GLM53_KDA_CKPT_AT_END: its end, when on the grid)
            if req_id in _KDA_PROVENANCE:  # its pages are not LMCache's (the provenance rule): no checkpoint from it
                self.kda_ckpt_skipped_provenance = getattr(self, "kda_ckpt_skipped_provenance", 0) + 1
                continue
            tokens = torch.as_tensor(model_input.input_tokens)[row, :end].tolist()
            if self.kda_store.has(tokens, end) or not self.kda_store.can_save():
                continue
            if _RESTORED_SPANS and self._on_replaced_span(tokens, end) is not None:
                with _PROVENANCE_LOCK:  # written after the restore: it pairs with the current pages
                    _POST_RESTORE_CKPTS.add(KDACheckpointStore.key(tokens, end))
            slot = self._slot_for(req_id, row, model_input)
            export_async = getattr(self, "_kda_export_async", None)
            if exact_end and end % bs:  # off the grid: the plain export (the model allows it under resume-as-decode)
                self.kda_store.save(tokens, end, self._kda_export(self._kv_cache_object, slot))
                self.pd_exact_end_exports = getattr(self, "pd_exact_end_exports", 0) + 1
                continue
            if callable(export_async):  # Stage 12: a device snapshot, saved by the page-DMA off this step
                producer = export_async(self._kv_cache_object, slot)
                if producer is not None:
                    self.kda_store.save_deferred(tokens, end, producer)
                    logger.debug("KDA checkpoint of %d tokens snapshotted from slot %d (page-DMA)", end, slot)
                    continue
            states = self._kda_export(self._kv_cache_object, slot)
            self.kda_store.save(tokens, end, states)
            logger.debug("KDA checkpoint of %d tokens saved from slot %d", end, slot)

    def _export_decode_boundary_checkpoints(
        self, model_input: Any, output: ModelRunnerOutput | None
    ) -> None:
        """Page boundaries a decode step committed: checkpoint at the committed
        position, never past it (module docstring; contract §4).

        A decode row at position ``c`` that committed ``m = n + 1`` tokens holds
        the state after ``B`` tokens for every page boundary ``c < B <= c + m``
        in candidate row ``B - c - 1`` (a plain decode: ``m = 1``, the slot's
        own row). A boundary the verify rows crossed past ``c + m`` is not
        committed: nothing is stored for it now, the next step sees it again.
        The export is the model's ``export_spec_boundary_state``; without it
        the boundary is skipped (counted, logged once) and hits keep resuming
        at the prefill boundaries.
        """
        if self.kda_store is None or output is None or getattr(self, "checkpoints_off", False):
            return
        if getattr(model_input, "prompt_lens", None) is not None:
            return  # a prefill build: _export_kda_checkpoints
        row_req_ids = getattr(model_input, "row_req_ids", None) or []
        sampled = getattr(output, "sampled_token_ids", None) or []
        index = getattr(output, "req_id_to_index", None) or {}
        if not row_req_ids or not sampled:
            return
        bs = next(iter(self.views.values())).block_size
        positions = torch.as_tensor(model_input.input_positions)
        num_valid_drafts = getattr(model_input, "num_valid_drafts", None)
        for row, req_id in enumerate(row_req_ids):
            idx = index.get(req_id)
            if idx is None or idx >= len(sampled):
                continue
            block = [int(t) for t in sampled[idx]]
            first = int(positions[row].reshape(-1)[0])
            if first < 0 or not block:
                continue
            drafts = (
                int(torch.as_tensor(num_valid_drafts)[row])
                if num_valid_drafts is not None
                else 0
            )
            speculative = num_valid_drafts is not None
            committed, deferred = step_boundary_checkpoints(
                first, len(block), bs, num_valid_drafts=drafts
            )
            self.decode_boundaries_deferred += len(deferred)
            if committed and req_id in _KDA_PROVENANCE:  # the provenance rule (see _export_kda_checkpoints)
                self.kda_ckpt_skipped_provenance = getattr(self, "kda_ckpt_skipped_provenance", 0) + len(committed)
                continue
            for boundary, candidate_row in committed:
                if not decode_boundary_wanted(boundary):
                    self.decode_boundaries_skipped_every += 1
                    continue
                tokens = self._committed_prefix(req_id, first, block, boundary)
                if tokens is None or self.kda_store.has(tokens, boundary):
                    continue
                if not callable(self._boundary_export):
                    self.decode_boundaries_skipped += 1
                    if not self._skipped_boundary_logged:
                        self._skipped_boundary_logged = True
                        logger.info(
                            "decode-time KDA checkpoint at %d tokens skipped: %s has "
                            "no export_spec_boundary_state (prefill boundaries and "
                            "chunk stores are unaffected)",
                            boundary,
                            type(self._kv_cache_object).__name__,
                        )
                    continue
                slot = self._slot_for(req_id, row, model_input)
                states = self._boundary_export(
                    self._kv_cache_object,
                    slot,
                    n_tokens=boundary,
                    candidate_row=candidate_row if speculative else None,
                )
                self.kda_store.save(tokens, boundary, states)
                self.decode_boundaries_checkpointed += 1
                logger.debug(
                    "KDA checkpoint of %d tokens saved from slot %d candidate row %s",
                    boundary,
                    slot,
                    candidate_row if speculative else "own",
                )

    def _committed_prefix(
        self, req_id: str, first: int, block: list[int], boundary: int
    ) -> list[int] | None:
        """Token ids ``[:boundary]`` of a decode row: the request's committed
        history up to its position ``first`` plus the block this step committed."""
        history = self._tokens_of(req_id)
        if history is None or len(history) < first + 1:
            return None
        prefix = [int(t) for t in list(history)[: first + 1]] + block
        if len(prefix) < boundary:
            return None
        return prefix[:boundary]

    def _collect(self, scheduler_output: SchedulerOutput) -> KVConnectorOutput:
        """``get_finished`` and the per-step reports; metadata must be bound."""
        output = KVConnectorOutput()
        output.finished_sending, output.finished_recving = self.connector.get_finished(
            set(scheduler_output.finished_req_ids)
        )
        if _leave_drain():
            output.finished_sending = self._leave_report(output.finished_sending)
        output.invalid_block_ids = self.connector.get_block_ids_with_load_errors()
        output.kv_connector_stats = self.connector.get_kv_connector_stats()
        output.kv_cache_events = self.connector.get_kv_connector_kv_cache_events()
        output.kv_connector_worker_meta = self.connector.build_connector_worker_meta()
        return output

    def finish(
        self,
        scheduler_output: SchedulerOutput,
        output: ModelRunnerOutput,
        model_input: Any | None,
    ) -> ModelRunnerOutput:
        """Save the step's pages, collect ``get_finished`` and attach the result."""
        if model_input is not None:
            self.save(scheduler_output, model_input, output)
        meta = self._metadata(scheduler_output)
        with self._lock:
            self.connector.bind_connector_metadata(meta)
            try:
                kv_output = self._collect(scheduler_output)
            finally:
                self.connector.clear_connector_metadata()
        if output is None:
            return ModelRunnerOutput.with_kv_conn_output_only(kv_output)
        output.kv_connector_output = kv_output
        return output

    def no_forward(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """A step with no scheduled tokens still moves KV.

        Mirrors upstream ``kv_connector_no_forward``.

        GLM53_KV_NOFWD_FLUSH=budget (Stage 12 W4 F2c, 2026-10-25; default all): the empty step keeps the hold / grace /
        budget rules of begin_step instead of flushing every held page store and draining the store worker. The runner
        step right after a prompt's last chunk has no model input and lands here (the cf3h-g run's engine stall stack:
        no_forward -> flush_deferred_save -> page reads, 0.44 s after the last prefill), so the full flush + drain sat
        on that prompt's first token: the stored 1M needle waited 51.4 s for 129 held steps
        (doc/ttnn/12-fast-start/w4_host). A held request leaving or preempted still flushes everything.
        """
        if os.environ.get("GLM53_KV_NOFWD_FLUSH", "all") == "budget":
            self.flush_deferred_save(scheduler_output, budget=_store_budget())
        else:
            self.flush_deferred_save()
        meta = self._metadata(scheduler_output)
        forward_context, _ = self._step_context(None)
        with self._lock:
            self.connector.bind_connector_metadata(meta)
            try:
                self.connector.start_load_kv(forward_context)
                kv_output = self._collect(scheduler_output)
            finally:
                self.connector.clear_connector_metadata()
        return ModelRunnerOutput.with_kv_conn_output_only(kv_output)

    def wrap_finish(
        self,
        scheduler_output: SchedulerOutput,
        model_input: Any | None,
        finish: Callable[..., ModelRunnerOutput],
    ) -> Callable[..., ModelRunnerOutput]:
        """Wrap a deferred sampler so its output carries the connector output."""

        def _finish(grammar_output: Any) -> ModelRunnerOutput:
            output = finish(grammar_output)
            if isinstance(output, ModelRunnerOutput):
                return self.finish(scheduler_output, output, model_input)
            # Async wrapper: the save/collect runs when it resolves.
            hook = getattr(output, "set_post_output_hook", None)
            if hook is None:
                raise TypeError(
                    f"{type(output).__name__} cannot carry a KV connector output"
                )
            hook(lambda resolved: self.finish(scheduler_output, resolved, model_input))
            return output

        return _finish

    def wrap_finish_mixed(
        self,
        scheduler_output: SchedulerOutput,
        prefill_input: Any,
        decode_input: Any,
        finish: Callable[..., ModelRunnerOutput],
    ) -> Callable[..., ModelRunnerOutput]:
        """``wrap_finish`` for a mixed prefill+decode step (one output, two inputs).

        The step is seen once. The decode rows get what a decode step's save
        gives them, now: their committed page-boundary checkpoints (the next
        forward changes their slots' state) and, for a connector reading the
        host buffers, the d2h copy of the pages they wrote. Then the prefill
        part is saved and the step's connector output collected exactly as for
        a prefill step (``finish``): chunk checkpoints and page stores, off the
        first-token path when deferred. The page stores bind the step's one
        connector metadata, which covers every request of the step.
        """

        def _finish(grammar_output: Any) -> ModelRunnerOutput:
            output = finish(grammar_output)
            if not isinstance(output, ModelRunnerOutput):
                raise TypeError(
                    "a mixed prefill+decode step is sampled synchronously; got "
                    f"{type(output).__name__}"
                )
            self._export_decode_boundary_checkpoints(decode_input, output)
            if getattr(self, "_generic_sync", False):
                self._sync_pages(decode_input, prefix=False, direction="d2h")
            return self.finish(scheduler_output, output, prefill_input)

        return _finish

    def shutdown(self) -> None:
        try:
            self._segment_drop(set(getattr(self, "_segment_parked", None) or ()))  # unreleased: never stored
            self.flush_deferred_save()
            if getattr(self, "store_worker", None) is not None:
                self.store_worker.close()  # every queued store lands before LMCache shuts down
        finally:
            self.connector.shutdown()


# ---------------------------------------------------------------------------
# Worker-process connector lifecycle
# ---------------------------------------------------------------------------


class _StoreWorker:
    """GLM53_KV_STORE_WORKER: one thread runs the LMCache page stores the engine thread hands over (FIFO), each a
    (step metadata, page snapshot) job; submit() blocks only while more than max_bytes of snapshots are in flight
    (back-pressure); drain() waits for every queued store; close() drains and stops the thread"""

    def __init__(self, host: Any, max_bytes: int):
        self.host, self.max_bytes = host, int(max_bytes)
        self.q: queue.Queue = queue.Queue()
        self.cv = threading.Condition()
        self.inflight = 0
        self.last_bytes = 0  # the last job's size: the estimate of the next one (has_room)
        self.stats = dict(jobs=0, done=0, errors=0, bytes=0, submit_wait_s=0.0, store_s=0.0)
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="tt-kv-store-worker", daemon=True)
        self.thread.start()

    def submit(self, meta: Any, job: dict) -> None:
        n = int(job.get("nbytes", 0))
        t0 = time.perf_counter()
        with self.cv:
            while self.inflight > 0 and self.inflight + n > self.max_bytes:
                self.cv.wait()
            self.inflight += n
        self.stats["submit_wait_s"] += time.perf_counter() - t0
        self.last_bytes = n
        self.stats["jobs"] += 1
        self.stats["bytes"] += n
        self.q.put((meta, job, n))

    def busy(self) -> bool:
        """a submitted store has not finished (queued or running)"""
        with self.cv:
            return self.inflight > 0

    def has_room(self) -> bool:
        """a submit of a job like the last one would not wait for the in-flight cap"""
        with self.cv:
            return self.inflight == 0 or self.inflight + self.last_bytes <= self.max_bytes

    def _run(self) -> None:
        while True:
            item = self.q.get()
            try:
                if item is None:
                    return
                meta, job, n = item
                t0 = time.perf_counter()
                try:
                    self.host._run_store_job(meta, job)
                    self.stats["done"] += 1
                except Exception as exc:  # noqa: BLE001 - a failed store drops its pages, never the engine
                    self.stats["errors"] += 1
                    self.error = exc
                    logger.exception("KV store worker: a page store failed (its pages are not in LMCache)")
                finally:
                    done = getattr(self.host, "_job_done", None)
                    if callable(done):
                        done(job)  # M1: its requests' delay-freed blocks may go once nothing else holds them
                self.stats["store_s"] += time.perf_counter() - t0
                with self.cv:
                    self.inflight -= n
                    self.cv.notify_all()
            finally:
                self.q.task_done()

    def drain(self) -> None:
        self.q.join()

    def wait_idle(self, timeout_s: float) -> bool:
        """GLM53_KV_LOOKUP_DEFER: wait at most timeout_s for every submitted store to finish; True when idle"""
        with self.cv:
            return self.cv.wait_for(lambda: self.inflight == 0, timeout=max(0.0, float(timeout_s)))

    def close(self) -> None:
        self.drain()
        self.q.put(None)
        self.thread.join(timeout=30)


class TTSingleProcessGroup:
    """World-size-1 stand-in for vLLM's ``GroupCoordinator`` (findings J1/J2).

    The TT worker is one process per engine and never calls
    ``init_distributed_environment``: a real ``GroupCoordinator`` would build
    ``torch.device("tt:0")``, which torch rejects. LMCache's worker role still
    reads ``get_tp_group()`` (``broadcast`` / ``broadcast_object`` of its
    lookup results), ``get_tensor_model_parallel_rank()`` and
    ``get_pp_group().is_last_rank``, and upstream's
    ``ensure_kv_transfer_initialized`` broadcasts ``engine_id`` over the TP
    group. With one rank every collective is the identity, which is exactly
    what this object implements.
    """

    def __init__(self) -> None:
        self.rank = 0
        self.rank_in_group = 0
        self.local_rank = 0
        self.world_size = 1
        self.ranks = [0]
        self.is_first_rank = True
        self.is_last_rank = True
        self.device = torch.device("cpu")
        self.device_group = None
        self.cpu_group = None

    def broadcast(self, tensor: Any, src: int = 0) -> Any:
        return tensor

    def broadcast_object(self, obj: Any = None, src: int = 0) -> Any:
        return obj

    def broadcast_object_list(self, obj_list: Any, src: int = 0) -> Any:
        return obj_list

    def all_reduce(self, tensor: Any) -> Any:
        return tensor

    def all_gather(self, tensor: Any, dim: int = -1) -> Any:
        return tensor

    def gather(self, tensor: Any, dst: int = 0, dim: int = -1) -> Any:
        return tensor

    def barrier(self) -> None:
        return None

    def send_object(self, obj: Any, dst: int) -> None:
        raise RuntimeError("single-process TT group has no peer to send to")

    def recv_object(self, src: int) -> Any:
        raise RuntimeError("single-process TT group has no peer to receive from")


_INSTALLED_GROUPS: dict[str, Any] | None = None


def install_tt_single_process_groups() -> bool:
    """Install ``TTSingleProcessGroup`` as vLLM's TP and PP groups if none exist.

    Sets ``parallel_state._TP`` / ``_PP`` so every accessor, including
    references imported before this call (``from vllm.distributed.parallel_state
    import get_pp_group`` in LMCache's adapter), resolves to the stub. Returns
    ``True`` when the stub was installed by this call; a real group (a
    distributed launch) is left alone.
    """
    global _INSTALLED_GROUPS
    from vllm.distributed import parallel_state as ps

    if ps._TP is not None or _INSTALLED_GROUPS is not None:
        return False
    stub = TTSingleProcessGroup()
    _INSTALLED_GROUPS = {"_TP": ps._TP, "_PP": ps._PP}
    ps._TP = stub
    if ps._PP is None:
        ps._PP = stub
    logger.info("TT KV connector: single-process TP/PP groups installed (world 1)")
    return True


def uninstall_tt_single_process_groups() -> None:
    global _INSTALLED_GROUPS
    if _INSTALLED_GROUPS is None:
        return
    from vllm.distributed import parallel_state as ps

    ps._TP = _INSTALLED_GROUPS["_TP"]
    ps._PP = _INSTALLED_GROUPS["_PP"]
    _INSTALLED_GROUPS = None


def tt_single_process_groups_installed() -> bool:
    return _INSTALLED_GROUPS is not None


def ensure_tt_kv_transfer_initialized(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> KVConnectorBase_V1 | None:
    """Create the worker-side connector for this process.

    Upstream ``ensure_kv_transfer_initialized`` broadcasts ``engine_id`` over
    the tensor-parallel group, and LMCache's worker role keeps using the TP /
    PP groups afterwards; the TT worker never initialises them (one worker per
    engine, parallelism is the model's). When no group exists a world-size-1
    ``TTSingleProcessGroup`` is installed first, then upstream's own path
    creates the connector through ``KVConnectorFactory``.
    """
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None or not kv_transfer_config.is_kv_transfer_instance:
        return None
    if kv_transfer_state.has_kv_transfer_group():
        return kv_transfer_state.get_kv_transfer_group()
    install_tt_single_process_groups()
    ensure_short_tmpdir_for_lmcache(vllm_config)
    kv_transfer_state.ensure_kv_transfer_initialized(vllm_config, kv_cache_config)
    connector = kv_transfer_state.get_kv_transfer_group()
    fix_lmcache_torch_device(connector)
    logger.info(
        "TT worker KV connector initialised: %s (engine_id=%s, kv_role=%s)",
        type(connector).__name__,
        kv_transfer_config.engine_id,
        kv_transfer_config.kv_role,
    )
    return connector


LMCACHE_SHUTDOWN_TIMEOUT_S = 10.0


def shutdown_lmcache_engine(
    connector: Any, timeout_s: float = LMCACHE_SHUTDOWN_TIMEOUT_S
) -> bool:
    """Stop LMCache's engine threads behind vLLM's ``LMCacheConnectorV1``.

    The wrapper does not forward ``shutdown`` to its adapter
    (``LMCacheConnectorV1Impl.shutdown`` -> ``stop_services``), so the lookup /
    offload servers and the engine's periodic threads outlive the worker and
    the EngineCore only dies at vLLM's shutdown timeout (seen on the
    mock-cluster rehearsal). The adapter's shutdown itself can block forever
    (LMCache-tt's ``zmq_server.close`` joins a receiver stuck in
    ``recv_multipart``), so it runs on a daemon thread with a bounded wait.
    Returns whether the adapter's shutdown finished within the wait.
    """
    impl = getattr(connector, "_lmcache_engine", None)
    stop = getattr(impl, "shutdown", None)
    if not callable(stop):
        return False

    def _run() -> None:
        try:
            stop()
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            logger.warning("LMCache adapter shutdown raised: %r", exc)

    worker = threading.Thread(target=_run, name="tt-lmcache-shutdown", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        logger.warning(
            "LMCache adapter shutdown did not finish within %.0f s (its zmq "
            "services keep blocking); continuing the worker shutdown",
            timeout_s,
        )
        return False
    return True


def ensure_tt_kv_transfer_shutdown(*, stop_lmcache: bool = True) -> None:
    """Shut the worker-side connector down and drop the single-process groups.

    ``stop_lmcache`` forwards the shutdown to LMCache's adapter (its services
    outlive the worker otherwise and the EngineCore dies only at vLLM's
    shutdown timeout: mock-cluster rehearsal ``a1_noshim`` 15 min vs
    ``a1_noshim2`` 10 s). In-process engines whose scheduler-role lookup client
    is still alive in the same process (the direct-scheduler tests) pass
    ``False``: LMCache-tt's ``stop_services`` then blocks for good (its offload
    thread's ``recv`` on a socket closed from another thread) and starves even
    a bounded join.
    """
    if stop_lmcache and kv_transfer_state.has_kv_transfer_group():
        shutdown_lmcache_engine(kv_transfer_state.get_kv_transfer_group())
    kv_transfer_state.ensure_kv_transfer_shutdown()
    uninstall_tt_single_process_groups()


TT_KV_SHAPE_HINT_KEY = "tt_kv_shape_hint"
TT_LMCACHE_TMPDIR_KEY = "tt_lmcache_tmpdir"
# ``$TMPDIR/engine_<uuid36>_service_offload_<rpc port>`` must fit a Unix socket
# path (107 bytes): the prefix alone is ~70 bytes, so TMPDIR itself has to be
# short (finding J8).
TT_LMCACHE_MAX_TMPDIR_LEN = 24


def _is_lmcache_connector(kv_transfer_config: Any) -> bool:
    name = str(getattr(kv_transfer_config, "kv_connector", "") or "")
    return name.startswith("LMCache")


# Connectors whose stores never contain a draft row (module docstring):
# ExampleConnector / SharedStorageConnector store a new request's prompt tokens.
_PROMPT_ONLY_STORE_CONNECTORS = ("ExampleConnector", "SharedStorageConnector")


def connector_spec_decode_mode(kv_transfer_config: Any) -> str:
    """How ``kv_transfer_config``'s connector behaves under speculative decoding.

    Returns the mode line to log, or raises ``ValueError`` for a connector the
    host cannot vouch for. LMCache's scheduler side keys and masks its stores
    on vLLM's ``all_token_ids`` (committed tokens; drafts are never in it) and
    ``num_computed_tokens`` (rolled back by the rejected count before the next
    schedule), and its worker side stores in ``wait_for_save`` after the step's
    forward: every stored chunk, prompt or generation
    (``save_decode_cache``), holds committed tokens with final latent rows.
    vLLM's ExampleConnector stores a new request's prompt only. Any other
    connector may store from the step's forward context, whose slot mapping
    covers the verify rows, so it is refused.
    """
    name = str(getattr(kv_transfer_config, "kv_connector", "") or "")
    if _is_lmcache_connector(kv_transfer_config):
        return (
            "chunk stores on committed tokens only (LMCache keys and masks its "
            "stores on vLLM's accepted tokens and stores after the step; "
            "generation chunks included when save_decode_cache is on), KDA "
            "checkpoints at prefill page boundaries, decode-time boundaries "
            "deferred to the committed position"
        )
    if name in _PROMPT_ONLY_STORE_CONNECTORS:
        return (
            f"{name} stores a new request's prompt pages only; KDA checkpoints "
            "at prefill page boundaries"
        )
    raise ValueError(
        f"--kv-transfer-config with speculative decoding on TT is validated for "
        f"LMCacheConnectorV1 and {'/'.join(_PROMPT_ONLY_STORE_CONNECTORS)} only: a "
        "verify step writes latent rows past the committed position, and "
        f"{name!r} is not known to store committed tokens only"
    )


def ensure_short_tmpdir_for_lmcache(vllm_config: Any) -> str | None:
    """Force a short ``TMPDIR`` for LMCache's ZMQ ipc sockets (finding J8).

    LMCache binds ``$TMPDIR/engine_<uuid>_service_offload_<port>``; a TMPDIR of
    more than ~24 characters overflows the 107-byte socket path limit and the
    engine dies at its first offload. Order: ``kv_connector_extra_config
    ["tt_lmcache_tmpdir"]``, the current TMPDIR when short enough, else ``/tmp``.
    Returns the directory in force, or ``None`` when no LMCache connector is set.
    """
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None or not _is_lmcache_connector(kv_transfer_config):
        return None
    import tempfile  # noqa: PLC0415

    extra = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    current = os.environ.get("TMPDIR") or tempfile.gettempdir()
    chosen = extra.get(TT_LMCACHE_TMPDIR_KEY) or current
    if len(chosen) > TT_LMCACHE_MAX_TMPDIR_LEN:
        logger.warning(
            "TMPDIR %r is %d characters; LMCache's ipc socket path would exceed "
            "107 bytes (J8): using /tmp (set kv_connector_extra_config[%r] to "
            "choose another short directory)",
            chosen,
            len(chosen),
            TT_LMCACHE_TMPDIR_KEY,
        )
        chosen = "/tmp"
    os.makedirs(chosen, exist_ok=True)
    if os.environ.get("TMPDIR") != chosen:
        os.environ["TMPDIR"] = chosen
        tempfile.tempdir = None  # drop the cached default
    return chosen


def fix_lmcache_torch_device(connector: Any) -> bool:
    """LMCache's adapter copies ``vllm_config.device_config.device`` as its torch
    device; on TT that is the ttnn ``MeshDevice`` and ``slot_mapping.to(device)``
    raises at the first save (finding J9). The TT connector moves pages through
    host buffers: its torch side is the CPU. Returns whether a change was made.
    """
    impl = getattr(connector, "_lmcache_engine", None)
    device = getattr(impl, "device", None)
    if impl is None or device is None or isinstance(device, (torch.device, str)):
        return False
    impl.device = torch.device("cpu")
    logger.info("LMCache connector torch device %r -> cpu (TT host buffers)", device)
    return True


def tt_kv_shape_hint_from_specs(specs: dict[str, KVCacheSpec]) -> dict[str, Any] | None:
    """What LMCache needs to size the TT latent (finding J5).

    LMCache derives ``kv_shape`` from vLLM's ``model_config`` (for glm5_next:
    ``use_mla False``, 45 layers, 64 x 64 heads), which is not the latent cache
    geometry. The plugin publishes the KV spec's view under
    ``kv_connector_extra_config["tt_kv_shape_hint"]`` and the TT GPU connector
    rewrites LMCache's metadata from it: ``(num MLA layers, 1, chunk, 1,
    latent + reserved)`` with ``use_mla True``.
    """
    mla = {
        name: spec for name, spec in specs.items() if isinstance(spec, MLAAttentionSpec)
    }
    if not mla:
        return None
    latents = sorted({int(s.head_size) for s in mla.values()})
    block_sizes = sorted({int(s.block_size) for s in mla.values()})
    dtypes = sorted({str(s.dtype) for s in mla.values()})
    if len(latents) != 1 or len(block_sizes) != 1 or len(dtypes) != 1:
        raise ValueError(
            "MLA KV specs must agree on head_size / block_size / dtype for the "
            f"LMCache hint; got {latents}, {block_sizes}, {dtypes}"
        )
    names = sorted(mla, key=lambda n: (_parse_layer_index(n) or 0, n))
    return {
        "num_layers": len(names),
        "latent_dim": latents[0],
        "tokens_per_page": block_sizes[0],
        "dtype": dtypes[0].replace("torch.", ""),
        "use_mla": True,
        "layer_names": names,
    }


def validate_tt_kv_transfer_config(
    vllm_config: VllmConfig, model_class: Any = None
) -> None:
    """Config-time admission for ``--kv-transfer-config`` on TT.

    Refuses the execution modes the host does not cover: single-process
    lane-DP (no scheduler-side connector), block-output models (their steps
    write whole output canvases past the sampled position) and, under
    speculative decoding, a connector not known to store committed tokens only
    (``connector_spec_decode_mode``). Everything else is the connector's own
    business; vLLM itself disables the hybrid KV manager for connectors that do
    not support it.
    """
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return
    from vllm_tt_plugin.config import (
        is_tt_block_output_model,
        uses_tt_lane_coordinator,
    )

    if uses_tt_lane_coordinator(vllm_config):
        raise ValueError(
            "--kv-transfer-config is not supported with the single-process TT "
            "lane coordinator (tt_data_parallel > 1 in one engine); run one "
            "engine per mesh (standard DP) instead"
        )
    if is_tt_block_output_model(vllm_config):
        raise ValueError(
            "--kv-transfer-config is not supported for block-output models "
            "(output_tokens_per_step > 1)"
        )
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is not None:
        mode = connector_spec_decode_mode(kv_transfer_config)
        logger.info(
            "KV connector with speculative decoding (K=%s): %s",
            getattr(speculative_config, "num_speculative_tokens", "?"),
            mode,
        )
    extra = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    if extra.get(KDACheckpointStore.EXTRA_CONFIG_KEY) and not getattr(
        vllm_config.scheduler_config, "enable_chunked_prefill", False
    ):
        raise ValueError(
            "tt_kda_checkpoint_dir needs chunked prefill: the scheduler ends a "
            "prompt's prefill on its last page boundary to checkpoint the KDA "
            "state there, then runs the tail as a second chunk"
        )
    # The connectors this host targets (LMCacheConnectorV1, ExampleConnector)
    # do not implement SupportsHMA; vLLM auto-disables the hybrid manager only
    # when the flag is unset, and the TT platform reports hybrid support, so
    # pin it here. The GLM spec is a single group either way.
    scheduler_config = vllm_config.scheduler_config
    if not getattr(scheduler_config, "disable_hybrid_kv_cache_manager", False):
        scheduler_config.disable_hybrid_kv_cache_manager = True
        logger.info("KV connector configured: hybrid KV cache manager disabled")
    ensure_short_tmpdir_for_lmcache(vllm_config)  # J8
    # J5: publish the KV spec's latent geometry for LMCache's TT connector.
    hook = getattr(model_class, "get_kv_cache_spec", None)
    if callable(hook):
        specs = hook(vllm_config)
        hint = tt_kv_shape_hint_from_specs(specs) if specs else None
        if hint is not None:
            kv_transfer_config.kv_connector_extra_config[TT_KV_SHAPE_HINT_KEY] = hint
            logger.info(
                "KV connector: tt_kv_shape_hint = %d MLA layer(s) x latent %d "
                "(%s), %d-token pages",
                hint["num_layers"],
                hint["latent_dim"],
                hint["dtype"],
                hint["tokens_per_page"],
            )
    logger.info(
        "TT KV connector configured: %s (kv_role=%s); pages are %d-token vLLM "
        "blocks in the TT paged layout",
        kv_transfer_config.kv_connector,
        kv_transfer_config.kv_role,
        int(vllm_config.cache_config.block_size),
    )
