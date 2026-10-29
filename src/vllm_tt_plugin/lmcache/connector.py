# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""LMCache device-side connector for the Tenstorrent plugin.

``VLLMPagedMemTTConnector`` implements LMCache's ``GPUConnectorInterface`` over
the plugin's paged latent caches. There is no GPU: the "GPU" side is the host
transfer buffer vLLM's ``KVConnectorBase_V1.set_host_xfer_buffer_ops`` path is
built for. A page is moved in two hops:

  LMCache memory object (CPU tier) <-> host transfer buffer (per layer, the same
  ``[num_blocks, 1, 64, 512]`` bf16 shape as the device tensor, indexed by vLLM's
  ``slot_mapping = block_id * 64 + offset``) <-> device latent tensor (ttnn), via
  the plugin's ``CopyBlocksOp`` (``copy_blocks(src, dst, src_ids, dst_ids,
  "h2d" | "d2h")``).

The connector owns the first hop and calls the op for the second; without an op
the host buffers are the cache (host-only / mock path). Memory objects are
``KV_MLA_FMT`` ``[1, num_layers, num_tokens, 544]`` bf16: per token row 512 latent
elements followed by 32 reserved elements for the pooled indexer keys (zero in v1
unless pooled-key buffers are registered). See ``page_contract``.

The vLLM adapter (``lmcache.integration.vllm.vllm_v1_adapter``) calls
``to_gpu``/``from_gpu`` with ``kvcaches=list(kv_caches.values())`` (the registered
per-layer tensors, DSA layer order) and ``slot_mapping`` for the whole token
sequence; ``start``/``end`` index into it. Chunk boundaries fall on page
boundaries (chunk_size is a multiple of 64), every page of a chunk must be whole
and contiguous in the slot mapping, and only the trailing page of a partial
chunk may be partial.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata

from vllm_tt_plugin.lmcache import page_contract as pc

logger = init_logger(__name__)

# vLLM's CopyBlocksOp:
#   (src_tensors, dst_tensors, src_block_ids, dst_block_ids, direction)
CopyBlocksOp = Callable[
    [Mapping[str, Any], Mapping[str, Any], list[int], list[int], str], None
]

# (page id, number of valid tokens in that page)
PageSpan = tuple[int, int]


def _takes_untilize(read: Callable) -> bool:
    """the model's page-DMA read_pages accepts untilize= (page_pool.read_pages(untilize=False) -> TiledPages)"""
    import inspect  # noqa: PLC0415

    try:
        return "untilize" in inspect.signature(read).parameters
    except (TypeError, ValueError):
        return False


def _zero_copy_store() -> bool:
    """GLM53_KV_STORE_ZEROCOPY=1 (Stage 12, program 2026-10-26T07:25:07Z: the store at pool speed; default off; tile
    objects only): a store's whole pages are SAVEd by the page-DMA straight into its memory objects in the pinned pool
    (LatentHitDma.save_resident) -- no ring read, no copy-out, no fill; only the 4 KB pooled halves are written on the
    host. The host runs such stores on the calling thread at the flush (the pages are valid there), not on the store
    worker: they land at once, so a repeat's lookup finds them without waiting."""
    return _tile_mode() and os.environ.get("GLM53_KV_STORE_ZEROCOPY", "0") == "1"


def _tile_mode() -> bool:
    """GLM53_KV_TILE_OBJECTS=1 (Stage 12 lever (c) step 2, program 2026-10-26T03:59:21Z; default off): a memory
    object's whole pages are stored in the device's TILE order -- per layer and page 64 KB of latent tiles then the 4 KB
    pooled half (the model's page_pool.tile_object_pages: the same bytes as 64 token rows, so the object's size and token
    slicing are LMCache's) -- and a partial last page row-major. A hit whose objects live in the model's pinned pool
    (LMCache's CPU tier there: GLM53_LMCACHE_POOL_CPU_GB + the fork's external buffer) pages in by descriptors at their
    own bytes (LatentHitDma.submit_resident: no host copy). Needs the page-DMA store read (read_pages(untilize=False))
    and its own key namespace (the tiers' storage dirs per target: row-major objects are never read by this path)."""
    return os.environ.get("GLM53_KV_TILE_OBJECTS", "0") == "1"


TILE_KEY_NAMESPACE = "#tt-tile-v2"


def check_key_namespace(model_name: str | None) -> None:
    """the tile-object key namespace (program 2026-10-26T03:59:21Z DECISION 2 / 16:12:35Z B2): with
    GLM53_KV_TILE_OBJECTS=1 every LMCache key must carry "#tt-tile-v2" (the fork's extra_config key_namespace, appended
    to the key's model_name on every tier), so a tile-mode lookup can never read a row-major object of the k line --
    and a row-major engine must not carry it. Refuses at the connector host's start (model_name None = no LMCache
    engine on this side: nothing to check)."""
    if model_name is None:
        return
    if _tile_mode() and not str(model_name).endswith(TILE_KEY_NAMESPACE):
        raise RuntimeError(f"GLM53_KV_TILE_OBJECTS=1 needs the LMCache key namespace {TILE_KEY_NAMESPACE!r} "
                           f"(extra_config key_namespace) -- keys are {model_name!r}")  # fmt: skip
    if not _tile_mode() and "#tt-tile" in str(model_name):
        raise RuntimeError(f"LMCache keys carry a tile namespace ({model_name!r}) but GLM53_KV_TILE_OBJECTS is off")


def _untilize_page_bytes(tiles: torch.Tensor) -> torch.Tensor:
    """[m, 65,536] uint8 tile-order latent pages (bf16 32 x 32 tiles of 16 x 16 faces: element (tr, tc, fr, fc, r, c)
    = row tr 32 + fr 16 + r, column tc 32 + fc 16 + c) -> [m, 64, 1,024] uint8 row-major"""
    m = int(tiles.shape[0])
    t = tiles.contiguous().view(torch.int16).view(m, 2, 16, 2, 2, 16, 16)
    return t.permute(0, 1, 3, 5, 2, 4, 6).reshape(m, 64, 512).contiguous().view(torch.uint8).view(m, 64, 1024)


def _tilize_page_bytes(rows: torch.Tensor) -> torch.Tensor:
    """inverse of _untilize_page_bytes: [m, 64, 1,024] uint8 row-major -> [m, 65,536] uint8 tile order"""
    m = int(rows.shape[0])
    r = rows.contiguous().view(torch.int16).view(m, 2, 2, 16, 16, 2, 16)
    return r.permute(0, 1, 4, 2, 5, 3, 6).reshape(m, -1).contiguous().view(torch.uint8)


class VLLMPagedMemTTConnector(GPUConnectorInterface):
    """LMCache ``GPUConnectorInterface`` over TT paged latent host buffers."""

    def __init__(
        self,
        num_layers: int = pc.NUM_DSA_LAYERS,
        tokens_per_page: int = pc.TOKENS_PER_PAGE,
        latent_bytes_per_token: int = pc.LATENT_BYTES_PER_TOKEN_LAYER,
        pooled_bytes_per_token: int = pc.POOLED_BYTES_PER_TOKEN_LAYER,
        layer_names: Sequence[str] | None = None,
        copy_blocks: CopyBlocksOp | None = None,
        device_kv_caches: Mapping[str, Any] | None = None,
        pooled_key_buffers: Mapping[str, torch.Tensor]
        | Sequence[torch.Tensor]
        | None = None,
        use_gpu: bool = False,
    ):
        self.num_layers = int(num_layers)
        self.tokens_per_page = int(tokens_per_page)
        self.latent_bytes_per_token = int(latent_bytes_per_token)
        self.pooled_bytes_per_token = int(pooled_bytes_per_token)
        self.bytes_per_token = self.latent_bytes_per_token + self.pooled_bytes_per_token
        self.hidden_elems = self.bytes_per_token // pc.LATENT_DTYPE.itemsize
        self.layer_names = (
            list(layer_names)
            if layer_names is not None
            else pc.dsa_layer_names(self.num_layers)
        )
        if len(self.layer_names) != self.num_layers:
            raise ValueError("layer_names must have one entry per layer")
        self.use_gpu = use_gpu  # LMCache's "intermediate GPU buffer" flag; unused on TT
        self.copy_blocks: CopyBlocksOp | None = copy_blocks
        # Kept by reference on purpose: a CopyBlocksOp may replace entries (the
        # ttnn-host mock does) and the connector must see the replacement.
        self.device_kv_caches: Mapping[str, Any] = (
            device_kv_caches if device_kv_caches is not None else {}
        )
        self.sharded_pool: Any = None
        # GLM53_KV_POOLED_PAGES (Stage 12): every page's pooled indexer keys ride in its reserved slice. STORE: pooled_source
        # (page ids -> per layer [n, 4,096] uint8, the hit-DMA's layer order) fills it; LOAD: the slices are kept by page
        # id (loaded_pooled) for the host, which hands them to the hit's checkpoint (the model loads them into the
        # request's state slot when it places the request)
        self.pooled_source: Callable[[list[int]], Sequence[Any]] | None = None
        # store path P2 (GLM53_KV_STORE_ONE_DMA): page ids -> the model's (targets, finish) of their pooled reads
        self.pooled_plan: Callable[[list[int]], Any] | None = None
        self.loaded_pooled: dict[int, torch.Tensor] = {}  # page -> [num_layers, 4,096] uint8 (connector layer order)
        self._pooled_held: list[MemoryObj] = []  # tile objects whose pooled halves loaded_pooled VIEWS (refcount held)
        self.kvcaches: list[torch.Tensor] | None = None
        self.num_blocks: int | None = None
        self._bound_id: int | None = None
        self._host_bytes: list[
            torch.Tensor
        ] = []  # per layer [num_blocks, page latent bytes] uint8
        self._host_dict: dict[str, torch.Tensor] = {}
        self._pooled_bytes: list[torch.Tensor] | None = None
        # Byte / page counters (A3 harness convention: row bytes = tokens x 1,024 B
        # per layer).
        self.stats: dict[str, int] = {
            "to_gpu_tokens": 0,
            "from_gpu_tokens": 0,  # token rows x layers moved memobj <-> host
            "to_gpu_row_bytes": 0,
            "from_gpu_row_bytes": 0,  # those rows x latent bytes
            "h2d_pages": 0,
            "d2h_pages": 0,
            "h2d_calls": 0,
            "d2h_calls": 0,  # copy_blocks traffic
            "h2d_s": 0.0,  # host wall of the h2d hop (copy_blocks, or the page-DMA submit)
            "h2d_dma_pages": 0,  # pages submitted to the page-DMA kernel (GLM53_LMCACHE_PAGE_DMA)
            "h2d_dma_direct_pages": 0,  # of those, straight from the memory objects (no host buffer)
        }
        # GLM53_LMCACHE_PAGE_DMA=1: the model's LatentHitDma (tt/page_pool.py) takes the h2d hop: the host pages are
        # tilized into its pinned staging ring and the persistent page-DMA kernel loads them; the connector host waits
        # for it right before the forward (TTKVConnectorHost.load), so the DMA overlaps whatever runs until then.
        self.hit_dma: Any = None
        if pooled_key_buffers is not None:
            self.register_pooled_key_buffers(pooled_key_buffers)

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: torch.device | None = None,
        layout_hints: Any = None,
    ) -> VLLMPagedMemTTConnector:
        """LMCache's factory entry point (``CreateGPUConnector``).

        Finding J5: LMCache sizes ``metadata.kv_shape`` from vLLM's
        ``model_config`` (glm5_next: ``use_mla False``, 45 layers, 64 x 64
        heads), not from the latent cache. When the plugin published the KV
        spec's geometry as ``extra_config["tt_kv_shape_hint"]`` (A1
        ``validate_tt_kv_transfer_config``), the metadata is rewritten from it
        before the engine is built: ``(num MLA layers, 1, chunk, 1, latent +
        reserved)``, ``use_mla True``. Without a hint the metadata must already
        describe the TT latent.
        """
        extra = metadata.kv_connector_extra_config or {}
        hint = extra.get("tt_kv_shape_hint")
        latent_bytes = pc.LATENT_BYTES_PER_TOKEN_LAYER
        layer_names: Sequence[str] | None = None
        chunk = int(metadata.kv_shape[2])
        if hint:
            num_layers = int(hint["num_layers"])
            latent_dim = int(hint["latent_dim"])
            latent_bytes = latent_dim * pc.LATENT_DTYPE.itemsize
            hidden = (latent_bytes + pc.POOLED_BYTES_PER_TOKEN_LAYER) // (
                pc.LATENT_DTYPE.itemsize
            )
            layer_names = list(hint.get("layer_names") or ()) or None
            rewritten = (num_layers, 1, chunk, 1, hidden)
            if tuple(metadata.kv_shape) != rewritten or not metadata.use_mla:
                logger.info(
                    "VLLMPagedMemTTConnector: LMCache metadata kv_shape %s use_mla %s "
                    "-> %s use_mla True (tt_kv_shape_hint from the vLLM KV spec)",
                    tuple(metadata.kv_shape),
                    metadata.use_mla,
                    rewritten,
                )
                metadata.kv_shape = rewritten
                metadata.use_mla = True
        num_layers, kv_dim, chunk, kv_heads, head = metadata.kv_shape
        hidden = kv_heads * head
        expected_hidden = (latent_bytes + pc.POOLED_BYTES_PER_TOKEN_LAYER) // (
            pc.LATENT_DTYPE.itemsize
        )
        if kv_dim != 1 or not metadata.use_mla:
            raise ValueError(
                "TT latent cache is MLA-style: kv_shape[1] must be 1 and use_mla True"
            )
        if hidden != expected_hidden:
            raise ValueError(
                f"kv_shape hidden {hidden} != {expected_hidden} "
                f"({latent_bytes // pc.LATENT_DTYPE.itemsize} latent + "
                f"{pc.POOLED_BYTES_PER_TOKEN_LAYER // pc.LATENT_DTYPE.itemsize} "
                "reserved)"
            )
        if metadata.kv_dtype != pc.LATENT_DTYPE:
            raise ValueError(f"kv_dtype {metadata.kv_dtype} != {pc.LATENT_DTYPE}")
        tokens_per_page = int(
            extra.get(
                "tt_tokens_per_page",
                (hint or {}).get("tokens_per_page", pc.TOKENS_PER_PAGE),
            )
        )
        if chunk % tokens_per_page:
            raise ValueError(
                f"chunk {chunk} is not a multiple of the page ({tokens_per_page})"
            )
        logger.info(
            "VLLMPagedMemTTConnector: %d DSA layers, page %d tokens, chunk %d tokens, "
            "%d B/token/layer (device %s)",
            num_layers,
            tokens_per_page,
            chunk,
            latent_bytes + pc.POOLED_BYTES_PER_TOKEN_LAYER,
            device,
        )
        return cls(
            num_layers=num_layers,
            tokens_per_page=tokens_per_page,
            latent_bytes_per_token=latent_bytes,
            layer_names=layer_names,
            use_gpu=use_gpu,
        )

    def set_host_xfer_buffer_ops(self, copy_blocks: CopyBlocksOp | None) -> None:
        """Install the plugin's ``CopyBlocksOp`` (vLLM ``set_host_xfer_buffer_ops``)."""
        self.copy_blocks = copy_blocks

    def set_hit_dma(self, dma: Any) -> None:
        """Route the h2d hop through the model's page-DMA page-in (``dma.layer_names`` = its layer order)."""
        if dma is not None and sorted(dma.layer_names) != sorted(self.layer_names):
            raise ValueError(
                f"page-DMA layers {dma.layer_names} differ from the connector's {self.layer_names}"
            )
        self.hit_dma = dma

    def register_device_kv_caches(self, device_kv_caches: Mapping[str, Any]) -> None:
        """Device-side (ttnn) handles for the copy op, keyed like the host buffers."""
        self.device_kv_caches = device_kv_caches

    def enable_pooled_pages(self, source: Callable[[list[int]], Sequence[Any]] | None, plan=None) -> None:
        """GLM53_KV_POOLED_PAGES: ``source(page_ids)`` -> per layer (``hit_dma.layer_names`` order) ``[n, 4,096]`` uint8
        pooled-key halves of those pages (device-native, the model's ``kv_read_pooled_pages``); None turns it off.
        plan (store path P2): ``plan(page_ids)`` -> (targets, finish), the same read split for the page-DMA's
        ``save_resident_pooled`` (finish(its arrays) = source(page_ids))."""
        self.pooled_source = source
        self.pooled_plan = plan if source is not None else None
        self.release_loaded_pooled()

    def _pooled_layer_order(self) -> list[int]:
        """connector layer i -> the index of its array in the pooled source's (the hit-DMA's) layer order"""
        names = getattr(self.hit_dma, "layer_names", None) or self.layer_names
        return [list(names).index(n) for n in self.layer_names]

    def _pooled_for(self, page_ids: list[int]) -> dict[int, torch.Tensor]:
        """{page: [num_layers, 4,096] uint8 (connector layer order)} from pooled_source (one call for the union)"""
        uniq = self._unique(page_ids)
        if not uniq:
            return {}
        arr = self._pooled_array(uniq)
        return {p: arr[k] for k, p in enumerate(uniq)}

    def _pooled_array(self, uniq: list[int], got: Sequence[Any] | None = None) -> torch.Tensor:
        """[len(uniq), num_layers, 4,096] uint8 (connector layer order): pooled_source for distinct pages, one call
        (got: that call's result, already read -- store path P2's one-wait store)"""
        if got is None:
            got = self.pooled_source(uniq)
        order = self._pooled_layer_order()
        width = self.tokens_per_page * self.pooled_bytes_per_token
        arr = torch.stack([torch.as_tensor(got[j]).view(torch.uint8).reshape(len(uniq), -1) for j in order], dim=1)
        if tuple(arr.shape) != (len(uniq), self.num_layers, width):
            raise ValueError(f"pooled source returned {tuple(arr.shape)} for {len(uniq)} pages")
        self.stats["pooled_store_pages"] = self.stats.get("pooled_store_pages", 0) + len(uniq)
        return arr

    def _fill_pages(self, memory_obj: MemoryObj, pages: list[PageSpan], row_of: Callable[[int], int],
                    lat_all: torch.Tensor, pool_all: torch.Tensor | None) -> bool:
        """a store's memory object filled from page-major sources in ONE copy per column group for all layers (Stage
        12 put/get, program 2026-10-24T02:08:04Z: the T2 store was 2 Python copies per page per layer -- a 32K
        store 0.26 s of host time). lat_all [L, P, T, LB] / pool_all [L, P, T, PB] (None: zeros) uint8, page p at
        source row row_of(p). False (the caller's per-page path) unless every page but the last is full and the
        pages' rows are consecutive."""
        T, LB = self.tokens_per_page, self.latent_bytes_per_token
        if not pages or any(cnt != T for _p, cnt in pages[:-1]):
            return False
        ks = [row_of(p) for p, _ in pages]
        k0, m = ks[0], len(ks)
        if ks != list(range(k0, k0 + m)):
            return False
        n = sum(cnt for _, cnt in pages)
        rows = self._rows(memory_obj, n)
        L = self.num_layers
        rows[..., :LB].copy_(lat_all[:, k0 : k0 + m].reshape(L, m * T, LB)[:, :n])
        if pool_all is not None:
            rows[..., LB:].copy_(pool_all[:, k0 : k0 + m].reshape(L, m * T, -1)[:, :n])
        else:
            rows[..., LB:].zero_()
        memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
        return True

    def _tile_fill(self, memory_obj: MemoryObj, pages: list[PageSpan], row_of: Callable[[int], int],
                   tiles_all: torch.Tensor, pool_all: torch.Tensor | None) -> None:
        """a store's memory object in the tile-object format (_tile_mode): its whole pages as [64 KB latent tiles |
        4 KB pooled half], a partial last page row-major. tiles_all [L, P, 65,536] / pool_all [L, P, 4,096] (None:
        zeros) uint8, page p at source row row_of(p)."""
        T, LB, L = self.tokens_per_page, self.latent_bytes_per_token, self.num_layers
        if any(cnt != T for _p, cnt in pages[:-1]):
            raise ValueError("tile objects: only an object's last page may be partial")
        n = sum(cnt for _, cnt in pages)
        rows = self._rows(memory_obj, n)
        full = [p for p, cnt in pages if cnt == T]
        m = len(full)
        if m:
            idx = torch.as_tensor([row_of(p) for p in full], dtype=torch.int64)
            dst = rows[:, : m * T].view(L, m, T * self.bytes_per_token)
            dst[..., : T * LB].copy_(tiles_all.index_select(1, idx))
            if pool_all is not None:
                dst[..., T * LB :].copy_(pool_all.index_select(1, idx))
            else:
                dst[..., T * LB :].zero_()
        if m < len(pages):  # the partial last page: its tokens' rows, row-major
            page, cnt = pages[-1]
            k = row_of(page)
            rows[:, m * T :, :LB].copy_(_untilize_page_bytes(tiles_all[:, k])[:, :cnt])
            if pool_all is not None:
                rows[:, m * T :, LB:].copy_(pool_all[:, k].reshape(L, T, -1)[:, :cnt])
            else:
                rows[:, m * T :, LB:].zero_()
        memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
        self.stats["tile_object_pages"] = self.stats.get("tile_object_pages", 0) + m

    def _read_rows(self, memory_obj: MemoryObj, pages: list[PageSpan]) -> torch.Tensor:
        """an object's rows [L, n, bytes per token] row-major for the host paths: the object itself, or (_tile_mode)
        a row-major copy of its tile-order whole pages + its row-major partial last page"""
        n = sum(cnt for _, cnt in pages)
        rows = self._rows(memory_obj, n)
        if not _tile_mode():
            return rows
        T, LB, L = self.tokens_per_page, self.latent_bytes_per_token, self.num_layers
        m = sum(1 for _p, cnt in pages if cnt == T)
        out = torch.empty_like(rows)
        if m:
            src = rows[:, : m * T].view(L, m, T * self.bytes_per_token)
            out[:, : m * T, :LB].copy_(_untilize_page_bytes(src[..., : T * LB].reshape(L * m, -1)).view(L, m * T, LB))
            out[:, : m * T, LB:].copy_(src[..., T * LB :].reshape(L, m * T, -1))
        out[:, m * T :].copy_(rows[:, m * T :])
        return out

    def _keep_loaded_pooled(self, memory_obj: MemoryObj, pages: list[PageSpan]) -> None:
        """a LOAD's pooled slices by page id (whole pages; a partial page's slice is padded with zeros)"""
        if self.pooled_source is None:
            return
        T, LB = self.tokens_per_page, self.latent_bytes_per_token
        n = sum(cnt for _, cnt in pages)
        pooled = self._read_rows(memory_obj, pages)[..., LB:]  # [L, n, 64] uint8
        b = torch.zeros(self.num_layers, len(pages) * T, pooled.shape[-1], dtype=torch.uint8)  # one per object
        tok = 0
        for k, (_page, cnt) in enumerate(pages):
            b[:, k * T : k * T + cnt] = pooled[:, tok : tok + cnt]  # (whole pages: one span; a partial page padded)
            tok += cnt
        b = b.view(self.num_layers, len(pages), -1)
        for k, (page, _cnt) in enumerate(pages):
            self.loaded_pooled[int(page)] = b[:, k]

    def _keep_loaded_pooled_many(self, objs: list, rows: list | None = None) -> None:
        """_keep_loaded_pooled for a LOAD's objects at once (Stage 12 lever (c), whole pages only -- the page-DMA path):
        ONE buffer [L, pages, 4,096] filled by one strided copy per object, a page's entry a view of it (was a zeros
        buffer + a copy per object: 7.5 ms of a 32K hit)"""
        if self.pooled_source is None or not objs:
            return
        T, LB = self.tokens_per_page, self.latent_bytes_per_token
        width = T * self.pooled_bytes_per_token
        if _tile_mode() and os.environ.get("GLM53_KV_POOLED_VIEWS", "1") == "1":
            # tile objects keep a page's 4 KB pooled half contiguous: a page's entry is a VIEW of the object (no copy on
            # the engine thread: ~7 ms of a 32K hit), the object's LMCache refcount held until release_loaded_pooled
            # (the host's, after the step's KDA imports took -- copied -- their pages)
            for k, (memory_obj, pages) in enumerate(objs):
                m = len(pages)
                r = rows[k] if rows is not None else self._rows(memory_obj, m * T)  # (the caller's _rows: one per object)
                pooled = r.view(self.num_layers, m, T * self.bytes_per_token)[..., T * LB :]
                memory_obj.ref_count_up()
                self._pooled_held.append(memory_obj)
                for j, (page, _cnt) in enumerate(pages):
                    self.loaded_pooled[int(page)] = pooled[:, j]
            self.stats["pooled_view_pages"] = self.stats.get("pooled_view_pages", 0) + sum(len(p) for _o, p in objs)
            return
        n_pages = sum(len(pages) for _o, pages in objs)
        b = torch.empty(self.num_layers, n_pages, T, self.pooled_bytes_per_token, dtype=torch.uint8)
        k = 0
        for i, (memory_obj, pages) in enumerate(objs):
            m = len(pages)
            r = rows[i] if rows is not None else self._rows(memory_obj, m * T)
            if _tile_mode():  # the pages' 4 KB pooled halves (whole pages only on the page-DMA path)
                pooled = r.view(self.num_layers, m, T * self.bytes_per_token)[..., T * LB :]
            else:
                pooled = r[..., LB:]  # [L, m T, 64] strided
            b[:, k : k + m].copy_(pooled.reshape(self.num_layers, m, T, -1))
            for j, (page, _cnt) in enumerate(pages):
                self.loaded_pooled[int(page)] = b[:, k + j].view(self.num_layers, width)
            k += m

    def release_loaded_pooled(self) -> None:
        """drop every kept slice (the pages no hit row took) and the LMCache refcounts the pooled VIEWS held: the host
        calls it once the step's KDA imports took their pages (taken slices are copies, the objects are free to go)"""
        self.loaded_pooled.clear()
        held, self._pooled_held = self._pooled_held, []
        for o in held:
            o.ref_count_down()

    def take_loaded_pooled(self, page_ids: Sequence[int]) -> list[torch.Tensor] | None:
        """the host's side of a LOAD: per layer (the pooled source's = hit-DMA's order) [n, 4,096] uint8 of page_ids,
        popped; None when any page was not loaded with its slice (the hit then has no pooled keys)"""
        if any(int(p) not in self.loaded_pooled for p in page_ids):
            return None
        width = self.tokens_per_page * self.pooled_bytes_per_token
        # numpy's single-threaded stack (torch.stack of ~500 small strided slices pays intra-op thread overhead: 10-22
        # ms at torch's default 64 threads on d0 vs 5 ms at 8, lever_c/getprof)
        import numpy as np  # noqa: PLC0415

        arr = (torch.from_numpy(np.stack([self.loaded_pooled.pop(int(p)).numpy() for p in page_ids])) if len(page_ids)
               else torch.zeros(0, self.num_layers, width, dtype=torch.uint8))  # fmt: skip
        order = self._pooled_layer_order()
        inv = [order.index(j) for j in range(self.num_layers)]  # source layer j -> connector layer
        return [arr[:, inv[j]] for j in range(self.num_layers)]

    def register_sharded_pool(self, pool: Any) -> None:
        """GLM53_DSA_SHARD (vllm_tt_plugin.lmcache.sharded_pool.ShardedPagePool): a LOAD's pages are admitted at their
        request columns (position // page) before the h2d copy, so each lands on its unit's owner chip; a STORE reads
        each page from its owner. ``None`` restores the replicated behaviour."""
        self.sharded_pool = pool

    def register_pooled_key_buffers(
        self, buffers: Mapping[str, torch.Tensor] | Sequence[torch.Tensor]
    ) -> None:
        """Optional pooled-indexer-key host buffers (per layer, dim 0 = num_blocks)."""
        tensors = (
            list(buffers.values()) if isinstance(buffers, Mapping) else list(buffers)
        )
        if len(tensors) != self.num_layers:
            raise ValueError("one pooled-key buffer per layer expected")
        page_bytes = self.tokens_per_page * self.pooled_bytes_per_token
        self._pooled_bytes = [
            self._as_page_bytes(t, page_bytes, "pooled-key") for t in tensors
        ]

    def initialize_kvcaches_ptr(self, **kwargs) -> None:
        """Bind ``kvcaches`` (the per-layer host transfer buffers) when passed."""
        if "kvcaches" in kwargs and kwargs["kvcaches"] is not None:
            self._bind(kwargs["kvcaches"])

    def _bind(
        self, kvcaches: Mapping[str, torch.Tensor] | Sequence[torch.Tensor]
    ) -> None:
        if isinstance(kvcaches, Mapping):
            names, tensors = list(kvcaches.keys()), list(kvcaches.values())
        else:
            tensors = list(kvcaches)
            names = self.layer_names
        if id(kvcaches) == self._bound_id and self.kvcaches is not None:
            return
        if len(tensors) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} per-layer host buffers, got {len(tensors)}"
            )
        page_bytes = self.tokens_per_page * self.latent_bytes_per_token
        host_bytes = [self._as_page_bytes(t, page_bytes, "latent") for t in tensors]
        num_blocks = {hb.shape[0] for hb in host_bytes}
        if len(num_blocks) != 1:
            raise ValueError(
                f"host buffers disagree on num_blocks: {sorted(num_blocks)}"
            )
        self.kvcaches = tensors
        self.num_blocks = num_blocks.pop()
        self._host_bytes = host_bytes
        self._host_dict = dict(zip(names, tensors))
        self.layer_names = list(names)
        self._bound_id = id(kvcaches)

    @staticmethod
    def _as_page_bytes(t: torch.Tensor, page_bytes: int, what: str) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                f"{what} host buffer must be a torch.Tensor, got {type(t).__name__}"
            )
        if t.device.type != "cpu":
            raise ValueError(
                f"{what} host buffer must live on the host, got {t.device}"
            )
        if not t.is_contiguous():
            raise ValueError(f"{what} host buffer must be contiguous")
        flat = t.view(t.shape[0], -1)
        if flat.shape[1] * t.element_size() != page_bytes:
            raise ValueError(
                f"{what} host buffer has {flat.shape[1] * t.element_size()} B "
                "per block, "
                f"expected {page_bytes}"
            )
        return flat.view(torch.uint8)

    def _require_bound(self) -> None:
        if self.kvcaches is None:
            raise ValueError("'kvcaches' must be passed in kwargs or bound beforehand")

    # ------------------------------------------------------------- slots/pages
    def pages_from_slot_mapping(
        self, slot_mapping: Any, start: int, end: int
    ) -> list[PageSpan]:
        """Translate ``slot_mapping[start:end]`` into whole pages.

        Raises ``ValueError`` if the range does not start on a page boundary, a
        slot is negative (prefix overlap), a page is not contiguous in the
        mapping, or a page other than the last is partial.
        """
        n = end - start
        if n <= 0:
            return []
        T = self.tokens_per_page
        if start % T:
            raise ValueError(f"start {start} is not page aligned ({T})")
        slots = slot_mapping[start:end]
        slots = (
            slots.detach().to("cpu", torch.long)
            if isinstance(slots, torch.Tensor)
            else torch.as_tensor(list(slots), dtype=torch.long)
        )
        if slots.numel() != n:
            raise ValueError(
                f"slot_mapping covers {slots.numel()} tokens, range needs {n}"
            )
        pages: list[PageSpan] = []
        for k in range(math.ceil(n / T)):
            seg = slots[k * T : min((k + 1) * T, n)]
            first = int(seg[0])
            if first < 0:
                raise ValueError(
                    "negative slot inside the range (overlaps the cached prefix)"
                )
            if first % T:
                raise ValueError(
                    f"page {k} starts at slot {first}, not on a page boundary"
                )
            if not torch.equal(
                seg, torch.arange(first, first + seg.numel(), dtype=torch.long)
            ):
                raise ValueError(f"page {k} is not contiguous in the slot mapping")
            pages.append((first // T, int(seg.numel())))
        return pages

    def pages_of_ranges(self, slot_mapping: Any, starts: Sequence[int], ends: Sequence[int]) -> list[list[PageSpan]]:
        """pages_from_slot_mapping for every (start, end) of a batched call at once (Stage 12 lever (c): per object it
        was 4 torch.arange + torch.equal + tolist per page -- ~7 ms of a 32K hit's 125 objects): ONE conversion of the
        slot mapping and one vectorized check of every covered position (slot = its page's first slot + offset; that
        first slot >= 0 and page aligned; a page start is T-aligned since every start is), the spans from one tolist.
        Any violation (or an uncovered tail) -> the per-range translation, for its exact error."""
        T = self.tokens_per_page
        se = [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]
        live = [(s, e) for s, e in se if e > s]
        if not live or any(s % T for s, _e in live):
            return [self.pages_from_slot_mapping(slot_mapping, s, e) for s, e in se]
        lo, hi = min(s for s, _e in live), max(e for _s, e in live)
        full = (slot_mapping.detach().to("cpu", torch.long) if isinstance(slot_mapping, torch.Tensor)
                else torch.as_tensor(list(slot_mapping), dtype=torch.long))  # fmt: skip
        if full.numel() < hi:
            return [self.pages_from_slot_mapping(slot_mapping, s, e) for s, e in se]
        seg = full[lo:hi]
        off = torch.arange(hi - lo, dtype=torch.long) % T
        first = seg[torch.arange(hi - lo, dtype=torch.long) - off]
        bad = (seg != first + off) | (first < 0) | (first % T != 0)
        tiled = sorted(live) == live and all(live[i][1] == live[i + 1][0] for i in range(len(live) - 1))
        if not tiled:  # ranges with gaps / out of order: only covered positions count
            cov = torch.zeros(hi - lo, dtype=torch.bool)
            for s, e in live:
                cov[s - lo : e - lo] = True
            bad &= cov
        if bool(bad.any()):
            return [self.pages_from_slot_mapping(slot_mapping, s, e) for s, e in se]
        pid = (seg[::T] // T).tolist()
        return [[(pid[(s - lo) // T + k], min(T, e - s - k * T)) for k in range(-(-(e - s) // T))] if e > s else []
                for s, e in se]  # fmt: skip

    # ------------------------------------------------------------ memobj views
    def _rows(self, memory_obj: MemoryObj, n_tokens: int) -> torch.Tensor:
        """Byte view ``[num_layers, n_tokens, bytes_per_token]`` of a memory object."""
        t = memory_obj.tensor
        if t is None:
            raise AssertionError("memory object has no tensor")
        fmt = memory_obj.metadata.fmt
        if fmt not in (MemoryFormat.KV_MLA_FMT, MemoryFormat.UNDEFINED):
            raise ValueError(
                f"memory object must be KV_MLA_FMT for the TT connector, got {fmt}"
            )
        if not t.is_contiguous():
            raise ValueError("memory object tensor must be contiguous")
        want = self.num_layers * n_tokens * self.bytes_per_token
        have = t.numel() * t.element_size()
        if have < want:
            raise ValueError(
                f"memory object holds {have} B, {n_tokens} tokens need {want} B"
            )
        flat = t.reshape(-1).view(torch.uint8)[:want]
        return flat.view(self.num_layers, n_tokens, self.bytes_per_token)

    _BULK_PAGES = 256  # pages per gather / scatter: bounds the temporary to 256 pages x page bytes per layer

    def _bulk_ok(self, pages: list[PageSpan]) -> bool:
        """Stage 12 (n29h7 M1, program 2026-10-23T19:24:42Z: 143 of 150 engine stall samples in the per-page,
        per-layer slice copies): every page but the last is full, so a span's tokens are its pages' rows in order and
        one gather / scatter per layer moves them. GLM53_LMCACHE_BULK_COPY=0: the per-page loop."""
        T = self.tokens_per_page
        return (bool(pages) and all(cnt == T for _p, cnt in pages[:-1])
                and os.environ.get("GLM53_LMCACHE_BULK_COPY", "1") == "1")  # fmt: skip

    def _memobj_to_host(self, memory_obj: MemoryObj, pages: list[PageSpan]) -> None:
        self._keep_loaded_pooled(memory_obj, pages)
        n = sum(cnt for _, cnt in pages)
        rows = self._read_rows(memory_obj, pages)
        LB, T = self.latent_bytes_per_token, self.tokens_per_page
        latent, pooled = rows[..., :LB], rows[..., LB:]
        if self._bulk_ok(pages):
            ids = [p for p, _ in pages]
            for s0 in range(0, len(ids), self._BULK_PAGES):
                part = ids[s0 : s0 + self._BULK_PAGES]
                t0, t1 = s0 * T, min(n, (s0 + len(part)) * T)
                full = (t1 - t0) // T  # the part's full pages; a partial one can only be the span's last
                c = (t1 - t0) - full * T
                idx = torch.as_tensor(part[:full], dtype=torch.int64)
                for layer in range(self.num_layers):
                    stores = [(self._host_bytes[layer], latent)]
                    if self._pooled_bytes is not None:
                        stores.append((self._pooled_bytes[layer], pooled))
                    for hb, src in stores:
                        if full:
                            hb.index_copy_(0, idx, src[layer, t0 : t0 + full * T].contiguous().view(full, -1))
                        if c:  # the partial last page: its tokens, then zeros (no rows of an earlier request)
                            host = hb[part[full]].view(T, -1)
                            host[:c].copy_(src[layer, t0 + full * T : t1])
                            host[c:].zero_()
            self.stats["to_gpu_tokens"] += n * self.num_layers
            self.stats["to_gpu_row_bytes"] += n * self.num_layers * LB
            return
        tok = 0
        for page, cnt in pages:
            for layer in range(self.num_layers):
                host = self._host_bytes[layer][page].view(T, LB)
                host[:cnt].copy_(latent[layer, tok : tok + cnt])
                if cnt < T:
                    # a partial page (an unfull chunk's last page): the h2d moves the WHOLE page, so rows past the
                    # loaded tokens must not carry what this host page held for an earlier request (Stage 12, round-5
                    # cross-request leak check, program 2026-10-23T13:58:48Z)
                    host[cnt:].zero_()
                if self._pooled_bytes is not None:
                    pooled_host = self._pooled_bytes[layer][page].view(T, -1)
                    pooled_host[:cnt].copy_(pooled[layer, tok : tok + cnt])
                    if cnt < T:
                        pooled_host[cnt:].zero_()
            tok += cnt
        self.stats["to_gpu_tokens"] += n * self.num_layers
        self.stats["to_gpu_row_bytes"] += n * self.num_layers * LB

    def _host_to_memobj(
        self, memory_obj: MemoryObj, pages: list[PageSpan], src: dict | None = None, job: dict | None = None
    ) -> None:
        if _tile_mode():
            if job is None or not job.get("tile"):
                raise RuntimeError("GLM53_KV_TILE_OBJECTS: a store must go through the page-DMA read (tile pages)")
            st = job.get("_tile_stacked")
            if st is None:
                st = job["_tile_stacked"] = (torch.stack(list(job["bytes"])),
                                             torch.stack(list(job["pooled"])) if job.get("pooled") is not None else None)  # fmt: skip
            self._tile_fill(memory_obj, pages, job["row"].__getitem__, *st)
            return
        n = sum(cnt for _, cnt in pages)
        rows = self._rows(memory_obj, n)
        LB, T = self.latent_bytes_per_token, self.tokens_per_page
        latent, pooled = rows[..., :LB], rows[..., LB:]
        if src is None and self.pooled_source is not None and job is None:
            src = self._pooled_for([p for p, _ in pages])
        # a store-worker job (prefetch_store_pages): the pages' rows as the engine thread read them, never the shared
        # host buffers (a block may be read again for a later store while this one waits)
        host = job["bytes"] if job is not None else self._host_bytes
        pooled_host = job.get("pooled") if job is not None else self._pooled_bytes
        row_of = job["row"].__getitem__ if job is not None else (lambda p: p)
        if job is not None:  # one copy per column group for all layers (_fill_pages); the job's sources stacked once
            st = job.get("_stacked")
            if st is None:
                T_ = self.tokens_per_page
                lat_all = torch.stack([t.reshape(t.shape[0], T_, -1) for t in host])
                pool_all = (torch.stack([t.reshape(t.shape[0], T_, -1) for t in pooled_host])
                            if pooled_host is not None else None)  # fmt: skip
                st = job["_stacked"] = (lat_all, pool_all)
            if self._fill_pages(memory_obj, pages, row_of, *st):
                self.stats["from_gpu_tokens"] += n * self.num_layers
                self.stats["from_gpu_row_bytes"] += n * self.num_layers * LB
                return
        if self._bulk_ok(pages):
            ids = [p for p, _ in pages]
            for s0 in range(0, len(ids), self._BULK_PAGES):
                part = ids[s0 : s0 + self._BULK_PAGES]
                t0, t1 = s0 * T, min(n, (s0 + len(part)) * T)
                m = t1 - t0
                idx = torch.as_tensor([row_of(p) for p in part], dtype=torch.int64)
                for layer in range(self.num_layers):
                    latent[layer, t0:t1].copy_(host[layer].index_select(0, idx).view(len(part) * T, LB)[:m])
                    if src is not None:
                        g = torch.stack([src[p][layer].reshape(-1) for p in part]).view(len(part) * T, -1)
                        pooled[layer, t0:t1].copy_(g[:m])
                    elif pooled_host is not None:
                        g = pooled_host[layer].index_select(0, idx).view(len(part) * T, -1)
                        pooled[layer, t0:t1].copy_(g[:m])
                    else:
                        pooled[layer, t0:t1].zero_()
            self.stats["from_gpu_tokens"] += n * self.num_layers
            self.stats["from_gpu_row_bytes"] += n * self.num_layers * LB
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
            return
        tok = 0
        for page, cnt in pages:
            for layer in range(self.num_layers):
                latent[layer, tok : tok + cnt].copy_(
                    host[layer][row_of(page)].view(T, LB)[:cnt]
                )
                if src is not None:
                    pooled[layer, tok : tok + cnt].copy_(src[page][layer].view(T, -1)[:cnt])
                elif pooled_host is not None:
                    pooled[layer, tok : tok + cnt].copy_(
                        pooled_host[layer][row_of(page)].view(T, -1)[:cnt]
                    )
                else:
                    pooled[layer, tok : tok + cnt].zero_()
            tok += cnt
        self.stats["from_gpu_tokens"] += n * self.num_layers
        self.stats["from_gpu_row_bytes"] += n * self.num_layers * LB
        memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    # ------------------------------------------------------------- device hop
    def _dma_targets(
        self, page_ids: list[int], placed: Mapping[int, tuple[int, int]] | None = None
    ) -> tuple[list[int], dict[str, Any]]:
        """the page-DMA's block ids (+ kwargs) for logical pages: replicated = the ids themselves; the sharded pool
        (GLM53_DSA_SHARD) = each page's LOCAL block on its OWNER chip (``owners=``; LatentHitDma's per-chip descriptors).
        A LOAD passes ``placed`` (``_admit``'s placement); a STORE locates the mapped blocks (refused without content)."""
        if self.sharded_pool is None:
            return list(page_ids), {}
        if placed is None:
            placed = {p: self.sharded_pool.locate(p) for p in self._unique(page_ids)}
        try:
            owner_local = [placed[p] for p in page_ids]
        except KeyError as e:
            raise ValueError(f"page {e.args[0]} was not admitted on the sharded pool before its page-DMA load") from None
        return [int(lo) for _o, lo in owner_local], {"owners": [int(o) for o, _lo in owner_local]}

    def _t2_read(self) -> Any:
        """the model DMA's read_pages when the store's d2h goes through it (GLM53_KV_T2_STORE, default 1), else None"""
        read = getattr(self.hit_dma, "read_pages", None) if self.hit_dma is not None else None
        return read if callable(read) and os.environ.get("GLM53_KV_T2_STORE", "1") == "1" else None

    def _move(
        self, page_ids: list[int], direction: str, placed: Mapping[int, tuple[int, int]] | None = None
    ) -> None:
        if not page_ids:
            return
        if direction == "h2d" and self.hit_dma is not None:
            t0 = time.perf_counter()
            index = torch.as_tensor(page_ids, dtype=torch.int64)
            order = [self.layer_names.index(n) for n in self.hit_dma.layer_names]
            ids, kw = self._dma_targets(page_ids, placed if placed is not None else {})
            self._dma_load()(
                [self._host_bytes[i].index_select(0, index).numpy() for i in order], ids, **kw
            )
            self.stats["h2d_s"] += time.perf_counter() - t0
            self.stats["h2d_dma_pages"] += len(page_ids)
            self.stats["h2d_pages"] += len(page_ids)
            self.stats["h2d_calls"] += 1
            return
        read = self._t2_read()
        if direction == "d2h" and read is not None:
            # Stage 12 (program 2026-10-23T03:54Z item 2): the store's page reads through the page-DMA SAVE
            # direction (the model's LatentHitDma.read_pages: pages DMA'd from the save chip into the pinned ring,
            # untilized) instead of a SlotOps copy + to_torch per page (0.06 GB/s on n29h5)
            t0 = time.perf_counter()
            index = torch.as_tensor(page_ids, dtype=torch.int64)
            order = [self.layer_names.index(n) for n in self.hit_dma.layer_names]
            ids, kw = self._dma_targets(page_ids)
            got = read(ids, **kw)
            for j, i in enumerate(order):
                rows = torch.from_numpy(got[j].reshape(len(page_ids), -1).view("u1"))
                self._host_bytes[i].index_copy_(0, index, rows)
            self.stats["d2h_s"] = self.stats.get("d2h_s", 0.0) + time.perf_counter() - t0
            self.stats["d2h_dma_pages"] = self.stats.get("d2h_dma_pages", 0) + len(page_ids)
            self.stats["d2h_pages"] += len(page_ids)
            self.stats["d2h_calls"] += 1
            return
        if self.copy_blocks is None:
            return
        if direction == "h2d":
            t0 = time.perf_counter()
            self.copy_blocks(
                self._host_dict, self.device_kv_caches, page_ids, page_ids, "h2d"
            )
            self.stats["h2d_s"] += time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            self.copy_blocks(
                self.device_kv_caches, self._host_dict, page_ids, page_ids, "d2h"
            )
            self.stats["d2h_s"] = self.stats.get("d2h_s", 0.0) + time.perf_counter() - t0
        self.stats[f"{direction}_pages"] += len(page_ids)
        self.stats[f"{direction}_calls"] += 1

    def _dma_load(self) -> Any:
        """the model DMA's hit load: load_many (any size: ring-sized batches, Stage 12 -- a 1M-token hit is 11.5 GB per
        chip, the staging ring ~0.8 GB) when the model has it, else submit (one ring)"""
        return getattr(self.hit_dma, "load_many", None) or self.hit_dma.submit

    @staticmethod
    def _unique(ids: list[int]) -> list[int]:
        return list(dict.fromkeys(ids))

    # ------------------------------------------------------------ interface
    def _admit(self, spans: list[tuple[int, list[PageSpan]]]) -> dict[int, tuple[int, int]] | None:
        """the sharded pool: every page of a LOAD at its request column (``start // page + k``) before the h2d copy;
        returns {page: (owner chip, local block)} (None on the replicated pool)"""
        if self.sharded_pool is None:
            return None
        pairs: dict[int, int] = {}
        for start, pages in spans:
            for col, (page, _cnt) in zip(self.sharded_pool.columns_of(start, len(pages)), pages, strict=True):
                if pairs.setdefault(page, col) != col:
                    raise ValueError(f"page {page} loaded at two columns ({pairs[page]}, {col})")
        if not pairs:
            return {}
        items = list(pairs.items())
        return {b: (int(o), int(lo)) for (b, _j), (o, lo) in zip(items, self.sharded_pool.admit(items), strict=True)}

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs) -> None:
        """Memory object -> host buffer pages -> device (``copy_blocks`` h2d)."""
        self.initialize_kvcaches_ptr(**kwargs)
        self._require_bound()
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs")
        pages = self.pages_from_slot_mapping(kwargs["slot_mapping"], start, end)
        placed = self._admit([(start, pages)])  # the sharded pool: before ANY device write (the page-DMA's too)
        if self.hit_dma is not None and self._dma_to_gpu([memory_obj], [pages], placed):
            return
        self._memobj_to_host(memory_obj, pages)
        self._move([p for p, _ in pages], "h2d", placed)

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs) -> None:
        """Device (``copy_blocks`` d2h) -> host buffer pages -> memory object."""
        self.initialize_kvcaches_ptr(**kwargs)
        self._require_bound()
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs")
        pages = self.pages_from_slot_mapping(kwargs["slot_mapping"], start, end)
        job = self.current_store_job()
        if job is not None:  # the store worker: the engine thread already read these pages (no device call here)
            self._host_to_memobj(memory_obj, pages, job=job)
            return
        if self._dma_from_gpu([memory_obj], [pages]):
            return
        self._move([p for p, _ in pages], "d2h")
        self._host_to_memobj(memory_obj, pages)

    def _dma_from_gpu(self, memory_objs: list, spans: list) -> bool:
        """GLM53_KV_T2_STORE (with the model's page-DMA): the page union read once (read_pages; each page from its owner
        chip on the sharded pool) STRAIGHT into the memory objects -- the row-major host buffers are never written, so
        they stay uncommitted (a pool of P tokens would otherwise make 11 KiB x P of host RAM resident over a serve).
        The reserved pooled-key slice: the registered pooled buffers' rows, else zero. False without the read."""
        read = self._t2_read()
        if read is None:
            return False
        uniq = self._unique([p for pages in spans for p, _ in pages])
        if not uniq:
            return True
        t0 = time.perf_counter()
        ids, kw = self._dma_targets(uniq)
        if _tile_mode():
            return self._dma_from_gpu_tile(memory_objs, spans, uniq, read, ids, kw, t0)
        got = read(ids, **kw)
        T, LB = self.tokens_per_page, self.latent_bytes_per_token
        at = {p: k for k, p in enumerate(uniq)}
        by_layer = [None] * self.num_layers  # connector layer i -> [n, T, LB] uint8
        for j, name in enumerate(self.hit_dma.layer_names):
            by_layer[self.layer_names.index(name)] = torch.from_numpy(
                got[j].reshape(len(uniq), -1).view("u1")).view(len(uniq), T, LB)  # fmt: skip
        pooled_arr = self._pooled_array(uniq) if self.pooled_source is not None else None
        src = {p: pooled_arr[k] for k, p in enumerate(uniq)} if pooled_arr is not None else None
        # page-major sources for the one-copy-per-object fill (Stage 12 put/get): [L, U, T, *]
        lat_all = torch.stack(by_layer)
        if pooled_arr is not None:
            pool_all = pooled_arr.permute(1, 0, 2).reshape(self.num_layers, len(uniq), T, -1)
        elif self._pooled_bytes is not None:
            uidx = torch.as_tensor(uniq, dtype=torch.int64)
            pool_all = torch.stack([pb.index_select(0, uidx) for pb in self._pooled_bytes]).view(
                self.num_layers, len(uniq), T, -1)  # fmt: skip
        else:
            pool_all = None
        n_tok = 0
        for memory_obj, pages in zip(memory_objs, spans, strict=True):
            n = sum(cnt for _, cnt in pages)
            if self._fill_pages(memory_obj, pages, at.__getitem__, lat_all, pool_all):
                n_tok += n
                continue
            rows = self._rows(memory_obj, n)
            latent, pooled = rows[..., :LB], rows[..., LB:]
            tok = 0
            for page, cnt in pages:
                k = at[page]
                for layer in range(self.num_layers):
                    latent[layer, tok : tok + cnt].copy_(by_layer[layer][k, :cnt])
                    if src is not None:
                        pooled[layer, tok : tok + cnt].copy_(src[page][layer].view(T, -1)[:cnt])
                    elif self._pooled_bytes is not None:
                        pooled[layer, tok : tok + cnt].copy_(self._pooled_bytes[layer][page].view(T, -1)[:cnt])
                    else:
                        pooled[layer, tok : tok + cnt].zero_()
                tok += cnt
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
            n_tok += n
        self.stats["d2h_s"] = self.stats.get("d2h_s", 0.0) + time.perf_counter() - t0
        self.stats["d2h_dma_pages"] = self.stats.get("d2h_dma_pages", 0) + len(uniq)
        self.stats["d2h_dma_direct_pages"] = self.stats.get("d2h_dma_direct_pages", 0) + len(uniq)
        self.stats["d2h_pages"] += len(uniq)
        self.stats["d2h_calls"] += 1
        self.stats["from_gpu_tokens"] += n_tok * self.num_layers
        self.stats["from_gpu_row_bytes"] += n_tok * self.num_layers * LB
        return True

    def _tile_pages_of(self, got: list, n: int) -> torch.Tensor:
        """the page-DMA read's tile-order pages (read_pages(untilize=False): TiledPages, its layer order) -> [L, n,
        65,536] uint8 in the connector's layer order"""
        import numpy as np  # noqa: PLC0415

        out = [None] * self.num_layers
        for j, name in enumerate(self.hit_dma.layer_names):
            g = got[j]
            arr = np.ascontiguousarray(g.tiles if hasattr(g, "tiles") else g)
            out[self.layer_names.index(name)] = torch.from_numpy(arr.view(np.uint8)).view(n, -1)
        return torch.stack(out)

    def _dma_from_gpu_tile(self, memory_objs, spans, uniq, read, ids, kw, t0) -> bool:
        """_dma_from_gpu in the tile-object format: the page union read once in TILE order (no untilize anywhere) and
        written into the objects page-major (_tile_fill)"""
        if self._save_resident(memory_objs, spans, t0):
            return True
        if not _takes_untilize(read):
            raise RuntimeError("GLM53_KV_TILE_OBJECTS needs the page-DMA's read_pages(untilize=False)")
        tiles_all = self._tile_pages_of(read(ids, **dict(kw, untilize=False)), len(uniq))
        if self.pooled_source is not None:
            pool_all = self._pooled_array(uniq).permute(1, 0, 2)  # [L, U, 4,096]
        elif self._pooled_bytes is not None:
            uidx = torch.as_tensor(uniq, dtype=torch.int64)
            pool_all = torch.stack([pb.index_select(0, uidx).reshape(len(uniq), -1) for pb in self._pooled_bytes])
        else:
            pool_all = None
        at = {p: k for k, p in enumerate(uniq)}
        n_tok = 0
        for memory_obj, pages in zip(memory_objs, spans, strict=True):
            self._tile_fill(memory_obj, pages, at.__getitem__, tiles_all, pool_all)
            n_tok += sum(cnt for _, cnt in pages)
        self.stats["d2h_s"] = self.stats.get("d2h_s", 0.0) + time.perf_counter() - t0
        self.stats["d2h_dma_pages"] = self.stats.get("d2h_dma_pages", 0) + len(uniq)
        self.stats["d2h_dma_direct_pages"] = self.stats.get("d2h_dma_direct_pages", 0) + len(uniq)
        self.stats["d2h_pages"] += len(uniq)
        self.stats["d2h_calls"] += 1
        self.stats["from_gpu_tokens"] += n_tok * self.num_layers
        self.stats["from_gpu_row_bytes"] += n_tok * self.num_layers * self.latent_bytes_per_token
        return True

    def zero_copy_store(self) -> bool:
        """the host's routing question: do this connector's stores SAVE straight into their objects (then they run on
        the flushing thread, not the store worker)?"""
        return _zero_copy_store() and callable(getattr(self.hit_dma, "save_resident", None))

    def _save_resident(self, memory_objs: list, spans: list, t0: float) -> bool:
        """GLM53_KV_STORE_ZEROCOPY (see _zero_copy_store): the store's whole pages SAVEd straight into its objects when
        every page is whole and every object lives in the pinned pool; the pooled halves from the pooled source (else
        the registered pooled buffers, else zero). False: the caller's read + fill path."""
        T, LB, L = self.tokens_per_page, self.latent_bytes_per_token, self.num_layers
        save = getattr(self.hit_dma, "save_resident", None)
        holds = getattr(self.hit_dma, "holds", None)
        if not (_zero_copy_store() and callable(save) and callable(holds)):
            return False
        live = [(o, pages) for o, pages in zip(memory_objs, spans, strict=True) if pages]
        if not live or any(cnt != T for _o, pages in live for _p, cnt in pages):
            return False
        obj_rows = [self._rows(o, len(pages) * T) for o, pages in live]
        if not all(holds(r) for r in obj_rows):
            return False
        pages_all = [p for _o, pages in live for p, _cnt in pages]
        ids, kw = self._dma_targets(pages_all)
        order = [self.layer_names.index(n) for n in self.hit_dma.layer_names]
        objs = [(r, len(pages)) for r, (_o, pages) in zip(obj_rows, live, strict=True)]
        one = getattr(self.hit_dma, "save_resident_pooled", None)
        distinct = len(set(pages_all)) == len(pages_all)  # (14's P2 nit: the plan is per page, not per distinct page)
        if not distinct:
            self.stats["d2h_one_dma_repeated_pages"] = self.stats.get("d2h_one_dma_repeated_pages", 0) + 1
        if (self.pooled_source is not None and self.pooled_plan is not None and callable(one) and distinct
                and os.environ.get("GLM53_KV_STORE_ONE_DMA", "0") == "1"):  # fmt: skip
            # store path P2 (program 2026-10-26T23:48:49Z): the latent SAVEs and the pooled pages' reads in ONE
            # page-DMA wait (was: save_resident + wait, then the pooled source's read + its own wait)
            targets, finish = self.pooled_plan(pages_all)
            got = one(objs, ids, targets, layer_order=order, **kw)
            pool_all = self._pooled_array(pages_all, got=finish(got)).permute(1, 0, 2)  # [L, n, 4,096]
            self.stats["d2h_one_dma_saves"] = self.stats.get("d2h_one_dma_saves", 0) + 1
        else:
            save(objs, ids, layer_order=order, **kw)
            self.hit_dma.wait()  # the objects hold the pages from here (LMCache puts them after this call)
            pool_all = None
        if pool_all is not None:
            pass
        elif self.pooled_source is not None:
            pool_all = self._pooled_array(pages_all).permute(1, 0, 2)  # [L, n, 4,096]
        elif self._pooled_bytes is not None:
            idx = torch.as_tensor(pages_all, dtype=torch.int64)
            pool_all = torch.stack([pb.index_select(0, idx).reshape(len(pages_all), -1) for pb in self._pooled_bytes])
        else:
            pool_all = None
        k = 0
        for r, (memory_obj, pages) in zip(obj_rows, live, strict=True):
            m = len(pages)
            dst = r.view(L, m, T * self.bytes_per_token)[..., T * LB :]
            if pool_all is not None:
                dst.copy_(pool_all[:, k : k + m])
            else:
                dst.zero_()
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
            k += m
        n = len(pages_all)
        s = self.stats
        s["d2h_s"] = s.get("d2h_s", 0.0) + time.perf_counter() - t0
        s["d2h_resident_saves"] = s.get("d2h_resident_saves", 0) + 1
        s["d2h_dma_pages"] = s.get("d2h_dma_pages", 0) + n
        s["d2h_dma_direct_pages"] = s.get("d2h_dma_direct_pages", 0) + n
        s["tile_object_pages"] = s.get("tile_object_pages", 0) + n
        s["d2h_pages"] += n
        s["d2h_calls"] += 1
        s["from_gpu_tokens"] += n * T * L
        s["from_gpu_row_bytes"] += n * T * L * LB
        return True

    def _dma_to_gpu(self, memory_objs: list, spans: list, placed: Mapping[int, tuple[int, int]] | None = None) -> bool:
        """GLM53_LMCACHE_PAGE_DMA: the memory objects' latent rows go STRAIGHT to the page-DMA, no row-major host
        buffer in between: per layer one [pages, 64 x 1,024 B] array gathered from the objects (one vectorized copy;
        the model tilizes it into its pinned ring) -> hit_dma.submit (the sharded pool: each page to its owner chip's
        local block, ``placed`` = _admit's placement). False when a page is partial (the caller then takes the
        host-buffer path)."""
        T, LB = self.tokens_per_page, self.latent_bytes_per_token
        if not spans or any(cnt != T for pages in spans for _, cnt in pages):
            return False
        t0 = time.perf_counter()
        parts: list[torch.Tensor] = []
        block_ids: list[int] = []
        live = [(o, pages) for o, pages in zip(memory_objs, spans, strict=True) if pages]
        obj_rows = [self._rows(o, len(pages) * T) for o, pages in live]
        self._keep_loaded_pooled_many(live, obj_rows)
        tile = _tile_mode()
        for rows, (_memory_obj, pages) in zip(obj_rows, live, strict=True):
            if tile:  # [L, m pages, 69,632]: the whole pages' latent tiles stay where they are
                parts.append(rows)
            else:
                parts.append(rows[..., :LB])  # [L, n, LB] uint8 (row stride bytes/token)
            block_ids += [p for p, _ in pages]
        if not block_ids:
            return True
        order = [self.layer_names.index(n) for n in self.hit_dma.layer_names]
        ids, kw = self._dma_targets(block_ids, placed if placed is not None else {})
        resident = getattr(self.hit_dma, "submit_resident", None)
        holds = getattr(self.hit_dma, "holds", None)
        if tile and callable(resident) and callable(holds) and all(holds(r) for r in parts):
            # Stage 12 lever (c) step 2: the objects live in the model's pinned pool in tile order -> descriptors at
            # their own bytes, no host copy; the DMA reads them after the call, so wait before LMCache may free them
            resident([(r, r.shape[1] // T) for r in parts], ids, layer_order=order, **kw)
            after = getattr(self.hit_dma, "after_wait", None)
            if callable(after) and os.environ.get("GLM53_KV_RESIDENT_ASYNC", "1") == "1":
                # the objects held (LMCache refcount) until the DMA's next wait() -- the model's, before the step that
                # reads the pages -- so the page-in runs behind the engine's other work
                held = [o for o, _pages in live]
                for o in held:
                    o.ref_count_up()
                after(lambda objs=held: [o.ref_count_down() for o in objs])
                self.stats["h2d_resident_async"] = self.stats.get("h2d_resident_async", 0) + 1
            else:
                self.hit_dma.wait()
            self.stats["h2d_resident_submits"] = self.stats.get("h2d_resident_submits", 0) + 1
            parts = None
        elif tile:  # tile objects outside the pool: their latent untilized -> the row-major submit below
            L = self.num_layers
            parts = [_untilize_page_bytes(r.view(L, r.shape[1] // T, T * self.bytes_per_token)[..., : T * LB]
                                          .reshape(-1, T * LB)).view(L, r.shape[1], LB) for r in parts]  # fmt: skip
            self.stats["h2d_tile_untilized_pages"] = self.stats.get("h2d_tile_untilized_pages", 0) + len(block_ids)
        fused = getattr(self.hit_dma, "submit_objects", None)
        if parts is None:
            pass
        elif (callable(fused) and os.environ.get("GLM53_KV_LOAD_FUSED", "1") == "1"
                and self.num_layers * len(block_ids) <= int(getattr(self.hit_dma, "slots", 0))):  # fmt: skip
            # Stage 12 lever (c) (program 2026-10-26T00:15:39Z): no torch.cat -- the model tilizes each object's
            # strided latent rows straight into its ring slots (one copy per object for all layers: a 32K hit 33 ms
            # vs 116 ms for the cat + tilize, lever_c/bench_load_variants.py); hits past the ring keep the cat path
            fused([(r, r.shape[1] // T) for r in parts], ids, layer_order=order, **kw)
            self.stats["h2d_fused_submits"] = self.stats.get("h2d_fused_submits", 0) + 1
        else:
            # Stage 12 put/get (program 2026-10-24T01:33:39Z / 02:08:04Z): ONE cat of every object's latent rows (a
            # 32K hit was 256 objects x 11 per-layer reshape copies + 11 cats: 0.43 s of host time) -> per layer
            lat = torch.cat(parts, dim=1)  # [L, pages T, LB]
            self._dma_load()([lat[i].view(len(block_ids), T * LB).numpy() for i in order], ids, **kw)
        n_tok = len(block_ids) * T
        self.stats["h2d_s"] += time.perf_counter() - t0
        self.stats["h2d_dma_pages"] += len(block_ids)
        self.stats["h2d_dma_direct_pages"] = self.stats.get("h2d_dma_direct_pages", 0) + len(block_ids)
        self.stats["h2d_pages"] += len(block_ids)
        self.stats["h2d_calls"] += 1
        self.stats["to_gpu_tokens"] += n_tok * self.num_layers
        self.stats["to_gpu_row_bytes"] += n_tok * self.num_layers * LB
        return True

    def batched_to_gpu(
        self, memory_objs=None, starts=None, ends=None, **kwargs
    ) -> None:
        """All memory objects to the host buffers, then one h2d for the page union (with the page-DMA: straight from
        the objects, _dma_to_gpu)."""
        self.initialize_kvcaches_ptr(**kwargs)
        self._require_bound()
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs")
        memory_objs = list(memory_objs)
        starts, ends = list(starts), list(ends)
        spans = list(zip([int(s) for s in starts], self.pages_of_ranges(kwargs["slot_mapping"], starts, ends), strict=True))
        placed = self._admit(spans)  # the sharded pool: before ANY device write (the page-DMA's too)
        if self.hit_dma is not None and self._dma_to_gpu(memory_objs, [p for _s, p in spans], placed):
            return
        rest = list(range(len(memory_objs)))
        if self.hit_dma is not None and os.environ.get("GLM53_KV_SPLIT_PARTIAL", "1") == "1":
            # 14's SHOULD 3 (2026-10-27T01:20:03Z): a hit ending on a partial chunk (a preempted request's resume, an
            # unfull prompt tail) took the host path for EVERY object; the whole-page objects keep the page-DMA path
            # and only the objects with a partial page go through the host buffers (chunk boundaries are page
            # boundaries: no page is in two objects)
            T = self.tokens_per_page
            whole = [k for k, (_s, p) in enumerate(spans) if p and all(cnt == T for _, cnt in p)]
            if whole and len(whole) < len(spans) and self._dma_to_gpu(
                    [memory_objs[k] for k in whole], [spans[k][1] for k in whole], placed):  # fmt: skip
                keep = set(whole)
                rest = [k for k in rest if k not in keep]
                self.stats["h2d_split_partial"] = self.stats.get("h2d_split_partial", 0) + 1
        all_pages: list[int] = []
        for k in rest:
            memory_obj, (_start, pages) = memory_objs[k], spans[k]
            self._memobj_to_host(memory_obj, pages)
            all_pages.extend(p for p, _ in pages)
        self._move(self._unique(all_pages), "h2d", placed)

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs) -> None:
        """One d2h for the union of pages, then fill every memory object."""
        self.initialize_kvcaches_ptr(**kwargs)
        self._require_bound()
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs")
        spans = self.pages_of_ranges(kwargs["slot_mapping"], starts, ends)
        memory_objs = list(memory_objs)
        job = self.current_store_job()
        if job is not None:  # the store worker: the engine thread already read these pages (no device call here)
            for memory_obj, pages in zip(memory_objs, spans, strict=True):
                self._host_to_memobj(memory_obj, pages, job=job)
            return
        if self._dma_from_gpu(memory_objs, spans):
            return
        self._move(self._unique([p for pages in spans for p, _ in pages]), "d2h")
        src = self._pooled_for([p for pages in spans for p, _ in pages]) if self.pooled_source is not None else None
        for memory_obj, pages in zip(memory_objs, spans, strict=True):
            self._host_to_memobj(memory_obj, pages, src)

    # ------------------------------------------------------------- store worker (GLM53_KV_STORE_WORKER)
    def prefetch_store_pages(self, page_ids: Sequence[int]) -> dict:
        """Stage 12 (program 2026-10-23T21:13:14Z, the store off the engine thread): on the ENGINE thread, the
        device -> host read of a store's pages (the bulk d2h: _move, or the page-DMA read under GLM53_KV_T2_STORE) and a
        snapshot of their rows into job-owned host tensors; the store worker then runs LMCache's store with
        current_store_job() set, and from_gpu / batched_from_gpu copy from the snapshot -- no device call, no shared
        host buffer, on the worker. {"row": {page: snapshot row}, "bytes": per layer [n, page bytes] uint8,
        "pooled": per layer or None, "nbytes": int}"""
        self._require_bound()
        ids = self._unique([int(p) for p in page_ids])
        row = {p: k for k, p in enumerate(ids)}
        idx = torch.as_tensor(ids, dtype=torch.int64)
        read = self._t2_read()
        if ids and read is not None:
            order = [self.layer_names.index(n) for n in self.hit_dma.layer_names]
            dma_ids, kw = self._dma_targets(ids)
            if (os.environ.get("GLM53_KV_STORE_UNTILIZE_WORKER", "0") == "1" or _tile_mode()) and _takes_untilize(read):
                # Stage 12 W4 F3a: the ring's tile-order pages; the store worker untilizes them (complete_store_job) --
                # or, tile objects (GLM53_KV_TILE_OBJECTS), keeps them as they are
                kw = dict(kw, untilize=False)
            elif _tile_mode():
                raise RuntimeError("GLM53_KV_TILE_OBJECTS needs the page-DMA's read_pages(untilize=False)")
            got = read(dma_ids, **kw)
            layers: list = [None] * self.num_layers
            for j, i in enumerate(order):
                g = got[j]
                layers[i] = g if callable(getattr(g, "result", None)) else torch.from_numpy(
                    g.reshape(len(ids), -1).view("u1"))
            self.stats["d2h_dma_pages"] = self.stats.get("d2h_dma_pages", 0) + len(ids)
        elif ids and self._pooled_bytes is None and getattr(self.copy_blocks, "__name__", "") == "tt_copy_kv_blocks":
            # the plugin's own page copy: read every layer's pages STRAIGHT into the job (fresh host tensors), skipping
            # the shared host rows and the snapshot copy (d0 2026-10-23: ~90 of the 209 ms engine share per 8,192 rows).
            # GLM53_KV_STORE_ASYNC_READ (default on): the reads only ENQUEUED here (the sharded pool's _PendingPages:
            # copies + non-blocking reads + an event, in queue order behind the step that wrote the pages); the worker
            # waits for them in complete_store_job -- 9's mock n29h7p2gprof spent 94 % of the store in this read's wait
            from vllm_tt_plugin.kv_connector import _read_pages, _read_pages_async  # noqa: PLC0415  (kv_connector imports this module)

            read_one = _read_pages_async if os.environ.get("GLM53_KV_STORE_ASYNC_READ", "1") != "0" else _read_pages
            layers = [read_one(self.device_kv_caches[name], ids) for name in self.layer_names]
            layers = [
                t.contiguous().view(torch.uint8).reshape(len(ids), -1) if isinstance(t, torch.Tensor) else t
                for t in layers
            ]
            enq = sum(1 for t in layers if not isinstance(t, torch.Tensor))
            self.stats["d2h_enqueued"] = self.stats.get("d2h_enqueued", 0) + enq
            self.stats["d2h_pages"] += len(ids)
            self.stats["d2h_calls"] += 1
        else:
            if ids:
                self._move(ids, "d2h")
            layers = [hb.index_select(0, idx) for hb in self._host_bytes]
        pooled = [pb.index_select(0, idx) for pb in self._pooled_bytes] if self._pooled_bytes is not None else None
        if pooled is None and ids and self.pooled_source is not None:
            # GLM53_KV_POOLED_PAGES (cf3h-c): the pages' pooled keys through the pooled source (the model's page-DMA
            # read, on this engine thread like the latent read): [n, L, 4,096] -> per layer [n, 4,096]
            arr = self._pooled_array(ids)
            pooled = [arr[:, layer] for layer in range(self.num_layers)]
        nbytes = sum(int(t.numel()) if isinstance(t, torch.Tensor) else int(t.nbytes) for t in layers)
        nbytes += sum(int(t.numel()) for t in pooled) if pooled else 0
        if _tile_mode() and not (ids and read is not None):
            if ids:
                raise RuntimeError("GLM53_KV_TILE_OBJECTS: a store must go through the page-DMA read (tile pages)")
        return {"row": row, "bytes": layers, "pooled": pooled, "nbytes": nbytes, "tile": _tile_mode()}

    def complete_store_job(self, job: dict) -> None:
        """on the store worker, before LMCache's store: every page read prefetch_store_pages only enqueued waited for
        (its event; the GIL is released) and turned into the job's uint8 rows"""
        n, t0 = len(job["row"]), time.perf_counter()
        for i, t in enumerate(job["bytes"]):
            if not isinstance(t, torch.Tensor):
                if job.get("tile") and hasattr(t, "tiles"):  # tile objects: the read's tile order, kept as it is
                    import numpy as np  # noqa: PLC0415

                    job["bytes"][i] = torch.from_numpy(np.ascontiguousarray(t.tiles).view(np.uint8)).view(n, -1)
                else:
                    job["bytes"][i] = t.result().contiguous().view(torch.uint8).reshape(n, -1)
        self.stats["store_read_wait_s"] = self.stats.get("store_read_wait_s", 0.0) + time.perf_counter() - t0

    def _store_tl(self) -> threading.local:
        tl = self.__dict__.get("_store_job_tl")
        if tl is None:
            tl = self.__dict__.setdefault("_store_job_tl", threading.local())
        return tl

    def current_store_job(self) -> dict | None:
        return getattr(self._store_tl(), "job", None)

    def set_store_job(self, job: dict | None) -> None:
        """the calling thread's store job (the store worker sets it around LMCache's store; None clears it)"""
        self._store_tl().job = job

    def get_shape(self, num_tokens: int) -> torch.Size:
        """``[1, num_layers, num_tokens, 544]`` (KV_MLA_FMT, bf16)."""
        return torch.Size([1, self.num_layers, num_tokens, self.hidden_elems])
