"""KV connector support for the TT platform.

TT models allocate their KV caches as ttnn tensors and run their own attention. Upstream
connectors see the TT cache through three things defined here:

- `TTAttentionBackend`: connectors that ask vLLM for "the" attention backend
  (NixlConnector uses its name and supported kernel block sizes) get this descriptor. It
  never runs attention.
- Per-layer KV specs and shadow tensors: connectors size their host buffers from a torch
  tensor per layer. The shadow tensors have the block-major shape and dtype of one
  layer's paged cache and no storage; the bytes they stand for live on the device.
- A copy-op provider: host-buffer connectors move blocks between the device cache and
  their host buffers through a `CopyBlocksOp`. The provider (registered by the tt-kv
  package) builds it from the model's device caches, so this plugin has no
  device-transfer code of its own. A copy op with a `shadow_kv_caches()` method supplies
  the per-layer tensors connectors see instead of the ones below: LMCache moves blocks
  by the tensors' device and token/head axes, which only the provider knows.
- Host-buffer layouts: by default host buffers hold the device's own (tiled) bytes,
  which only another TT instance reads. A provider that converts while it copies
  declares the vLLM `KVCacheLayout`s it produces (`wire_layouts`); when the user selects
  one with vLLM's own `VLLM_KV_CACHE_LAYOUT`, the worker offers it to vLLM's layout
  resolution and the host buffers hold KV in that row-major layout, as on any other
  platform.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.attention.backend import AttentionBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


class TTAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "TT"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("TT models run their own attention")

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError("TT models run their own attention")

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return True


# (mesh_device, device_caches_by_layer, kv_cache_config) -> CopyBlocksOp
CopyOpProvider = Callable[[Any, dict[str, Any], "KVCacheConfig"], Callable]
_copy_op_provider: CopyOpProvider | None = None
_wire_layouts: tuple[str, ...] = ()
_copy_op: Any = None
# Received requests whose host -> device copy is still running: (request ids, load
# ticket).
_held_recvs: list[tuple[set[str], int]] = []
# Requests scheduled in steps whose saves were left running (see output_needs_saves).
_unwaited_saves: set[str] = set()


def register_copy_op_provider(
    provider: CopyOpProvider, wire_layouts: tuple[str, ...] = ()
) -> None:
    """`wire_layouts`: vLLM KVCacheLayout names the provider's copy op can hold host
    buffers in (it is then called with `wire_layout=<name>`); empty for device bytes
    only."""
    global _copy_op_provider, _wire_layouts
    _copy_op_provider = provider
    _wire_layouts = tuple(wire_layouts)


# kv_buffer_device="tt": a provider (registered by the tt-kv package) that turns the model's
# device caches into tensors a connector registers as device memory, and makes the NIXL
# backend that carries them findable.
_device_buffer_provider: Callable[..., Any] | None = None
_prepare_device_buffers: Callable[[], bool] | None = None


def register_device_buffer_provider(
    provider: Callable[..., Any], prepare: Callable[[], bool]
) -> None:
    """`provider(mesh_device, device_caches, kv_cache_config, kv_transfer_config,
    device_index)` returns a handle with `caches` (per-layer tensors on device `tt` whose
    data_ptr() a NIXL backend resolves to the device cache) and `wait_saves()` (the
    forward that wrote the blocks has completed), or None. `prepare()` makes the backend
    findable before NIXL agents are created."""
    global _device_buffer_provider, _prepare_device_buffers
    _device_buffer_provider = provider
    _prepare_device_buffers = prepare


# A provider whose transfers use device fabric another process brought up (a fabric
# island) decides the fabric config of the KV-transfer worker's mesh.
_fabric_config_provider: Callable[[int], Any] | None = None


def register_fabric_config_provider(provider: Callable[[int], Any]) -> None:
    """`provider(num_devices)` returns `(args, kwargs)` for `ttnn.set_fabric_config`, or
    None to keep the plugin's choice. Called before the mesh opens, also for a
    single-device mesh, and only when KV transfer is configured."""
    global _fabric_config_provider
    _fabric_config_provider = provider


def fabric_config_override(vllm_config: VllmConfig, num_devices: int) -> Any:
    if _fabric_config_provider is None or not kv_transfer_enabled(vllm_config):
        return None
    return _fabric_config_provider(num_devices)


_worker_config: VllmConfig | None = None


def note_vllm_config(vllm_config: VllmConfig) -> None:
    """The worker's config, for platform hooks vLLM calls without one (NIXL's device
    checks run while the connector is built, outside any current-config context)."""
    global _worker_config
    _worker_config = vllm_config


def current_vllm_config() -> VllmConfig | None:
    if _worker_config is not None:
        return _worker_config
    from vllm.config import get_current_vllm_config

    try:
        return get_current_vllm_config()
    except AssertionError:  # no config set in this context
        return None


def device_buffers_requested(vllm_config: VllmConfig | None) -> bool:
    cfg = getattr(vllm_config, "kv_transfer_config", None)
    return getattr(cfg, "kv_buffer_device", None) == "tt"


def prepare_device_buffers() -> bool:
    return _prepare_device_buffers is not None and bool(_prepare_device_buffers())


def kv_transfer_enabled(vllm_config: VllmConfig) -> bool:
    return getattr(vllm_config, "kv_transfer_config", None) is not None


# The layout every TT cache setup supports; host buffers then hold device bytes.
DEFAULT_KV_CACHE_LAYOUT = "LBNHC"
_LAYOUT_ALIASES = {"NHD": "LBNHC", "HND": "LBHNC"}


def host_buffer_wire_layout(vllm_config: VllmConfig) -> str | None:
    """The row-major layout host buffers hold: the one the user selected with
    `VLLM_KV_CACHE_LAYOUT`, when a connector is configured and the copy-op provider
    produces it. None: device bytes (also when the selected layout is unsupported;
    vLLM's layout resolution then refuses it)."""
    import vllm.envs as envs

    name = envs.VLLM_KV_CACHE_LAYOUT
    if name is None or not kv_transfer_enabled(vllm_config):
        return None
    name = _LAYOUT_ALIASES.get(name, name)
    return name if name in _wire_layouts else None


def supported_kv_cache_layouts(vllm_config: VllmConfig) -> list[str]:
    """What the TT worker reports to vLLM's layout resolution, most preferred first."""
    wire = host_buffer_wire_layout(vllm_config)
    if wire is None or wire == DEFAULT_KV_CACHE_LAYOUT:
        return [DEFAULT_KV_CACHE_LAYOUT]
    return [DEFAULT_KV_CACHE_LAYOUT, wire]


def layer_names(vllm_config: VllmConfig) -> list[str]:
    """One vLLM layer name per decoder layer, in the order the model allocates its
    caches."""
    n = vllm_config.model_config.get_num_layers_by_block_type(
        vllm_config.parallel_config, "attention"
    )
    return [f"model.layers.{i}.self_attn" for i in range(n)]


def shadow_kv_caches(kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
    """Storage-free per-layer tensors with the logical paged-cache shape connectors size
    from.

    Shape is [num_blocks, num_kv_heads, block_size, C]: K and V of one block packed in
    the last dimension, as upstream's packed host layout expects; C is 2 * head_size
    elements, or the spec's packed bytes per (head, token) in uint8 for block-float
    device caches. Only shape and dtype are read.
    """
    out: dict[str, torch.Tensor] = {}
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if is_state_spec(spec):
            for name in group.layer_names:
                out[name] = state_shadow(spec, kv_cache_config.num_blocks)
            continue
        packed = getattr(spec, "state_content_bytes", None)
        last = packed if packed is not None else 2 * spec.head_size
        dtype = torch.uint8 if packed is not None else spec.dtype
        shape = (kv_cache_config.num_blocks, spec.num_kv_heads, spec.block_size, last)
        for name in group.layer_names:
            out[name] = torch.empty((1,), dtype=dtype).expand(shape)
    return out


def is_state_spec(spec) -> bool:
    from vllm.v1.kv_cache_interface import MambaSpec

    return isinstance(spec, MambaSpec)


def state_shadow(spec, num_blocks: int) -> torch.Tensor:
    """Storage-free tensor for a recurrent-state layer: one page (``page_size_bytes``)
    per block.

    Connectors size the state host buffer from it (4-D, like attention layers) and
    address each state slot's conv and SSM regions from the spec, so only its bytes per
    block matter."""
    return torch.empty((1,), dtype=torch.uint8).expand(
        (num_blocks, 1, 1, spec.page_size_bytes)
    )


# Bytes of one 32x32 tile on the device for block-float formats (shared exponents per 16
# values).
_BLOCK_FLOAT_TILE_BYTES = {"bfloat8_b": 1088, "bfloat4_b": 576}


def device_kv_spec(spec, device_dtype: str, wire_layout: str | None = None):
    """The KV spec whose pages match the host-buffer bytes of one layer.

    Block-float caches pack each 32x32 tile with shared exponents, so a block holds more
    bytes than one per element: the spec declares them as packed uint8 content
    (`state_content_bytes`, K and V per head and token), which upstream connectors size
    host buffers and transfers from. With a wire layout the host buffers hold the spec's
    own dtype whatever the device stores (the copy op converts block-float on the way),
    so the spec is unchanged.
    """
    import dataclasses

    tile = _BLOCK_FLOAT_TILE_BYTES.get(device_dtype)
    if tile is None or wire_layout is not None:
        return spec
    if spec.head_size % 32 or spec.block_size % 32:
        raise ValueError(
            f"{device_dtype} KV needs head_size and block_size multiples of 32"
        )
    per_token = 2 * (spec.head_size // 32) * tile // 32
    return dataclasses.replace(spec, dtype=torch.uint8, state_content_bytes=per_token)


def init_single_rank_distributed(vllm_config: VllmConfig, init_method: str) -> None:
    """KV connectors query vLLM's TP/PP groups. TT runs one worker rank per engine, so
    build the trivial groups on gloo."""
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
        model_parallel_is_initialized,
    )
    from vllm.platforms import current_platform

    if model_parallel_is_initialized():
        return
    # GroupCoordinator builds torch.device(current_platform.device_name) for out-of-tree
    # platforms; "tt" is not a torch device type. The groups are gloo/CPU only, so
    # present them as CPU.
    # TODO(upstream vLLM): fall back to CPU when the platform has no torch device.
    cls = type(current_platform)
    saved = cls.device_name
    cls.device_name = "cpu"
    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=init_method,
            local_rank=0,
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1)
    finally:
        cls.device_name = saved


def register_device_caches(
    mesh_device: Any,
    device_caches: dict[str, Any],
    kv_cache_config: KVCacheConfig,
    wire_layout: str | None = None,
) -> None:
    """Hand the connector the shadow caches and, for host-buffer connectors, the copy
    op. `wire_layout`: `host_buffer_wire_layout()`; the copy op then fills host buffers
    in it."""
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
    )

    if not has_kv_transfer_group():
        return
    connector = get_kv_transfer_group()
    global _copy_op
    vllm_config = current_vllm_config()
    if device_buffers_requested(vllm_config):
        # The connector moves device KV itself (no host buffers, no copy op). The handle's
        # wait_saves() stands in for the saves: an output that can finish a request is
        # returned only after the forward that wrote its blocks completed, since the peer
        # may read them as soon as the request is reported finished.
        handle = None
        if _device_buffer_provider is not None:
            index = vllm_config.parallel_config.data_parallel_rank_local or 0
            handle = _device_buffer_provider(
                mesh_device,
                device_caches,
                kv_cache_config,
                vllm_config.kv_transfer_config,
                index,
            )
        if handle is None:
            raise RuntimeError(
                'kv_buffer_device="tt" needs the tt-kv package and its NIXL backend, with '
                '"TT" among the connector\'s backends'
            )
        _copy_op = handle
        connector.register_kv_caches(handle.caches)
        return
    if _copy_op_provider is None:
        raise RuntimeError(
            "A KV connector is configured but no TT copy-op provider is registered; "
            "install the tt-kv package"
        )
    # The copy op first: connectors may compute their P/D compatibility check while
    # registering caches, and the copy op's layout is part of it (see
    # kv_transfer_compat_factors).
    if wire_layout is not None:
        _copy_op = _copy_op_provider(
            mesh_device, device_caches, kv_cache_config, wire_layout=wire_layout
        )
    else:
        _copy_op = _copy_op_provider(mesh_device, device_caches, kv_cache_config)
    own = getattr(_copy_op, "shadow_kv_caches", None)
    connector.register_kv_caches(
        own() if own is not None else shadow_kv_caches(kv_cache_config)
    )
    connector.set_host_xfer_buffer_ops(_copy_op)


def shutdown() -> None:
    """Before the model and the mesh go: the connector (and its NIXL agents) first, as
    vLLM's GPU worker does, then the copy op or device-buffer handle, whose device
    programs must stop before the mesh closes (optional protocol: ``close()``)."""
    global _copy_op
    from vllm.distributed.kv_transfer import kv_transfer_state

    kv_transfer_state.ensure_kv_transfer_shutdown()
    op, _copy_op = _copy_op, None
    close = getattr(op, "close", None)
    if close is not None:
        close()


def kv_transfer_compat_factors() -> dict[str, str]:
    """Factors P and D must agree on beyond upstream's: the device cache layout the copy
    op moves. Empty before the copy op exists or if it does not describe its layout."""
    op = _copy_op
    if op is None or not hasattr(op, "layout_hash"):
        return {}
    return {"tt_kv_layout": op.layout_hash()}


def num_command_queues(vllm_config: VllmConfig) -> int:
    """A second command queue carries KV copies, so they never wait in the model's
    queue."""
    return 2 if kv_transfer_enabled(vllm_config) else 1


# Asynchronous copy ops (optional protocol): wait_saves(), take_load_ticket(),
# load_done(ticket). A synchronous op (plain CopyBlocksOp) needs neither of the two
# functions below.


def wait_saves() -> None:
    """Saves must land before an output that can finish a request is returned: the
    decode node then reads the host buffer. Saves complete in order, so this covers
    every earlier save."""
    if _copy_op is not None and hasattr(_copy_op, "wait_saves"):
        _copy_op.wait_saves()
    _unwaited_saves.clear()


def _scheduled_progress(scheduler_output):
    """(request id, tokens computed before the step, tokens scheduled in it) per
    scheduled request."""
    n = scheduler_output.num_scheduled_tokens
    for r in scheduler_output.scheduled_new_reqs:
        yield r.req_id, r.num_computed_tokens, n.get(r.req_id, 0)
    cached = scheduler_output.scheduled_cached_reqs
    for req_id, computed in zip(cached.req_ids, cached.num_computed_tokens):
        yield req_id, computed, n.get(req_id, 0)


def output_needs_saves(
    scheduler_output, num_prompt_tokens: Callable[[str], int | None]
) -> bool:
    """Whether the step's saves must complete before its output is returned.

    A step whose requests are all partial prefills (each stops short of its prompt)
    samples nothing and finishes no request, so its saves (the blocks of the chunk just
    computed) keep running while the next chunk is computed. The step that completes a
    prompt waits, and with it every earlier save. A request of unknown prompt length
    counts as complete.
    """
    progress = list(_scheduled_progress(scheduler_output))
    if not progress:
        return True
    for req_id, computed, scheduled in progress:
        total = num_prompt_tokens(req_id)
        if total is None or computed + scheduled >= total:
            return True
    return False


def defer_saves(scheduler_output) -> None:
    """Record the requests of a step whose saves were left running."""
    _unwaited_saves.update(r for r, _, _ in _scheduled_progress(scheduler_output))


def handle_preemptions(scheduler_output) -> None:
    """Before the step's forward, as vLLM's GPU runner does: a connector with
    asynchronous saves (LMCache's async store) finishes reading the blocks of preempted
    requests, which this step may overwrite."""
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
    )

    meta = getattr(scheduler_output, "kv_connector_metadata", None)
    if meta is not None and has_kv_transfer_group():
        get_kv_transfer_group().handle_preemptions(meta)


def note_block_ids(scheduler_output) -> None:
    """Tell the copy op which KV cache group owns the blocks this step may copy
    (optional protocol: ``note_block_ids(block_ids)`` per request, vLLM's per-group
    ``BlockIds``).

    With several groups sharing one block pool (hybrid models), host-buffer connectors
    copy one group's blocks per call but hand over every layer; knowing the owner lets
    the copy op move only that group's layers. Saves copy blocks scheduled in this step;
    loads copy blocks the connector metadata names (``reqs_to_recv`` / ``reqs_to_save``
    entries with ``local_block_ids``).
    """
    op = _copy_op
    note = getattr(op, "note_block_ids", None)
    if note is None:
        return
    for r in scheduler_output.scheduled_new_reqs:
        note(r.block_ids)
    for ids in scheduler_output.scheduled_cached_reqs.new_block_ids or ():
        note(ids)
    pending = [getattr(scheduler_output, "kv_connector_metadata", None)]
    while pending:
        meta = pending.pop()
        children = getattr(meta, "metadata", None)
        if isinstance(children, tuple):  # MultiConnector: one child per connector
            pending.extend(children)
        for attr in ("reqs_to_recv", "reqs_to_save"):
            for m in (getattr(meta, attr, None) or {}).values():
                note(getattr(m, "local_block_ids", None))


# Device-buffer self test (optional protocol: the handle's ``selftest(kind, block_ids)``
# reads blocks back from the device and logs one CRC per block). P checks a request's
# blocks when the connector offers them to D, D the blocks it received, once received.
_st_blocks: dict[str, list[int]] = {}
_st_recv: dict[str, list[int]] = {}


def _first_group(ids) -> list[int]:
    if ids and isinstance(ids[0], (list, tuple)):
        return list(ids[0])
    return list(ids or ())


def selftest_step(scheduler_output) -> None:
    op = _copy_op
    if op is None or not hasattr(op, "selftest"):
        return
    for r in scheduler_output.scheduled_new_reqs:
        _st_blocks[r.req_id] = _first_group(r.block_ids)
    cached = scheduler_output.scheduled_cached_reqs
    for req_id, ids in zip(cached.req_ids, cached.new_block_ids or ()):
        if ids is not None:
            _st_blocks.setdefault(req_id, []).extend(_first_group(ids))
    meta = getattr(scheduler_output, "kv_connector_metadata", None)
    for req_id in getattr(meta, "reqs_to_send", None) or {}:
        if req_id in _st_blocks:
            op.selftest("save", _st_blocks[req_id])
    for req_id, m in (getattr(meta, "reqs_to_recv", None) or {}).items():
        _st_recv[req_id] = _first_group(getattr(m, "local_block_ids", None))
    for req_id in scheduler_output.finished_req_ids or ():
        _st_blocks.pop(req_id, None)


def _selftest_received(kv_output) -> None:
    op = _copy_op
    if kv_output is None or op is None or not hasattr(op, "selftest"):
        return
    # A failed receive is reported received with its blocks invalid (recomputed later).
    invalid = set(getattr(kv_output, "invalid_block_ids", None) or ())
    for req_id in kv_output.finished_recving or ():
        ids = _st_recv.pop(req_id, None)
        if ids and not invalid.intersection(ids):
            op.selftest("load", ids)


def wait_saves_of_freed(scheduler_output) -> None:
    """Before a step's loads and forward, wait for running saves of requests whose
    blocks were freed since the last step (finished early, aborted or preempted).

    Their blocks may be reused in this step: a received request's host slots would
    otherwise be written by both the old save and the new receive. A later save to the
    same slot needs no wait, because saves complete in order.
    """
    if not _unwaited_saves:
        return
    freed = set(scheduler_output.finished_req_ids or ())
    freed |= set(getattr(scheduler_output, "preempted_req_ids", None) or ())
    if _unwaited_saves & freed:
        wait_saves()


def begin_step() -> None:
    """Before a step's forward is enqueued: the copy op may order the step's loads after
    the work enqueued so far rather than after this forward, which never touches blocks
    still being received (tt-kv spec 05 §3.2). Optional protocol, like the functions
    below."""
    op = _copy_op
    if op is not None and hasattr(op, "begin_step"):
        op.begin_step()


def gate_received(kv_output) -> None:
    """Report a received request only once its host -> device copy has completed.

    The connector reports a request as received right after the copy op returns. With an
    async op the copy may still be running, so such requests are held back (they stay
    waiting for remote KV in the scheduler, and other requests keep running) and
    released in a later step.
    """
    _selftest_received(kv_output)
    op = _copy_op
    if kv_output is None or op is None or not hasattr(op, "take_load_ticket"):
        return
    recv = set(kv_output.finished_recving or ())
    ticket = op.take_load_ticket()
    if ticket is not None and recv and not op.load_done(ticket):
        _held_recvs.append((recv, ticket))
        recv = set()
    kv_output.finished_recving = recv or None
    release_ready(kv_output)


def release_ready(kv_output) -> None:
    """Move held requests whose load has completed into this step's received set.

    Called when the step's output is built (after the forward, at sample time) as well
    as right after the connector lifecycle: a load posted early in the step has usually
    finished by then, so the request is scheduled one step sooner.
    """
    op = _copy_op
    if kv_output is None or op is None or not _held_recvs:
        return
    recv = set(kv_output.finished_recving or ())
    while _held_recvs and op.load_done(_held_recvs[0][1]):
        recv |= _held_recvs.pop(0)[0]
    kv_output.finished_recving = recv or None


def decode_join_sampling_ok(params) -> bool:
    """Whether a request may skip its prefill step and join decode (see
    TTScheduler._joins_decode).

    The model's prefill also initialises the request's device state slot: on-device
    sampling state (seeds, penalty history from the prompt tokens) and recurrent state.
    A request joining decode gets a slot without that initialisation, so only requests
    that need none of it qualify.
    """
    if params is None:
        return True
    return (
        params.prompt_logprobs is None
        and params.seed is None
        and params.presence_penalty == 0.0
        and params.frequency_penalty == 0.0
        and params.repetition_penalty == 1.0
    )
