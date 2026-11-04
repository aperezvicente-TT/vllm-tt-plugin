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
_copy_op: Any = None
# Received requests whose host -> device copy is still running: (request ids, load
# ticket).
_held_recvs: list[tuple[set[str], int]] = []
# Requests scheduled in steps whose saves were left running (see output_needs_saves).
_unwaited_saves: set[str] = set()


def register_copy_op_provider(provider: CopyOpProvider) -> None:
    global _copy_op_provider
    _copy_op_provider = provider


def kv_transfer_enabled(vllm_config: VllmConfig) -> bool:
    return getattr(vllm_config, "kv_transfer_config", None) is not None


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
        packed = getattr(spec, "state_content_bytes", None)
        last = packed if packed is not None else 2 * spec.head_size
        dtype = torch.uint8 if packed is not None else spec.dtype
        shape = (kv_cache_config.num_blocks, spec.num_kv_heads, spec.block_size, last)
        for name in group.layer_names:
            out[name] = torch.empty((1,), dtype=dtype).expand(shape)
    return out


# Bytes of one 32x32 tile on the device for block-float formats (shared exponents per 16
# values).
_BLOCK_FLOAT_TILE_BYTES = {"bfloat8_b": 1088, "bfloat4_b": 576}


def device_kv_spec(spec, device_dtype: str):
    """The KV spec whose pages match the device cache bytes of one layer.

    Block-float caches pack each 32x32 tile with shared exponents, so a block holds more
    bytes than one per element: the spec declares them as packed uint8 content
    (`state_content_bytes`, K and V per head and token), which upstream connectors size
    host buffers and transfers from.
    """
    import dataclasses

    tile = _BLOCK_FLOAT_TILE_BYTES.get(device_dtype)
    if tile is None:
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
) -> None:
    """Hand the connector the shadow caches and, for host-buffer connectors, the copy
    op."""
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
    )

    if not has_kv_transfer_group():
        return
    connector = get_kv_transfer_group()
    if _copy_op_provider is None:
        raise RuntimeError(
            "A KV connector is configured but no TT copy-op provider is registered; "
            "install the tt-kv package"
        )
    # The copy op first: connectors may compute their P/D compatibility check while
    # registering caches, and the copy op's layout is part of it (see
    # kv_transfer_compat_factors).
    global _copy_op
    _copy_op = _copy_op_provider(mesh_device, device_caches, kv_cache_config)
    own = getattr(_copy_op, "shadow_kv_caches", None)
    connector.register_kv_caches(
        own() if own is not None else shadow_kv_caches(kv_cache_config)
    )
    connector.set_host_xfer_buffer_ops(_copy_op)


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


def gate_received(kv_output) -> None:
    """Report a received request only once its host -> device copy has completed.

    The connector reports a request as received right after the copy op returns. With an
    async op the copy may still be running, so such requests are held back (they stay
    waiting for remote KV in the scheduler, and other requests keep running) and
    released in a later step.
    """
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
