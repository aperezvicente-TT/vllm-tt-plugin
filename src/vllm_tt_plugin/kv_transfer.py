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
  device-transfer code of its own.
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

    Shape is [num_blocks, num_kv_heads, block_size, 2 * head_size]: K and V of one block
    packed in the last dimension, as upstream's packed host layout expects. Only shape
    and dtype are read.
    """
    out: dict[str, torch.Tensor] = {}
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        shape = (
            kv_cache_config.num_blocks,
            spec.num_kv_heads,
            spec.block_size,
            2 * spec.head_size,
        )
        for name in group.layer_names:
            out[name] = torch.empty((1,), dtype=spec.dtype).expand(shape)
    return out


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
    connector.register_kv_caches(shadow_kv_caches(kv_cache_config))
    if _copy_op_provider is None:
        raise RuntimeError(
            "A KV connector is configured but no TT copy-op provider is registered; "
            "install the tt-kv package"
        )
    global _copy_op
    _copy_op = _copy_op_provider(mesh_device, device_caches, kv_cache_config)
    connector.set_host_xfer_buffer_ops(_copy_op)


def num_command_queues(vllm_config: VllmConfig) -> int:
    """A second command queue carries KV copies, so they never wait in the model's
    queue."""
    return 2 if kv_transfer_enabled(vllm_config) else 1


# Asynchronous copy ops (optional protocol): wait_saves(), take_load_ticket(),
# load_done(ticket). A synchronous op (plain CopyBlocksOp) needs neither of the two
# functions below.


def wait_saves() -> None:
    """Saves must land before the step's output is returned: returning it can finish a
    request on the prefill node, and the decode node then reads the host buffer."""
    if _copy_op is not None and hasattr(_copy_op, "wait_saves"):
        _copy_op.wait_saves()


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
