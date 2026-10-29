# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Page / page-table contract constants for the GLM-5.3-Flash latent cache.

Source: ``doc/ttnn/15-lmcache/page_contract.md`` rev 1 (A3, 2026-10-20 16:28Z) in
the glm53-flash-bringup workspace. Everything the LMCache connector assumes about
a page lives here; change this file when the contract revs.

Per DSA layer a page (= one vLLM block of 64 tokens) is the bf16 ``[64, 512]``
latent block (65,536 B) plus a reserved pooled-indexer-key block ``[16, 128]``
bf16 (4,096 B, zero-filled in v1). The LMCache-side layout is per token row:
``[512 latent | 32 reserved]`` bf16 = 1,088 B per token per layer, so a full page
is the contract's 69,632 B per layer and a partial chunk (LMCache sizes memory
objects by exact token count) needs no padding rule.
"""

from __future__ import annotations

import torch

# vLLM block == latent page (tt/generator.py BLOCK_SIZE; --block-size 64 enforced).
TOKENS_PER_PAGE = 64

# Latent c_t per token per DSA layer.
LATENT_DIM = 512
LATENT_DTYPE = torch.bfloat16
LATENT_BYTES_PER_TOKEN_LAYER = LATENT_DIM * 2  # 1,024
LATENT_BYTES_PER_PAGE_LAYER = TOKENS_PER_PAGE * LATENT_BYTES_PER_TOKEN_LAYER  # 65,536

# Pooled indexer keys (index_kpool = 4): one 128-row per 4 tokens, reserved in v1.
POOLED_KEY_ROWS_PER_PAGE = TOKENS_PER_PAGE // 4  # 16
POOLED_KEY_DIM = 128
POOLED_BYTES_PER_PAGE_LAYER = POOLED_KEY_ROWS_PER_PAGE * POOLED_KEY_DIM * 2  # 4,096
POOLED_BYTES_PER_TOKEN_LAYER = POOLED_BYTES_PER_PAGE_LAYER // TOKENS_PER_PAGE  # 64

# The 11 sparse_dsa layers of the 45-layer model: i % 4 == 3.
NUM_MODEL_LAYERS = 45
DSA_LAYER_INDICES = tuple(i for i in range(NUM_MODEL_LAYERS) if i % 4 == 3)
NUM_DSA_LAYERS = len(DSA_LAYER_INDICES)  # 11
assert NUM_DSA_LAYERS == 11

# LMCache-side token row: 512 latent + 32 reserved bf16 elements (= 64 B).
BYTES_PER_TOKEN_LAYER = (
    LATENT_BYTES_PER_TOKEN_LAYER + POOLED_BYTES_PER_TOKEN_LAYER
)  # 1,088
LMCACHE_HIDDEN_ELEMS = BYTES_PER_TOKEN_LAYER // 2  # 544
PAGE_BYTES_PER_LAYER = TOKENS_PER_PAGE * BYTES_PER_TOKEN_LAYER  # 69,632
PAGE_BYTES = NUM_DSA_LAYERS * PAGE_BYTES_PER_LAYER  # 765,952
assert PAGE_BYTES_PER_LAYER == LATENT_BYTES_PER_PAGE_LAYER + POOLED_BYTES_PER_PAGE_LAYER

# LMCache chunk = 4 pages (page_contract.md §6).
# 128 = 2 pages = the prefill chunk alignment and the prefix-cache snapshot boundary (J11): every prefill chunk end
# is then both an LMCache chunk boundary and a KDA checkpoint, so a later request's LMCache hit lands on one
LMCACHE_CHUNK_TOKENS = 2 * TOKENS_PER_PAGE  # 128

# vLLM KV-spec layer names (one MLAAttentionSpec per DSA layer).
VLLM_LAYER_NAME = "model.layers.{i}.self_attn"

# Host transfer buffer / device latent tensor shape per layer (dim 0 = num_blocks).
HOST_BUFFER_PAGE_SHAPE = (1, TOKENS_PER_PAGE, LATENT_DIM)
POOLED_BUFFER_PAGE_SHAPE = (1, POOLED_KEY_ROWS_PER_PAGE, POOLED_KEY_DIM)


def dsa_layer_names(num_layers: int = NUM_DSA_LAYERS) -> list[str]:
    """vLLM layer names of the first ``num_layers`` DSA layers, ascending."""
    if num_layers > NUM_DSA_LAYERS:
        raise ValueError(f"{num_layers} > {NUM_DSA_LAYERS} DSA layers")
    return [VLLM_LAYER_NAME.format(i=i) for i in DSA_LAYER_INDICES[:num_layers]]


def kv_shape(
    chunk_size: int = LMCACHE_CHUNK_TOKENS, num_layers: int = NUM_DSA_LAYERS
) -> tuple[int, int, int, int, int]:
    """``LMCacheMetadata.kv_shape`` = (num_layers, kv_dim, chunk, kv_heads, head_size).

    kv_dim 1 (MLA: one latent, no separate K/V), one head of 544 elements so
    ``metadata.get_shapes(n)`` yields ``[1, num_layers, n, 544]``.
    """
    if chunk_size % TOKENS_PER_PAGE:
        raise ValueError(
            f"chunk_size {chunk_size} must be a multiple of the page "
            f"({TOKENS_PER_PAGE})"
        )
    return (num_layers, 1, chunk_size, 1, LMCACHE_HIDDEN_ELEMS)


def lmcache_metadata(
    chunk_size: int = LMCACHE_CHUNK_TOKENS,
    num_layers: int = NUM_DSA_LAYERS,
    model_name: str = "zai-org/GLM-5.3-Flash",
    worker_id: int = 0,
    world_size: int = 1,
    **extra,
):
    """Build the ``LMCacheMetadata`` the TT connector expects (use_mla, bf16)."""
    from lmcache.v1.metadata import LMCacheMetadata

    return LMCacheMetadata(
        model_name=model_name,
        world_size=world_size,
        local_world_size=world_size,
        worker_id=worker_id,
        local_worker_id=worker_id,
        kv_dtype=LATENT_DTYPE,
        kv_shape=kv_shape(chunk_size, num_layers),
        use_mla=True,
        chunk_size=chunk_size,
        **extra,
    )
