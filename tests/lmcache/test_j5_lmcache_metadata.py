# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""J5: LMCache sizes its metadata from vLLM's model_config (glm5_next: use_mla False,
45 layers, 64 x 64 heads); with the plugin's ``tt_kv_shape_hint`` the TT connector
rewrites the metadata to the KV spec's latent geometry before the engine is built.
Exercises LMCache's real ``CreateGPUConnector`` entry (venv-lmcache)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("lmcache")

from lmcache.v1.metadata import LMCacheMetadata  # noqa: E402

from vllm_tt_plugin.lmcache import page_contract as pc  # noqa: E402
from vllm_tt_plugin.lmcache.connector import VLLMPagedMemTTConnector  # noqa: E402

GPU_CONNECTOR = "vllm_tt_plugin.lmcache.connector:VLLMPagedMemTTConnector"


def _glm_metadata(extra: dict) -> LMCacheMetadata:
    """What VllmServiceFactory.get_or_create_metadata builds for GLM-5.3-Flash."""
    return LMCacheMetadata(
        model_name="zai-org/GLM-5.3-Flash",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(45, 2, 256, 64, 64),
        use_mla=False,
        chunk_size=256,
        kv_connector_extra_config=extra,
    )


def _hint(latent: int = 512) -> dict:
    return {
        "num_layers": pc.NUM_DSA_LAYERS,
        "latent_dim": latent,
        "tokens_per_page": 64,
        "dtype": "bfloat16",
        "use_mla": True,
        "layer_names": pc.dsa_layer_names(),
    }


def test_without_hint_the_glm_metadata_is_refused():
    md = _glm_metadata({"gpu_connector": GPU_CONNECTOR})
    with pytest.raises(ValueError, match="use_mla True"):
        VLLMPagedMemTTConnector.from_metadata(md, False, torch.device("cpu"))


def test_hint_rewrites_the_metadata_to_the_kv_spec_geometry():
    md = _glm_metadata({"gpu_connector": GPU_CONNECTOR, "tt_kv_shape_hint": _hint()})
    conn = VLLMPagedMemTTConnector.from_metadata(md, False, torch.device("cpu"))
    assert tuple(md.kv_shape) == (11, 1, 256, 1, 544) and md.use_mla is True
    assert conn.num_layers == 11 and conn.latent_bytes_per_token == 1024
    assert conn.layer_names == pc.dsa_layer_names()
    assert tuple(conn.get_shape(64)) == (1, 11, 64, 544)


def test_hint_through_lmcache_create_gpu_connector():
    """The factory path LMCache's VllmServiceFactory takes on a cpu torch build."""
    from lmcache.utils import EngineType
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.gpu_connector import CreateGPUConnector

    md = _glm_metadata({"gpu_connector": GPU_CONNECTOR, "tt_kv_shape_hint": _hint()})
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256, local_cpu=True, max_local_cpu_size=0.1
    )
    config.extra_config = {"gpu_connector": GPU_CONNECTOR, "tt_tokens_per_page": 64}
    conn = CreateGPUConnector(config, md, EngineType.VLLM, layout_hints=None)
    assert isinstance(conn, VLLMPagedMemTTConnector)
    assert tuple(md.kv_shape) == (11, 1, 256, 1, 544) and md.use_mla


def test_hint_with_another_latent_width():
    md = _glm_metadata(
        {"gpu_connector": GPU_CONNECTOR, "tt_kv_shape_hint": _hint(latent=128)}
    )
    conn = VLLMPagedMemTTConnector.from_metadata(md, False, torch.device("cpu"))
    assert (
        tuple(md.kv_shape) == (11, 1, 256, 1, 160)
        and conn.latent_bytes_per_token == 256
    )
