# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

import inspect

from torch import nn
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.model_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import get_model_architecture

from vllm_tt_plugin.config import (
    get_tt_config,
    get_tt_data_parallel_size,
    get_tt_kv_cache_dtype,
    get_tt_max_batch_size,
)
from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)


class TTModelLoader(BaseModelLoader):
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        """Load a model with the given configurations."""

        device_config = vllm_config.device_config
        model_class, _ = get_model_architecture(model_config)
        optimizations = get_tt_config(vllm_config).get("optimizations", None)
        if optimizations is not None:
            assert optimizations in [
                "performance",
                "accuracy",
            ], f"""Invalid optimizations configuration `{optimizations}`,
            allowed values are 'performance' or 'accuracy'"""

        tt_data_parallel = get_tt_data_parallel_size(vllm_config)
        max_batch_size = get_tt_max_batch_size(vllm_config)

        model = model_class.initialize_vllm_model(
            model_config.hf_config,
            device_config.device,
            max_batch_size,
            max_seq_len=model_config.max_model_len,
            tt_data_parallel=tt_data_parallel,
            optimizations=optimizations,
            **_construction_kv_cache_dtype(model_class, vllm_config),
            **_construction_rope_layout(model_class, vllm_config),
        )
        return model

    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError


def _construction_kv_cache_dtype(model_class, vllm_config: VllmConfig) -> dict:
    """``kv_cache_dtype`` for models that fix their KV dtype when they are built.

    Such a model declares a ``kv_cache_dtype`` parameter on ``initialize_vllm_model``;
    an explicit ``--kv-cache-dtype`` then reaches it here as well as at KV allocation.
    Other models receive it at allocation only.
    """
    dtype = get_tt_kv_cache_dtype(vllm_config.cache_config.cache_dtype)
    if dtype is None:
        return {}
    params = inspect.signature(model_class.initialize_vllm_model).parameters
    return {"kv_cache_dtype": dtype} if "kv_cache_dtype" in params else {}


def _construction_rope_layout(model_class, vllm_config: VllmConfig) -> dict:
    """``use_hf_rope=True`` for a model that declares ``supports_hf_rope_kv`` when a KV
    connector is configured.

    The model then keeps K in the checkpoint's (Hugging Face, rotate-half) RoPE
    channel order, as vLLM's other platforms do, so the KV it writes means the same
    as theirs when a connector moves it to another platform. Without a connector the
    model keeps its own order.
    """
    capabilities = getattr(model_class, "model_capabilities", None) or {}
    if not capabilities.get("supports_hf_rope_kv"):
        return {}
    if getattr(vllm_config, "kv_transfer_config", None) is None:
        return {}
    return {"use_hf_rope": True}
