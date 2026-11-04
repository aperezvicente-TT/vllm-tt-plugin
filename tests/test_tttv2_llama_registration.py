# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""TT_LLAMA_TEXT_VER=tt_transformers_v2: Llama text from tt-metal's TTTv2 generators.

Covers registration, the startup checkpoint check, the construction-time
``kv_cache_dtype`` the loader forwards to models that fix their KV dtype when they are
built, and (with tt-metal's TTTv2 tree on PYTHONPATH) the plugin's prefill/decode call
shape through the real generator facade.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import vllm.config  # noqa: F401  # finish vLLM init before the plugin package

import vllm_tt_plugin.platform as tt_platform
from vllm_tt_plugin import loader as tt_loader
from vllm_tt_plugin.model_input import TTSamplingParams

TTTV2_LLAMA = "models.common.models.llama3_8b.generator:Llama3Generator"


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    from vllm.model_executor.models.registry import ModelRegistry

    out: dict[str, str] = {}
    monkeypatch.setattr(
        ModelRegistry, "get_supported_archs", staticmethod(lambda: list(out))
    )
    monkeypatch.setattr(
        ModelRegistry,
        "register_model",
        staticmethod(lambda arch, target: out.setdefault(arch, target)),
    )
    monkeypatch.delenv("TT_MODEL_CLASS_OVERRIDES", raising=False)
    monkeypatch.delenv("EXTRA_MODELS_DIR", raising=False)
    return out


def test_selector_registers_the_tttv2_llama_generator(registered, monkeypatch):
    monkeypatch.setenv("TT_LLAMA_TEXT_VER", "tt_transformers_v2")

    tt_platform.register_tt_models()

    assert registered["TTLlamaForCausalLM"] == TTTV2_LLAMA
    # Only the Llama text family moves; other families keep their built-ins.
    assert registered["TTQwen2ForCausalLM"].startswith("models.tt_transformers.")


def test_unset_selector_keeps_tt_transformers(registered, monkeypatch):
    monkeypatch.delenv("TT_LLAMA_TEXT_VER", raising=False)

    tt_platform.register_tt_models()

    assert registered["TTLlamaForCausalLM"] == (
        "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM"
    )


def test_invalid_selector_lists_tttv2(registered, monkeypatch):
    monkeypatch.setenv("TT_LLAMA_TEXT_VER", "nope")

    with pytest.raises(ValueError, match="tt_transformers_v2"):
        tt_platform.register_tt_models()


def _tttv2_class():
    module, name = TTTV2_LLAMA.split(":")
    return type(name, (), {"__module__": module})


def _config(model: str):
    return SimpleNamespace(model_config=SimpleNamespace(model=model))


@pytest.mark.parametrize(
    "model",
    [
        "meta-llama/Llama-3.1-8B-Instruct",
        "/models/Llama-3.1-8B-Instruct/",
    ],
)
def test_checkpoint_check_accepts_supported_ids(monkeypatch, model):
    monkeypatch.setenv("TT_LLAMA_TEXT_VER", "tt_transformers_v2")

    tt_platform._validate_tttv2_llama_checkpoint(_config(model), _tttv2_class())


def test_checkpoint_check_names_the_supported_set(monkeypatch):
    monkeypatch.setenv("TT_LLAMA_TEXT_VER", "tt_transformers_v2")

    with pytest.raises(ValueError, match="supported: Llama-3.1-8B-Instruct"):
        tt_platform._validate_tttv2_llama_checkpoint(
            _config("meta-llama/Llama-3.2-3B-Instruct"), _tttv2_class()
        )


def test_checkpoint_check_ignores_other_selections(monkeypatch):
    other = _config("meta-llama/Llama-3.2-3B-Instruct")
    monkeypatch.setenv("TT_LLAMA_TEXT_VER", "tt_transformers")
    tt_platform._validate_tttv2_llama_checkpoint(other, _tttv2_class())

    # A TT_MODEL_CLASS_OVERRIDES target for the architecture is the operator's choice.
    monkeypatch.setenv("TT_LLAMA_TEXT_VER", "tt_transformers_v2")
    tt_platform._validate_tttv2_llama_checkpoint(other, type("Other", (), {}))


class _BuildsWithKVDtype:
    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, kv_cache_dtype=None
    ):
        pass


class _AllocatesWithKVDtype:
    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, **kwargs):
        pass


def _cache(dtype):
    return SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=dtype))


def test_loader_forwards_explicit_kv_dtype_to_construction_time_models():
    import ttnn

    assert tt_loader._construction_kv_cache_dtype(
        _BuildsWithKVDtype, _cache("bfloat16")
    ) == {"kv_cache_dtype": ttnn.bfloat16}


@pytest.mark.parametrize(
    ("model_class", "dtype"),
    [
        (_BuildsWithKVDtype, "auto"),
        # Not a declared parameter: the dtype reaches this model at allocation only.
        (_AllocatesWithKVDtype, "bfloat16"),
    ],
)
def test_loader_sends_no_kv_dtype_otherwise(model_class, dtype):
    assert tt_loader._construction_kv_cache_dtype(model_class, _cache(dtype)) == {}


def test_loader_rejects_unsupported_kv_dtype():
    with pytest.raises(ValueError, match="not supported on TT"):
        tt_loader._construction_kv_cache_dtype(_BuildsWithKVDtype, _cache("fp8"))


def test_tttv2_generator_matches_the_plugin_contract():
    """Against the tt-metal tree on PYTHONPATH, when it has the TTTv2 generator."""
    import inspect

    generator = pytest.importorskip("models.common.models.llama3_8b.generator")
    cls = generator.Llama3Generator
    if "kv_cache_dtype" not in inspect.signature(cls.initialize_vllm_model).parameters:
        pytest.skip(
            "tt-metal on PYTHONPATH predates the TTTv2 vLLM kv_cache_dtype change"
        )

    assert "kv_cache_dtype" in inspect.signature(cls.allocate_kv_cache).parameters
    caps = cls.model_capabilities
    for key in (
        "supports_prefix_caching",
        "supports_chunked_prefill",
        "supports_async_decode",
        "supports_sample_on_device",
    ):
        assert caps.get(key) is True, key


# The plugin's own prefill and decode submissions, through the real TTTv2 generator
# facade and its VLLMAdapter (argument normalization and validation), into a target that
# records what it gets.


class _RecordingTarget:
    def __init__(self, block_size):
        import ttnn

        self.paged_kv_cache_config = SimpleNamespace(
            block_size=block_size, dtype=ttnn.bfloat16
        )
        self.eager_execution = "eager"
        self.traced_prefill_execution = "traced-prefill"
        self.traced_decode_execution = "traced-decode"
        self.calls = []

    def prefill_forward(self, **kwargs):
        self.calls.append(("prefill", kwargs))
        return torch.zeros((1, 1))

    def decode_forward(self, **kwargs):
        self.calls.append(("decode", kwargs))
        return torch.zeros((1, 1))


def _tttv2_generator(block_size=64):
    import inspect

    generator = pytest.importorskip("models.common.models.llama3_8b.generator")
    import ttnn
    from models.common.llm_runtime.config import PagedKVCacheConfig, TraceConfig

    if (
        "kv_cache_dtype"
        not in inspect.signature(
            generator.Llama3Generator.initialize_vllm_model
        ).parameters
    ):
        pytest.skip(
            "tt-metal on PYTHONPATH predates the TTTv2 vLLM kv_cache_dtype change"
        )
    adapter = generator.VLLMAdapter(
        generator.VLLMAdapterConfig.resolve(
            trace=TraceConfig("all"),
            paged_kv_cache=PagedKVCacheConfig(
                block_size=block_size, max_num_blocks=256, dtype=ttnn.bfloat16
            ),
            expected_num_layers=32,
            expected_kv_heads_per_device=8,
            expected_head_dim=128,
            model_kv_cache_dtype=ttnn.bfloat16,
            request_state_fields=("prompt_tokens", "output_tokens", "slot_remap"),
        )
    )
    target = _RecordingTarget(block_size)
    return generator.Llama3Generator(target, adapter), target


def _sampling(rows=1):
    return TTSamplingParams(
        temperature=torch.zeros(rows),
        top_k=torch.ones(rows, dtype=torch.int32),
        top_p=torch.ones(rows),
        presence_penalty=torch.zeros(rows),
        frequency_penalty=torch.zeros(rows),
        repetition_penalty=torch.ones(rows),
        seed=torch.full((rows,), -1),
        num_logprobs=torch.full((rows,), -2),
        enable_log_probs=torch.zeros(rows, dtype=torch.bool),
    )


def test_plugin_prefill_reaches_tttv2_with_a_block_aligned_resume():
    """D resumes a fully transferred 300-token prompt at 299 (vLLM recomputes the last
    token); the TTTv2 planner continues only from block-aligned starts, so the facade
    floors it to 256."""
    from vllm_tt_plugin.model_runner import TTModelRunner

    model, target = _tttv2_generator(block_size=64)
    runner = SimpleNamespace(
        model=model,
        kv_caches=object(),
        trace_mode="all",
        request_specific_rope=False,
        tt_per_lane_max_num_seqs=32,
        async_decode=SimpleNamespace(note_prefill_submitted=lambda: None),
    )
    model_input = SimpleNamespace(
        input_tokens=torch.ones((1, 300), dtype=torch.int32),
        block_tables=torch.arange(8, dtype=torch.int32).view(1, 8),
        block_tables_per_layer=None,
        prompt_lens=np.array([300]),
        input_positions=np.array([299]),
        multi_modal_kwargs={},
        stage_inputs=None,
        perform_device_sampling=True,
        tt_sampling_params=_sampling(),
        prefill_empty_slots=[3],
    )

    TTModelRunner.submit_prefill(runner, model_input, [1])

    ((kind, kwargs),) = target.calls
    assert kind == "prefill"
    assert kwargs["execution"] == "traced-prefill"
    assert kwargs["start_pos"].tolist() == [256]
    assert kwargs["prompt_lens"].tolist() == [300]
    assert kwargs["empty_slots"] == [3]
    assert kwargs["kv_cache"] is runner.kv_caches


def test_plugin_decode_reaches_tttv2_including_a_decode_join_row():
    """Row 1 is a request whose prompt KV arrived remotely: it joins decode at position
    len - 1 in a slot no prefill touched. To the model it is an ordinary decode row."""
    from vllm_tt_plugin.async_decode import TTAsyncDecodeController

    model, target = _tttv2_generator()
    runner = SimpleNamespace(
        model=model,
        kv_caches=object(),
        trace_mode="all",
        request_specific_rope=False,
        supports_stage_outputs=False,
        async_decode_scheduling=False,
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
    )
    model_input = SimpleNamespace(
        input_tokens=torch.tensor([[11], [22]], dtype=torch.int32),
        input_positions=torch.tensor([40, 299], dtype=torch.int32),
        block_tables=torch.tensor(
            [[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]], dtype=torch.int32
        ),
        block_tables_per_group=None,
        block_tables_per_layer=None,
        unpadded_batch_size=2,
        tt_sampling_params=_sampling(2),
        perform_device_sampling=True,
        prompt_tokens=None,
        output_tokens=None,
        decode_layout_changed=True,
        num_valid_drafts=None,
        accepted_counts=None,
        spec_mode=None,
        slot_remap=torch.tensor([0, 1], dtype=torch.int32),
        prompt_lens=None,
    )

    TTAsyncDecodeController(runner).submit_decode(model_input, read_from_device=True)

    ((kind, kwargs),) = target.calls
    assert kind == "decode"
    assert kwargs["execution"] == "traced-decode"
    assert kwargs["tokens"].tolist() == [11, 22]
    assert kwargs["start_pos"].tolist() == [40, 299]
    # Legacy (version-0) reload contract: a layout change reloads through reset_batch.
    assert kwargs["reset_batch"] is True
    assert kwargs["slot_remap"].tolist() == [0, 1]


def test_checkpoint_name_of_hub_cache_snapshot():
    from vllm_tt_plugin.platform import _checkpoint_name

    snap = (
        "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct"
        "/snapshots/0e9e39f2"
    )
    assert _checkpoint_name(snap) == "Llama-3.1-8B-Instruct"
    assert (
        _checkpoint_name("meta-llama/Llama-3.1-8B-Instruct") == "Llama-3.1-8B-Instruct"
    )
    assert _checkpoint_name("/models/Llama-3.1-8B-Instruct/") == "Llama-3.1-8B-Instruct"
