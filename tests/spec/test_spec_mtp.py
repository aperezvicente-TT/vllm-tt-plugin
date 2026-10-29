# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM's ``mtp`` method is the model's own MTP head, drafting on device.

On TT an MTP head is part of the loaded target model, so ``mtp`` runs through
the model-owned drafter contract that ``custom_class`` also selects: the
runner verifies with ``decode_forward(..., spec_mode="argmax_ids")`` and asks
``propose_draft_tokens`` for the next drafts, and no upstream proposer or draft
checkpoint is built.

Upstream does build one. ``SpeculativeConfig.__post_init__`` defaults
``model`` to the target checkpoint, builds a second ``ModelConfig`` from it
with ``runner="draft"``, and raises ``NotImplementedError`` for a target type
it ships no MTP drafter architecture for, which is every TT model it has never
heard of (``glm5_next`` in vLLM 0.26). So these tests build real
``SpeculativeConfig`` and ``VllmConfig`` objects, as
``test_spec_async_admission.py`` does, to show the TT construction patch, the
admission and the runner selection running for real.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

# vLLM's bootstrap resolves the platform plugin, which imports plugin modules.
import vllm  # noqa: F401
import vllm.platforms

from tests.spec.fake_spec_model import FakeSpecModel
from vllm_tt_plugin.config import get_tt_spec_plan
from vllm_tt_plugin.platform import (
    _install_tt_mtp_speculative_config_patch,
    _uninstall_tt_mtp_speculative_config_patch,
)
from vllm_tt_plugin.spec_admission import MTP_DRAFT_METHOD
from vllm_tt_plugin.spec_decode import (
    HIDDEN_HANDOFF_ON_DEVICE,
    SPEC_REQUIREMENT_DEVICE_PROPOSE,
    SPEC_REQUIREMENT_HIDDEN_FEED,
)

ARCHITECTURE = "TTMtpHeadSpecModel"
NO_HIDDEN_FEED_ARCHITECTURE = "TTMtpNoHiddenFeedSpecModel"
DRAFT_LEN = 3


class MtpHeadModel(FakeSpecModel):
    """Declares what the GLM-5.3-Flash adapter declares for its MTP head.

    The drafter proposes on device and reads the target hidden state, which
    never leaves the device. Ordinary decode overlaps, but the speculative
    verify does not have the deferred readback, so a launch has to be
    synchronous.
    """

    max_supported_num_seqs = 8
    model_capabilities = {
        "supports_spec_decode": True,
        "spec_requirements": (
            SPEC_REQUIREMENT_DEVICE_PROPOSE,
            SPEC_REQUIREMENT_HIDDEN_FEED,
        ),
        "spec_hidden_handoff": (HIDDEN_HANDOFF_ON_DEVICE,),
        "supports_async_decode": True,
        "supports_async_spec_decode": False,
    }

    # vLLM's registry classifies an architecture by these members; a class it
    # cannot place is classified as pooling. None of them is ever called.
    def __init__(self, *, vllm_config=None, prefix=""):  # noqa: D107
        del vllm_config, prefix
        super().__init__()

    def embed_input_ids(self, input_ids):
        raise NotImplementedError("the TT worker does not call this")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("the TT worker does not call this")

    def compute_logits(self, hidden_states):
        raise NotImplementedError("the TT worker does not call this")


class NoHiddenFeedModel(MtpHeadModel):
    """Proposes on device but declares no hidden-state feed."""

    model_capabilities = {
        **MtpHeadModel.model_capabilities,
        "spec_requirements": (SPEC_REQUIREMENT_DEVICE_PROPOSE,),
    }


@pytest.fixture(scope="module", autouse=True)
def registered_architectures():
    from vllm.model_executor.models.registry import ModelRegistry

    for architecture, class_name in (
        (ARCHITECTURE, "MtpHeadModel"),
        (NO_HIDDEN_FEED_ARCHITECTURE, "NoHiddenFeedModel"),
    ):
        if architecture not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(
                architecture, f"tests.spec.test_spec_mtp:{class_name}"
            )
    yield


@pytest.fixture(autouse=True)
def unpatched_construction():
    """Each test installs the patch itself, so the default is upstream's.

    Restored afterwards too: the patch is process-wide, and the TT platform
    hook installs it whenever it runs.
    """
    _uninstall_tt_mtp_speculative_config_patch()
    yield
    _uninstall_tt_mtp_speculative_config_patch()


def _checkpoint(directory, architecture=ARCHITECTURE):
    """A checkpoint-shaped directory for the real ``ModelConfig``.

    The model type is one vLLM ships no MTP drafter for, as ``glm5_next`` is,
    and the config carries the MTP depth a checkpoint with its own head does.
    """
    (directory / "config.json").write_text(
        json.dumps(
            {
                "architectures": [architecture],
                "model_type": "llama",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "num_key_value_heads": 2,
                "num_nextn_predict_layers": 1,
                "vocab_size": 512,
                "max_position_embeddings": 2048,
                "torch_dtype": "bfloat16",
            }
        )
    )
    return directory


@pytest.fixture
def checkpoint(tmp_path):
    return _checkpoint(tmp_path)


def _model_config(checkpoint):
    from vllm.config import ModelConfig

    return ModelConfig(
        model=str(checkpoint),
        tokenizer=str(checkpoint),
        skip_tokenizer_init=True,
        trust_remote_code=False,
        dtype="bfloat16",
        seed=0,
        max_model_len=2048,
        enforce_eager=True,
        # Stated, as in test_spec_async_admission: an inferred runner sends the
        # automatic async selection down its pooling branch.
        runner="generate",
    )


def _speculative_config(model_config, parallel_config, **overrides):
    from vllm.config.speculative import SpeculativeConfig

    return SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=parallel_config,
        **{
            "method": MTP_DRAFT_METHOD,
            "num_speculative_tokens": DRAFT_LEN,
            **overrides,
        },
    )


def _build_config(checkpoint, *, async_scheduling=False, **speculative_overrides):
    """A real ``VllmConfig``, running the real gate and the TT hook."""
    from vllm.config import CacheConfig, ParallelConfig, SchedulerConfig, VllmConfig

    model_config = _model_config(checkpoint)
    parallel_config = ParallelConfig()
    return VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=CacheConfig(
            block_size=32, gpu_memory_utilization=0.9, cache_dtype="auto"
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_model_len=2048,
            is_encoder_decoder=False,
            async_scheduling=async_scheduling,
        ),
        speculative_config=_speculative_config(
            model_config, parallel_config, **speculative_overrides
        ),
        additional_config={"tt": {}},
    )


# region What upstream does with mtp


def test_upstream_cannot_build_mtp_for_a_target_it_ships_no_drafter_for(checkpoint):
    """The failure the TT patch exists for, reproduced so the rest proves it.

    Upstream builds a draft ``ModelConfig`` from the target checkpoint, finds
    a model type with no MTP drafter architecture, and gives up.
    """
    from vllm.config import ParallelConfig

    with pytest.raises(NotImplementedError) as excinfo:
        _speculative_config(_model_config(checkpoint), ParallelConfig())

    assert "Unsupported speculative method: 'mtp'" in str(excinfo.value)


# endregion What upstream does with mtp

# region The TT construction


def test_mtp_is_built_without_a_draft_model(checkpoint):
    """The target stands in for the draft, as it does for ``custom_class``.

    Identity, not equality: a second ``ModelConfig`` built from the same
    checkpoint would compare equal on most fields and still be the draft
    model this construction must not build.
    """
    from vllm.config import ParallelConfig

    _install_tt_mtp_speculative_config_patch()
    model_config = _model_config(checkpoint)
    parallel_config = ParallelConfig()

    config = _speculative_config(model_config, parallel_config)

    assert config.method == MTP_DRAFT_METHOD
    assert config.num_speculative_tokens == DRAFT_LEN
    # Defaulted to the target checkpoint, exactly as upstream defaults it.
    assert config.model == model_config.model
    assert config.draft_model_config is model_config
    assert config.draft_parallel_config is parallel_config
    # Still an MTP method to upstream's scheduler, which keys its own
    # lookahead and prefix-cache behaviour on it.
    assert config.use_eagle()


def test_dflash_is_built_as_the_model_owned_drafter(checkpoint):
    """``dflash`` (the Apache-2.0 DFlash2 drafter run inside the loaded
    model's step) builds no draft model either: 'model' defaults to the
    drafter's id, the draft configs ARE the target's, and the method keeps
    vLLM's name (the scheduler's lookahead keys on it)."""
    from vllm.config import ParallelConfig

    from vllm_tt_plugin.spec_admission import DFLASH_DRAFT_METHOD, DFLASH_DRAFTER_ID

    _install_tt_mtp_speculative_config_patch()
    model_config = _model_config(checkpoint)
    parallel_config = ParallelConfig()

    config = _speculative_config(model_config, parallel_config, method=DFLASH_DRAFT_METHOD)

    assert config.method == DFLASH_DRAFT_METHOD
    assert config.num_speculative_tokens == DRAFT_LEN
    assert config.model == DFLASH_DRAFTER_ID
    assert config.draft_model_config is model_config
    assert config.draft_parallel_config is parallel_config
    assert config.use_eagle()


def test_a_deprecated_mtp_alias_still_becomes_mtp(checkpoint):
    """Upstream rewrites every other MTP name to ``mtp``; so does TT."""
    from vllm.config import ParallelConfig

    _install_tt_mtp_speculative_config_patch()
    model_config = _model_config(checkpoint)

    config = _speculative_config(model_config, ParallelConfig(), method="deepseek_mtp")

    assert config.method == MTP_DRAFT_METHOD
    assert config.draft_model_config is model_config


def test_other_methods_keep_upstream_construction(checkpoint):
    """Only MTP methods are taken over: ngram still gets upstream's defaults."""
    from vllm.config import ParallelConfig

    _install_tt_mtp_speculative_config_patch()

    config = _speculative_config(
        _model_config(checkpoint), ParallelConfig(), method="ngram"
    )

    assert config.method == "ngram"
    assert (config.prompt_lookup_min, config.prompt_lookup_max) == (5, 5)


def test_the_construction_patch_is_installed_only_on_tt(monkeypatch):
    """On any other platform ``mtp`` has to build the upstream drafter."""
    from vllm.config.speculative import SpeculativeConfig

    original = SpeculativeConfig.__post_init__
    monkeypatch.setattr(vllm.platforms, "current_platform", SimpleNamespace())

    _install_tt_mtp_speculative_config_patch()

    assert SpeculativeConfig.__post_init__ is original


def test_installing_twice_changes_nothing():
    """Installed from three entry points and on every configuration re-entry."""
    from vllm.config.speculative import SpeculativeConfig

    original = SpeculativeConfig.__post_init__
    _install_tt_mtp_speculative_config_patch()
    once = SpeculativeConfig.__post_init__
    _install_tt_mtp_speculative_config_patch()

    assert SpeculativeConfig.__post_init__ is once
    assert once.__wrapped__ is original
    _uninstall_tt_mtp_speculative_config_patch()
    assert SpeculativeConfig.__post_init__ is original


# endregion The TT construction

# region Admission through the real hook


def test_an_mtp_launch_is_admitted_and_publishes_its_plan(checkpoint):
    """The launch the GLM-5.3-Flash adapter serves: mtp, no async scheduling."""
    _install_tt_mtp_speculative_config_patch()

    config = _build_config(checkpoint, async_scheduling=False)

    assert config.speculative_config.method == MTP_DRAFT_METHOD
    assert config.scheduler_config.async_scheduling is False
    plan = get_tt_spec_plan(config)
    assert plan is not None
    assert plan.effective_k == DRAFT_LEN
    assert config.scheduler_config.scheduler_cls == (
        "vllm_tt_plugin.scheduler.TTScheduler"
    )


def test_an_mtp_launch_left_to_select_async_scheduling_is_refused(checkpoint):
    """Upstream turns asynchronous scheduling on for ``mtp`` by itself.

    ``mtp`` is one of the EAGLE/MTP methods upstream schedules asynchronously,
    so a launch that does not say ``--no-async-scheduling`` arrives at the TT
    hook with it on. A model that overlaps ordinary decode but declares no
    deferred verify is then refused, by name, rather than served wrongly.
    """
    _install_tt_mtp_speculative_config_patch()

    with pytest.raises(Exception) as excinfo:
        _build_config(checkpoint, async_scheduling=None)

    message = str(excinfo.value)
    assert "supports_async_spec_decode" in message
    assert "--no-async-scheduling" in message


def test_the_target_checkpoint_may_be_named_explicitly(checkpoint):
    _install_tt_mtp_speculative_config_patch()

    config = _build_config(checkpoint, model=str(checkpoint))

    assert config.speculative_config.model == str(checkpoint)
    assert get_tt_spec_plan(config) is not None


def test_a_separate_mtp_checkpoint_is_refused(checkpoint):
    """TT loads no draft weights, so naming some would misreport the drafter."""
    _install_tt_mtp_speculative_config_patch()

    with pytest.raises(Exception) as excinfo:
        _build_config(checkpoint, model="some-org/separate-mtp-head")

    message = str(excinfo.value)
    assert "some-org/separate-mtp-head" in message
    assert str(checkpoint) in message
    assert "propose_draft_tokens" in message


def test_a_model_without_a_hidden_feed_is_refused_mtp(tmp_path):
    """An MTP head reads the target hidden state by construction."""
    _install_tt_mtp_speculative_config_patch()
    checkpoint = _checkpoint(tmp_path, NO_HIDDEN_FEED_ARCHITECTURE)

    with pytest.raises(Exception) as excinfo:
        _build_config(checkpoint)

    assert SPEC_REQUIREMENT_HIDDEN_FEED in str(excinfo.value)


# endregion Admission through the real hook

# region The runner


def test_the_runner_drafts_from_the_model(checkpoint):
    """The runner the worker builds for this config asks the model to draft.

    The drafts come from ``propose_draft_tokens``, selected by the method's
    requirements, and nothing builds a host proposer for it.
    """
    from vllm_tt_plugin.model_runner import TTModelRunner

    _install_tt_mtp_speculative_config_patch()
    config = _build_config(checkpoint)

    runner = TTModelRunner(
        config,
        mesh_device=None,
        trace_mode="none",
        enable_model_warmup=False,
        num_devices=1,
    )

    assert runner._spec_method == MTP_DRAFT_METHOD
    assert runner._spec_drafts_from_model is True
    assert runner._num_speculative_tokens == DRAFT_LEN
    assert runner._ngram_proposer is None


# endregion The runner
