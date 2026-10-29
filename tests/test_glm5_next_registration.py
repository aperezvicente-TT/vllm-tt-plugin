# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import pytest
import vllm.config  # noqa: F401  # finish vLLM init before the plugin package,

# whose bare import re-enters vllm.platforms mid-initialization
import vllm_tt_plugin.platform as tt_platform

GLM5_NEXT_TARGET = (
    "models.autoports.zai_org_glm_5_3_flash.tt.generator_vllm:Glm5NextForCausalLM"
)


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    from vllm.model_executor.models.registry import ModelRegistry

    registry: dict[str, str] = {}
    monkeypatch.setattr(
        ModelRegistry, "get_supported_archs", staticmethod(lambda: list(registry))
    )
    monkeypatch.setattr(
        ModelRegistry,
        "register_model",
        staticmethod(lambda arch, target: registry.__setitem__(arch, target)),
    )
    monkeypatch.delenv("TT_MODEL_CLASS_OVERRIDES", raising=False)
    return registry


def test_glm5_next_registers_the_bare_and_tt_architecture_names(registered):
    """The checkpoint declares Glm5NextForConditionalGeneration, which vLLM
    0.26.0 does not ship. ModelConfig resolves the architecture before
    check_and_update_config adds the TT prefix, so the bare name must be
    registered as well, or it resolves to the Transformers multimodal
    fallback."""
    tt_platform.register_tt_models()

    assert registered["Glm5NextForConditionalGeneration"] == GLM5_NEXT_TARGET
    assert registered["TTGlm5NextForConditionalGeneration"] == GLM5_NEXT_TARGET


def test_glm5_next_registration_is_idempotent_and_yields_to_overrides(
    registered, monkeypatch: pytest.MonkeyPatch
):
    target = "pkg.mod:OtherGlm5Next"
    monkeypatch.setenv(
        "TT_MODEL_CLASS_OVERRIDES", f"Glm5NextForConditionalGeneration={target}"
    )

    tt_platform.register_tt_models()
    tt_platform.register_tt_models()

    assert registered["Glm5NextForConditionalGeneration"] == target
    assert registered["TTGlm5NextForConditionalGeneration"] == target
