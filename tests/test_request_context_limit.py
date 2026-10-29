# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``tt_request_context_limit``: a request limit below the advertised context.

A model may serve ``max_model_len`` for most of its layers while some layers
compute only a shorter context faithfully for now (GLM-5.3-Flash's MLA/DSA
layers in the dense-exact regime). Admission must refuse longer requests and
resolve default ``max_tokens`` against the limit, on every entry point, instead
of letting them reach the device.
"""

import pytest
from vllm.sampling_params import SamplingParams

from tests.test_block_request_validation import (
    ARModel,
    _ar_config,
    _patch_model_resolution,
    _patch_upstream_boundary,
)
from vllm_tt_plugin.config import (
    get_tt_request_context_limit,
    store_tt_request_context_limit,
)
from vllm_tt_plugin.platform import TTPlatform

MAX_MODEL_LEN = 1024


def _model(limit):
    return type(
        "LimitedModel",
        (ARModel,),
        {
            "model_capabilities": {
                **ARModel.model_capabilities,
                "tt_request_context_limit": limit,
            }
        },
    )


@pytest.fixture(autouse=True)
def block_contract():
    """Neutralize the block-output module's autouse contract (not imported here)."""


def _startup(monkeypatch, limit):
    config = _ar_config()
    config.model_config.max_model_len = MAX_MODEL_LEN
    _patch_model_resolution(monkeypatch, _model(limit))
    TTPlatform.check_and_update_config(config)
    return config


def _validate(prompt_len, max_tokens):
    TTPlatform.validate_request(
        {"prompt_token_ids": [1] * prompt_len}, SamplingParams(max_tokens=max_tokens)
    )


def test_startup_stores_the_limit_and_installs_the_default_patch(monkeypatch):
    input_processor = _patch_upstream_boundary(monkeypatch)
    config = _startup(monkeypatch, 640)

    assert get_tt_request_context_limit(config) == 640
    assert hasattr(input_processor, "_tt_original_process_inputs")


@pytest.mark.parametrize("limit", [0, MAX_MODEL_LEN, MAX_MODEL_LEN + 1])
def test_a_limit_that_does_not_bind_is_no_limit(monkeypatch, limit):
    input_processor = _patch_upstream_boundary(monkeypatch)
    config = _startup(monkeypatch, limit)

    assert get_tt_request_context_limit(config) == 0
    assert not hasattr(input_processor, "_tt_original_process_inputs")
    _validate(prompt_len=1000, max_tokens=24)


@pytest.mark.parametrize("limit", [True, -1, "640", 640.0])
def test_invalid_limits_fail_at_startup(monkeypatch, limit):
    with pytest.raises(ValueError, match="tt_request_context_limit"):
        _startup(monkeypatch, limit)


def test_the_store_rejects_what_the_platform_would_not_resolve():
    config = _ar_config()
    with pytest.raises(ValueError, match="request_context_limit"):
        store_tt_request_context_limit(config, -1)


def test_admission_refuses_requests_over_the_limit(monkeypatch):
    # keep the config alive: the platform holds it weakly, as for a live engine
    config = _startup(monkeypatch, 640)

    _validate(prompt_len=600, max_tokens=40)
    _validate(prompt_len=639, max_tokens=None)
    with pytest.raises(
        ValueError, match=r"600 prompt \+ 41 output tokens exceeds .* limit of 640"
    ):
        _validate(prompt_len=600, max_tokens=41)
    with pytest.raises(ValueError, match=r"640 prompt \+ 1 output tokens"):
        _validate(prompt_len=640, max_tokens=None)
    assert get_tt_request_context_limit(config) == 640


def test_default_max_tokens_resolve_against_the_limit_on_the_clone(monkeypatch):
    input_processor = _patch_upstream_boundary(monkeypatch)
    config = _startup(monkeypatch, 640)
    processor = type("P", (), {})()
    processor.vllm_config = config
    processor.model_config = config.model_config
    shared = SamplingParams(max_tokens=None)

    request = input_processor.InputProcessor.process_inputs(
        processor, "r0", {"prompt_token_ids": [1] * 600}, shared
    )

    assert request.sampling_params.max_tokens == 40
    assert shared.max_tokens is None  # the caller-owned params stay untouched
    explicit = input_processor.InputProcessor.process_inputs(
        processor, "r1", {"prompt_token_ids": [1] * 600}, SamplingParams(max_tokens=7)
    )
    assert explicit.sampling_params.max_tokens == 7


def test_frontend_max_output_tokens_follow_the_limit(monkeypatch):
    config = _startup(monkeypatch, 640)
    platform = TTPlatform()

    assert platform.get_max_output_tokens(600) == 40
    assert platform.get_max_output_tokens(639) == 1
    with pytest.raises(ValueError, match=r"640 prompt \+ 1 output tokens"):
        platform.get_max_output_tokens(640)
    assert get_tt_request_context_limit(config) == 640


def test_no_limit_keeps_the_upstream_frontend_default(monkeypatch):
    config = _startup(monkeypatch, 0)

    assert TTPlatform().get_max_output_tokens(600) > MAX_MODEL_LEN
    assert get_tt_request_context_limit(config) == 0


def test_admission_refuses_more_parallel_samples_than_the_batch(monkeypatch):
    """n > max_num_seqs is a 400 (tests/serving malformed/huge_n: n=999 ran 999
    sequences past the client's timeout); n up to the batch is admitted"""
    config = _startup(monkeypatch, 0)
    cap = int(config.scheduler_config.max_num_seqs)
    prompt = {"prompt_token_ids": [1] * 8}
    TTPlatform.validate_request(prompt, SamplingParams(max_tokens=1, n=cap))
    with pytest.raises(ValueError, match=rf"n={cap + 1} parallel samples exceed max_num_seqs={cap}"):
        TTPlatform.validate_request(prompt, SamplingParams(max_tokens=1, n=cap + 1))
