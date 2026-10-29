# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A structured /v1/completions request under --reasoning-parser glm45 is
constrained from its first token (the structured/choice serving check).

The real GLM-5.3-Flash tokenizer and vLLM's real glm45 reasoner and
StructuredOutputManager gate (should_fill_bitmask / _get_reasoner) run here;
only the manager's construction is replaced (it needs a full vllm_config)."""

import os
from types import SimpleNamespace

import pytest

TOKENIZER = os.environ.get(
    "GLM53_TOKENIZER_DIR",
    "/opt/glm53/share/llm-wksp/models/models--zai-org--GLM-5.3-Flash/snapshots/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
)
pytestmark = pytest.mark.skipif(not os.path.isdir(TOKENIZER), reason="GLM-5.3-Flash tokenizer not on this host")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture
def mgr(tok):
    from vllm.reasoning import ReasoningParserManager
    from vllm.v1.structured_output import StructuredOutputManager

    class Mgr:  # the real gate methods over a hand-built manager state
        reasoner_cls = ReasoningParserManager.get_reasoning_parser("glm45")
        enable_in_reasoning = False
        tokenizer = tok
        _get_reasoner = StructuredOutputManager._get_reasoner
        should_fill_bitmask = StructuredOutputManager.should_fill_bitmask

    return Mgr()


def _request(prompt_ids, reasoning_parser_kwargs=None):
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams
    from vllm.v1.structured_output.request import StructuredOutputRequest

    so = StructuredOutputRequest.from_sampling_params(
        SamplingParams(max_tokens=4, structured_outputs=StructuredOutputsParams(choice=["red", "blue"]))
    )
    so.reasoning_parser_kwargs = reasoning_parser_kwargs
    return SimpleNamespace(request_id="r0", prompt_token_ids=list(prompt_ids), structured_output_request=so)


def _apply(mgr, req):
    from vllm_tt_plugin.scheduler import TTScheduler

    sched = SimpleNamespace(structured_output_manager=mgr, _reasoning_marker_ids=TTScheduler._reasoning_marker_ids)
    TTScheduler._raw_prompt_reasoning_state(sched, req)


def test_raw_completion_prompt_is_constrained_from_the_first_token(tok, mgr):
    ids = tok("Pick a color: ", add_special_tokens=False)["input_ids"]
    # vLLM alone: the thinking-on reasoner waits for '</think>', so the grammar never engages
    assert mgr.should_fill_bitmask(_request(ids)) is False
    req = _request(ids)
    _apply(mgr, req)
    assert req.structured_output_request.reasoning_ended is True
    assert mgr.should_fill_bitmask(req) is True


def test_raw_prompt_that_opens_reasoning_keeps_vllms_check(tok, mgr):
    think = tok.convert_tokens_to_ids("<think>")
    req = _request(tok("Q: pick.", add_special_tokens=False)["input_ids"] + [think])
    _apply(mgr, req)
    assert req.structured_output_request.reasoning_ended is None
    assert mgr.should_fill_bitmask(req) is False  # the model reasons first, as the prompt asked


def test_raw_prompt_with_closed_reasoning_keeps_vllms_check(tok, mgr):
    start, end = tok.convert_tokens_to_ids(["<think>", "</think>"])
    req = _request(tok("Q: pick.", add_special_tokens=False)["input_ids"] + [start, end])
    _apply(mgr, req)
    assert req.structured_output_request.reasoning_ended is None
    assert mgr.should_fill_bitmask(req) is True  # vLLM finds the end marker itself


def test_chat_requests_are_untouched(tok, mgr):
    prompt = tok.apply_chat_template([{"role": "user", "content": "pick"}], add_generation_prompt=True, tokenize=True)
    for kwargs in ({"chat_template_kwargs": {"enable_thinking": False}}, {"chat_template_kwargs": None}):
        req = _request(prompt, kwargs)
        _apply(mgr, req)
        assert req.structured_output_request.reasoning_ended is None


def test_unstructured_and_in_reasoning_requests_are_untouched(tok, mgr):
    req = SimpleNamespace(request_id="r1", prompt_token_ids=[1, 2], structured_output_request=None)
    _apply(mgr, req)  # no structured request: nothing to do
    mgr.enable_in_reasoning = True
    req = _request(tok("x", add_special_tokens=False)["input_ids"])
    _apply(mgr, req)
    assert req.structured_output_request.reasoning_ended is None
