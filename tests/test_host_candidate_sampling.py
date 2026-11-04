# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host sampling from device-side top-K candidates (``host_candidate_sampling``)."""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

from tests.test_model_runner import _batch_with_one_request, _fake_runner, _prepare
from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.input_batch import SamplingInputBatch
from vllm_tt_plugin.model_input import TTHostCandidates, TTSamplingParams
from vllm_tt_plugin.model_runner import TTModelRunner

K = 256


def _sampling(temperature, top_k, top_p):
    rows = len(temperature)
    return TTSamplingParams(
        temperature=torch.tensor(temperature, dtype=torch.float32),
        top_k=torch.tensor(top_k, dtype=torch.int32),
        top_p=torch.tensor(top_p, dtype=torch.float32),
        presence_penalty=torch.zeros(rows),
        frequency_penalty=torch.zeros(rows),
        repetition_penalty=torch.ones(rows),
        seed=torch.full((rows,), -1),
        num_logprobs=torch.full((rows,), -2),
        enable_log_probs=torch.zeros(rows, dtype=torch.bool),
    )


class _CandidateModel:
    """What the device returns: raw top-K logits, their ids, logsumexp(logits / T)."""

    model_capabilities = {"host_candidate_sampling": {"k": K}}

    def __init__(self, logits):
        self.logits = logits
        self.full_reads = 0

    def candidates(self, temperature):
        values, indices = torch.topk(self.logits, K, dim=-1)
        t = torch.where(temperature > 0, temperature, torch.ones_like(temperature))
        lse = torch.logsumexp(self.logits / t.unsqueeze(1), dim=-1)
        return TTHostCandidates(values=values, indices=indices, lse=lse)

    def read_decode_candidate_logits(self):
        self.full_reads += 1
        return self.logits.clone()


def _runner(model, vocab):
    runner = SimpleNamespace(
        model=model,
        vocab_size=vocab,
        host_candidate_fallback_rows=0,
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        _is_adaptive_block_output=False,
        tt_per_lane_max_num_seqs=0,
    )
    runner._sample_host_candidates = (
        lambda *args: TTModelRunner._sample_host_candidates(runner, *args)
    )
    return runner


def _sample(runner, params, batch):
    candidates = runner.model.candidates(params.temperature)
    runner.tt_per_lane_max_num_seqs = batch
    tokens, logprobs = TTModelRunner._get_output_tokens(
        runner,
        tt_out=candidates,
        tt_log_probs=None,
        sampling_params=params,
        model_input=SimpleNamespace(grammar_bitmask=[None]),
        batch_size_per_dp=[batch],
        perform_device_sampling=False,
        is_decode=True,
    )
    assert logprobs == [None]
    return tokens[0]


def _reference_probs(logits, temperature, top_k, top_p):
    scaled = logits / temperature.unsqueeze(1)
    return apply_top_k_top_p(scaled.clone(), top_k.long(), top_p).softmax(dim=-1)


@pytest.mark.parametrize(
    "temperature, top_k, top_p",
    [(0.7, 4096, 0.9), (1.0, 50, 1.0), (1.3, 100, 0.8), (0.5, 4096, 0.95)],
)
def test_candidate_tokens_follow_the_full_vocabulary_distribution(
    temperature, top_k, top_p
):
    torch.manual_seed(0)
    vocab, samples = 4096, 40000
    row = torch.randn(vocab) * 3.0
    model = _CandidateModel(row.expand(samples, vocab).contiguous())
    runner = _runner(model, vocab)
    params = _sampling([temperature] * samples, [top_k] * samples, [top_p] * samples)

    tokens = _sample(runner, params, samples).view(-1)

    assert tokens.dtype == torch.int32
    assert model.full_reads == 0
    expected = _reference_probs(
        row.unsqueeze(0), params.temperature[:1], params.top_k[:1], params.top_p[:1]
    )[0]
    seen = torch.bincount(tokens.long(), minlength=vocab).float() / samples
    assert bool((seen[expected == 0] == 0).all())
    assert (seen - expected).abs().sum() / 2 < 0.02


def test_each_row_uses_its_own_temperature_and_cut():
    torch.manual_seed(1)
    vocab, reps = 2048, 20000
    rows = torch.randn(2, vocab) * 2.0
    logits = rows.repeat_interleave(reps, dim=0)
    model = _CandidateModel(logits)
    runner = _runner(model, vocab)
    params = _sampling(
        [0.6] * reps + [1.5] * reps,
        [vocab] * reps + [8] * reps,
        [0.7] * reps + [1.0] * reps,
    )

    tokens = _sample(runner, params, 2 * reps).view(-1)

    for r in range(2):
        sel = slice(r * reps, (r + 1) * reps)
        expected = _reference_probs(
            rows[r : r + 1],
            params.temperature[sel][:1],
            params.top_k[sel][:1],
            params.top_p[sel][:1],
        )[0]
        seen = torch.bincount(tokens[sel].long(), minlength=vocab).float() / reps
        assert bool((seen[expected == 0] == 0).all())
        assert (seen - expected).abs().sum() / 2 < 0.03


def test_undecidable_rows_read_the_full_logits():
    torch.manual_seed(2)
    vocab = 4096
    peaked = torch.randn(vocab) * 4.0
    # Nearly uniform: the top 256 tokens hold ~6% of the mass, far below p.
    flat = torch.randn(vocab) * 1e-3
    model = _CandidateModel(torch.stack([peaked, flat, peaked]))
    runner = _runner(model, vocab)
    params = _sampling([1.0] * 3, [vocab] * 3, [0.9] * 3)

    tokens = _sample(runner, params, 3).view(-1)

    assert model.full_reads == 1
    assert runner.host_candidate_fallback_rows == 1
    for r, row in enumerate((peaked, flat, peaked)):
        expected = _reference_probs(
            row.unsqueeze(0), params.temperature[:1], params.top_k[:1], params.top_p[:1]
        )[0]
        assert expected[tokens[r]] > 0


def test_fallback_rows_keep_the_full_vocabulary_distribution():
    torch.manual_seed(3)
    vocab, samples = 1024, 30000
    # ~30% of the mass lies outside the top 256, so top_p=0.9 is undecidable.
    row = (
        torch.cat([torch.full((16,), 3.0), torch.zeros(vocab - 16)])
        + torch.randn(vocab) * 0.1
    )
    model = _CandidateModel(row.expand(samples, vocab).contiguous())
    runner = _runner(model, vocab)
    params = _sampling([1.0] * samples, [vocab] * samples, [0.9] * samples)

    tokens = _sample(runner, params, samples).view(-1)

    assert runner.host_candidate_fallback_rows == samples
    expected = _reference_probs(
        row.unsqueeze(0), params.temperature[:1], params.top_k[:1], params.top_p[:1]
    )[0]
    seen = torch.bincount(tokens.long(), minlength=vocab).float() / samples
    assert bool((seen[expected == 0] == 0).all())
    # Hundreds of kept tokens: compare the head and the tail mass, not each token.
    assert abs(seen[:16].sum() - expected[:16].sum()) < 0.01
    assert (seen[:16] - expected[:16]).abs().max() < 0.01


# --- Routing ----------------------------------------------------------------


def _routing_runner(sampling, *, capability=True, **batch):
    flags = dict(no_penalties=True, no_allowed_token_ids=True, max_num_logprobs=None)
    flags.update(batch)
    return SimpleNamespace(
        model=SimpleNamespace(
            model_capabilities={"host_candidate_sampling": {"k": K}}
            if capability
            else {}
        ),
        model_config=SimpleNamespace(logits_processors=[]),
        input_batch=SimpleNamespace(sampling=sampling, **flags),
        vocab_size=128000,
    )


def _rows(temperature=0.8, top_k=128000, top_p=0.9):
    sampling = SamplingInputBatch(4)
    sampling.temperature[:] = temperature
    sampling.top_k[:] = top_k
    sampling.top_p[:] = top_p
    return sampling


@pytest.mark.parametrize(
    "temperature, top_k, top_p, expected",
    [
        (0.8, 128000, 0.9, True),  # top-p
        (0.8, 40, 1.0, True),  # top-k inside the candidates
        (0.8, K, 1.0, True),
        (0.8, K + 1, 1.0, False),  # top-k wider than the candidates, no top-p
        (0.8, 128000, 1.0, False),  # plain temperature sampling needs the whole row
        (0.8, 1000, 0.9, False),  # top-p over a top-k set wider than the candidates
        (0.0, 1, 0.0, False),  # greedy keeps the bitwise argmax path
        (1.0, 1, 0.0, True),  # top_k=1 at T > 0
    ],
)
def test_routing_by_sampling_params(temperature, top_k, top_p, expected):
    runner = _routing_runner(_rows(temperature, top_k, top_p))
    assert TTModelRunner.check_host_candidate_sampling(runner, [0, 2]) is expected


def test_one_row_off_the_candidate_path_routes_the_whole_step():
    sampling = _rows()
    sampling.temperature[2] = 0.0
    runner = _routing_runner(sampling)
    assert TTModelRunner.check_host_candidate_sampling(runner, [0, 1])
    assert not TTModelRunner.check_host_candidate_sampling(runner, [0, 2])


def test_seeded_rows_keep_their_host_generators():
    sampling = _rows()
    sampling.seed[1] = 1234
    assert not TTModelRunner.check_host_candidate_sampling(
        _routing_runner(sampling), [0, 1]
    )
    sampling = _rows()
    sampling.generators[1] = torch.Generator()
    assert not TTModelRunner.check_host_candidate_sampling(
        _routing_runner(sampling), [0, 1]
    )


@pytest.mark.parametrize(
    "batch",
    [
        dict(no_penalties=False),
        dict(no_allowed_token_ids=False),
        dict(max_num_logprobs=0),
        dict(max_num_logprobs=5),
    ],
)
def test_full_logits_features_keep_the_full_logits_path(batch):
    assert not TTModelRunner.check_host_candidate_sampling(
        _routing_runner(_rows(), **batch), [0]
    )


def test_bad_words_logits_processors_and_missing_capability():
    sampling = _rows()
    sampling.bad_words_token_ids = {0: [[5]]}
    assert not TTModelRunner.check_host_candidate_sampling(
        _routing_runner(sampling), [0]
    )
    sampling = _rows()
    sampling.has_active_logitsprocs = lambda: True
    assert not TTModelRunner.check_host_candidate_sampling(
        _routing_runner(sampling), [0]
    )
    runner = _routing_runner(_rows())
    runner.model_config.logits_processors = ["custom"]
    assert not TTModelRunner.check_host_candidate_sampling(runner, [0])
    assert not TTModelRunner.check_host_candidate_sampling(
        _routing_runner(_rows(), capability=False), [0]
    )
    assert not TTModelRunner.check_host_candidate_sampling(_routing_runner(_rows()), [])


@pytest.mark.parametrize("candidates_ok", [True, False])
def test_decode_build_marks_candidate_steps(candidates_ok):
    batch, request = _batch_with_one_request(
        prompt_len=8,
        output_len=1,
        num_computed_tokens=8,
        sampling_params=SamplingParams(temperature=0.8, top_p=0.9),
    )
    runner = _fake_runner(batch, request)
    asked = []
    runner.check_host_candidate_sampling = (
        lambda rows: asked.append(rows) or candidates_ok
    )

    model_input = _prepare(runner, ("r", 1, 8, 1))

    assert model_input.prompt_lens is None
    assert asked == [[0]]
    assert model_input.host_candidate_sampling is candidates_ok


def test_prefill_and_device_sampled_builds_never_use_candidates():
    batch, request = _batch_with_one_request(
        prompt_len=8,
        output_len=0,
        num_computed_tokens=0,
        sampling_params=SamplingParams(temperature=0.8, top_p=0.9),
    )
    runner = _fake_runner(batch, request)
    runner.check_host_candidate_sampling = lambda rows: True
    assert not _prepare(runner, ("r", 8, 0, 0)).host_candidate_sampling

    batch, request = _batch_with_one_request(
        prompt_len=8,
        output_len=1,
        num_computed_tokens=8,
        sampling_params=SamplingParams(temperature=0.8, top_p=0.9),
    )
    runner = _fake_runner(batch, request)
    runner.check_host_candidate_sampling = lambda rows: True
    runner.check_perform_device_sampling = lambda **_: True
    assert not _prepare(runner, ("r", 1, 8, 1)).host_candidate_sampling


# --- Submission and readback --------------------------------------------------


class _DeviceCandidates:
    """Stands in for the model's device-side output, which only the model can read."""


class _SubmittingModel:
    decode_input_update_contract = 1
    model_capabilities = {
        "supports_async_decode": True,
        "host_candidate_sampling": {"k": K},
    }

    def __init__(self):
        self.calls = []

    def decode_forward(self, **kwargs):
        self.calls.append(kwargs)
        return [_DeviceCandidates()]

    def process_decode_output_host(self, tt_out, is_tokens=False, **kwargs):
        assert isinstance(tt_out[0], _DeviceCandidates) and not is_tokens and not kwargs
        return torch.zeros(2, K), torch.zeros(2, K, dtype=torch.int64), torch.zeros(2)


def _submission_input(host_candidates):
    page_table = torch.tensor([[0]], dtype=torch.int32)
    return SimpleNamespace(
        input_tokens=torch.zeros((2, 1), dtype=torch.int32),
        input_positions=torch.zeros((2,), dtype=torch.int32),
        block_tables=page_table,
        block_tables_per_group=[page_table],
        block_tables_per_layer=None,
        unpadded_batch_size=2,
        tt_sampling_params=_sampling([0.7, 0.0], [K, 1], [0.9, 1.0]),
        perform_device_sampling=False,
        host_candidate_sampling=host_candidates,
        prompt_tokens=None,
        output_tokens=None,
        decode_layout_changed=False,
        num_valid_drafts=None,
        accepted_counts=None,
        spec_mode=None,
        slot_remap=None,
    )


def _controller(model):
    runner = SimpleNamespace(
        model=model,
        trace_mode="decode_only",
        kv_caches=object(),
        request_specific_rope=False,
        supports_stage_outputs=False,
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
    )
    return TTAsyncDecodeController(runner)


def test_candidate_submission_passes_temperatures_and_finalizes_candidates():
    model = _SubmittingModel()
    controller = _controller(model)

    submission = controller.submit_decode(
        _submission_input(True), read_from_device=False
    )
    finalized = controller.finalize_decode(submission)

    call = model.calls[-1]
    assert call["host_candidates"] is True
    assert call["candidate_temperature"].tolist() == pytest.approx([0.7, 0.0])
    assert call["reload_inputs"] is True
    assert "sampling_params" not in call
    assert isinstance(finalized.tt_out, TTHostCandidates)
    assert finalized.tt_out.values.shape == (2, K)
    assert finalized.tt_log_probs is None


def test_full_logits_submission_keeps_its_call_shape():
    model = _SubmittingModel()
    controller = _controller(model)

    controller.submit_decode(_submission_input(False), read_from_device=False)

    assert "host_candidates" not in model.calls[-1]
    assert "candidate_temperature" not in model.calls[-1]
