# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

from vllm_tt_plugin.host_sampler import TTTopKTopPSampler


@pytest.mark.parametrize("batch", [1, 4, 7, 32])
@pytest.mark.parametrize("fp64", [False, True])
@pytest.mark.parametrize(
    "mode", ["raw_logprobs", "processed_logits", "processed_logprobs"]
)
@pytest.mark.parametrize("filters", [False, True])
def test_parallel_sampling_preserves_tokens_scores_and_rng(batch, fp64, mode, filters):
    torch.manual_seed(197)
    logits = torch.randn(batch, 4101)
    logits[:, 0] = -torch.inf
    native = TopKTopPSampler(mode, fp64)
    parallel = TTTopKTopPSampler(mode, fp64)
    seeds = {i: torch.Generator().manual_seed(123 + i) for i in range(batch)}
    copies = {
        i: torch.Generator().set_state(gen.get_state()) for i, gen in seeds.items()
    }
    k = torch.full((batch,), 80, dtype=torch.int32) if filters else None
    p = torch.full((batch,), 0.9) if filters else None
    global_state = torch.get_rng_state().clone()
    try:
        for _ in range(3):
            expected, expected_scores = native.forward_native(
                logits.clone(), seeds, k, p
            )
            actual, actual_scores = parallel(logits.clone(), copies, k, p)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if expected_scores is None:
                assert actual_scores is None
            else:
                torch.testing.assert_close(
                    actual_scores, expected_scores, rtol=0, atol=0
                )
            assert all(
                torch.equal(seeds[i].get_state(), copies[i].get_state()) for i in seeds
            )
            assert torch.equal(torch.get_rng_state(), global_state)
        assert parallel._sampling_pool is not None
        assert parallel._noise_cache
    finally:
        if parallel._sampling_pool is not None:
            parallel._sampling_pool.shutdown()


# Unseeded requests are sampled by inverse CDF (same distribution, not upstream's RNG
# stream): see test_unseeded_sampling_matches_distribution.
@pytest.mark.parametrize("kind", ["partial", "shared"])
def test_fallback_keeps_global_and_request_rng_state(kind):
    torch.manual_seed(9)
    logits = torch.randn(8, 4101)
    native = TopKTopPSampler()
    parallel = TTTopKTopPSampler()
    generators = {i: torch.Generator().manual_seed(20 + i) for i in range(8)}
    if kind == "unseeded":
        generators = {}
    elif kind == "partial":
        generators = {i: gen for i, gen in generators.items() if i % 2}
    else:
        generators = dict.fromkeys(range(8), generators[0])
    initial = {i: gen.get_state().clone() for i, gen in generators.items()}
    global_initial = torch.get_rng_state().clone()
    expected, _ = native.forward_native(logits.clone(), generators, None, None)
    final = {i: gen.get_state().clone() for i, gen in generators.items()}
    global_final = torch.get_rng_state().clone()
    for i, gen in generators.items():
        gen.set_state(initial[i])
    torch.set_rng_state(global_initial)
    actual, _ = parallel(logits.clone(), generators, None, None)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), global_final)
    assert all(torch.equal(gen.get_state(), final[i]) for i, gen in generators.items())
    assert parallel._sampling_pool is None


def test_unseeded_sampling_matches_distribution():
    """Unseeded rows sample softmax(logits) after top-k/top-p, as upstream does.

    The distribution matches upstream; individual tokens do not.
    """
    torch.manual_seed(3)
    vocab = 5000
    logits = torch.full((3, vocab), -30.0)
    # Four live tokens.
    logits[:, :4] = torch.tensor([2.0, 1.0, 0.5, -1.0]).log_softmax(0)
    sampler = TTTopKTopPSampler()
    want = torch.softmax(logits[0], 0)[:4]
    counts = torch.zeros(4)
    for _ in range(4000):
        tok, _ = sampler(logits.clone(), {}, None, None)
        assert int(tok.max()) < 4
        counts += torch.bincount(tok, minlength=vocab)[:4].float()
    freq = counts / counts.sum()
    assert torch.allclose(freq, want, atol=0.01), (freq, want)
    # top-p keeps the smallest prefix whose preceding mass is below p: here tokens 0 and
    # 1 only.
    p = torch.full((3,), 0.6)
    for _ in range(200):
        tok, _ = sampler(logits.clone(), {}, None, p)
        assert int(tok.max()) <= 1
    assert sampler._sampling_pool is None


def _candidates(logits, cand=256):
    vals, idx = torch.topk(logits, cand, dim=-1, sorted=True)
    return vals, idx, torch.logsumexp(logits, dim=-1)


def test_candidate_kept_set_matches_upstream():
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    from vllm_tt_plugin.host_sampler import _candidate_top_k_top_p

    g = torch.Generator().manual_seed(1)
    rows, vocab = 16, 32000
    seen_exact = seen_inexact = 0
    for scale, temp in ((8.0, 0.3), (6.0, 0.6), (3.0, 0.6), (1.0, 1.0), (0.2, 1.0)):
        logits = torch.randn(rows, vocab, generator=g) * scale / temp
        p = torch.rand(rows, generator=g) * 0.95 + 0.04
        mixed = torch.randint(1, 300, (rows,), generator=g)
        mixed[::3] = vocab  # top-k off on these rows
        for k in (None, torch.randint(1, 257, (rows,), generator=g), mixed):
            ref = torch.isfinite(
                apply_top_k_top_p_pytorch(logits.clone(), k, p.clone(), True)
            )
            vals, idx, lse = _candidates(logits)
            _, _, keep, exact = _candidate_top_k_top_p(vals, lse, k, p, vocab)
            got = torch.zeros_like(ref).scatter_(1, idx, keep)
            assert torch.equal(got[exact], ref[exact]), (scale, temp, k is None)
            if k is not None:
                on = (k < vocab) & (k <= 256)
                assert bool(exact[on].all())
            seen_exact += int(exact.sum())
            seen_inexact += int((~exact).sum())
    assert seen_exact and seen_inexact


def test_candidate_sampling_matches_distribution():
    from vllm_tt_plugin.host_sampler import (
        _candidate_top_k_top_p,
        sample_from_candidates,
    )

    g = torch.Generator().manual_seed(2)
    vocab = 20000
    logits = torch.randn(4, vocab, generator=g) * 4.0
    # A few dozen tokens carry the kept mass.
    logits[:, :12] += torch.linspace(9.0, 6.0, 12)
    k = torch.tensor([vocab, 8, vocab, 50])
    p = torch.tensor([0.9, 0.95, 0.5, 0.8])
    vals, idx, lse = _candidates(logits)
    _, probs, keep, exact = _candidate_top_k_top_p(vals, lse, k, p, vocab)
    assert bool(exact.all())
    want = torch.zeros(4, vocab).scatter_(1, idx, probs * keep)
    want /= want.sum(dim=1, keepdim=True)
    torch.manual_seed(5)
    draws = 40000
    counts = torch.zeros(4, vocab)
    for _ in range(draws // 1000):
        rep = lambda t: t.repeat_interleave(1000, dim=0)
        tok, ex = sample_from_candidates(
            rep(vals), rep(idx), rep(lse), rep(k), rep(p), vocab
        )
        assert bool(ex.all())
        counts.index_put_(
            (torch.arange(4).repeat_interleave(1000), tok),
            torch.ones(4000),
            accumulate=True,
        )
    assert not bool((counts > 0)[want == 0].any())  # never a dropped token
    tv = 0.5 * (counts / draws - want).abs().sum(dim=1)
    assert bool((tv < 0.03).all()), tv


def test_candidate_fallback_rows_use_full_vocabulary_path():
    """Flat rows whose top 256 tokens hold under p use `fast_apply_top_k_top_p`."""
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    from vllm_tt_plugin import host_sampler

    g = torch.Generator().manual_seed(3)
    vocab = 8192
    logits = torch.randn(6, vocab, generator=g) * 0.1  # flat
    logits[::2] *= 60.0  # peaked: decided by the candidates
    p = torch.full((6,), 0.9)
    vals, idx, lse = _candidates(logits)
    _, exact = host_sampler.sample_from_candidates(vals, idx, lse, None, p, vocab)
    assert exact.tolist() == [True, False] * 3

    calls = []
    real = host_sampler.fast_apply_top_k_top_p

    def counting(x, k, p):
        calls.append(x.shape[0])
        return real(x, k, p)

    host_sampler.fast_apply_top_k_top_p = counting
    try:
        sampler = TTTopKTopPSampler()
        ref = torch.isfinite(
            apply_top_k_top_p_pytorch(logits.clone(), None, p.clone(), True)
        )
        for _ in range(50):
            tok, scores = sampler(logits.clone(), {}, None, p)
            assert scores is None
            assert bool(ref.gather(1, tok.unsqueeze(1)).all())
    finally:
        host_sampler.fast_apply_top_k_top_p = real
    assert calls == [3] * 50

    # The fallback rows' frequencies follow the renormalised kept distribution.
    rows = torch.tensor([1, 3, 5])
    want = torch.softmax(logits[rows].masked_fill(~ref[rows], -float("inf")), dim=-1)
    torch.manual_seed(4)
    draws = 20000
    counts = torch.zeros(3, vocab)
    for _ in range(draws // 500):
        tok, _ = sampler(
            logits[rows].repeat_interleave(500, dim=0),
            {},
            None,
            torch.full((1500,), 0.9),
        )
        counts.index_put_(
            (torch.arange(3).repeat_interleave(500), tok),
            torch.ones(1500),
            accumulate=True,
        )
    assert not bool((counts > 0)[want == 0].any())
    # ~7400 kept tokens per row and 20000 draws: compare in 64 equal-probability buckets
    # instead.
    order = want.argsort(dim=1)
    cw = want.gather(1, order).cumsum(1)
    bucket = (cw * 64).clamp(max=63.999).long()
    fb = torch.zeros(3, 64).scatter_add_(1, bucket, counts.gather(1, order) / draws)
    wb = torch.zeros(3, 64).scatter_add_(1, bucket, want.gather(1, order))
    assert bool((0.5 * (fb - wb).abs().sum(dim=1) < 0.04).all())


def test_candidate_path_only_for_unseeded_raw_modes():
    from vllm_tt_plugin import host_sampler

    logits = torch.randn(4, 8192) * 5
    p = torch.full((4,), 0.9)
    calls = []
    real = host_sampler.sample_from_candidates
    host_sampler.sample_from_candidates = lambda *a: calls.append(1) or real(*a)
    try:
        for mode in (
            "raw_logprobs",
            "raw_logits",
            "processed_logits",
            "processed_logprobs",
        ):
            calls.clear()
            tok, scores = TTTopKTopPSampler(mode)(logits.clone(), {}, None, p)
            assert (calls == [1]) == mode.startswith("raw"), mode
            assert (scores is None) == mode.startswith("raw"), mode
        calls.clear()
        TTTopKTopPSampler()(
            logits.clone(), {}, None, None
        )  # top-k/no top-p: upstream path
        TTTopKTopPSampler()(
            logits.clone(), {0: torch.Generator().manual_seed(0)}, None, p
        )
        assert calls == []
    finally:
        host_sampler.sample_from_candidates = real


def test_fast_top_k_top_p_matches_upstream():
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    from vllm_tt_plugin.host_sampler import fast_apply_top_k_top_p

    g = torch.Generator().manual_seed(0)
    vocab = 32000
    for scale, temp in (
        (3.0, 0.6),
        (1.0, 1.0),
        (0.2, 1.0),
        (8.0, 0.3),
    ):  # sharp ... very flat
        logits = torch.randn(16, vocab, generator=g) * scale / temp
        p = torch.rand(16, generator=g) * 0.95 + 0.04
        for k in (None, torch.randint(1, 300, (16,), generator=g)):
            ref = apply_top_k_top_p_pytorch(
                logits.clone(), k, p.clone(), allow_cpu_sync=True
            )
            got = fast_apply_top_k_top_p(logits.clone(), k, p.clone())
            assert torch.equal(torch.isfinite(ref), torch.isfinite(got)), (
                scale,
                temp,
                k is None,
            )
            kept = torch.isfinite(ref)
            assert torch.equal(ref[kept], got[kept])
