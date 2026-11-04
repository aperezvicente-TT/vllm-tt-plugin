# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Native CPU sampling with bounded, speculative noise for seeded requests."""

import os
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

import torch
from vllm.v1.sample.ops.topk_topp_sampler import (
    TopKTopPSampler,
    apply_top_k_top_p,
    random_sample,
    sample_with_exponential_noise,
)
from vllm.v1.sample.sampler import Sampler

# Candidate counts tried per row before falling back to a full-vocabulary sort.
_TOP_P_CANDIDATES = (256, 4096)


def _candidate_top_k_top_p(vals, lse, k, p, vocab):
    """Top-k then top-p over each row's top K tokens, `vals` [B, K] sorted descending.

    `lse` [B] is the logsumexp of the whole row and `vocab` its length (k >= vocab is no
    top-k). Returns the top-k-masked vals, their probabilities, the kept-token mask (a
    prefix of each row) and the rows it is exact for: top-k with k <= K, or no top-k and
    a kept prefix ending inside K.
    """
    batch, cand = vals.shape
    if k is not None:
        k_rows = k.to(torch.long).clamp(max=vocab)
        pos = torch.arange(cand).unsqueeze(0)
        vals = vals.masked_fill(pos >= k_rows.unsqueeze(1), -float("inf"))
        topk_active = k_rows < vocab
        topk_exact = topk_active & (k_rows <= cand)
    else:
        topk_active = topk_exact = torch.zeros(batch, dtype=torch.bool)
    # Normaliser: the top-k set itself where top-k applies, the whole row otherwise.
    norm = torch.where(topk_active, torch.logsumexp(vals.float(), dim=-1), lse.float())
    probs = torch.exp(vals.float() - norm.unsqueeze(1))
    keep = vals != -float("inf")
    if p is None:
        return vals, probs, keep, topk_exact
    # Mass ranked strictly above each token.
    before = torch.cumsum(probs, dim=-1) - probs
    pp = p.to(probs.dtype).unsqueeze(1)
    keep &= before < pp
    keep[:, 0] = True
    exact = topk_exact | (~topk_active & (before[:, -1] + probs[:, -1] >= pp[:, 0]))
    return vals, probs, keep, exact


def _top_k_top_p_from_candidates(logits, k, p, cand):
    """Kept-token mask over each row's top `cand` tokens; the rows it is exact for."""
    vals, idx = torch.topk(logits, cand, dim=-1, sorted=True)
    lse = torch.logsumexp(logits.float(), dim=-1)
    vals, _, keep, exact = _candidate_top_k_top_p(vals, lse, k, p, logits.shape[1])
    return vals, idx, keep, exact


def _top_p_keep_by_histogram(logits, p, buckets=2048):
    """Exact top-p kept-token mask (no top-k) without sorting the vocabulary.

    Logits are binned by value; the mass above each bin is a prefix sum over bins. The
    first bin whose above-mass is below p is the only bin straddling the cutoff: higher
    bins are kept, lower bins dropped, and only that bin's tokens are sorted to place
    the cutoff exactly.
    """
    rows, vocab = logits.shape
    lg = logits.float()
    lse = torch.logsumexp(lg, dim=-1, keepdim=True)
    probs = torch.exp(lg - lse)
    lo = lg.amin(dim=-1, keepdim=True)
    hi = lg.amax(dim=-1, keepdim=True)
    scale = (buckets - 1) / (hi - lo).clamp_min(1e-30)
    bin_of = ((lg - lo) * scale).to(torch.long).clamp_(0, buckets - 1)
    mass = torch.zeros(rows, buckets, dtype=probs.dtype).scatter_add_(1, bin_of, probs)
    above = mass.flip(1).cumsum(1).flip(1) - mass  # mass in strictly higher bins
    pp = p.to(probs.dtype).unsqueeze(1)
    # The first bin below p is the cutoff bin.
    cut = (above < pp).to(torch.int8).argmax(dim=1, keepdim=True)
    keep = bin_of > cut
    in_cut = bin_of == cut
    width = int(in_cut.sum(dim=1).max().item())
    cand, cidx = torch.topk(
        lg.masked_fill(~in_cut, -float("inf")), width, dim=-1, sorted=True
    )
    cprob = torch.exp(cand - lse)
    before = above.gather(1, cut) + torch.cumsum(cprob, dim=-1) - cprob
    cand_keep = (before < pp) & (cand != -float("inf"))
    # Entries past a row's bin size are -inf picks of arbitrary indices: OR keeps
    # their state.
    keep.scatter_(1, cidx, keep.gather(1, cidx) | cand_keep)
    keep.scatter_(1, lg.argmax(dim=-1, keepdim=True), True)  # at least one token
    return keep


def fast_apply_top_k_top_p(logits, k, p):
    """`apply_top_k_top_p` without sorting the vocabulary when the answer is the same.

    vLLM's CPU path sorts every row of the vocabulary for top-p (most of a decode step
    at 128K vocab and 32 rows). Top-p keeps, in descending order, each token whose
    preceding probability mass is below p, so with exact probabilities (from the
    full-row logsumexp) the kept set lies inside the top K tokens whenever those hold at
    least p of the mass. Rows are tried with growing K; only rows still undecided use
    the upstream sort. Top-k is applied first and top-p sees the renormalised top-k
    distribution, as upstream. Returns full-vocabulary logits with dropped tokens at
    -inf, like `apply_top_k_top_p`.
    """
    if p is None or logits.device.type != "cpu" or logits.dim() != 2:
        return apply_top_k_top_p(logits, k, p)
    batch, vocab = logits.shape
    out = torch.full_like(logits, -float("inf"))
    rows = torch.arange(batch)
    for cand in _TOP_P_CANDIDATES:
        if k is not None:
            cand = max(cand, int(k[rows].max().item()))
        if cand * 4 >= vocab:
            break
        sub_k = None if k is None else k[rows]
        vals, idx, keep, exact = _top_k_top_p_from_candidates(
            logits[rows], sub_k, p[rows], cand
        )
        done = rows[exact]
        if len(done):
            kept = torch.where(
                keep[exact], vals[exact], torch.full_like(vals[exact], -float("inf"))
            )
            out[done] = out[done].scatter_(1, idx[exact], kept)
        rows = rows[~exact]
        if not len(rows):
            return out
    if k is None or bool((k[rows] >= logits.shape[1]).all()):
        sub = logits[rows]
        keep = _top_p_keep_by_histogram(sub, p[rows])
        out[rows] = sub.masked_fill(~keep, -float("inf"))
        return out
    out[rows] = apply_top_k_top_p(
        logits[rows].clone(), None if k is None else k[rows], p[rows]
    )
    return out


def sample_unseeded(probs):
    """One token per row from `probs` by inverse-CDF sampling (rows without a seed).

    Same distribution as vLLM's exponential-noise sampling, which exists to avoid a
    device sync and draws one random number per vocabulary entry; on the CPU that draw
    (like `torch.multinomial`) dominates a decode step at large vocabularies. Here: one
    uniform draw per row and a binary search in the row's cumulative sum. `right=True`
    never selects a zero-probability token.
    """
    cdf = probs.cumsum(dim=-1)
    u = torch.rand(probs.shape[0], 1, dtype=cdf.dtype) * cdf[:, -1:]
    return (
        torch.searchsorted(cdf, u, right=True).clamp_(max=probs.shape[1] - 1).view(-1)
    )


def sample_from_candidates(vals, idx, lse, k, p, vocab):
    """One token per row drawn from its top-K candidates; the rows where it is exact.

    `vals` [B, K] are temperature-scaled logits sorted descending and `idx` [B, K] their
    vocabulary ids; `lse`, `k`, `p` and `vocab` as in `_candidate_top_k_top_p`. Tokens
    of rows that are not exact are -1. The full logits are never read, so the candidates
    may come from the device.
    """
    _, probs, keep, exact = _candidate_top_k_top_p(vals, lse, k, p, vocab)
    # Bounding the index by the kept prefix length guards against u * total rounding up
    # to total, which would otherwise land on a dropped candidate.
    cdf = probs.masked_fill(~keep, 0.0).cumsum(dim=-1)
    u = torch.rand(vals.shape[0], 1, dtype=cdf.dtype) * cdf[:, -1:]
    pos = torch.searchsorted(cdf, u, right=True)
    pos = torch.minimum(pos, keep.sum(dim=1, keepdim=True) - 1)
    return idx.gather(1, pos).view(-1).masked_fill(~exact, -1), exact


@dataclass
class _PrefetchedNoise:
    state: torch.Tensor
    vocab: int
    size_bytes: int
    future: Future


class TTTopKTopPSampler(TopKTopPSampler):
    def __init__(self, logprobs_mode="raw_logprobs", use_fp64_gumbel=False):
        super().__init__(logprobs_mode, use_fp64_gumbel)
        self.forward = self.forward_native
        self._sampling_pool = None
        self._noise_cache = OrderedDict()
        self._cached_bytes = 0
        self._max_prefetch_bytes = 64 * 1024 * 1024
        self._sampling_lock = Lock()

    def _prepare_noise(self, state, vocab):
        # Workers only advance private clones. Preparing or discarding noise
        # must not consume a live request's RNG, including after cancellation.
        clone = torch.Generator(device="cpu").set_state(state)
        dtype = torch.float64 if self.use_fp64_gumbel else torch.float32

        def generate():
            noise = torch.empty(vocab, dtype=dtype)
            noise.exponential_(generator=clone)
            return noise, clone.get_state()

        size = vocab * (8 if self.use_fp64_gumbel else 4)
        return _PrefetchedNoise(
            state, vocab, size, self._sampling_pool.submit(generate)
        )

    def forward_native(self, logits, generators, k, p):
        batch, vocab = logits.shape
        if (
            logits.device.type != "cpu"
            or batch == 0
            or vocab < 4096
            or set(generators) != set(range(batch))
            or len({id(gen) for gen in generators.values()}) != batch
            or any(gen.device.type != "cpu" for gen in generators.values())
        ):
            return self._forward_unseeded(logits, generators, k, p)

        # A request generator may move to a different batch row. Cache by the
        # generator itself, and verify its complete state before reusing noise.
        with self._sampling_lock:
            return self._sample_seeded(logits, generators, k, p)

    def _forward_unseeded(self, logits, generators, k, p):
        """`forward_native` minus two full-vocabulary costs that change no result.

        The top-p sort becomes exact candidate selection, and per-entry sampling noise
        becomes one draw per row for rows without a seed.
        """
        if logits.device.type != "cpu":
            return super().forward_native(logits, generators, k, p)
        if (
            p is not None
            and not generators
            and self.logprobs_mode in ("raw_logprobs", "raw_logits")
            and _TOP_P_CANDIDATES[0] * 4 < logits.shape[1]
        ):
            return self._sample_candidates(logits, k, p), None
        logits = (
            fast_apply_top_k_top_p(logits, k, p)
            if p is not None
            else apply_top_k_top_p(logits, k, p)
        )
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        if generators:
            # Some rows carry a seed: keep upstream's noise stream for all rows.
            return random_sample(
                probs, generators, self.use_fp64_gumbel
            ), logits_to_return
        return sample_unseeded(probs), logits_to_return

    @staticmethod
    def _sample_candidates(logits, k, p):
        """Unseeded top-p sampling that needs no processed logits.

        The full vocabulary is read only by topk and logsumexp. Rows the candidates
        cannot decide take the full-vocabulary path.
        """
        vals, idx = torch.topk(logits, _TOP_P_CANDIDATES[0], dim=-1, sorted=True)
        lse = torch.logsumexp(logits.float(), dim=-1)
        tokens, exact = sample_from_candidates(vals, idx, lse, k, p, logits.shape[1])
        if not bool(exact.all()):
            rows = (~exact).nonzero().view(-1)
            sub = fast_apply_top_k_top_p(
                logits[rows], None if k is None else k[rows], p[rows]
            )
            tokens[rows] = sample_unseeded(sub.softmax(dim=-1, dtype=torch.float32))
        return tokens

    def _sample_seeded(self, logits, generators, k, p):
        batch, vocab = logits.shape
        if self._sampling_pool is None:
            available = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else (os.cpu_count() or 1)
            )
            self._sampling_pool = ThreadPoolExecutor(
                max_workers=min(32, available), thread_name_prefix="tt-sampling"
            )

        prepared = []
        for row in range(batch):
            generator = generators[row]
            state = generator.get_state()
            entry = self._noise_cache.pop(generator, None)
            if entry is not None:
                self._cached_bytes -= entry.size_bytes
            if (
                entry is None
                or entry.vocab != vocab
                or not torch.equal(entry.state, state)
            ):
                if entry is not None:
                    entry.future.cancel()
                entry = self._prepare_noise(state, vocab)
            prepared.append(entry)

        # Resolve generation before submitting sampling tasks to the same pool;
        # a worker must never wait on work queued behind other waiting workers.
        noise_and_states = [entry.future.result() for entry in prepared]

        def sample_row(row):
            values = apply_top_k_top_p(
                logits[row : row + 1],
                k[row : row + 1] if k is not None else None,
                p[row : row + 1] if p is not None else None,
            )
            scores = None
            if self.logprobs_mode == "processed_logits":
                scores = values
            elif self.logprobs_mode == "processed_logprobs":
                scores = values.log_softmax(dim=-1, dtype=torch.float32)
            probs = values.softmax(dim=-1, dtype=torch.float32)
            noise, next_state = noise_and_states[row]
            generators[row].set_state(next_state)
            token = sample_with_exponential_noise(probs, noise.unsqueeze(0))
            return token, scores

        # Each Torch operation may also use intra-op threads. Group rows into
        # a few tasks rather than dispatching one competing Torch job per row.
        if batch == 1:
            results = [sample_row(0)]
        else:
            available = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else (os.cpu_count() or 1)
            )
            workers = max(1, min(4, available // torch.get_num_threads(), batch))
            chunk_size = (batch + workers - 1) // workers

            def sample_chunk(start):
                return [
                    sample_row(row)
                    for row in range(start, min(start + chunk_size, batch))
                ]

            results = [
                result
                for chunk in self._sampling_pool.map(
                    sample_chunk, range(0, batch, chunk_size)
                )
                for result in chunk
            ]

        # This work can overlap the next TT forward pass. The cache is bounded
        # by noise bytes, including pending futures; evicted clones are harmless.
        noise_bytes = vocab * (8 if self.use_fp64_gumbel else 4)
        prefetch_count = min(batch, self._max_prefetch_bytes // noise_bytes)
        for row in range(batch - prefetch_count, batch):
            generator = generators[row]
            entry = self._prepare_noise(noise_and_states[row][1], vocab)
            self._noise_cache[generator] = entry
            self._cached_bytes += entry.size_bytes
            while self._cached_bytes > self._max_prefetch_bytes:
                _, old = self._noise_cache.popitem(last=False)
                self._cached_bytes -= old.size_bytes
                old.future.cancel()

        tokens = torch.cat([result[0] for result in results])
        scores = (
            torch.cat([result[1] for result in results])
            if results[0][1] is not None
            else None
        )
        return tokens, scores


class TTHostSampler(Sampler):
    @staticmethod
    def apply_temperature(logits, temp, all_random):
        # CPU metadata needs no synchronization. Avoid touching the entire
        # vocabulary when division would leave every value unchanged.
        if (
            all_random
            and logits.device.type == "cpu"
            and temp.device.type == "cpu"
            and torch.all(temp == 1.0).item()
        ):
            return logits
        return Sampler.apply_temperature(logits, temp, all_random)


def create_host_sampler():
    sampler = TTHostSampler()
    sampler.topk_topp_sampler = TTTopKTopPSampler(
        sampler.logprobs_mode, sampler.use_fp64_gumbel
    )
    return sampler
