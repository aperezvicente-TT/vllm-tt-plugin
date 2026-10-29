"""GLM53_PREEMPT_SAVE (program 2026-10-27T00:05:27Z (2); doc/ttnn/12-fast-start/preempt/design.md): no recompute
preemption. The scheduler records a preempted request's computed KV before vLLM frees its blocks. The worker's
begin_step stores [0, n) through LMCache (a save-only request over the recorded blocks) plus an exact KDA checkpoint at
n. On resume the hit covers the request's whole history up to n. Host only: fakes for the inner connector and the KDA
store; the real TTScheduler for the hook."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from vllm_tt_plugin import kv_connector as K


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(K, "_PREEMPT_SAVES", {})
    monkeypatch.setattr(K, "_PREEMPT_RESUME", {})
    monkeypatch.setattr(K, "_KDA_PROVENANCE", {})
    monkeypatch.delenv("GLM53_RESUME_AS_DECODE", raising=False)
    monkeypatch.setenv("GLM53_PREEMPT_SAVE", "1")
    monkeypatch.delenv("GLM53_PREEMPT_SAVE_MAX_TOKENS", raising=False)


def test_record_and_bounds(monkeypatch):
    assert K.record_preempt_save("a", 130, ([1, 2, 3],), list(range(200)))
    rec = K._PREEMPT_SAVES["a"]
    assert rec["n"] == 130 and rec["block_ids"] == [[1, 2, 3]] and rec["token_ids"] == list(range(130))
    monkeypatch.setenv("GLM53_PREEMPT_SAVE_MAX_TOKENS", "100")
    assert not K.record_preempt_save("b", 130, ([1, 2, 3],), list(range(200)))
    assert K._PREEMPT_SAVES["__over__"] == ["b"]  # counted, falls back to recompute
    assert not K.record_preempt_save("s", 64, ([1],), list(range(60)))  # 14 B2: a history shorter than n
    assert K._PREEMPT_SAVES["__short__"] == ["s"]
    assert K._PREEMPT_RESUME == {"a": None}  # recorded: its lookup waits for the worker's save
    monkeypatch.setenv("GLM53_PREEMPT_SAVE", "0")
    assert not K.record_preempt_save("c", 10, ([1],), list(range(10)))


def test_a_resumed_request_hits_its_history_not_only_its_prompt(monkeypatch):
    fresh = SimpleNamespace(prompt_token_ids=[1, 2, 3], all_token_ids=[1, 2, 3], num_preemptions=0)
    resumed = SimpleNamespace(prompt_token_ids=[1, 2, 3], all_token_ids=[1, 2, 3, 9, 8], num_preemptions=1)
    assert K._hit_tokens(fresh) == [1, 2, 3] and K._hit_tokens(resumed) == [1, 2, 3, 9, 8]
    monkeypatch.setenv("GLM53_PREEMPT_SAVE", "0")
    assert K._hit_tokens(resumed) == [1, 2, 3]  # off: unchanged


class _Store:
    def __init__(self, exact=()):
        self.exact, self.saved = set(exact), []

    def has(self, tokens, n):
        return n in self.exact

    def longest_checkpointed(self, tokens, aligned, block):
        return (aligned // 128) * 128

    def save(self, tokens, n, states):
        self.saved.append((list(tokens), n, states))
        self.exact.add(n)


def test_an_exact_checkpoint_is_the_resume_point(monkeypatch):
    c = K.TTPageAlignedConnector.__new__(K.TTPageAlignedConnector)
    c._inner, c._block_size, c._kda_store = SimpleNamespace(), 64, _Store(exact={300, 320})
    toks = list(range(400))
    assert c.aligned_hit(toks, 300, exact=True) == 256  # off the grid: the model cannot continue there (9 (h))
    assert c.aligned_hit(toks, 320, exact=True) == 320  # on the page grid: an exact checkpoint is usable as is
    monkeypatch.setenv("GLM53_RESUME_AS_DECODE", "1")
    assert c.aligned_hit(toks, 300, exact=True) == 300  # resume-as-decode: not rounded to the page / checkpoint grid
    assert c.__dict__["preempt_exact_hits"] == 2
    assert c.aligned_hit(toks, 300) == 256  # a fresh request: the grid, as before
    assert c.aligned_hit(toks, 290, exact=True) == 256  # no exact checkpoint there: the grid


class _Inner:
    def __init__(self):
        self.bound, self.layers, self.waits = [], [], 0

    def bind_connector_metadata(self, meta):
        self.bound.append(meta)

    def clear_connector_metadata(self):
        pass

    def save_kv_layer(self, name, cache, attn):
        self.layers.append(name)

    def wait_for_save(self):
        self.waits += 1


def _host(slot=5):
    host = K.TTKVConnectorHost.__new__(K.TTKVConnectorHost)
    host.connector, host._lock = _Inner(), threading.Lock()
    host.views = {"L0": SimpleNamespace(block_size=64), "L1": SimpleNamespace(block_size=64)}
    host.connector_caches = {"L0": object(), "L1": object()}
    host._step_context = lambda mi: (None, {"L0": None, "L1": None})
    host._metadata = lambda so: SimpleNamespace(requests=["the step's own"], other=1)
    host.kda_store, host._kv_cache_object = _Store(), object()
    host._kda_export = lambda kv, s: ("state of slot", s)
    host._boundary_export = lambda kv, s, n_tokens, candidate_row: ("candidate row", s, n_tokens, candidate_row)
    host.spec_k = 3
    host._slot_of = lambda rid: slot
    return host


def test_begin_step_stores_the_recorded_range_and_an_exact_checkpoint(monkeypatch):
    monkeypatch.setenv("GLM53_RESUME_AS_DECODE", "1")
    host = _host()
    K.record_preempt_save("r", 130, ([4, 7, 9],), list(range(1000, 1200)))
    host._preempt_store(SimpleNamespace(preempted_req_ids={"r"}))
    (meta,) = host.connector.bound
    (rm,) = meta.requests  # a save-only request; the step's own metadata object is untouched
    assert rm.req_id == "r" and rm.token_ids == list(range(1000, 1130)) and rm.save_spec.can_save
    assert rm.save_spec.skip_leading_tokens == 0 and rm.is_last_prefill  # the unfull tail is stored too
    want = [b * 64 + i for b in (4, 7, 9) for i in range(64)][:130]
    assert rm.slot_mapping.tolist() == want
    assert host.connector.layers == ["L0", "L1"] and host.connector.waits == 1
    assert host.kda_store.saved == [(list(range(1000, 1130)), 130, ("state of slot", 5))]
    assert host.preempt_saves == 1 and K._PREEMPT_SAVES == {} and K._PREEMPT_RESUME == {"r": 130}


def test_a_request_not_preempted_this_step_is_kept_then_dropped():
    host = _host()
    K.record_preempt_save("x", 64, ([1],), list(range(64)))
    for _ in range(3):
        host._preempt_store(SimpleNamespace(preempted_req_ids=set()))
        assert host.connector.bound == []
    host._preempt_store(SimpleNamespace(preempted_req_ids=set()))
    assert "x" not in K._PREEMPT_SAVES  # aged out (its step never reached a worker)


def test_a_failed_save_falls_back_to_recompute_without_raising():
    host = _host(slot=None)  # no slot known: the checkpoint cannot be exported
    K.record_preempt_save("r", 64, ([1],), list(range(64)))
    host._preempt_store(SimpleNamespace(preempted_req_ids={"r"}))
    assert host.preempt_saves_failed == 1


def test_the_provenance_rule_skips_the_checkpoint(monkeypatch):
    host = _host()
    K._KDA_PROVENANCE["r"] = 64
    K.record_preempt_save("r", 64, ([1],), list(range(64)))
    host._preempt_store(SimpleNamespace(preempted_req_ids={"r"}))
    assert host.connector.waits == 1 and host.kda_store.saved == [] and host.preempt_ckpt_skipped_provenance == 1


def test_the_scheduler_records_before_vllm_frees_the_blocks(monkeypatch):
    from tests.test_mixed_steps import _commit, _request, _scheduler

    scheduler = _scheduler(mixed=False)
    scheduler.add_request(_request("p", 200, max_tokens=8))
    _commit(scheduler, scheduler.schedule())
    req = scheduler.requests["p"]
    n, blocks = int(req.num_computed_tokens), scheduler.kv_cache_manager.get_block_ids("p")
    assert n > 0 and blocks and blocks[0]
    scheduler.running.remove(req)  # (vLLM pops it from running before _preempt_request)
    monkeypatch.setattr(scheduler, "connector", object())  # (a connector is configured: the hook records)
    scheduler._preempt_request(req, 0.0)
    rec = K._PREEMPT_SAVES["p"]
    assert rec["n"] == n and rec["block_ids"][0] == list(blocks[0]) and rec["token_ids"] == list(req.all_token_ids[:n])
    assert req.num_computed_tokens == 0 and scheduler.preempt_saves_recorded == 1
    torch.manual_seed(0)


def test_without_resume_as_decode_the_save_snaps_to_the_page_grid():
    """9 / 14 (00:53Z): without the model's resume-as-decode, an off-grid continuation is refused, so the save stores
    KV [0, floor64(n)) only; the slot holds the state after n, so no checkpoint is exported (an earlier decode-boundary
    export at the snapped point serves the resume, else the hit walks down: counted)"""
    host = _host()
    K.record_preempt_save("r", 130, ([4, 7, 9],), list(range(1000, 1200)))
    host._preempt_store(SimpleNamespace(preempted_req_ids={"r"}))
    (meta,) = host.connector.bound
    (rm,) = meta.requests
    assert rm.token_ids == list(range(1000, 1128)) and len(rm.slot_mapping) == 128
    assert host.kda_store.saved == [] and host.preempt_snap_no_ckpt == 1 and host.preempt_saves_snapped == 1
    assert K._PREEMPT_RESUME == {"r": 128}
    host2 = _host()
    host2.kda_store = _Store(exact={128})  # the decode-boundary export already took 128
    K.record_preempt_save("q", 130, ([4, 7, 9],), list(range(1000, 1200)))
    host2._preempt_store(SimpleNamespace(preempted_req_ids={"q"}))
    assert not getattr(host2, "preempt_snap_no_ckpt", 0) and K._PREEMPT_RESUME["q"] == 128


def test_a_verifying_slot_exports_from_its_candidate_row(monkeypatch):
    """14 B2: a slot whose last step verified drafts holds its committed state in candidate row n - c - 1 of the shared
    state, not in its own row: the save exports through export_spec_boundary_state"""
    monkeypatch.setenv("GLM53_RESUME_AS_DECODE", "1")
    host = _host()
    mi = SimpleNamespace(row_req_ids=["r"], input_positions=torch.tensor([[127, 128, 129, 130]]), num_valid_drafts=[3])
    host._note_decode_rows(mi)  # the step's load: verify rows from position 127
    K.record_preempt_save("r", 129, ([4, 7, 9],), list(range(1000, 1200)))
    host._preempt_store(SimpleNamespace(preempted_req_ids={"r"}))
    assert host.kda_store.saved[0][2] == ("candidate row", 5, 129, 1) and host.preempt_ckpts_spec == 1
    # a plain decode row: the slot's own row, as before
    host = _host()
    host._note_decode_rows(SimpleNamespace(row_req_ids=["p"], input_positions=torch.tensor([[128]])))
    K.record_preempt_save("p", 129, ([4, 7, 9],), list(range(1000, 1200)))
    host._preempt_store(SimpleNamespace(preempted_req_ids={"p"}))
    assert host.kda_store.saved[0][2] == ("state of slot", 5)
    host = _host()  # its own row is not the state after n: refused, recompute
    host._note_decode_rows(SimpleNamespace(row_req_ids=["q"], input_positions=torch.tensor([[100]])))
    K.record_preempt_save("q", 129, ([4, 7, 9],), list(range(1000, 1200)))
    host._preempt_store(SimpleNamespace(preempted_req_ids={"q"}))
    assert host.preempt_saves_failed == 1 and host.kda_store.saved == []
    # a candidate row outside 0..K: refused, recompute
    host = _host()
    host._note_decode_rows(SimpleNamespace(row_req_ids=["x"], input_positions=mi.input_positions, num_valid_drafts=[3]))
    K.record_preempt_save("x", 140, ([4, 7, 9],), list(range(1000, 1200)))
    host._preempt_store(SimpleNamespace(preempted_req_ids={"x"}))
    assert host.preempt_saves_failed == 1 and "x" not in K._PREEMPT_RESUME


def test_the_scheduler_clamps_n_to_the_committed_history(monkeypatch):
    """14 B2: under async scheduling num_computed_tokens counts the in-flight drafts; the record never passes the
    request's committed history"""
    from tests.test_mixed_steps import _commit, _request, _scheduler

    scheduler = _scheduler(mixed=False)
    scheduler.add_request(_request("p", 200, max_tokens=8))
    _commit(scheduler, scheduler.schedule())
    req = scheduler.requests["p"]
    req.num_computed_tokens = len(req.all_token_ids) + 3  # (3 drafts of the in-flight step)
    scheduler.running.remove(req)
    monkeypatch.setattr(scheduler, "connector", object())
    scheduler._preempt_request(req, 0.0)
    assert K._PREEMPT_SAVES["p"]["n"] == len(req.all_token_ids)


class _ChunkInner:
    """LMCache's lookup as served, over a REAL ChunkedTokenDatabase (chunk 128, unfull chunks kept): the longest
    prefix of stored chunk keys of request.all_token_ids; a full-history hit gives back its last token (the adapter's
    rule); is_async False (LMCacheConnectorV1)"""

    def __init__(self):
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.token_database import ChunkedTokenDatabase

        self.db = ChunkedTokenDatabase(LMCacheEngineConfig.from_defaults(chunk_size=128, save_unfull_chunk=True), None)
        self.stored = set()

    def store(self, tokens):
        self.stored |= {h for _s, _e, h in self.db.process_tokens(tokens=tokens, make_key=False)}

    def get_num_new_matched_tokens(self, request, num_computed):
        hit = 0
        for _s, e, h in self.db.process_tokens(tokens=request.all_token_ids, make_key=False):
            if h not in self.stored:
                break
            hit = e
        need = hit - num_computed - (1 if hit == request.num_tokens else 0)
        return max(need, 0), False


def _resumed(hist, prompt):
    return SimpleNamespace(request_id="r", prompt_token_ids=hist[:prompt], all_token_ids=hist, num_preemptions=1,
                           num_tokens=len(hist), num_prompt_tokens=prompt, sampling_params=None, kv_transfer_params=None)


@pytest.mark.parametrize("as_decode", [True, False])
def test_the_resume_lookup_reaches_the_save_through_lmcaches_chunk_keys(monkeypatch, as_decode):
    """14 B1 / 9 (e), over LMCache's chunk keys: the save stored [0, n); the resumed history is n + 1 tokens (the last
    sampled token never computed), whose last chunk key [896, 1001) misses the stored unfull chunk [896, 1000) -- the
    old lookup hit 896. Cut to the resume point the lookup hits it. With resume-as-decode n = 1000 (off the grid, the
    1-token row decodes); without it the save snapped to 960 and the continuation starts on the grid (the model's plan
    check refuses an off-grid start, as without the flag, and is never asked to plan one)."""
    from vllm_tt_plugin import plan_check

    if as_decode:
        monkeypatch.setenv("GLM53_RESUME_AS_DECODE", "1")
    refused = []

    def model_plan(start, n):  # the model without the flag: continuations start on the 64-token page grid
        if start % 64 and not as_decode:
            refused.append(start)
            return f"start {start} off the page grid"
        return None

    monkeypatch.setattr(plan_check, "_CHECK", [model_plan])
    hist = list(range(5000, 6001))  # n + 1 = 1,001
    inner = _ChunkInner()
    saved = 1000 if as_decode else 960
    c = K.TTPageAlignedConnector(inner, 64, _Store(exact={saved}))
    K.record_preempt_save("r", 1000, ([1] * 16,), hist)
    assert c.get_num_new_matched_tokens(_resumed(hist, 600), 0) == (None, False)  # its save has not run yet
    inner.store(hist[:saved])  # the worker's save (LMCache store [0, saved) + the checkpoint) ...
    K._PREEMPT_RESUME["r"] = saved  # ... and its resume point
    assert inner.get_num_new_matched_tokens(_resumed(hist, 600), 0)[0] == 896  # the old lookup: the whole history
    assert c.get_num_new_matched_tokens(_resumed(hist, 600), 0) == (saved, False)
    assert c.__dict__["preempt_exact_hits"] == 1 and refused == [] and c.__dict__["preempt_lookup_waits"] == 1


# --- Form B v2 (P / D over Mooncake): D waits for P's stores (kv_transfer_params "tt.expect_hit")


class _Cleared:
    def __init__(self):
        self.cleared = []

    def clear_lookup_status(self, rid):
        self.cleared.append(rid)


def _d_request(hist, expect):
    r = _resumed(hist, len(hist))  # D's prompt = P's prompt + token 1
    r.num_preemptions = 0
    r.kv_transfer_params = {"tt.expect_hit": expect}
    return r


def test_d_waits_for_ps_stores_then_hits_their_exact_end(monkeypatch):
    """D's lookup stays undetermined while the remote holds fewer than the expected n tokens (vLLM retries the next
    step; LMCache's cached lookup count is cleared each time), then hits n exactly over the history cut to n (D's
    prompt is P's prompt + token 1: the uncut lookup would stop at the last full chunk)"""
    from vllm_tt_plugin import plan_check

    monkeypatch.setenv("GLM53_RESUME_AS_DECODE", "1")
    monkeypatch.setattr(plan_check, "_CHECK", [])
    monkeypatch.setattr(K, "_EXPECT_T0", {})
    hist = list(range(7000, 8001))  # P's prompt 1,000 + token 1
    inner = _ChunkInner()
    inner._lmcache_engine = SimpleNamespace(lookup_client=_Cleared())
    c = K.TTPageAlignedConnector(inner, 64, _Store(exact={1000}))
    req = _d_request(hist, 1000)
    inner.store(hist[:512])  # P's stores are landing: half of them so far
    assert c.get_num_new_matched_tokens(req, 0) == (None, False)
    assert inner._lmcache_engine.lookup_client.cleared == ["r"] and c.__dict__["expect_waits"] == 1
    inner.store(hist[:1000])  # all of them, the unfull last chunk [896, 1000) included
    assert c.get_num_new_matched_tokens(req, 0) == (1000, False)
    assert c.__dict__["preempt_exact_hits"] == 1 and "r" not in K._EXPECT_T0


def test_d_takes_what_is_there_after_the_wait(monkeypatch):
    from vllm_tt_plugin import plan_check

    monkeypatch.setenv("GLM53_KV_EXPECT_WAIT_S", "0.2")
    monkeypatch.setattr(plan_check, "_CHECK", [])
    monkeypatch.setattr(K, "_EXPECT_T0", {})
    hist = list(range(7000, 8001))
    inner = _ChunkInner()
    c = K.TTPageAlignedConnector(inner, 64, _Store())
    req = _d_request(hist, 1000)
    inner.store(hist[:512])
    assert c.get_num_new_matched_tokens(req, 0) == (None, False)
    import time as _t

    _t.sleep(0.25)
    n, _a = c.get_num_new_matched_tokens(req, 0)
    assert n == 512 and c.__dict__["expect_timeouts"] == 1  # proceeds on the grid hit it has (counted)


def test_no_expectation_no_wait(monkeypatch):
    """a request without tt.expect_hit looks up as before (one call, no cut)"""
    from vllm_tt_plugin import plan_check

    monkeypatch.setattr(plan_check, "_CHECK", [])
    hist = list(range(7000, 8001))
    inner = _ChunkInner()
    c = K.TTPageAlignedConnector(inner, 64, _Store())
    r = _resumed(hist, len(hist))
    r.num_preemptions = 0
    inner.store(hist[:512])
    assert c.get_num_new_matched_tokens(r, 0) == (512, False)
    assert "expect_waits" not in c.__dict__


class _CkptStore(_Store):
    def can_save(self):
        return True


def _ckpt_host(rows):
    host = _host()
    host.kda_store = _CkptStore()
    host._prefill_rows = lambda mi: rows
    host._slot_for = lambda rid, row, mi: 3
    host._kda_export_async = None
    return host


def test_a_p_request_exports_its_exact_prompt_end(monkeypatch):
    """Form B v2, the P side: a P request (tt.pd_role P, under resume-as-decode) checkpoints its EXACT prompt end n
    (off the page grid) so D imports at n and decodes token 1; any other request keeps the page-grid rule"""
    from vllm_tt_plugin import plan_check

    monkeypatch.setenv("GLM53_RESUME_AS_DECODE", "1")
    monkeypatch.setenv("GLM53_KDA_CKPT_EVERY", "0")
    monkeypatch.setenv("GLM53_LMCACHE_MIN_STORE_TOKENS", "0")
    monkeypatch.setattr(K, "_PD_EXACT_END", {})
    monkeypatch.setattr(K, "_EXPECT_T0", {})
    monkeypatch.setattr(plan_check, "_CHECK", [])
    hist = list(range(9000, 10000))  # 1,000 tokens (off the 64 grid)
    p_req = _resumed(hist, 1000)
    p_req.num_preemptions, p_req.request_id = 0, "p"
    p_req.kv_transfer_params = {"tt.pd_role": "P"}
    inner = _ChunkInner()
    K.TTPageAlignedConnector(inner, 64, _Store()).get_num_new_matched_tokens(p_req, 0)  # (P's own lookup: a miss)
    assert K._PD_EXACT_END == {"p": 1000}
    mi = SimpleNamespace(input_tokens=torch.tensor([hist]))
    host = _ckpt_host([(0, "p", 512, 1000)])
    host._export_kda_checkpoints(mi)
    assert host.kda_store.saved == [(hist, 1000, ("state of slot", 3))] and host.pd_exact_end_exports == 1
    host2 = _ckpt_host([(0, "q", 512, 1000)])  # not a P request: 1,000 is off the grid -> no checkpoint
    host2._export_kda_checkpoints(mi)
    assert host2.kda_store.saved == []
    monkeypatch.setenv("GLM53_RESUME_AS_DECODE", "0")  # the model refuses off-grid exports without the flag
    K._PD_EXACT_END.clear()
    K.TTPageAlignedConnector(inner, 64, _Store()).get_num_new_matched_tokens(p_req, 0)
    assert K._PD_EXACT_END == {}


def test_d_also_waits_for_ps_exact_checkpoint(monkeypatch):
    """14's 03:41:08Z MUST: P's KV can land before its async-written exact checkpoint; D waits for both (counted
    expect_waits_ckpt), so the exact resume is not lost to a silent whole-prompt re-prefill"""
    from vllm_tt_plugin import plan_check

    monkeypatch.setenv("GLM53_RESUME_AS_DECODE", "1")
    monkeypatch.setattr(plan_check, "_CHECK", [])
    monkeypatch.setattr(K, "_EXPECT_T0", {})
    hist = list(range(7000, 8001))
    inner = _ChunkInner()
    store = _Store()  # P's checkpoint file not written yet
    c = K.TTPageAlignedConnector(inner, 64, store)
    req = _d_request(hist, 1000)
    inner.store(hist[:1000])  # the KV landed
    assert c.get_num_new_matched_tokens(req, 0) == (None, False) and c.__dict__["expect_waits_ckpt"] == 1
    store.exact.add(1000)  # P's writer finished
    assert c.get_num_new_matched_tokens(req, 0) == (1000, False)


class _PooledInner(_Inner):
    """the TT GPU connector's pooled source during a store: each saved page's pooled keys through the host's
    _pooled_pairs (GLM53_KV_POOLED_PAGES)"""

    def __init__(self, host, pages):
        super().__init__()
        self.host, self.pages, self.pairs = host, pages, []

    def save_kv_layer(self, name, cache, attn):
        super().save_kv_layer(name, cache, attn)
        self.pairs.append(K.TTKVConnectorHost._pooled_pairs(self.host, self.pages))


def test_a_preempt_save_maps_its_own_pages_pooled_keys(monkeypatch):
    """v3b6 (the blk2 kill, 2026-10-29T01:53Z): a store finds each page's pooled keys through the STORED STEP's rows;
    a preempted request is in none of them, so every preempt save raised 'pooled keys of page N: no row of the stored
    step covers it' and the request recomputed from 0. The save maps its own pages: block j -> (its slot, column j)."""
    monkeypatch.setenv("GLM53_RESUME_AS_DECODE", "1")
    host = _host(slot=5)
    host.pooled_pages, host._pooled_map, host._pooled_map_input = True, {}, None  # (an inline store just reset it)
    host._page_columns = lambda mi: {}
    host.connector = _PooledInner(host, [4, 7, 9])
    K.record_preempt_save("r", 130, ([4, 7, 9, 11],), list(range(1000, 1200)))
    host._preempt_store(SimpleNamespace(preempted_req_ids={"r"}))
    assert host.preempt_saves == 1 and not getattr(host, "preempt_saves_failed", 0)
    assert host.connector.pairs == [[(5, 0), (5, 1), (5, 2)]] * 2  # (two layers; 130 tokens = 3 pages of 64)
    assert host._pooled_map == {} and host._pooled_map_input is None  # the step's own map is back
    assert K._PREEMPT_RESUME == {"r": 130}
