"""KDACheckpointStore async writer (Stage 12, gx2c6 2026-10-23): the step only serializes to memory; one background writer
puts the bytes on disk (9's 86c1e5e bounds then evict); queued entries are visible to every store of the root."""

import os
import time

import pytest
import torch

from vllm_tt_plugin import kv_connector as K


@pytest.fixture(autouse=True)
def _tail_split_on(monkeypatch):
    """these tests check the resume-point rule at the last GRID boundary on short prompts: GLM53_PF_TAIL_SPLIT_MIN off
    (its last-chunk-end rule has its own tests, tests/test_tail_split_min.py)"""
    monkeypatch.setenv("GLM53_PF_TAIL_SPLIT_MIN", "0")


@pytest.fixture(autouse=True)
def _no_provenance_marks():
    """P1-1 marks are per request id: an earlier file's request "r" (tests/test_kv_connector.py, no request_finished)
    must not mark this file's requests"""
    K._KDA_PROVENANCE.clear()
    yield
    K._KDA_PROVENANCE.clear()


def _states(n=4096, v=1.0):
    return [{"L": 0, "S": torch.full((n,), float(v)), "C": torch.zeros(4)}]


def test_async_save_is_visible_before_the_write_and_to_a_second_store(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "1")
    a = K.KDACheckpointStore(str(tmp_path))
    b = K.KDACheckpointStore(str(tmp_path))  # the scheduler-side store of the same process
    toks = list(range(256))
    st = _states()
    a.save(toks, 128, st)
    st[0]["S"].fill_(-1.0)  # the exporter's object changes after save: the checkpoint is the snapshot at save time
    st = _states()
    assert a.has(toks, 128) and b.has(toks, 128)  # pending or written: both see it
    got = b.load(toks, 128)
    assert got is not None and torch.equal(got[0]["S"], st[0]["S"])
    a.flush()
    assert os.path.exists(a.path(toks, 128)) and a.stats["saved"] == 1 and a.stats["clone_s"] > 0
    assert not K._KDA_PENDING[a.root]  # nothing left in flight
    assert torch.equal(K.KDACheckpointStore(str(tmp_path)).load(toks, 128)[0]["S"], st[0]["S"])  # from disk


def test_writer_error_surfaces_on_the_next_save(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "1")
    monkeypatch.setenv("GLM53_KDA_CKPT_RAM_GB", "0")  # with a RAM tier the snapshot would stay valid (and has() True)
    s = K.KDACheckpointStore(str(tmp_path))

    def boom(*a, **k):  # the step only clones: the serialize runs in the writer alone
        raise OSError("No space left on device")

    monkeypatch.setattr(K, "_blob_dump", boom)
    toks = list(range(256))
    s.save(toks, 128, _states())
    s._queue.join()
    monkeypatch.undo()
    assert not s.has(toks, 128)  # the failed entry is not reported as stored
    with pytest.raises(RuntimeError, match="writer failed"):
        s.save(toks, 256, _states())


def test_every_policy_saves_the_interval_and_the_prompts_last_grid_boundary(tmp_path, monkeypatch):
    """GLM53_KDA_CKPT_EVERY=512, grid 128: a 1,000-token prompt prefilled in 128-token chunks checkpoints at 512 (the
    interval) and 896 (its last grid boundary: the next turn's hit resumes there), nowhere else; 0 = every boundary."""
    from types import SimpleNamespace

    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    prompt = list(range(1000))
    chunks = [(s, min(s + 128, 1000)) for s in range(0, 1000, 128)]

    def run(every):
        monkeypatch.setenv("GLM53_KDA_CKPT_EVERY", str(every))
        store = K.KDACheckpointStore(str(tmp_path / f"e{every}"), min_free_bytes=0)
        saved = []
        host = SimpleNamespace(kda_store=store, _kda_export=lambda kv, slot: [{"L": 0, "S": torch.zeros(2)}],
                               _kv_cache_object=None, views={"l": SimpleNamespace(block_size=64)},
                               _tokens_of=lambda rid: prompt, _ckpt_grid=128, _slot_for=lambda *a: 0)  # fmt: skip
        for s, e in chunks:
            host._prefill_rows = lambda mi, s=s, e=e: [(0, "r", s, e)]
            mi = SimpleNamespace(input_tokens=torch.tensor([prompt[:e]]))
            K.TTKVConnectorHost._export_kda_checkpoints(host, mi)
        for s, e in chunks:
            if store.has(prompt, e):
                saved.append(e)
        return saved

    assert run(512) == [512, 896]
    assert run(0) == [128, 256, 384, 512, 640, 768, 896]


def test_ram_tier_serves_hits_without_a_file_load_and_refills_from_disk(tmp_path, monkeypatch):
    """GLM53_KDA_CKPT_RAM_GB: a saved checkpoint is loaded from RAM (no file load); a disk-only one (another process
    wrote it) is loaded from its file once and then cached; the cap evicts the least recently used from RAM only."""
    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    per = _states(n=64 * 1024)  # 256 KB of states
    monkeypatch.setenv("GLM53_KDA_CKPT_RAM_GB", str(600 * 1024 / (1 << 30)))  # room for two
    s = K.KDACheckpointStore(str(tmp_path / "a"), min_free_bytes=0)
    toks = list(range(1024))
    s.save(toks, 128, per)
    loads = []
    real_load = K._blob_load
    monkeypatch.setattr(K, "_blob_load", lambda *a, **k: (loads.append(1), real_load(*a, **k))[1])
    got = s.load(toks, 128)
    assert not loads and s.stats["ram_hits"] == 1 and torch.equal(got[0]["S"], per[0]["S"])
    per[0]["S"].fill_(5.0)  # the exporter's object changes later: the RAM copy is the snapshot
    assert not torch.equal(s.load(toks, 128)[0]["S"], per[0]["S"])
    s.save(toks, 256, _states(n=64 * 1024, v=2))
    s.save(toks, 384, _states(n=64 * 1024, v=3))  # the RAM tier holds two: 128 (LRU) leaves RAM, stays on disk
    assert s.has(toks, 128) and s.load(toks, 128) is not None and loads == [1] and s.stats["disk_loads"] == 1
    assert s.load(toks, 128) is not None and loads == [1]  # cached again after the disk load
    fresh = K.KDACheckpointStore(str(tmp_path / "a"), min_free_bytes=0)  # same root, same process: the RAM tier is shared
    assert fresh.load(toks, 384)[0]["S"][0] == 3 and loads == [1]


def test_save_deferred_is_visible_at_once_and_its_producer_runs_once(tmp_path, monkeypatch):
    """Stage 12: export_kda_state_async's producer (a device snapshot the page-DMA saves) -- has() at once; the writer
    or a load() that comes first runs it exactly once; the RAM tier and the file then hold its states."""
    import threading

    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "1")
    s = K.KDACheckpointStore(str(tmp_path), min_free_bytes=0)
    toks = list(range(512))
    calls, gate = [], threading.Event()

    def producer():
        gate.wait(5)
        calls.append(1)
        return _states(v=7)

    s.save_deferred(toks, 256, producer)
    assert s.has(toks, 256) and s.stats["deferred"] == 1
    gate.set()
    got = s.load(toks, 256)  # may run the producer itself (before the writer) -- never twice
    s.flush()
    assert calls == [1] and got[0]["S"][0] == 7
    assert os.path.exists(s.path(toks, 256)) and s.load(toks, 256)[0]["S"][0] == 7
    fresh = K.KDACheckpointStore(str(tmp_path), min_free_bytes=0)
    assert fresh.load(toks, 256)[0]["S"][0] == 7  # from the shared RAM tier or the file


def test_save_deferred_sync_mode_and_a_failing_producer(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    s = K.KDACheckpointStore(str(tmp_path / "s"), min_free_bytes=0)
    toks = list(range(256))
    s.save_deferred(toks, 128, lambda: _states(v=3))
    assert os.path.exists(s.path(toks, 128))
    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "1")
    monkeypatch.setenv("GLM53_KDA_CKPT_RAM_GB", "0")
    a = K.KDACheckpointStore(str(tmp_path / "a"), min_free_bytes=0)

    def boom():
        raise TimeoutError("state save not complete")

    a.save_deferred(toks, 128, boom)
    a._queue.join()
    assert not a.has(toks, 128)  # never reported as stored
    with pytest.raises(RuntimeError, match="writer failed"):
        a.save_deferred(toks, 256, lambda: _states())


def test_export_site_takes_the_async_path_and_falls_back(tmp_path, monkeypatch):
    """_export_kda_checkpoints: a model producer -> save_deferred, export_kda_state is not called; None -> sync export"""
    from types import SimpleNamespace

    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    monkeypatch.setenv("GLM53_KDA_CKPT_EVERY", "0")
    prompt = list(range(256))
    for async_ret, want_sync in ((lambda: _states(v=9), False), (None, True)):
        store = K.KDACheckpointStore(str(tmp_path / str(want_sync)), min_free_bytes=0)
        sync_calls = []
        host = SimpleNamespace(kda_store=store, _kv_cache_object=None, views={"l": SimpleNamespace(block_size=64)},
                               _kda_export=lambda kv, slot: (sync_calls.append(slot), _states(v=1))[1],
                               _kda_export_async=lambda kv, slot, r=async_ret: r,
                               _tokens_of=lambda rid: prompt, _ckpt_grid=128, _slot_for=lambda *a: 4,
                               _prefill_rows=lambda mi: [(0, "r", 0, 128)])  # fmt: skip
        K.TTKVConnectorHost._export_kda_checkpoints(host, SimpleNamespace(input_tokens=torch.tensor([prompt[:128]])))
        assert store.has(prompt, 128) and (sync_calls == [4]) == want_sync
        assert store.load(prompt, 128)[0]["S"][0] == (1 if want_sync else 9)


def test_provenance_rule_marks_requests_that_recompute_lmcache_chunks(tmp_path, monkeypatch):
    """P1-1: LMCache full chunks past the KDA-aligned hit = pages of another computation -> no checkpoint from that
    request (n29h5c a1: lookup 512, KDA hit 128); a multi-turn turn 2 (lookup = turn 1's prompt, its last 128-chunk is
    turn 1's checkpoint) is not marked; the knob off marks nothing; request_finished clears the mark."""
    from types import SimpleNamespace

    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    monkeypatch.delenv("GLM53_KDA_CKPT_PROVENANCE", raising=False)
    store = K.KDACheckpointStore(str(tmp_path), min_free_bytes=0)
    prompt = list(range(1000, 1586))
    store.save(prompt, 128, _states())  # the only checkpoint: 128

    class Inner:
        def __init__(self, ext):
            self.ext, self.finished = ext, []

        def get_num_new_matched_tokens(self, request, n):
            return self.ext, False

        def request_finished(self, request, block_ids):
            self.finished.append(request.request_id)
            return False, None

    def ask(rid, ext, toks=prompt):
        c = K.TTPageAlignedConnector(Inner(ext), 128, store)
        got = c.get_num_new_matched_tokens(SimpleNamespace(request_id=rid, prompt_token_ids=toks), 0)
        return c, got

    c, got = ask("a1", 512)  # LMCache 512 tokens, KDA hit 128 -> it recomputes [128, 512) of LMCache's chunks
    assert got == (128, False) and K._KDA_PROVENANCE["a1"] == 512
    c.request_finished(SimpleNamespace(request_id="a1"), [])
    assert "a1" not in K._KDA_PROVENANCE and c._inner.finished == ["a1"]
    store.save(prompt, 512, _states())
    _, got = ask("a2", 585)  # the same prompt again: lookup len - 1, hit 512 = the last full chunk -> not marked
    assert got == (512, False) and "a2" not in K._KDA_PROVENANCE
    _, got = ask("turn2", 586, prompt + list(range(7, 300)))  # turn 2: turn 1's 586 tokens stored, checkpoint 512
    assert got == (512, False) and "turn2" not in K._KDA_PROVENANCE
    _, got = ask("none", 0)
    assert got == (0, False) and "none" not in K._KDA_PROVENANCE
    monkeypatch.setenv("GLM53_KDA_CKPT_PROVENANCE", "0")
    _, got = ask("off", 512, list(range(5000, 5586)))  # no checkpoint for this prompt -> hit 0, rule off
    assert "off" not in K._KDA_PROVENANCE


def test_marked_requests_export_no_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    monkeypatch.setenv("GLM53_KDA_CKPT_EVERY", "0")
    prompt = list(range(256))
    store = K.KDACheckpointStore(str(tmp_path), min_free_bytes=0)
    host = SimpleNamespace(kda_store=store, _kv_cache_object=None, views={"l": SimpleNamespace(block_size=64)},
                           _kda_export=lambda kv, slot: _states(), _tokens_of=lambda rid: prompt, _ckpt_grid=128,
                           _slot_for=lambda *a: 0, _prefill_rows=lambda mi: [(0, "m", 0, 128)])  # fmt: skip
    K._KDA_PROVENANCE["m"] = 256
    try:
        K.TTKVConnectorHost._export_kda_checkpoints(host, SimpleNamespace(input_tokens=torch.tensor([prompt[:128]])))
        assert not store.has(prompt, 128) and host.kda_ckpt_skipped_provenance == 1
    finally:
        K._KDA_PROVENANCE.pop("m", None)
    K.TTKVConnectorHost._export_kda_checkpoints(host, SimpleNamespace(input_tokens=torch.tensor([prompt[:128]])))
    assert store.has(prompt, 128)


def test_lmcache_load_window_is_the_aligned_hit(tmp_path, monkeypatch):
    """n29h5c 32K request (program 05:14Z): LMCache matched 31,999 tokens, the KDA checkpoint was at 1,408 -> vLLM
    computes from 1,408 and LMCache's load window is 1,408 rounded up to its chunk (1,536: whole chunks), not 32,000,
    with LMCache's own assert on the count it gets back; GLM53_LMCACHE_LOAD_ALIGNED=0 = the old raw window."""
    from types import SimpleNamespace

    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    store = K.KDACheckpointStore(str(tmp_path), min_free_bytes=0)
    prompt = list(range(32000))
    store.save(prompt, 1408, _states())

    class Engine:  # LMCache's scheduler-side adapter: load_specs + the assert of update_state_after_alloc
        def __init__(self):
            self.load_specs = {}
            self._lmcache_chunk_size = 256

        def update(self, request, n):
            spec = self.load_specs[request.request_id]
            recalc = 1 if spec.lmcache_cached_tokens == request.num_tokens else 0
            assert n == spec.lmcache_cached_tokens - spec.vllm_cached_tokens - recalc, (n, spec)
            spec.can_load = True

    class Inner:
        def __init__(self):
            self._lmcache_engine = Engine()

        def get_num_new_matched_tokens(self, request, n):
            self._lmcache_engine.load_specs[request.request_id] = SimpleNamespace(
                vllm_cached_tokens=n, lmcache_cached_tokens=32000, can_load=False)
            return 31999, False

        def update_state_after_alloc(self, request, blocks, n):
            self._lmcache_engine.update(request, n)

    req = SimpleNamespace(request_id="r32k", prompt_token_ids=prompt, num_tokens=32000)
    for knob, window in (("1", 1536), ("0", 32000)):  # 1,408 rounded up to LMCache's 256-token chunk
        monkeypatch.setenv("GLM53_LMCACHE_LOAD_ALIGNED", knob)
        c = K.TTPageAlignedConnector(Inner(), 128, store)
        ext, _ = c.get_num_new_matched_tokens(req, 0)
        assert ext == 1408
        c.update_state_after_alloc(req, None, ext)
        spec = c._inner._lmcache_engine.load_specs["r32k"]
        assert spec.lmcache_cached_tokens == window and spec.can_load
    K._KDA_PROVENANCE.pop("r32k", None)
def test_every_policy_resume_point_is_strictly_before_an_aligned_prompts_end(tmp_path, monkeypatch):
    """EVERY=8192, a grid-aligned prompt (1,024 = 8 x 128): the resume point is 896, the last grid boundary STRICTLY
    before the end, not 1,024 itself. A repeat of the prompt must compute >= 1 token, so a checkpoint at 1,024 never serves it
    (n29h5c's r1 cell: a 4,096-token prompt checkpointed only 4,096 and its repeat hit nothing)."""
    from types import SimpleNamespace

    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    monkeypatch.setenv("GLM53_KDA_CKPT_EVERY", "8192")  # the served value: no interval boundary inside the prompt
    monkeypatch.setenv("GLM53_KDA_CKPT_AT_END", "0")  # the pre-cf3h-d rule (aligned: the next test)
    prompt = list(range(1024))
    store = K.KDACheckpointStore(str(tmp_path / "aligned"), min_free_bytes=0)
    host = SimpleNamespace(kda_store=store, _kda_export=lambda kv, slot: [{"L": 0, "S": torch.zeros(2)}],
                           _kv_cache_object=None, views={"l": SimpleNamespace(block_size=64)},
                           _tokens_of=lambda rid: prompt, _ckpt_grid=128, _slot_for=lambda *a: 0)  # fmt: skip
    ends = list(range(128, 1025, 128))
    for e in ends:
        host._prefill_rows = lambda mi, s=e - 128, e=e: [(0, "r", s, e)]
        K.TTKVConnectorHost._export_kda_checkpoints(host, SimpleNamespace(input_tokens=torch.tensor([prompt[:e]])))
    assert [e for e in ends if store.has(prompt, e)] == [896]


def test_cold_request_gets_no_hit_stores_nothing_and_exports_no_checkpoint(tmp_path, monkeypatch):
    """kv_transfer_params {"tt.cold": true} (Stage 9 round-3 review P1-1, the cold len_control probe): LMCache holds
    the prefix and a KDA checkpoint exists, yet the request gets 0 external tokens, carries lmcache.skip_save and is
    marked (no checkpoint); the inner lookup still runs once; the same prompt without the switch hits."""
    from types import SimpleNamespace

    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    store = K.KDACheckpointStore(str(tmp_path), min_free_bytes=0)
    prompt = list(range(2000, 2586))
    store.save(prompt, 512, _states())

    class Inner:
        def __init__(self):
            self.calls = 0

        def get_num_new_matched_tokens(self, request, n):
            self.calls += 1
            return 585, False

    def req(rid, kv):
        sp = SimpleNamespace(extra_args={"kv_transfer_params": dict(kv)} if kv is not None else None)
        return SimpleNamespace(request_id=rid, prompt_token_ids=prompt, num_prompt_tokens=len(prompt),
                               sampling_params=sp, kv_transfer_params=kv)

    c = K.TTPageAlignedConnector(Inner(), 128, store)
    r = req("cold", {"tt.cold": True})
    try:
        assert c.get_num_new_matched_tokens(r, 0) == (0, False)
        assert c._inner.calls == 1 and c.cold_requests == 1
        assert r.sampling_params.extra_args["kv_transfer_params"]["lmcache.skip_save"] is True
        assert "cold" in K._KDA_PROVENANCE and "cold" not in c._reported
    finally:
        K._KDA_PROVENANCE.pop("cold", None)
    assert c.get_num_new_matched_tokens(req("warm", None), 0) == (512, False)
    K._KDA_PROVENANCE.pop("warm", None)


def test_prefill_save_is_deferred_and_flushed_in_step_order(monkeypatch):
    """GLM53_KV_SAVE_DEFER (default): a prefill step's save (LMCache store + KDA export) runs at the next step's
    begin_step, the scheduler's next lookup (flush_deferred_saves), no_forward or shutdown -- never inside the step
    that returns the first token; a later step flushes it first (step order); decode steps save at once"""
    from types import SimpleNamespace

    monkeypatch.delenv("GLM53_KV_SAVE_DEFER", raising=False)
    host = K.TTKVConnectorHost.__new__(K.TTKVConnectorHost)
    host._deferred_save, host.saves_deferred = None, 0
    done = []
    host._save_now = lambda so, mi, out=None: done.append(so)
    host._save_state = lambda mi, out=None: None
    host._save_pages = lambda so, mi: done.append(so)
    pre = lambda tag: (tag, SimpleNamespace(prompt_lens=[1]))  # noqa: E731
    dec = lambda tag: (tag, SimpleNamespace(prompt_lens=None))  # noqa: E731
    host.save(*pre("p1"))
    assert done == [] and host.saves_deferred == 1
    K.flush_deferred_saves()  # the scheduler's next lookup
    assert done == ["p1"]
    host.save(*pre("p2"))
    host.save(*pre("p3"))  # p2 first, then p3 waits
    assert done == ["p1", "p2"]
    host.save(*dec("d1"))  # a decode step: p3, then d1 at once
    assert done == ["p1", "p2", "p3", "d1"]
    host.save(*pre("p4"))
    host.connector, host._metadata, host._lock = SimpleNamespace(bind_connector_metadata=lambda m: None,
                                                                 clear_connector_metadata=lambda: None,
                                                                 handle_preemptions=lambda m: None,
                                                                 shutdown=lambda: done.append("shutdown")), (lambda so: None), __import__("threading").Lock()  # fmt: skip
    host.begin_step(SimpleNamespace())  # the next step's begin
    assert done[-1] == "p4"
    host.save(*pre("p5"))
    host.shutdown()
    assert done[-2:] == ["p5", "shutdown"]
    monkeypatch.setenv("GLM53_KV_SAVE_DEFER", "0")
    host.save(*pre("p6"))
    assert done[-1] == "p6"


def test_store_hold_keeps_a_chunked_prompts_page_stores_until_its_first_token(monkeypatch):
    """GLM53_KV_STORE_HOLD=1: the KDA / DSA checkpoints of each prefill step still run one step later (begin_step), but
    the LMCache page stores of a prompt's intermediate chunks are held until the step after its last chunk, then run in
    step order; a held request leaving (finished / preempted in the coming step) flushes them first; a lookup flushes"""
    from types import SimpleNamespace

    monkeypatch.setenv("GLM53_KV_STORE_HOLD", "1")
    monkeypatch.setenv("GLM53_KV_STORE_PRIORITY", "0")  # every held store at once (the per-step budget: the next test)
    monkeypatch.delenv("GLM53_KV_SAVE_DEFER", raising=False)
    host = K.TTKVConnectorHost.__new__(K.TTKVConnectorHost)
    host._deferred_save, host.saves_deferred = None, 0
    states, pages = [], []
    host._save_state = lambda mi, out=None: states.append(mi.tag)
    host._save_pages = lambda so, mi: pages.append(mi.tag)
    host._save_now = lambda so, mi, out=None: (states.append(mi.tag), pages.append(mi.tag))
    host._tokens_of = lambda rid: list(range(300))  # a 300-token prompt: chunks end 128, 256, 300
    host._prefill_rows = lambda mi: [(0, "r", mi.start, mi.end)]
    host.connector = SimpleNamespace(bind_connector_metadata=lambda m: None, clear_connector_metadata=lambda: None,
                                     handle_preemptions=lambda m: None)  # fmt: skip
    host._metadata, host._lock = (lambda so: None), __import__("threading").Lock()
    step = lambda tag, s0, e: (SimpleNamespace(finished_req_ids=set()),  # noqa: E731
                               SimpleNamespace(prompt_lens=[e], row_req_ids=["r"], start=s0, end=e, tag=tag))
    nxt = SimpleNamespace(finished_req_ids=set())
    host.save(*step("c1", 0, 128))
    host.begin_step(nxt)
    assert states == ["c1"] and pages == []  # c1's checkpoint ran, its store is held (no first token yet)
    host.save(*step("c2", 128, 256))
    host.begin_step(nxt)
    assert states == ["c1", "c2"] and pages == []
    host.save(*step("c3", 256, 300))  # the last chunk: the first token leaves with this step
    host.begin_step(nxt)
    assert states == ["c1", "c2", "c3"] and pages == ["c1", "c2", "c3"]  # all stores, in step order, after the token
    # a held request that leaves flushes at once
    host.save(*step("d1", 0, 128))
    host.begin_step(SimpleNamespace(finished_req_ids={"r"}))
    assert pages[-1] == "d1"
    # a lookup flushes everything held
    host.save(*step("e1", 0, 128))
    host.begin_step(nxt)
    assert pages[-1] == "d1"
    K.flush_deferred_saves()
    assert pages[-1] == "e1"
    monkeypatch.setenv("GLM53_KV_STORE_HOLD", "0")  # off: every page store runs with its state, one step later
    host.save(*step("f1", 0, 128))
    host.begin_step(nxt)
    assert pages[-1] == "f1"


def test_store_priority_flushes_one_held_step_per_engine_step(monkeypatch):
    """GLM53_KV_STORE_PRIORITY (default 1, n29h7 M1 2026-10-23: every interleaved decode step flushed every held store on
    the engine thread): while the last held prompt is mid-prefill nothing flushes, decode steps included; after its
    first token the held steps drain one per engine step (begin_step or a decode step's save), oldest first; a held
    request leaving still flushes everything at once; a host whose checkpoints cannot resume skips the lookup flush"""
    from types import SimpleNamespace

    monkeypatch.setenv("GLM53_KV_STORE_HOLD", "1")
    monkeypatch.delenv("GLM53_KV_STORE_PRIORITY", raising=False)
    monkeypatch.delenv("GLM53_KV_SAVE_DEFER", raising=False)
    host = K.TTKVConnectorHost.__new__(K.TTKVConnectorHost)
    host._deferred_save, host.saves_deferred = None, 0
    pages = []
    host._save_state = lambda mi, out=None: None
    host._save_pages = lambda so, mi: pages.append(mi.tag)
    host._save_now = lambda so, mi, out=None: pages.append(mi.tag)
    host._tokens_of = lambda rid: list(range(400))  # chunks end 128, 256, 384, 400
    host._prefill_rows = lambda mi: [(0, "r", mi.start, mi.end)] if mi.prompt_lens is not None else []
    host.connector = SimpleNamespace(bind_connector_metadata=lambda m: None, clear_connector_metadata=lambda: None,
                                     handle_preemptions=lambda m: None)  # fmt: skip
    host._metadata, host._lock = (lambda so: None), __import__("threading").Lock()
    nxt = SimpleNamespace(finished_req_ids=set())
    pre = lambda tag, s0, e: (nxt, SimpleNamespace(prompt_lens=[e], row_req_ids=["r"], start=s0, end=e, tag=tag))  # noqa: E731
    dec = lambda tag: (nxt, SimpleNamespace(prompt_lens=None, row_req_ids=["o"], tag=tag))  # noqa: E731
    for tag, s0, e in (("c1", 0, 128), ("c2", 128, 256)):
        host.save(*pre(tag, s0, e))
        host.begin_step(nxt)
        host.save(*dec("d-" + tag))  # another user's decode step between the chunks
    assert pages == ["d-c1", "d-c2"]  # mid-prefill: no held store flushed, decode steps included
    host.save(*pre("c3", 256, 384))
    host.begin_step(nxt)
    host.save(*pre("c4", 384, 400))  # the last chunk (its first token leaves with this step)
    host.begin_step(nxt)
    assert pages == ["d-c1", "d-c2", "c1"]  # one held step per engine step from here, oldest first
    host.save(*dec("d1"))
    assert pages[-2:] == ["c2", "d1"]
    host.begin_step(nxt)
    assert pages[-1] == "c3"
    host.begin_step(SimpleNamespace(finished_req_ids={"r"}))  # the held request leaves: everything at once
    assert pages[-1] == "c4" and not host._held_pages
    host.save(*pre("e1", 0, 128))
    host.begin_step(nxt)
    host.checkpoints_off = True
    K.flush_deferred_saves()  # no hit can resume on this cache: the lookup leaves the held store alone
    assert pages[-1] == "c4" and host._held_pages
    host.checkpoints_off = False
    K.flush_deferred_saves()
    assert pages[-1] == "e1"


def test_per_step_hold_check_reads_the_history_length_not_the_tokens(monkeypatch):
    """Stage 12 (program 2026-10-23T19:45:01Z: per-step host work O(1) in the context): with the runner's len_of,
    the held-store check never builds the request's token list (a 1M-token history per engine step)"""
    from types import SimpleNamespace

    monkeypatch.setenv("GLM53_KV_STORE_HOLD", "1")
    host = K.TTKVConnectorHost.__new__(K.TTKVConnectorHost)

    def boom(rid):
        raise AssertionError("the token list was built")

    host._tokens_of, host._len_of = boom, (lambda rid: 1_048_576)
    host._prefill_rows = lambda mi: [(0, "r", mi.start, mi.end)]
    held = [(None, SimpleNamespace(row_req_ids=["r"], start=0, end=8192))]
    assert host._may_hold(held, SimpleNamespace(finished_req_ids=set())) is True  # mid-prefill: hold
    held = [(None, SimpleNamespace(row_req_ids=["r"], start=1_040_384, end=1_048_576))]
    assert host._may_hold(held, SimpleNamespace(finished_req_ids=set())) is False
    runner = SimpleNamespace(requests={"r": SimpleNamespace(prompt_token_ids=[1] * 300, output_token_ids=[2] * 5)})
    assert K._runner_request_len(runner, "r") == 305 == len(K._runner_request_tokens(runner, "r"))
    assert K._runner_request_len(runner, "x") is None


@pytest.mark.parametrize("at_end, want", [("aligned", [896, 1024]), ("all", [896, 1024]), ("0", [896])])
def test_every_policy_also_checkpoints_an_aligned_prompts_end_under_ckpt_at_end(tmp_path, monkeypatch, at_end, want):
    """GLM53_KDA_CKPT_AT_END aligned / all (cf3h-d): a chunk that ends exactly at a grid-aligned prompt's end is
    checkpointed there (the scheduler no longer splits such a prompt, so in service that is its only chunk end); the
    last boundary strictly before the end keeps its rule (an unaligned prompt's split still lands there)"""
    from types import SimpleNamespace

    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    monkeypatch.setenv("GLM53_KDA_CKPT_EVERY", "8192")
    monkeypatch.setenv("GLM53_KDA_CKPT_AT_END", at_end)
    prompt = list(range(1024))
    store = K.KDACheckpointStore(str(tmp_path / at_end), min_free_bytes=0)
    host = SimpleNamespace(kda_store=store, _kda_export=lambda kv, slot: [{"L": 0, "S": torch.zeros(2)}],
                           _kv_cache_object=None, views={"l": SimpleNamespace(block_size=64)},
                           _tokens_of=lambda rid: prompt, _ckpt_grid=128, _slot_for=lambda *a: 0)  # fmt: skip
    ends = list(range(128, 1025, 128))
    for e in ends:
        host._prefill_rows = lambda mi, s=e - 128, e=e: [(0, "r", s, e)]
        K.TTKVConnectorHost._export_kda_checkpoints(host, SimpleNamespace(input_tokens=torch.tensor([prompt[:e]])))
    assert [e for e in ends if store.has(prompt, e)] == want


def test_ckpt_at_end_rejects_an_unknown_value(monkeypatch):
    monkeypatch.setenv("GLM53_KDA_CKPT_AT_END", "end")
    with pytest.raises(ValueError, match="GLM53_KDA_CKPT_AT_END"):
        K.kda_ckpt_at_end_mode()


# ---------------------------------------------------------------- the one-replay store (GLM53_KDA_CKPT_IN_REPLAY)
def _snap_host(tmp_path, prompt, exports):
    from types import SimpleNamespace

    store = K.KDACheckpointStore(str(tmp_path / "snap"), min_free_bytes=0)

    def export(kv, slot, at=None):  # 12's M2: export_kda_state(kv, slot, at=cut) reads the snapshot staging
        exports.append((slot, at, "snapshot"))
        return [{"L": 0, "S": torch.zeros(2)}]

    host = SimpleNamespace(kda_store=store, _kda_export=export, _kv_cache_object=None,
                           views={"l": SimpleNamespace(block_size=64)}, _tokens_of=lambda rid: prompt, _ckpt_grid=128,
                           _slot_for=lambda *a: 3, checkpoints_off=False,
                           _kda_snapshot_ready=lambda kv, slot, cut: True)  # fmt: skip
    host._prefill_rows = lambda mi: K.TTKVConnectorHost._prefill_rows(host, mi)
    return host, store


def _model_input(**kw):
    """a REAL TTModelInput (frozen): the connector must replace its fields, never assign them -- the 2026-10-25 cf3h-g
    image mock died on `model_input.kda_snapshot_at = ...` that a mutable SimpleNamespace stand-in had let pass"""
    from dataclasses import MISSING, fields

    from vllm_tt_plugin.model_input import TTModelInput

    base = {f.name: None for f in fields(TTModelInput) if f.default is MISSING and f.default_factory is MISSING}
    base.update(kw)
    return TTModelInput(**base)


@pytest.mark.parametrize("n, start, end, want", [(2048, 0, 2048, 1920), (2000, 0, 2000, 1920), (2176, 0, 2048, 0),
                                                   (2176, 2048, 2176, 0), (4100, 2048, 4100, 4096), (900, 0, 900, 0)])
def test_in_replay_marks_the_cut_inside_the_chunk_and_exports_it_from_the_snapshot(tmp_path, monkeypatch, n, start,
                                                                                   end, want):
    """GLM53_KDA_CKPT_IN_REPLAY=1: a prefill chunk that covers the prompt's cut floor((len - 1) / 128) x 128 strictly
    inside gets kda_snapshot_at = the cut; after the step the checkpoint (tokens[:cut], cut) is exported with
    source="snapshot". A chunk ending ON the cut (2,176: [0, 2048)) needs no snapshot; a prompt under the store floor
    (900 < 1,024) none at all."""
    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    monkeypatch.setenv("GLM53_KDA_CKPT_IN_REPLAY", "1")
    monkeypatch.setenv("GLM53_KDA_CKPT_EVERY", "1048576")
    monkeypatch.setenv("GLM53_LMCACHE_MIN_STORE_TOKENS", "1024")
    prompt = list(range(n))
    exports = []
    host, store = _snap_host(tmp_path, prompt, exports)
    mi = _model_input(prompt_lens=[end], row_req_ids=["r"], input_positions=torch.tensor([start]),
                      input_tokens=torch.tensor([prompt[:end]]))  # fmt: skip
    out = K.TTKVConnectorHost._mark_kda_snapshots(host, mi)  # frozen: a replaced copy, the input untouched
    assert out.kda_snapshot_at == ([want] if want else None) and mi.kda_snapshot_at is None
    assert (out is mi) == (not want)
    K.TTKVConnectorHost._export_kda_snapshots(host, out)
    assert exports == ([(3, want, "snapshot")] if want else [])
    assert store.has(prompt, want) if want else True


def test_in_replay_turns_off_the_split_and_the_end_rule(tmp_path, monkeypatch):
    """the scheduler never splits a storing prompt (aligned or not), and a chunk ending at an aligned prompt's end is
    not checkpointed there (the snapshot at the cut serves repeats and continuations alike)"""
    monkeypatch.setenv("GLM53_KDA_CKPT_IN_REPLAY", "1")
    monkeypatch.setenv("GLM53_KDA_CKPT_AT_END", "aligned")
    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    monkeypatch.setenv("GLM53_KDA_CKPT_EVERY", "8192")
    prompt = list(range(1024))
    exports = []
    host, store = _snap_host(tmp_path, prompt, exports)
    host._kda_export = lambda kv, slot: [{"L": 0, "S": torch.zeros(2)}]
    from types import SimpleNamespace

    K.TTKVConnectorHost._export_kda_checkpoints(
        host, SimpleNamespace(prompt_lens=[1024], row_req_ids=["r"], input_positions=torch.tensor([0]),
                              input_tokens=torch.tensor([prompt])))  # fmt: skip
    assert not store.has(prompt, 1024)


def test_in_replay_needs_a_model_that_snapshots(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("GLM53_KDA_CKPT_IN_REPLAY", "1")
    with pytest.raises(RuntimeError, match="kda_checkpoint_in_replay"):
        K.TTKVConnectorHost.from_runner(SimpleNamespace(_is_lane_mode=False, model=SimpleNamespace()))


def test_in_replay_skips_a_snapshot_the_model_says_is_not_ready(tmp_path, monkeypatch):
    """12's M2: kda_snapshot_ready False -> no export, no store entry (counted); the tag stays the model's"""
    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    monkeypatch.setenv("GLM53_KDA_CKPT_IN_REPLAY", "1")
    prompt = list(range(2048))
    exports = []
    host, store = _snap_host(tmp_path, prompt, exports)
    host._kda_snapshot_ready = lambda kv, slot, cut: False
    mi = _model_input(prompt_lens=[2048], row_req_ids=["r"], input_positions=torch.tensor([0]),
                      input_tokens=torch.tensor([prompt]), kda_snapshot_at=[1920])  # fmt: skip
    K.TTKVConnectorHost._export_kda_snapshots(host, mi)
    assert exports == [] and not store.has(prompt, 1920) and host.kda_snapshots_not_ready == 1



def _blob_zoo():
    g = torch.Generator().manual_seed(0)
    return {
        "num_tokens": 16100,
        "states": [
            {"L": 0, "S": torch.randn(4, 128, 128, generator=g, dtype=torch.float32), "C": torch.randn(3, 7).bfloat16()},
            {"L": 3, "S": torch.randn(2, 64, 64, generator=g).to(torch.bfloat16), "C": torch.arange(10, dtype=torch.int32)},
        ],
        "extra": (torch.tensor(3.5), torch.zeros(0, 4), torch.tensor([True, False]), None, "kda", 2.5, -1,
                  K.np.arange(6, dtype=K.np.int64).reshape(2, 3), K.np.float32(1.25), {7: "int key"}),
        "nan": float("nan"),
    }  # fmt: skip


def _same(a, b):
    if isinstance(a, torch.Tensor):
        def bits(t):  # bit for bit (NaN payloads, -0.0)
            return t.reshape(-1).view(torch.uint8) if t.numel() and t.dtype != torch.bool else t

        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(bits(a), bits(b))
    if isinstance(a, K.np.ndarray):
        return isinstance(b, K.np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and (a == b).all()
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and a != a:
        return isinstance(b, float) and b != b
    return a == b and type(a) is type(b) or (isinstance(a, K.np.generic) and a.item() == b)


def test_raw_format_round_trip_is_bitwise_and_loads_without_pickle(tmp_path, monkeypatch):
    """GLM53_KDA_CKPT_FORMAT=raw (default, lever 1): skeleton + raw bytes; every dtype / shape / container comes back
    bit for bit, and the loader never calls torch.load (no pickle under the GIL)"""
    blob = _blob_zoo()
    p = str(tmp_path / "x.pt")
    assert K._blob_dump(blob, p) == "raw"
    with open(p, "rb") as f:
        assert f.read(len(K._RAW_MAGIC)) == K._RAW_MAGIC
    monkeypatch.setattr(K.torch, "load", lambda *a, **k: pytest.fail("raw checkpoint went through torch.load"))
    got = K._blob_load(p)
    assert _same(blob, got)
    s = got["states"][0]["S"]
    s.add_(1.0)  # a loaded tensor is an ordinary writable CPU tensor
    assert s.is_contiguous() and s.device.type == "cpu"


def test_old_torch_save_checkpoints_still_load_and_the_format_switch(tmp_path, monkeypatch):
    """files written before the raw format (torch.save) load through the same path; GLM53_KDA_CKPT_FORMAT=torch keeps
    the old writer; a blob the skeleton cannot carry falls back to torch.save"""
    blob = _blob_zoo()
    old = str(tmp_path / "old.pt")
    torch.save(blob, old)
    assert _same(blob, K._blob_load(old))
    monkeypatch.setenv("GLM53_KDA_CKPT_FORMAT", "torch")
    assert K._blob_dump(blob, str(tmp_path / "t.pt")) == "torch"
    monkeypatch.delenv("GLM53_KDA_CKPT_FORMAT")
    odd = {"num_tokens": 1, "states": {(0, 1): torch.ones(2)}}  # a tuple key: no JSON skeleton
    assert K._blob_dump(odd, str(tmp_path / "o.pt")) == "torch"
    assert torch.equal(K._blob_load(str(tmp_path / "o.pt"))["states"][(0, 1)], torch.ones(2))


def test_store_writes_raw_by_default_and_reads_back_from_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "1")
    monkeypatch.setenv("GLM53_KDA_CKPT_RAM_GB", "0")
    a = K.KDACheckpointStore(str(tmp_path), min_free_bytes=0)
    toks = list(range(512))
    st = [{"L": i, "S": torch.randn(64, 64).bfloat16(), "C": torch.randn(4, 8)} for i in range(3)]
    a.save(toks, 256, st)
    a.flush()
    assert a.stats.get("written_raw") == 1 and not a.stats.get("written_torch")
    with open(a.path(toks, 256), "rb") as f:
        assert f.read(len(K._RAW_MAGIC)) == K._RAW_MAGIC
    got = K.KDACheckpointStore(str(tmp_path), min_free_bytes=0).load(toks, 256)
    assert _same(st, got)


def test_the_served_snapshot_blob_is_raw(tmp_path, monkeypatch):
    """the structure StackModel.collect_snapshot returns (a KDA pack + a DSA blob with string tensor keys) goes the raw
    way, not the torch.save fallback"""
    states = [
        dict(kda=dict(kind="glm53_kda_pack_v1", layers=[0, 1, 2, 4], n_s=8, S=torch.randn(8, 4, 8, 32, 32),
                      C=torch.randn(8, 4, 3, 32, 32).bfloat16())),
        dict(dsa=dict(kind="glm53_dsa_v1", unit_tokens=128, units=126, n_tokens=16100, chips=8, replicated=False,
                      layout="unit", pools=True, tensors={"L3.pools": torch.randn(1, 504, 32, 32).bfloat16(),
                                                          "L3.ring": torch.randn(8, 2, 32, 32).bfloat16()})),
    ]  # fmt: skip
    blob = {"num_tokens": 16100, "states": states}
    p = str(tmp_path / "s.pt")
    monkeypatch.setattr(K, "_RAW_FALLBACK_WARNED", False)  # (an earlier test's tuple-key blob set it)
    assert K._blob_dump(blob, p) == "raw" and not K._RAW_FALLBACK_WARNED
    assert _same(blob, K._blob_load(p))


class _Handle:
    """a stand-in for the model's KDASnapshotHandle: valid until invalidate()"""

    def __init__(self):
        self.ok = True

    def valid(self):
        return self.ok


def _snap_producer(states_fn, handle, next_pos):
    def producer():
        return {"next_pos": next_pos, "states": states_fn()}

    producer.kda_snapshot, producer.next_pos = handle, next_pos
    return producer


def test_the_latest_snapshot_export_is_the_only_snapshot_producer(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    s = K.KDACheckpointStore(str(tmp_path), min_free_bytes=0)
    toks = list(range(1024))
    h1 = _Handle()
    p1 = _snap_producer(lambda: _states(v=1), h1, 256)
    s.save_deferred(toks, 256, p1)
    assert s.snapshot_producer(toks, 256) is p1 and s.snapshot_producer(toks, 128) is None
    p2 = _snap_producer(lambda: _states(v=2), _Handle(), 512)
    s.save_deferred(toks, 512, p2)  # the snapshot buffers now hold 512: 256 is a host checkpoint only
    assert s.snapshot_producer(toks, 256) is None and s.snapshot_producer(toks, 512) is p2
    assert K.KDACheckpointStore(str(tmp_path), min_free_bytes=0).snapshot_producer(toks, 512) is p2  # per root
    s.save_deferred(toks, 768, lambda: {"next_pos": 768, "states": _states(v=3)})  # no handle: nothing resident
    assert s.snapshot_producer(toks, 512) is None and s.snapshot_producer(toks, 768) is None


def test_a_back_to_back_hit_imports_from_the_snapshot_without_waiting_for_the_export(tmp_path, monkeypatch):
    """lever 1 (b), the 16,100 back-to-back case: the cold request's export is still in the writer (its page-DMA save +
    untilize blocked here) when the hit's import runs -- the hit takes the model's snapshot marker at once (the
    producer never runs on the engine thread); the writer then finishes the host copy as before"""
    import threading
    from types import SimpleNamespace

    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "1")
    s = K.KDACheckpointStore(str(tmp_path), min_free_bytes=0)
    toks = list(range(16100))
    gate, ran = threading.Event(), []

    def slow_states():
        ran.append(threading.current_thread().name)
        assert gate.wait(10)
        return _states(v=4)

    h = _Handle()
    s.save_deferred(toks, 16000, _snap_producer(slow_states, h, 16000))
    asked = []

    def kda_snapshot_states(producer, fallback):  # the model adapter's hook (generator_vllm.kda_snapshot_states)
        asked.append(producer)
        if not producer.kda_snapshot.valid():
            return None
        return {"next_pos": producer.next_pos, "states": [{"kda_snapshot": producer.kda_snapshot, "fallback": fallback}]}

    host = SimpleNamespace(kda_store=s, _kda_snap_states=kda_snapshot_states)
    t0 = time.perf_counter()
    got = K.TTKVConnectorHost._kda_snapshot_import(host, toks[:16000], 16000)
    took = time.perf_counter() - t0
    assert got["next_pos"] == 16000 and got["states"][0]["kda_snapshot"] is h and took < 0.5
    assert threading.current_thread().name not in ran and host.kda_snapshot_imports == 1
    gate.set()
    s.flush()  # the writer resolved the export: the host copy is on disk / in RAM for later hits
    assert s.load(toks[:16000], 16000)["states"][0]["S"][0] == 4
    fb = got["states"][0]["fallback"]()  # a snapshot that went stale before the prefill: the host checkpoint
    assert fb["next_pos"] == 16000 and fb["states"][0]["S"][0] == 4
    h.ok = False
    assert K.TTKVConnectorHost._kda_snapshot_import(host, toks[:16000], 16000) is None  # stale: the host path
    assert K.TTKVConnectorHost._kda_snapshot_import(host, toks[:15872], 15872) is None  # not the latest export
    off = SimpleNamespace(kda_store=s, _kda_snap_states=None)  # GLM53_KDA_SNAP_IMPORT=0
    assert K.TTKVConnectorHost._kda_snapshot_import(off, toks[:16000], 16000) is None


def test_step_phase_timers_log_a_slow_step_and_keep_totals(monkeypatch):
    """GLM53_KV_PHASE_LOG_MS: per-phase host time of a connector step (wave 2's unaccounted 127 s): one INFO line at or
    above the threshold, cumulative totals in phase_s, the KDA import sub-phases folded in"""
    from types import SimpleNamespace

    H = K.TTKVConnectorHost
    monkeypatch.setenv("GLM53_KV_PHASE_LOG_MS", "5")
    host = SimpleNamespace(_prefill_rows=lambda mi: [1, 2, 3])
    for fn in ("_phase_begin", "_phase", "_phase_end"):
        setattr(host, fn, getattr(H, fn).__get__(host))
    ph = host._phase_begin("load")
    time.sleep(0.003)
    host._phase(ph, "start_load_kv")
    host._kda_import_phase = {"store_load": 0.004}
    time.sleep(0.004)
    host._phase(ph, "kda_imports")
    lines = []
    monkeypatch.setattr(K.logger, "info", lambda msg, *a: lines.append(msg % a))  # the plugin logger does not propagate
    host._phase_end(ph, model_input=object())
    assert set(host.phase_s) == {"load.start_load_kv", "load.kda_imports", "load.kda_store_load"}
    assert host.phase_s["load.start_load_kv"] >= 0.003 and host._kda_import_phase is None
    assert len(lines) == 1 and "step phases (load" in lines[0] and "prefill rows 3" in lines[0], lines
    monkeypatch.setenv("GLM53_KV_PHASE_LOG_MS", "0")
    assert host._phase_begin("load") is None  # off: no timing at all


@pytest.mark.parametrize("ahead", ["1", "0"])
def test_resolve_ahead_puts_a_checkpoint_in_ram_while_an_earlier_file_is_written(tmp_path, monkeypatch, ahead):
    """GLM53_KDA_CKPT_RESOLVE_AHEAD (the k4a / k4b 32,000-token repeats resolved their stale-snapshot 31,872 checkpoint
    inline for 612 / 873 ms; here a slow file write stands in for a busy writer): with it, checkpoint B's producer runs
    on the resolver thread while
    the writer is still writing A's file, and a hit on B is a RAM hit; without it (0), as before, the hit runs B's
    producer itself on the calling thread. Either way every producer runs exactly once and both files are written."""
    import threading

    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "1")
    monkeypatch.setenv("GLM53_KDA_CKPT_RESOLVE_AHEAD", ahead)
    s = K.KDACheckpointStore(str(tmp_path), min_free_bytes=0)
    writing, release = threading.Event(), threading.Event()
    orig = s._write_file

    def slow_write(path, data, blob=None):  # the first file write stalls (a slow NFS write) until released
        if not writing.is_set():
            writing.set()
            release.wait(10)
        orig(path, data, blob)

    monkeypatch.setattr(s, "_write_file", slow_write)
    toks = list(range(1024))
    ran = {"a": [], "b": []}
    s.save_deferred(toks, 256, lambda: (ran["a"].append(threading.current_thread().name), _states(v=1))[1])
    assert writing.wait(5)  # the writer is inside A's file write
    s.save_deferred(toks, 512, lambda: (ran["b"].append(threading.current_thread().name), _states(v=2))[1])
    if ahead == "1":
        deadline = time.time() + 5
        while not ran["b"] and time.time() < deadline:
            time.sleep(0.01)
        hits = s.stats["ram_hits"]
        got = s.load(toks, 512)  # B is in RAM although A's file is still being written
        assert got[0]["S"][0] == 2 and s.stats["ram_hits"] == hits + 1
        assert ran["b"] == ["tt-kda-ckpt-resolver"] and s.stats.get("resolved_ahead", 0) >= 1
    else:
        got = s.load(toks, 512)  # as before: the hit resolves B itself
        assert got[0]["S"][0] == 2 and ran["b"] == [threading.current_thread().name]
    release.set()
    s.flush()
    assert len(ran["a"]) == 1 and len(ran["b"]) == 1
    assert os.path.exists(s.path(toks, 256)) and os.path.exists(s.path(toks, 512))


def test_tail_split_min_checkpoints_a_short_prompt_at_its_last_chunk_end(tmp_path, monkeypatch):
    """GLM53_PF_TAIL_SPLIT_MIN (program 2026-10-28T02:28:53Z (2)): a prompt below it runs its last chunk whole (no
    split at the last grid boundary), so its resume point is the last CHUNK end, the one whose next model chunk
    (GLM53_PF_MODEL_CHUNK, 2,048) reaches the end: a 2,526-token prompt in 2,048-token chunks checkpoints at 2,048
    (a repeat recomputes 478 tokens); at or above the minimum the last-grid-boundary rule stands."""
    from types import SimpleNamespace

    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    monkeypatch.setenv("GLM53_KDA_CKPT_EVERY", "1048576")

    def run(n, tmin):
        monkeypatch.setenv("GLM53_PF_TAIL_SPLIT_MIN", str(tmin))
        prompt = list(range(n))
        chunks = [(s, min(s + 2048, n)) for s in range(0, n, 2048)]
        store = K.KDACheckpointStore(str(tmp_path / f"n{n}t{tmin}"), min_free_bytes=0)
        host = SimpleNamespace(kda_store=store, _kda_export=lambda kv, slot: [{"L": 0, "S": torch.zeros(2)}],
                               _kv_cache_object=None, views={"l": SimpleNamespace(block_size=64)},
                               _tokens_of=lambda rid: prompt, _ckpt_grid=128, _slot_for=lambda *a: 0)  # fmt: skip
        for s, e in chunks:
            host._prefill_rows = lambda mi, s=s, e=e: [(0, "r", s, e)]
            K.TTKVConnectorHost._export_kda_checkpoints(host, SimpleNamespace(input_tokens=torch.tensor([prompt[:e]])))
        return [e for s, e in chunks if store.has(prompt, e)]

    assert run(2526, 8192) == [2048]  # the last chunk end
    assert run(6000, 8192) == [4096]  # (2,048 + 2,048 < 6,000: only the last one)
    assert run(2526, 0) == []  # off: the last grid boundary (2,432) is no chunk end of this plan (the split makes it one)
    assert run(9000, 8192) == []  # at / above the minimum: the old rule (its split chunk end is what it checkpoints)


def test_tail_split_min_reads_the_env():
    import os

    old = os.environ.pop("GLM53_PF_TAIL_SPLIT_MIN", None)
    try:
        assert K.tail_split_min() == 8192  # default on
        os.environ["GLM53_PF_TAIL_SPLIT_MIN"] = "0"
        assert K.tail_split_min() == 0
    finally:
        os.environ.pop("GLM53_PF_TAIL_SPLIT_MIN", None)
        if old is not None:
            os.environ["GLM53_PF_TAIL_SPLIT_MIN"] = old
