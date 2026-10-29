# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""cf3h-e (program 2026-10-24T20:38:48Z (ii), approved 20:43:55Z): GLM53_KDA_CKPT_PROVENANCE=restore.

cf3h-d's window: the first served G[:32000] stored LMCache pages through 32,000 with no KDA checkpoint inside it, so
every later G[:16100] matched LMCache 16,000 against the 7,936 checkpoint, was barred by the provenance rule and never
checkpointed 16,000 (16,100 repeat 4.27 s vs cf3h-b's 0.80 s). Under restore, such a request removes LMCache's chunk
keys for the span it recomputes, [w, m), from every tier at its first prefill step; its own pages are then stored there
and its checkpoints are exported as a first-time request's. Guards (served kv_load_failure_policy = fail): another
request's admitted load or another running prefill sharing a chunk key past w, or a key still held after the removal,
keeps the bar. Real LMCache tiers here: local CPU, local disk and the fs remote dir as served."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.token_database import ChunkedTokenDatabase

from vllm_tt_plugin import kv_connector as K
from vllm_tt_plugin.lmcache import page_contract as pc

CHUNK = pc.LMCACHE_CHUNK_TOKENS


@pytest.fixture(autouse=True)
def _clean_registries(monkeypatch):
    # LMCache's token database seeds its NONE_HASH from PYTHONHASHSEED at construction; the restart test builds a second
    # one and compares the keys, so the seed is pinned here (review round 17 O6: without it that test failed)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    regs = (K._KDA_PROVENANCE, K._KDA_RESTORE, K._LMC_LOADS, K._LMC_WRITERS, K._RESTORED_SPANS, K._POST_RESTORE_CKPTS)
    for r in regs:
        r.clear()
    yield
    for r in regs:
        r.clear()


def _tiers(base: str):
    """the served tiers: local CPU, local disk, the fs remote dir (legacy url form, as the served yaml)"""
    os.makedirs(os.path.join(base, "cold"), exist_ok=True)
    os.makedirs(os.path.join(base, "disk"), exist_ok=True)
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=CHUNK, local_cpu=True, max_local_cpu_size=0.25, local_disk=f"file://{base}/disk/",
        max_local_disk_size=0.25, remote_url=f"fs://localhost:0{base}/cold", remote_serde="naive",
        save_unfull_chunk=True, lmcache_instance_id=f"s9-restore-{os.getpid()}-{time.time_ns()}")  # fmt: skip
    md = pc.lmcache_metadata()
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return config, md, loop, CreateStorageBackends(config, md, loop, "cpu")


def _close(backends, loop):
    for b in backends.values():
        try:
            b.close()
        except Exception:  # noqa: BLE001
            pass
    loop.call_soon_threadsafe(loop.stop)


def _remote(backends):
    return next(b for name, b in backends.items() if "Remote" in name or "Remote" in type(b).__name__)


def _seed(backends, md, keys, tiers=("LocalCPUBackend", "LocalDiskBackend", "remote")):
    """put every key into the given tiers and wait until each tier holds it"""
    cpu = backends["LocalCPUBackend"]
    targets = [_remote(backends) if t == "remote" else backends[t] for t in tiers]
    for i, k in enumerate(keys):
        obj = cpu.allocate(md.get_shapes(CHUNK), md.get_dtypes(), fmt=MemoryFormat.KV_MLA_FMT)
        obj.tensor.reshape(-1).view(torch.uint8).fill_(i + 1)
        for t in targets:
            t.submit_put_task(k, obj)
    deadline = time.time() + 30
    while time.time() < deadline and not all(t.contains(k) for t in targets for k in keys):
        time.sleep(0.05)
    assert all(t.contains(k) for t in targets for k in keys), "a tier did not take the seeded chunks"


def _held(backends, key) -> list:
    return [name for name, b in backends.items() if b.contains(key)]


class _Host(K.TTKVConnectorHost):
    """the worker-side host with only what the provenance restore reads"""

    def __init__(self, backends=None, db=None, tokens_of=None):  # noqa: D107 -- no device state
        engine = SimpleNamespace(storage_manager=SimpleNamespace(storage_backends=backends), token_database=db)
        self.connector = SimpleNamespace(_lmcache_engine=SimpleNamespace(lmcache_engine=engine))
        self._tokens_of = tokens_of or (lambda rid: None)
        self._len_of = lambda rid: None
        for c in ("kda_ckpt_skipped_provenance", "kda_provenance_restored", "kda_provenance_restored_tokens",
                  "kda_provenance_restore_skipped_load", "kda_provenance_restore_skipped_writer",
                  "kda_provenance_restore_failed"):  # fmt: skip
            setattr(self, c, 0)


def _rows(*rows):
    """a prefill build: rows (req_id, start, prompt end)"""
    return SimpleNamespace(row_req_ids=[r[0] for r in rows], input_positions=torch.tensor([r[1] for r in rows]),
                           prompt_lens=torch.tensor([r[2] for r in rows]))  # fmt: skip


def _prompt(n, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 100000, (n,), generator=g).tolist()


def _keys(db, tokens, m):
    return [(int(s), int(e), k) for s, e, k in db.process_tokens(tokens=list(tokens[:m]))]


def test_restore_mode_marks_a_restore_candidate_with_its_load_window(tmp_path, monkeypatch):
    """the scheduler side: LMCache 512 tokens, KDA hit 128 -> barred, and under restore a candidate with w = the hit
    rounded up to LMCache's chunk (its load window) and m = the full match; mode 1 = the bar alone; mode 0 = no rule"""

    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    store = K.KDACheckpointStore(str(tmp_path), min_free_bytes=0)
    prompt = _prompt(586)
    store.save(prompt, 128, [{"L": 0, "S": torch.zeros(4)}])

    class Inner:
        def get_num_new_matched_tokens(self, request, n):
            return 512, False

    def ask(rid, mode):
        monkeypatch.setenv("GLM53_KDA_CKPT_PROVENANCE", mode)
        c = K.TTPageAlignedConnector(Inner(), 128, store)
        return c.get_num_new_matched_tokens(SimpleNamespace(request_id=rid, prompt_token_ids=prompt), 0)

    assert ask("r", "restore") == (128, False)
    assert K._KDA_PROVENANCE["r"] == 512 and K._KDA_RESTORE["r"][:2] == (128, 512) and K._KDA_RESTORE["r"][2] is prompt
    assert ask("b", "1") == (128, False) and K._KDA_PROVENANCE["b"] == 512 and "b" not in K._KDA_RESTORE
    assert ask("o", "0") == (128, False) and "o" not in K._KDA_PROVENANCE and "o" not in K._KDA_RESTORE
    monkeypatch.setenv("GLM53_KDA_CKPT_PROVENANCE", "yes")
    with pytest.raises(ValueError, match="GLM53_KDA_CKPT_PROVENANCE"):
        K.kda_ckpt_provenance_mode()


def test_restore_replaces_the_span_on_every_tier_and_lifts_the_bar(tmp_path):
    config, md, loop, backends = _tiers(str(tmp_path))
    try:
        db = ChunkedTokenDatabase(config, md)
        toks = _prompt(6 * CHUNK)
        w, m = 2 * CHUNK, 5 * CHUNK  # KDA hit 256, LMCache match 640: R recomputes [256, 640)
        keys = _keys(db, toks, m)
        _seed(backends, md, [k for _s, _e, k in keys])
        K._KDA_PROVENANCE["R"] = m
        host = _Host(backends, db)
        assert host._restore_span("R", w, m, toks) is True
        for s, _e, k in keys:  # [w, m) gone from all three tiers, the chunks below w untouched
            assert _held(backends, k) == [] if s >= w else len(_held(backends, k)) == 3, (s, _held(backends, k))
        assert "R" not in K._KDA_PROVENANCE  # the bar is lifted: R's checkpoints export as a first-time request's
        assert (host.kda_provenance_restored, host.kda_provenance_restored_tokens) == (1, m - w)
    finally:
        _close(backends, loop)


def test_concurrent_loads_and_writers_guard_the_span(tmp_path):
    """the CONCURRENT case (20:43:55Z condition 1): R (restore candidate, span [256, 640)) and S share the prefix;
    S's admitted load window 512 crosses R's span -> R keeps its bar, nothing is removed, skipped_load counts it. A load
    that ends at w, or one of another prompt, does not block. After S's load step S is a writer (its prefill pages are
    being stored) -> a second candidate keeps its bar (skipped_writer); once S's last chunk is saved, a third restores."""
    config, md, loop, backends = _tiers(str(tmp_path))
    try:
        db = ChunkedTokenDatabase(config, md)
        toks = _prompt(6 * CHUNK)
        other = _prompt(6 * CHUNK, seed=99)
        w, m = 2 * CHUNK, 5 * CHUNK
        keys = [k for _s, _e, k in _keys(db, toks, m)]
        _seed(backends, md, keys)
        s_toks = toks[: 4 * CHUNK + 7]  # S: the same prefix, a shorter prompt
        host = _Host(backends, db, tokens_of=lambda rid: {"S": s_toks}.get(rid))
        K._LMC_LOADS.update({"S": (s_toks, 4 * CHUNK), "S_at_w": (toks, w), "U": (other, m)})
        for rid in ("R1", "R2", "R3"):
            K._KDA_PROVENANCE[rid] = m
        assert host._restore_span("R1", w, m, toks) is False
        assert host.kda_provenance_restore_skipped_load == 1 and all(len(_held(backends, k)) == 3 for k in keys)
        assert K._KDA_PROVENANCE["R1"] == m
        # S's step runs: its load is done, it becomes a writer until its last prefill chunk is saved
        host._provenance_step(_rows(("S", 4 * CHUNK, len(s_toks))))
        assert "S" not in K._LMC_LOADS and "S" in K._LMC_WRITERS
        assert host._restore_span("R2", w, m, toks) is False
        assert host.kda_provenance_restore_skipped_writer == 1 and all(len(_held(backends, k)) == 3 for k in keys)
        K._LMC_WRITERS.pop("S")  # (TTKVConnectorHost._save_now after S's last chunk)
        assert host._restore_span("R3", w, m, toks) is True  # S_at_w (window = w) and U (another prompt) never blocked
        assert host.kda_provenance_restored == 1 and "R3" not in K._KDA_PROVENANCE and K._KDA_PROVENANCE["R2"] == m
        assert [len(_held(backends, k)) for k in keys] == [3, 3, 0, 0, 0]
    finally:
        _close(backends, loop)


def test_a_restore_candidate_is_handled_at_its_first_prefill_step(tmp_path):
    """_provenance_step: the candidate's own load and writer entries never block it; the step pops the candidate"""
    config, md, loop, backends = _tiers(str(tmp_path))
    try:
        db = ChunkedTokenDatabase(config, md)
        toks = _prompt(6 * CHUNK)
        w, m = 2 * CHUNK, 5 * CHUNK
        keys = [k for _s, _e, k in _keys(db, toks, m)]
        _seed(backends, md, keys)
        host = _Host(backends, db, tokens_of=lambda rid: toks)
        K._KDA_PROVENANCE["R"] = m
        K._KDA_RESTORE["R"] = (w, m, toks)
        K._LMC_LOADS["R"] = (toks, w)
        host._provenance_step(_rows(("R", w, len(toks))))
        assert "R" not in K._KDA_RESTORE and "R" not in K._KDA_PROVENANCE and host.kda_provenance_restored == 1
        assert [len(_held(backends, k)) for k in keys] == [3, 3, 0, 0, 0]
    finally:
        _close(backends, loop)


def test_a_failed_removal_keeps_the_bar():
    class Stuck:  # a tier that cannot remove
        def contains(self, key, pin=False):
            return True

        def remove(self, key, force=True):
            raise OSError("read-only file system")

    class DB:
        def process_tokens(self, tokens=None, **kw):
            for s in range(0, len(tokens), CHUNK):
                yield s, min(s + CHUNK, len(tokens)), ("key", s)

    toks = list(range(5 * CHUNK))
    K._KDA_PROVENANCE["F"] = 5 * CHUNK
    host = _Host({"RemoteBackend": Stuck()}, DB())
    assert host._restore_span("F", 2 * CHUNK, 5 * CHUNK, toks) is False
    assert host.kda_provenance_restore_failed == 1 and K._KDA_PROVENANCE["F"] == 5 * CHUNK
    host = _Host(None, None)  # no LMCache engine (the ExampleConnector / mock path)
    assert host._restore_span("F", 2 * CHUNK, 5 * CHUNK, toks) is False and host.kda_provenance_restore_failed == 1


def test_removed_keys_stay_removed_on_the_remote_dir_after_a_restart(tmp_path):
    """the restart case (20:43:55Z condition 1): the fs remote dir is the restart-reuse tier; a key removed there is
    absent for the next engine on the same dir, a kept key is still there"""
    base = str(tmp_path)
    config, md, loop, backends = _tiers(base)
    db = ChunkedTokenDatabase(config, md)
    toks = _prompt(6 * CHUNK)
    w, m = 2 * CHUNK, 5 * CHUNK
    keys = [k for _s, _e, k in _keys(db, toks, m)]
    try:
        _seed(backends, md, keys)
        K._KDA_PROVENANCE["R"] = m
        assert _Host(backends, db)._restore_span("R", w, m, toks) is True
    finally:
        _close(backends, loop)
    config2, md2, loop2, backends2 = _tiers(base)  # the restart: a new engine's tiers on the same dirs
    try:
        db2 = ChunkedTokenDatabase(config2, md2)
        assert [k for _s, _e, k in _keys(db2, toks, m)] == keys  # the same keys (the hash seed is the process's)
        remote = _remote(backends2)
        assert [remote.contains(k) for k in keys] == [True, True, False, False, False]
    finally:
        _close(backends2, loop2)


def test_shares_chunk_past():
    a = list(range(1000))
    assert K._shares_chunk_past(a, a[:600], 256, CHUNK)
    assert not K._shares_chunk_past(a, a[:256], 256, CHUNK)  # ends at w: no key past w
    assert K._shares_chunk_past(a, a[:300], 256, CHUNK)  # a partial chunk past w (its key hashes [0, 300))
    b = a[:300] + [7] * 700
    assert not K._shares_chunk_past(a, b, 256, CHUNK)  # differs inside the first chunk past w
    assert K._shares_chunk_past(a, a[:383] + [7], 256, CHUNK) is False


def test_mixed_provenance_imports_are_counted(tmp_path, monkeypatch):
    """ruling A (2026-10-24T20:58:54Z): after a restore of [256, 640), a hit on a checkpoint deeper than 256 that was
    written BEFORE the restore (the first writer's 640) is mixed provenance and counts; a checkpoint exported after the
    restore on that prefix (512) pairs with the current pages and does not; a prefix that does not share the span
    (another prompt) never counts"""
    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    monkeypatch.setenv("GLM53_KDA_CKPT_EVERY", "0")
    config, md, loop, backends = _tiers(str(tmp_path / "tiers"))
    try:
        db = ChunkedTokenDatabase(config, md)
        toks = _prompt(6 * CHUNK)
        other = _prompt(6 * CHUNK, seed=5)
        w, m = 2 * CHUNK, 5 * CHUNK
        _seed(backends, md, [k for _s, _e, k in _keys(db, toks, m)])
        store = K.KDACheckpointStore(str(tmp_path / "kda"), min_free_bytes=0)
        st = [{"L": 0, "S": torch.zeros(4)}]
        store.save(toks, 640, st)  # the first writer's deeper checkpoint, before the restore
        store.save(other, 640, st)
        host = _Host(backends, db)
        host.kda_store, host._kv_cache_object, host._ckpt_grid = store, None, CHUNK
        host._kda_import = lambda kv, slot, states: None
        host._kda_export = lambda kv, slot: st
        host._slot_for = lambda *a: 0
        host.views = {"l": SimpleNamespace(block_size=64)}
        K._KDA_PROVENANCE["R"] = m
        assert host._restore_span("R", w, m, toks) is True

        def imp(rid, t, start):
            so = SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(req_id=rid, num_computed_tokens=start)],
                                 scheduled_cached_reqs=None)  # fmt: skip
            host._import_kda_checkpoints(so, SimpleNamespace(row_req_ids=[rid], input_positions=torch.tensor([start]),
                                                             prompt_lens=torch.tensor([len(t)]),
                                                             input_tokens=torch.tensor([t])))  # fmt: skip

        imp("X", toks, 640)
        assert host.kda_provenance_mixed_imports == 1
        imp("Y", other, 640)  # another prompt: not on the replaced span
        assert host.kda_provenance_mixed_imports == 1
        # R's own checkpoint after the restore (its prefill chunk ending at 512): pairs with the current pages
        host._prefill_rows = lambda mi: [(0, "R", 384, 512)]
        host._tokens_of = lambda rid: toks
        K.TTKVConnectorHost._export_kda_checkpoints(host, SimpleNamespace(input_tokens=torch.tensor([toks[:512]])))
        assert store.has(toks, 512) and K.KDACheckpointStore.key(toks[:512], 512) in K._POST_RESTORE_CKPTS
        del host._prefill_rows
        imp("Z", toks, 512)
        assert host.kda_provenance_mixed_imports == 1
    finally:
        K._RESTORED_SPANS.clear()
        K._POST_RESTORE_CKPTS.clear()
        _close(backends, loop)


def test_a_zero_hit_restore_candidate_saves_from_its_window(tmp_path, monkeypatch):
    """cf3h-e mock r3 (2026-10-24T21:17Z): a 16,100-token prefix of a stored 32,000-token prompt matched LMCache 16,000
    and no KDA checkpoint (hit 0); the worker removed [0, 16000), but LMCache's adapter saves a request from its load
    spec's cached count, which stayed the raw 16,000 because vLLM took 0 external tokens -> a hole, the repeat hit 0.
    The candidate's spec count is now its window w (0 here); a request under the bar alone (mode 1) keeps the raw count."""
    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    store = K.KDACheckpointStore(str(tmp_path), min_free_bytes=0)
    prompt = _prompt(16100)

    class Engine:
        def __init__(self):
            self.load_specs = {}
            self._lmcache_chunk_size = 128

    class Inner:
        def __init__(self):
            self._lmcache_engine = Engine()
            self.updated = []

        def get_num_new_matched_tokens(self, request, n):
            self._lmcache_engine.load_specs[request.request_id] = SimpleNamespace(
                vllm_cached_tokens=n, lmcache_cached_tokens=16000, can_load=False)
            return 16000, False

        def update_state_after_alloc(self, request, blocks, n):
            self.updated.append((request.request_id, n))

    for mode, want in (("restore", 0), ("1", 16000)):
        monkeypatch.setenv("GLM53_KDA_CKPT_PROVENANCE", mode)
        inner = Inner()
        c = K.TTPageAlignedConnector(inner, 128, store)
        req = SimpleNamespace(request_id=f"r-{mode}", prompt_token_ids=prompt, num_tokens=16100)
        ext, _ = c.get_num_new_matched_tokens(req, 0)
        assert ext == 0 and K._KDA_PROVENANCE[req.request_id] == 16000
        c.update_state_after_alloc(req, None, 0)
        assert inner._lmcache_engine.load_specs[req.request_id].lmcache_cached_tokens == want, mode
        assert inner.updated == [(req.request_id, 0)]
