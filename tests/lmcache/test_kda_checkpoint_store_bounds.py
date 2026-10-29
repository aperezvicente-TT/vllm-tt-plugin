# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The KDA checkpoint store is disk-bounded (2026-10-23: ~148 MB per checkpoint, 7.6 GB in 9 min of serving; a full
root disk SIGBUS-killed the next launch): an LRU byte cap that never evicts a recently used checkpoint (the scheduler
counted it with has(); the worker loads it a step later), and no save under the free-space floor."""

import os
import time

import pytest
import torch

from vllm_tt_plugin.kv_connector import KDACheckpointStore


@pytest.fixture(autouse=True)
def _sync_checkpoint_writes(monkeypatch):
    """these tests move / delete checkpoint files right after a save: synchronous writes, no RAM tier (the async writer has its own
    tests, test_kda_ckpt_store_async.py)"""
    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    monkeypatch.setenv("GLM53_KDA_CKPT_RAM_GB", "0")  # disk semantics: no RAM tier behind the files

MB = 1 << 20


def _states(mb):
    return [{"L": 0, "S": torch.zeros(mb * MB // 4, dtype=torch.float32)}]


def _age(store, ids, n, seconds):
    t = time.time() - seconds
    os.utime(store.path(ids, n), (t, t))


def test_lru_cap_evicts_the_oldest_unused_checkpoints(tmp_path):
    s = KDACheckpointStore(str(tmp_path), max_bytes=3 * MB + MB // 2, min_free_bytes=0)
    ids = list(range(512))
    for n in (64, 128, 192):
        s.save(ids, n, _states(1))
        _age(s, ids, n, 3600 - n)  # 64 oldest
    assert s.has(ids, 64)  # a recent use refreshes the LRU clock of 64
    s.save(ids, 256, _states(1))
    assert s.has(ids, 64) and not s.has(ids, 128) and s.has(ids, 192) and s.has(ids, 256)
    assert s.evicted == 1


def test_recently_used_checkpoints_are_never_evicted(tmp_path):
    s = KDACheckpointStore(str(tmp_path), max_bytes=MB, min_free_bytes=0)
    ids = list(range(512))
    s.save(ids, 64, _states(1))
    s.save(ids, 128, _states(1))  # over the cap, but 64 was used < EVICT_MIN_AGE_S ago
    assert s.has(ids, 64) and s.has(ids, 128) and s.evicted == 0


def test_no_save_under_the_free_space_floor(tmp_path):
    s = KDACheckpointStore(str(tmp_path), max_bytes=1 << 40, min_free_bytes=1 << 60)
    ids = list(range(512))
    assert not s.can_save()
    assert s.save(ids, 64, _states(1)) is None and not s.has(ids, 64)
    assert s.saves_skipped_low_disk >= 1


def test_env_bounds(tmp_path, monkeypatch):
    monkeypatch.setenv(KDACheckpointStore.MAX_GB_ENV, "2")
    monkeypatch.setenv(KDACheckpointStore.MIN_FREE_GB_ENV, "0.5")
    s = KDACheckpointStore(str(tmp_path))
    assert s.max_bytes == 2 << 30 and s.min_free_bytes == 512 * MB


def test_decode_boundaries_follow_the_checkpoint_period(monkeypatch):
    """GLM53_KDA_CKPT_EVERY bounds the decode-time checkpoints too: only boundaries at a multiple of N (n29h5c's
    ITL P99 405 ms was a 148 MB export every 64 generated tokens); unset / 0 keeps every boundary"""
    from vllm_tt_plugin.kv_connector import decode_boundary_wanted

    monkeypatch.delenv("GLM53_KDA_CKPT_EVERY", raising=False)
    assert all(decode_boundary_wanted(b) for b in (64, 128, 192, 8192))
    monkeypatch.setenv("GLM53_KDA_CKPT_EVERY", "8192")
    assert [b for b in range(64, 3 * 8192 + 1, 64) if decode_boundary_wanted(b)] == [8192, 16384, 24576]
    assert decode_boundary_wanted(192, every=0) and not decode_boundary_wanted(192, every=128)


def test_short_prompts_are_marked_skip_save_by_the_scheduler_proxy(monkeypatch):
    """GLM53_LMCACHE_MIN_STORE_TOKENS = N: the proxy sets LMCache's per-request lmcache.skip_save on a prompt shorter
    than N before the inner connector tracks it; longer prompts and N = 0 are untouched; a lookup still runs"""
    from types import SimpleNamespace

    from vllm_tt_plugin.kv_connector import TTPageAlignedConnector, lmcache_min_store_tokens

    seen = []
    inner = SimpleNamespace(get_num_new_matched_tokens=lambda r, n: (seen.append(r.request_id), (0, False))[1])
    proxy = TTPageAlignedConnector(inner, block_size=64)

    def req(rid, n, extra=None):
        return SimpleNamespace(request_id=rid, num_prompt_tokens=n, prompt_token_ids=list(range(n)),
                               sampling_params=SimpleNamespace(extra_args=extra))

    monkeypatch.delenv("GLM53_LMCACHE_MIN_STORE_TOKENS", raising=False)
    assert lmcache_min_store_tokens() == 0
    r0 = req("off", 128)
    proxy.get_num_new_matched_tokens(r0, 0)
    assert r0.sampling_params.extra_args is None

    monkeypatch.setenv("GLM53_LMCACHE_MIN_STORE_TOKENS", "1024")
    short, long_ = req("short", 128, {"kv_transfer_params": {"lmcache.tag": "x"}}), req("long", 2048)
    proxy.get_num_new_matched_tokens(short, 0)
    proxy.get_num_new_matched_tokens(long_, 0)
    assert short.sampling_params.extra_args["kv_transfer_params"] == {"lmcache.tag": "x", "lmcache.skip_save": True}
    assert long_.sampling_params.extra_args is None
    assert seen == ["off", "short", "long"]  # every request is still looked up
