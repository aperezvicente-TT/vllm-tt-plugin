"""Stage 12 lever (c) step 3: the warm start after a relaunch (vllm_tt_plugin.lmcache.warm_start). The real
LMCacheEngine with an fs:// cold tier: engine A stores a prompt's chunks (CPU tier + the cold dir), A is destroyed (the
relaunch), engine B's warm-up reads the cold tier's newest objects into its CPU tier -- B's first hit is then served by
the CPU tier with the same bytes; the byte cap / tier fraction stop it early; the KDA RAM tier fills from the newest
checkpoint files up to its budget without evicting; both are off by default."""

from __future__ import annotations

import os
import threading

import pytest
import torch
from lmcache.utils import EngineType, mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import CreateGPUConnector

from vllm_tt_plugin.lmcache import page_contract as pc
from vllm_tt_plugin.lmcache import warm_start as W
from vllm_tt_plugin.lmcache.mock_paged_cache import MockTTPagedLatentCache

from .tt_lmcache_testing import BLOCK_SIZE as BS

T = 1024  # 4 chunks of 256


def _engine(name, remote, model_name=None):
    cfg = LMCacheEngineConfig.from_defaults(
        chunk_size=256, local_cpu=True, max_local_cpu_size=0.5, save_unfull_chunk=True,
        remote_url=f"fs://localhost:0{remote}/", remote_serde="naive", lmcache_instance_id=name,
        extra_config={"gpu_connector": "vllm_tt_plugin.lmcache.connector:VLLMPagedMemTTConnector",
                      "tt_tokens_per_page": BS, "save_only_first_rank": False})  # fmt: skip
    md = pc.lmcache_metadata(kv_connector_extra_config={"tt_tokens_per_page": BS},
                             **({"model_name": model_name} if model_name else {}))  # fmt: skip
    conn = CreateGPUConnector(cfg, md, EngineType.VLLM)
    eng = LMCacheEngineBuilder.get_or_create(name, cfg, md, conn, mock_up_broadcast_fn, mock_up_broadcast_object_fn)
    eng.post_init()
    return eng, conn


def _stored(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    remote = str(tmp_path / "cold")
    os.makedirs(remote)
    g = torch.Generator().manual_seed(4)
    toks = torch.randint(1000, 150000, (T,), generator=g)
    cache = MockTTPagedLatentCache(2 * T // BS + 8, backend="torch")
    for n in cache.layer_names:
        cache.device_caches[n].copy_(torch.randn(cache.device_caches[n].shape, generator=g).to(cache.device_caches[n].dtype))
    slots = lambda b0: torch.tensor([b * BS + i for b in range(b0, b0 + T // BS) for i in range(BS)])  # noqa: E731
    eng, conn = _engine("warmA", remote)
    conn.set_host_xfer_buffer_ops(cache.copy_blocks)
    conn.register_device_kv_caches(cache.device_caches)
    eng.store(toks, kvcaches=cache.host_list, slot_mapping=slots(1), req_id="A")
    for _ in range(200):  # the remote put is asynchronous
        if len(os.listdir(remote)) >= T // 256:
            break
        threading.Event().wait(0.05)
    LMCacheEngineBuilder.destroy("warmA")
    return remote, toks, cache, slots


def test_remote_dirs_parse_the_fs_url():
    class E:
        class config:
            remote_url = "fs://localhost:0/mnt/a/b/, fs://localhost:0/c"

    assert W.remote_dirs(E) == ["/mnt/a/b/", "/c"]
    E.config.remote_url = "lm://host:1234"
    assert W.remote_dirs(E) == []


def test_warm_cpu_tier_then_the_first_hit_is_a_cpu_hit_with_the_same_bytes(tmp_path, monkeypatch):
    remote, toks, cache, slots = _stored(tmp_path, monkeypatch)
    eng, conn = _engine("warmB", remote)
    try:
        conn.set_host_xfer_buffer_ops(cache.copy_blocks)
        conn.register_device_kv_caches(cache.device_caches)
        cpu = eng.storage_manager.storage_backends["LocalCPUBackend"]
        st = W.warm_lmcache_cpu_tier(eng, 1 << 30)
        assert st["loaded"] == st["keys"] == T // 256 and st["stopped"] is None
        assert len(cpu.hot_cache) == T // 256
        assert all(o.get_ref_count() == 1 for o in cpu.hot_cache.values())  # held by the hot cache only
        again = W.warm_lmcache_cpu_tier(eng, 1 << 30)
        assert again["keys"] == 0  # already resident
        remote_get = eng.storage_manager.storage_backends["RemoteBackend"].batched_get_blocking
        calls = []
        eng.storage_manager.storage_backends["RemoteBackend"].batched_get_blocking = lambda ks: calls.append(ks) or remote_get(ks)
        hit = eng.lookup(toks)
        ret = eng.retrieve(toks[:hit], kvcaches=cache.host_list, slot_mapping=slots(T // BS + 4)[:hit], req_id="B")
        assert hit == T and int(ret.sum()) == T and not calls  # served by the CPU tier
        for n in cache.layer_names:
            assert torch.equal(cache.device_tokens(n, slots(T // BS + 4)).view(torch.int16),
                               cache.device_tokens(n, slots(1)).view(torch.int16))  # fmt: skip
    finally:
        LMCacheEngineBuilder.destroy("warmB")


def test_warm_cpu_tier_stops_at_the_byte_cap(tmp_path, monkeypatch):
    remote, _toks, _cache, _slots = _stored(tmp_path, monkeypatch)
    eng, _conn = _engine("warmC", remote)
    try:
        one = os.path.getsize(next(os.path.join(remote, f) for f in os.listdir(remote)))
        st = W.warm_lmcache_cpu_tier(eng, 2 * one + one // 2)
        assert st["loaded"] == 2 and st["bytes"] <= 2 * one + one // 2
    finally:
        LMCacheEngineBuilder.destroy("warmC")


def test_start_is_off_by_default_and_never_raises(monkeypatch):
    monkeypatch.delenv("GLM53_LMCACHE_WARM_GB", raising=False)
    monkeypatch.delenv("GLM53_KDA_CKPT_WARM", raising=False)
    assert W.start(object(), None) is None
    monkeypatch.setenv("GLM53_LMCACHE_WARM_GB", "1")
    stats: dict = {}
    t = W.start(object(), None, stats)  # no storage manager: reported, not raised
    t.join(10)
    assert stats["done"] and stats["lmcache"]["stopped"] == "no fs remote / CPU tier"


def test_warm_kda_ram_tier_newest_first_up_to_the_budget(tmp_path):
    from collections import OrderedDict

    root = tmp_path / "kda"
    root.mkdir()
    blobs = {}
    for i in range(5):
        p = str(root / f"k{i}.pt")
        open(p, "wb").close()
        os.utime(p, (1000 + i, 1000 + i))  # k4 newest
        blobs[p] = {"states": torch.zeros(1024, dtype=torch.uint8), "num_tokens": 64}

    class Store:
        ram_bytes = 3 * 1024 + 100
        _ram: OrderedDict = OrderedDict()

        def __init__(self):
            self.root = str(root)
            self._ram = OrderedDict()

        def _ram_put(self, path, states):
            self._ram[path] = (states, states.numel())

    s = Store()
    st = W.warm_kda_ram_tier(s, load=lambda p: blobs[p])
    assert st["loaded"] == 3 and st["stopped"] == "RAM tier full"
    assert sorted(os.path.basename(p) for p in s._ram) == ["k2.pt", "k3.pt", "k4.pt"]


def test_warm_skips_objects_of_another_key_namespace(tmp_path, monkeypatch):
    """S2 (program 2026-10-26T16:30:44Z): the cold dir holds the k line's ROW-MAJOR objects (the plain namespace);
    a tile-mode engine ("#tt-tile-v2" in its keys' model_name) must not pull any of them into its CPU tier"""
    remote, _toks, _cache, _slots = _stored(tmp_path, monkeypatch)
    n_files = len([f for f in os.listdir(remote) if f.endswith(".data")])
    assert n_files == T // 256
    eng, _conn = _engine("warmNS", remote, model_name="zai-org/GLM-5.3-Flash#tt-tile-v2")
    try:
        st = W.warm_lmcache_cpu_tier(eng, 1 << 30)
        assert st["keys"] == 0 and st["loaded"] == 0 and st["skipped_namespace"] == n_files, st
    finally:
        LMCacheEngineBuilder.destroy("warmNS")
