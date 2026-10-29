# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""GLM53_KV_STORE_WORKER: the process's FIRST store can go through the worker path. The GPU connector's host buffers
are bound only inside LMCache's own to_gpu / from_gpu, and that path reads the pages before any LMCache call; a served
image mock died there ('kvcaches' must be passed in kwargs or bound beforehand) at its first deferred store. The host
now binds the buffers LMCache was registered with first. The real VLLMPagedMemTTConnector over the mock paged cache;
no device, no vLLM engine."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import vllm_tt_plugin.kv_connector as K
from vllm_tt_plugin.lmcache import page_contract as pc
from vllm_tt_plugin.lmcache.connector import VLLMPagedMemTTConnector
from vllm_tt_plugin.lmcache.mock_paged_cache import MockTTPagedLatentCache

from .test_store_worker import _Conn, _meta


def _host(monkeypatch, cache: MockTTPagedLatentCache):
    monkeypatch.setenv("GLM53_KV_STORE_WORKER", "1")
    gpu = VLLMPagedMemTTConnector.from_metadata(pc.lmcache_metadata())
    gpu.set_host_xfer_buffer_ops(cache.copy_blocks)
    gpu.register_device_kv_caches(cache.device_caches)
    h = K.TTKVConnectorHost.__new__(K.TTKVConnectorHost)
    h.connector, h.tt_gpu_connector = _Conn(), gpu
    h.pooled_pages, h._generic_sync = False, False
    h.views = {"l0": SimpleNamespace(block_size=64)}
    h._lock, h._metadata = threading.Lock(), (lambda so: so)
    h._step_context = lambda mi: (None, {"l0": None})
    h.connector_caches = dict(cache.host_buffers)  # what the host registers with LMCache (registration order)
    h._deferred_save, h._held_pages = None, []
    h._enable_store_worker(h.connector)
    return h


def test_the_first_store_on_the_worker_binds_the_registered_host_buffers(monkeypatch):
    cache = MockTTPagedLatentCache(48, seed=7)
    h = _host(monkeypatch, cache)
    assert h.tt_gpu_connector.kvcaches is None  # the served process at its first store: nothing bound yet
    h._save_pages(_meta(64, list(range(64 * 4))), None)  # pages 1-3 past the saved first page
    h.store_worker.drain()
    assert h.store_worker.stats["done"] == 1
    bound = h.tt_gpu_connector.kvcaches
    assert [t.data_ptr() for t in bound] == [t.data_ptr() for t in cache.host_buffers.values()]


def test_a_store_after_lmcache_bound_the_buffers_keeps_that_binding(monkeypatch):
    cache = MockTTPagedLatentCache(48, seed=8)
    h = _host(monkeypatch, cache)
    h.tt_gpu_connector.initialize_kvcaches_ptr(kvcaches=cache.host_list)  # LMCache's own call came first
    first = h.tt_gpu_connector.kvcaches
    h._save_pages(_meta(64, list(range(64 * 4))), None)
    h.store_worker.drain()
    assert h.tt_gpu_connector.kvcaches is first and h.store_worker.stats["done"] == 1


def test_the_worker_runs_with_pooled_pages_and_its_jobs_carry_the_pooled_keys(monkeypatch):
    """Stage 12 cf3h-c (program 2026-10-24T02:20:57Z): with GLM53_KV_POOLED_PAGES the worker is no longer refused;
    the engine thread's prefetch reads the pages' pooled keys through the pooled source (the stored step's map set
    first) and the job carries them"""
    import torch

    cache = MockTTPagedLatentCache(48, seed=9)
    store = {n: torch.randint(0, 256, (48, 4096), dtype=torch.uint8) for n in cache.layer_names}
    calls = []

    def source(ids):
        calls.append((threading.current_thread().name, list(ids)))
        return [store[n][torch.as_tensor(ids)] for n in cache.layer_names]

    h = _host(monkeypatch, cache)
    h.pooled_pages = True
    h.tt_gpu_connector.enable_pooled_pages(source)
    h._enable_store_worker(h.connector)
    assert h.store_worker is not None, "the worker runs with pooled pages"
    seen = {}
    orig = h.tt_gpu_connector.prefetch_store_pages

    def spy(pages):
        seen["map_input"] = h._pooled_map_input
        job = orig(pages)
        seen["job"] = job
        return job

    h.tt_gpu_connector.prefetch_store_pages = spy
    mi = SimpleNamespace(tag="stored step")
    h._save_pages(_meta(64, list(range(64 * 4))), mi)
    h.store_worker.drain()
    assert seen["map_input"] is mi  # the pooled source maps this step's columns
    assert calls and calls[0][0] == threading.current_thread().name  # read on the engine thread
    job = seen["job"]
    assert job["pooled"] is not None and len(job["pooled"]) == len(cache.layer_names)
    for j, n in enumerate(cache.layer_names):
        assert torch.equal(job["pooled"][j], store[n][torch.as_tensor(calls[0][1])])
