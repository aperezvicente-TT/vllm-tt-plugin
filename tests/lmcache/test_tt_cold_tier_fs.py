# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 12 (program 2026-10-23T13:07:10Z, n29h7s1): the LMCache COLD level on the share (LMCache-tt's fs remote
backend) as the served tier yaml configures it. LMCache-tt's legacy remote_url parser (connector.parse_remote_url)
REQUIRES a host and a port, so its own docstring example ``fs:///path`` is refused ("Invalid URL ...: missing host")
and the RemoteBackend runs without a connection (n29h7s1 served RAM + NVMe only). Store + retrieve contract-sized page
chunks through the backends LMCache builds (CreateStorageBackends, as the served engine): the broken form, the
working legacy form ``fs://localhost:0/<path>`` (the adapter uses only the path), and the non-deprecated plugin form
(remote_storage_plugins [fs] + extra_config remote_storage_plugin.fs.base_path)."""

from __future__ import annotations

import asyncio
import os
import threading
import time

import pytest
import torch
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend import CreateStorageBackends

from vllm_tt_plugin.lmcache import page_contract as pc

CHUNK = pc.LMCACHE_CHUNK_TOKENS


def _key(i: int) -> CacheEngineKey:
    return CacheEngineKey(model_name="zai-org/GLM-5.3-Flash", world_size=1, worker_id=0, chunk_hash=5000 + i,
                          dtype=pc.LATENT_DTYPE)  # fmt: skip


def _backends(**kw):
    config = LMCacheEngineConfig.from_defaults(chunk_size=CHUNK, local_cpu=True, max_local_cpu_size=0.25,
                                               remote_serde="naive", lmcache_instance_id="s12-cold-fs", **kw)  # fmt: skip
    md = pc.lmcache_metadata()
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return CreateStorageBackends(config, md, loop, "cpu"), md, loop


def _remote(backends):
    return next(b for name, b in backends.items() if name.startswith("RemoteBackend") or "Remote" in type(b).__name__)


def _roundtrip(backends, md, base: str, n: int = 3) -> list:
    cpu, remote = backends["LocalCPUBackend"], _remote(backends)
    shapes, dtypes = md.get_shapes(CHUNK), md.get_dtypes()
    datas, keys = [], [_key(i) for i in range(n)]
    for i, k in enumerate(keys):
        obj = cpu.allocate(shapes, dtypes, fmt=MemoryFormat.KV_MLA_FMT)
        g = torch.Generator().manual_seed(77 + i)
        data = torch.randint(0, 256, (obj.get_size(),), generator=g, dtype=torch.uint8)
        obj.tensor.reshape(-1).view(torch.uint8).copy_(data)
        datas.append(data)
        remote.submit_put_task(k, obj)
    deadline = time.time() + 30
    while time.time() < deadline and not all(remote.contains(k) for k in keys):
        time.sleep(0.05)
    assert all(remote.contains(k) for k in keys), "the cold level did not take the chunks"
    files = [os.path.join(dp, f) for dp, _d, fs in os.walk(base) for f in fs]
    assert len(files) >= n, files
    out = []
    for k, data in zip(keys, datas, strict=True):
        got = remote.get_blocking(k)
        assert got is not None, k
        assert torch.equal(got.tensor.reshape(-1).view(torch.uint8)[: data.numel()], data)
        got.ref_count_down()
        out.append(k)
    return files


def _close(backends, loop):
    for b in backends.values():
        try:
            b.close()
        except Exception:
            pass
    loop.call_soon_threadsafe(loop.stop)


def test_fs_url_without_host_is_refused_by_lmcache(tmp_path):
    """the n29h7s1 form: LMCache builds the RemoteBackend but never connects (every put / contains misses)"""
    backends, md, loop = _backends(remote_url=f"fs://{tmp_path}")
    try:
        remote = _remote(backends)
        assert getattr(remote, "connection", None) is None
    finally:
        _close(backends, loop)


@pytest.mark.parametrize("form", ["legacy_url", "plugin"])
def test_cold_level_stores_and_retrieves_page_chunks(tmp_path, form):
    base = str(tmp_path / "cold")
    os.makedirs(base)
    if form == "legacy_url":
        backends, md, loop = _backends(remote_url=f"fs://localhost:0{base}")
    else:
        backends, md, loop = _backends(remote_storage_plugins=["fs"],
                                       extra_config={"remote_storage_plugin.fs.base_path": base})  # fmt: skip
    try:
        files = _roundtrip(backends, md, base)
        assert all(f.startswith(base) for f in files)
    finally:
        _close(backends, loop)
