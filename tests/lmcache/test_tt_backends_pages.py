# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A2: LMCache local_cpu_backend / local_disk_backend store and retrieve of
contract-sized page chunks on the CPU-only build (template: LMCache
tests/v1/storage_backend/test_local_cpu_backend.py,
test_local_disk_backend.py)."""

from __future__ import annotations

import asyncio
import os
import threading
import time

import torch
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend

from vllm_tt_plugin.lmcache import page_contract as pc

from .tt_lmcache_testing import evidence

CHUNK = pc.LMCACHE_CHUNK_TOKENS
PAGES_PER_CHUNK = CHUNK // pc.TOKENS_PER_PAGE  # 2 since J11 (128-token chunks)


def _config(tmp_path=None, max_cpu_gb=0.25):
    kw = dict(
        chunk_size=CHUNK,
        local_cpu=True,
        max_local_cpu_size=max_cpu_gb,
        lmcache_instance_id="a2-backend-test",
    )
    if tmp_path is not None:
        kw.update(local_disk=str(tmp_path), max_local_disk_size=1.0)
    return LMCacheEngineConfig.from_defaults(**kw)


def _key(i: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="zai-org/GLM-5.3-Flash",
        world_size=1,
        worker_id=0,
        chunk_hash=1000 + i,
        dtype=pc.LATENT_DTYPE,
    )


def _fill(obj, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, 256, (obj.get_size(),), generator=g, dtype=torch.uint8)
    obj.tensor.reshape(-1).view(torch.uint8).copy_(data)
    return data


def _bytes(obj) -> torch.Tensor:
    return obj.tensor.reshape(-1).view(torch.uint8)


def test_local_cpu_backend_page_chunks():
    config = _config()
    md = pc.lmcache_metadata()
    backend = LocalCPUBackend(config, md, dst_device="cpu")
    try:
        shapes, dtypes = md.get_shapes(CHUNK), md.get_dtypes()
        assert shapes == [torch.Size([1, 11, CHUNK, 544])]
        objs, datas = [], []
        for i in range(8):
            obj = backend.allocate(shapes, dtypes, fmt=MemoryFormat.KV_MLA_FMT)
            assert obj is not None and obj.get_size() == PAGES_PER_CHUNK * pc.PAGE_BYTES
            datas.append(_fill(obj, i))
            objs.append(obj)
        t0 = time.perf_counter()
        backend.batched_submit_put_task([_key(i) for i in range(8)], objs)
        put_s = time.perf_counter() - t0
        for i in range(8):
            assert backend.contains(_key(i))
            got = backend.get_blocking(_key(i))
            assert got is not None and got.metadata.fmt == MemoryFormat.KV_MLA_FMT
            assert torch.equal(_bytes(got), datas[i])
            got.ref_count_down()
        assert not backend.contains(_key(99))
        assert backend.remove(_key(0)) and not backend.contains(_key(0))
        evidence(
            dict(
                test="test_local_cpu_backend_page_chunks",
                chunks=8,
                chunk_bytes=PAGES_PER_CHUNK * pc.PAGE_BYTES,
                put_s=round(put_s, 4),
                allocator=type(backend.memory_allocator).__name__,
                status="pass",
            )
        )
    finally:
        backend.close()


def test_local_disk_backend_page_chunks(tmp_path):
    config = _config(tmp_path)
    md = pc.lmcache_metadata()
    cpu = LocalCPUBackend(config, md, dst_device="cpu")
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    disk = LocalDiskBackend(config, loop, cpu, dst_device="cpu", metadata=md)
    try:
        shapes, dtypes = md.get_shapes(CHUNK), md.get_dtypes()
        objs, datas = [], []
        for i in range(4):
            obj = cpu.allocate(shapes, dtypes, fmt=MemoryFormat.KV_MLA_FMT)
            datas.append(_fill(obj, 100 + i))
            objs.append(obj)
        t0 = time.perf_counter()
        disk.batched_submit_put_task([_key(i) for i in range(4)], objs)
        deadline = time.time() + 30
        while time.time() < deadline and not all(
            disk.contains(_key(i)) and os.path.exists(disk._key_to_path(_key(i)))
            for i in range(4)
        ):
            time.sleep(0.05)
        put_s = time.perf_counter() - t0
        files = sorted(os.listdir(disk.path))
        assert len(files) == 4, files
        on_disk = sum(os.path.getsize(os.path.join(disk.path, f)) for f in files)
        assert on_disk >= 4 * PAGES_PER_CHUNK * pc.PAGE_BYTES
        for obj in objs:
            obj.ref_count_down()
        for i in range(4):
            got = disk.get_blocking(_key(i))
            assert got is not None, i
            assert torch.equal(_bytes(got), datas[i])
            got.ref_count_down()
        assert disk.get_blocking(_key(42)) is None
        evidence(
            dict(
                test="test_local_disk_backend_page_chunks",
                chunks=4,
                chunk_bytes=PAGES_PER_CHUNK * pc.PAGE_BYTES,
                on_disk_bytes=on_disk,
                put_s=round(put_s, 4),
                path=disk.path,
                status="pass",
            )
        )
    finally:
        disk.close()
        cpu.close()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
