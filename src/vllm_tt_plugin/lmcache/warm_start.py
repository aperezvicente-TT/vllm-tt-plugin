"""Stage 12 lever (c) step 3: the warm start after a relaunch (program 2026-10-26T03:56:09Z; probe
doc/ttnn/12-fast-start/lever_c/warmstart_probe.py).

A relaunch keeps only the stable tiers: LMCache's fs:// cold tier on the share (the CPU tier dies with the process,
the local_disk tier is per run) and the KDA checkpoint files (their RAM tier dies too). The first hit of a stored
prefix then reads its 125 objects (32K) from the share (1.2-3.4 s) and its checkpoint (0.41 s cold per 148 MB). Off
the request path, in one background thread after the connector is wired, this refills both:

* GLM53_LMCACHE_WARM_GB (default 0 = off): the cold tier's newest objects (file mtime) into the LMCache CPU tier (the
  pinned pool in tile mode) up to that many GB, never past GLM53_LMCACHE_WARM_FRACTION (default 0.8) of the tier --
  LMCache's own RemoteBackend.batched_get_blocking (allocates in the CPU tier) + LocalCPUBackend.batched_submit_put_task
  (the hot cache holds its own reference; the get's is dropped). Stops at the first allocation failure.
* GLM53_KDA_CKPT_WARM (default 0): the newest checkpoint files into the KDA store's RAM tier up to its budget
  (GLM53_KDA_CKPT_RAM_GB), never evicting.

A request racing the warm-up reads the cold tier itself as before (a duplicate put is dropped by the CPU backend).
Probe on host-8chip: the relaunched first 32K hit 22.5 ms after the warm-up vs 1.29 s without.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def remote_dirs(engine: Any) -> list[str]:
    """the fs:// cold tier's directories of an LMCache engine ('' entries dropped); [] when it has no fs remote"""
    url = str(getattr(getattr(engine, "config", None), "remote_url", None) or "")
    out = []
    for part in url.split(","):
        part = part.strip()
        if not part.startswith("fs://"):
            continue
        rest = part[len("fs://") :]
        path = "/" + rest.split("/", 1)[1] if "/" in rest else ""
        if path.strip("/"):
            out.append(path)
    return out


def warm_lmcache_cpu_tier(engine: Any, max_bytes: int, *, batch: int = 8, stop: threading.Event | None = None) -> dict:
    """the cold tier's newest objects into the CPU tier, up to max_bytes (see the module doc) -> stats"""
    from lmcache.utils import CacheEngineKey  # noqa: PLC0415

    st: dict = dict(keys=0, loaded=0, bytes=0, list_s=0.0, load_s=0.0, stopped=None)
    sm = getattr(engine, "storage_manager", None)
    backends = getattr(sm, "storage_backends", None) or {}
    remote = next((b for n, b in backends.items() if "Remote" in n), None)
    cpu = backends.get("LocalCPUBackend")
    dirs = remote_dirs(engine)
    if remote is None or cpu is None or not dirs:
        st["stopped"] = "no fs remote / CPU tier"
        return st
    # S2 (program 2026-10-26T16:30:44Z): only objects of THIS engine's key namespace -- the key's model_name carries
    # the layout namespace ("#tt-tile-v2" for tile objects, the LMCache fork's key_namespace), so a row-major object of
    # the k line in the same cold dir is never pulled into a tile-mode CPU tier (nor the reverse)
    want = getattr(getattr(engine, "metadata", None), "model_name", None)
    st["skipped_namespace"] = 0
    t0 = time.perf_counter()
    files = []
    for d in dirs:
        for f in Path(d).glob("*.data"):
            try:
                s = f.stat()
            except OSError:
                continue
            files.append((s.st_mtime, s.st_size, f))
    files.sort(key=lambda x: x[0], reverse=True)
    keys, sizes, total = [], [], 0
    for _m, size, f in files:
        if total + size > max_bytes:
            break
        try:
            k = CacheEngineKey.from_string(f.stem.replace("-SEP-", "/"))
        except Exception:  # noqa: BLE001 -- a foreign file in the dir
            continue
        if want is not None and k.model_name != want:
            st["skipped_namespace"] += 1
            continue
        if cpu.contains(k):
            continue
        keys.append(k)
        sizes.append(size)
        total += size
    st["keys"], st["list_s"] = len(keys), time.perf_counter() - t0
    t1 = time.perf_counter()
    for i in range(0, len(keys), max(1, batch)):
        if stop is not None and stop.is_set():
            st["stopped"] = "stop"
            break
        ks = keys[i : i + batch]
        objs = remote.batched_get_blocking(ks)
        live = [(k, o, z) for k, o, z in zip(ks, objs, sizes[i : i + batch]) if o is not None]
        if live:
            cpu.batched_submit_put_task([k for k, _o, _z in live], [o for _k, o, _z in live])
            for _k, o, z in live:
                o.ref_count_down()
                st["bytes"] += z
            st["loaded"] += len(live)
        if len(live) < len(ks):  # the CPU tier refused an allocation (full) or a file went away: stop here
            st["stopped"] = "allocation / read failed"
            break
    st["load_s"] = time.perf_counter() - t1
    return st


def warm_kda_ram_tier(store: Any, *, stop: threading.Event | None = None, load: Any = None) -> dict:
    """the newest KDA checkpoint files into the store's RAM tier up to its budget (never evicting) -> stats"""
    st: dict = dict(files=0, loaded=0, bytes=0, load_s=0.0, stopped=None)
    ram_bytes = int(getattr(store, "ram_bytes", 0) or 0)
    root = getattr(store, "root", None)
    if ram_bytes <= 0 or not root or not os.path.isdir(root):
        st["stopped"] = "no RAM tier / root"
        return st
    if load is None:
        from vllm_tt_plugin.kv_connector import _blob_load as load  # noqa: PLC0415
    from vllm_tt_plugin.kv_connector import _KDA_LOCK, _states_nbytes  # noqa: PLC0415

    paths = []
    for name in os.listdir(root):
        if name.endswith(".pt"):
            p = os.path.join(root, name)
            try:
                paths.append((os.stat(p).st_mtime, p))
            except OSError:
                continue
    paths.sort(reverse=True)
    st["files"] = len(paths)
    t0 = time.perf_counter()
    for _m, p in paths:
        if stop is not None and stop.is_set():
            st["stopped"] = "stop"
            break
        with _KDA_LOCK:
            have = p in store._ram
            used = sum(v[1] for v in store._ram.values())
        if have:
            continue
        try:
            blob = load(p)
        except Exception as e:  # noqa: BLE001 -- a partial / foreign file: skip it
            logger.warning("KDA warm start: skipping %s (%r)", p, e)
            continue
        n = _states_nbytes(blob["states"])
        if used + n > ram_bytes:
            st["stopped"] = "RAM tier full"
            break
        store._ram_put(p, blob["states"])
        st["loaded"] += 1
        st["bytes"] += n
    st["load_s"] = time.perf_counter() - t0
    return st


def start(engine: Any, kda_store: Any = None, stats: dict | None = None) -> threading.Thread | None:
    """GLM53_LMCACHE_WARM_GB / GLM53_KDA_CKPT_WARM: one daemon thread running both warm-ups (the CPU tier first);
    None when both are off. ``stats`` (if given) receives {"lmcache": ..., "kda": ..., "done": bool}."""
    gb = float(os.environ.get("GLM53_LMCACHE_WARM_GB", "0") or 0)
    kda = os.environ.get("GLM53_KDA_CKPT_WARM", "0") == "1" and kda_store is not None
    if gb <= 0 and not kda:
        return None
    out = stats if stats is not None else {}
    out["done"] = False

    def run() -> None:
        try:
            if gb > 0:
                cfg = getattr(engine, "config", None)
                tier = float(getattr(cfg, "max_local_cpu_size", 0) or 0) * (1 << 30)
                frac = float(os.environ.get("GLM53_LMCACHE_WARM_FRACTION", "0.8"))
                cap = int(min(gb * (1 << 30), frac * tier)) if tier > 0 else int(gb * (1 << 30))
                out["lmcache"] = warm_lmcache_cpu_tier(engine, cap, batch=int(os.environ.get("GLM53_LMCACHE_WARM_BATCH", "8")))
                logger.info("LMCache warm start: %s", out["lmcache"])
            if kda:
                out["kda"] = warm_kda_ram_tier(kda_store)
                logger.info("KDA checkpoint warm start: %s", out["kda"])
        except Exception as e:  # noqa: BLE001 -- the warm-up is an optimization: never take the server down
            out["error"] = repr(e)
            logger.warning("warm start failed: %r", e)
        finally:
            out["done"] = True

    t = threading.Thread(target=run, name="glm53-warm-start", daemon=True)
    t.start()
    return t
