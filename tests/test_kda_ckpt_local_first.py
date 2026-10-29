"""Program 2026-10-28T05:07:22Z (1) + (2): the k4d5 warm 16,100 repeat read its 16,000 KDA checkpoint from the
NFS share (372 ms of kda_store_load).
(1) GLM53_KDA_CKPT_LOCAL_DIR: checkpoints are written local-first and mirrored to the share root in the background (the
    restart truth); has() / load() prefer the local copy and fall back to the share.
(2) GLM53_KDA_CKPT_RECENT_N: the last N saved checkpoints keep their states in the RAM tier whatever its byte LRU does,
    so a hit on an earlier prefix than the latest export is an in-memory hit.
Both keep the same bytes: the loaded states equal the saved ones."""

import os

import pytest
import torch

from vllm_tt_plugin import kv_connector as K


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("GLM53_KDA_CKPT_LOCAL_DIR", "GLM53_KDA_CKPT_RECENT_N", "GLM53_KDA_CKPT_RAM_GB", "GLM53_KDA_CKPT_ASYNC",
              "GLM53_KDA_CKPT_LOCAL_MAX_GB", "GLM53_KDA_CKPT_LOCAL_MIN_FREE_GB"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GLM53_KDA_CKPT_MIN_FREE_GB", "0")
    monkeypatch.setenv("GLM53_KDA_CKPT_LOCAL_MIN_FREE_GB", "0")


def _states(v):
    return [{"L": 0, "S": torch.full((4, 4), float(v)), "C": torch.arange(5) + v}]


def _same(a, b):
    return all(torch.equal(x["S"], y["S"]) and torch.equal(x["C"], y["C"]) for x, y in zip(a, b))


def test_local_first_writes_locally_mirrors_to_the_share_and_reads_local_first(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_KDA_CKPT_LOCAL_DIR", str(tmp_path / "local"))
    monkeypatch.setenv("GLM53_KDA_CKPT_RAM_GB", "0")  # (every load from a file)
    share = str(tmp_path / "share")
    st = K.KDACheckpointStore(share)
    toks = list(range(500))
    st.save(toks, 256, _states(3))
    st.flush()
    share_file = st.path(toks, 256)
    local_file = st.lpath(share_file)
    assert os.path.dirname(local_file) == st.local_root and os.path.exists(local_file) and os.path.exists(share_file)
    assert st.stats.get("mirrored") == 1
    assert _same(st.load(toks, 256), _states(3)) and st.stats.get("local_loads") == 1 and not st.stats.get("disk_loads")
    os.remove(local_file)  # (a local disk wiped: the share copy serves)
    assert st.has(toks, 256) and _same(st.load(toks, 256), _states(3)) and st.stats.get("disk_loads") == 1


def test_a_restarted_server_with_an_empty_local_dir_reads_the_share(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_KDA_CKPT_RAM_GB", "0")
    share = str(tmp_path / "share")
    toks = list(range(300))
    first = K.KDACheckpointStore(share)
    first.save(toks, 128, _states(5))  # written straight to the share (local-first off)
    first.flush()
    monkeypatch.setenv("GLM53_KDA_CKPT_LOCAL_DIR", str(tmp_path / "fresh_local"))
    st = K.KDACheckpointStore(share)
    assert st.has(toks, 128) and not os.path.exists(st.lpath(st.path(toks, 128)))
    assert _same(st.load(toks, 128), _states(5)) and st.stats.get("disk_loads") == 1


def test_off_by_default_nothing_changes(tmp_path):
    st = K.KDACheckpointStore(str(tmp_path / "share"))
    assert st.local_root is None and st.lpath("x") is None and st.recent_n == 0
    toks = list(range(200))
    st.save(toks, 64, _states(1))
    st.flush()
    assert os.path.exists(st.path(toks, 64)) and _same(st.load(toks, 64), _states(1))


def test_the_recent_n_saves_stay_in_ram_past_the_byte_lru(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_KDA_CKPT_RAM_GB", "1e-12")  # ~0 bytes: the plain LRU keeps nothing
    monkeypatch.setenv("GLM53_KDA_CKPT_RECENT_N", "2")
    st = K.KDACheckpointStore(str(tmp_path / "share"))
    toks = list(range(600))
    for n, v in ((128, 1), (256, 2), (384, 3)):
        st.save(toks, n, _states(v))
    st.flush()
    ram = K._KDA_RAM[st.root]
    assert set(ram) == {st.path(toks, 256), st.path(toks, 384)}  # the last two saves, pinned
    assert _same(st.load(toks, 256), _states(2)) and _same(st.load(toks, 384), _states(3))
    assert st.stats["ram_hits"] == 2 and not st.stats.get("disk_loads")
    assert _same(st.load(toks, 128), _states(1)) and st.stats["disk_loads"] == 1  # the older one: a file read


def test_recent_n_pins_even_with_the_ram_tier_off(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_KDA_CKPT_RAM_GB", "0")
    monkeypatch.setenv("GLM53_KDA_CKPT_RECENT_N", "1")
    st = K.KDACheckpointStore(str(tmp_path / "share"))
    toks = list(range(300))
    st.save(toks, 64, _states(7))
    st.save(toks, 128, _states(8))
    st.flush()
    assert list(K._KDA_RAM[st.root]) == [st.path(toks, 128)]
    assert _same(st.load(toks, 128), _states(8)) and st.stats["ram_hits"] == 1
