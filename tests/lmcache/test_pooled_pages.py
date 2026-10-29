# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 12 (GLM53_KV_POOLED_PAGES; doc/ttnn/12-fast-start/paged_pooled_keys.md): every LMCache page carries its 64
tokens' DSA pooled keys in the reserved 4 KiB slice. STORE: the connector asks its pooled source (the host's page ->
(slot, column) map + the model's page-DMA read) and writes the slice token-wise; LOAD: it keeps each page's slice by
page id, the host assembles the hit's units and attaches them to the checkpoint (the model loads them into the slot it
places the request in). Host only: the sharded fake mesh + a fake page-DMA, and the host helpers on stand-ins."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from vllm_tt_plugin.kv_connector import TTKVConnectorHost
from vllm_tt_plugin.lmcache import page_contract as pc

from .test_sharded_pool import _dma_mesh, _memobj, _store
from .tt_lmcache_testing import BLOCK_SIZE, LAYER_NAMES, prompt, slot_mapping

POOLED = BLOCK_SIZE * pc.POOLED_BYTES_PER_TOKEN_LAYER  # 4,096


def _slice(page: int, layer: int) -> np.ndarray:
    """the pooled keys a fake device holds for page `page` of DSA layer `layer` (never all-zero)"""
    return ((np.arange(POOLED, dtype=np.uint32) * 7 + page * 131 + layer * 17) % 251 + 1).astype(np.uint8)


class _Source:
    """the host's pooled source: page ids -> per layer (the DMA's order) [n, 4,096]"""

    def __init__(self, layer_names):
        self.layer_names, self.calls = list(layer_names), []

    def __call__(self, page_ids):
        self.calls.append(list(page_ids))
        return [np.stack([_slice(p, LAYER_NAMES.index(n)) for p in page_ids]) for n in self.layer_names]


def test_store_writes_each_pages_pooled_slice_and_a_load_keeps_them():
    a, dma_a = _dma_mesh()
    src = _Source(dma_a.layer_names)
    a.conn.enable_pooled_pages(src)
    n = 512
    blocks_a = [3, 9, 14, 20, 27, 31, 40, 55]
    chunks, objs = _store(a, a.prefill(blocks_a, prompt(n, seed=31)), n)
    assert src.calls == [blocks_a], "one source call for the store's page union"
    for (s, _e), obj in zip(chunks, objs):
        for li in range(len(LAYER_NAMES)):
            got = obj.tensor[0, li, :, pc.LATENT_DIM :].contiguous().view(torch.uint8).reshape(-1, POOLED)
            for k in range(got.shape[0]):
                assert np.array_equal(got[k].numpy(), _slice(blocks_a[s // BLOCK_SIZE + k], li)), (s, li, k)
    assert a.conn.stats["pooled_store_pages"] == len(blocks_a)
    # the hit on another mesh: the slices are kept by ITS page ids, in the source's layer order
    b, dma_b = _dma_mesh()
    b.conn.enable_pooled_pages(_Source(dma_b.layer_names))
    blocks_b = [60, 61, 62, 63, 64, 65, 66, 67]
    b.conn.batched_to_gpu(objs, [s for s, _ in chunks], [e for _, e in chunks], kvcaches=b.host,
                          slot_mapping=slot_mapping(blocks_b, n))  # fmt: skip
    assert sorted(b.conn.loaded_pooled) == sorted(blocks_b)
    got = b.conn.take_loaded_pooled(blocks_b[:6])
    for j, name in enumerate(dma_b.layer_names):
        li = LAYER_NAMES.index(name)
        assert np.array_equal(got[j].numpy(), np.stack([_slice(p, li) for p in blocks_a[:6]])), name
    assert sorted(b.conn.loaded_pooled) == blocks_b[6:], "taken slices are popped"
    assert b.conn.take_loaded_pooled([999]) is None


def test_partial_page_load_keeps_a_zero_padded_slice():
    a, dma_a = _dma_mesh()
    a.conn.enable_pooled_pages(_Source(dma_a.layer_names))
    n = 300
    obj = _memobj(n)
    a.conn.from_gpu(obj, 0, n, kvcaches=a.host, slot_mapping=a.prefill([4, 5, 6, 7, 8], prompt(n, seed=32)))
    b, dma_b = _dma_mesh()
    b.conn.enable_pooled_pages(_Source(dma_b.layer_names))
    b.conn.to_gpu(obj, 0, n, kvcaches=b.host, slot_mapping=slot_mapping([40, 41, 42, 43, 44], n))
    last = b.conn.take_loaded_pooled([44])
    j = dma_b.layer_names.index(LAYER_NAMES[0])
    tail = (n % BLOCK_SIZE) * pc.POOLED_BYTES_PER_TOKEN_LAYER
    assert np.array_equal(last[j][0, :tail].numpy(), _slice(8, 0)[:tail]) and not last[j][0, tail:].any()


def _host_stub(conn=None, slots=None):
    view = SimpleNamespace(block_size=BLOCK_SIZE)
    h = SimpleNamespace(views={"l": view}, _group_for=lambda v: 0, tt_gpu_connector=conn, pooled_missing=0,
                        _slot_for=lambda req_id, row, mi: slots[req_id])  # fmt: skip
    return h


def test_page_columns_cover_each_rows_written_pages_only():
    mi = SimpleNamespace(row_req_ids=["A", "B", None], input_positions=torch.tensor([128, 0, 0]),
                         prompt_lens=torch.tensor([300, 64, 0]), block_tables_per_group=None,
                         block_tables=torch.tensor([[7, 8, 9, 10, 11, 99], [20, 21, 0, 0, 0, 0], [5, 5, 5, 5, 5, 5]]))  # fmt: skip
    h = _host_stub(slots={"A": 2, "B": 5})
    got = TTKVConnectorHost._page_columns(h, mi)
    # A covers positions < 300 -> columns 0..4 (block 99 past its end is not mapped); B covers < 64 -> column 0
    assert got == {7: (2, 0), 8: (2, 1), 9: (2, 2), 10: (2, 3), 11: (2, 4), 20: (5, 0)}
    h._pooled_map, h._read_pooled = got, lambda pairs, pages=None: (pairs, pages)
    assert TTKVConnectorHost._pooled_source(h, [9, 20]) == ([(2, 2), (5, 0)], [9, 20])
    h2 = _host_stub(slots={})
    h2._pooled_map, h2._read_pooled = got, lambda pairs: pairs  # a model hook without pages= (8508f94e970)
    assert TTKVConnectorHost._pooled_source(h2, [9, 20]) == [(2, 2), (5, 0)]
    try:
        TTKVConnectorHost._pooled_source(h, [21])
        raise AssertionError("an unmapped page must raise")
    except RuntimeError as e:
        assert "no row" in str(e)


def test_with_pooled_attaches_whole_units_in_device_order():
    """the hit's pages -> units: unit u = pages 2u, 2u + 1, tile c of the unit = [half 0 of tile c | half 1 of tile c]
    (page_pool.pooled_units); an odd page count pads the last unit's second half with zeros"""
    conn = SimpleNamespace(loaded_pooled={}, take_loaded_pooled=None)
    rng = np.random.default_rng(3)
    pages = {b: torch.from_numpy(rng.integers(1, 256, (11, POOLED), dtype=np.uint8)) for b in (7, 8, 9)}

    def take(ids):
        return [torch.stack([pages[i][j] for i in ids]) for j in range(11)]

    conn.take_loaded_pooled = take
    h = _host_stub(conn)
    mi = SimpleNamespace(block_tables_per_group=None, block_tables=torch.tensor([[7, 8, 9, 0]]))
    states = {"next_pos": 192, "states": ["kda"]}
    out = TTKVConnectorHost._with_pooled(h, states, 0, 192, mi)
    assert "pooled" not in states and out["states"] == ["kda"] and out["pooled"]["units"] == 2
    assert out["pooled"]["pages"] == 3
    for j in range(11):
        d = out["pooled"]["data"][j]
        assert d.shape == (2, 8192)
        h0, h1 = pages[7][j].numpy().reshape(4, 1024), pages[8][j].numpy().reshape(4, 1024)
        assert np.array_equal(d[0].reshape(4, 2, 1024)[:, 0], h0) and np.array_equal(d[0].reshape(4, 2, 1024)[:, 1], h1)
        assert np.array_equal(d[1].reshape(4, 2, 1024)[:, 0], pages[9][j].numpy().reshape(4, 1024))
        assert not d[1].reshape(4, 2, 1024)[:, 1].any()
    # a page stored without its slice (zeros) or not loaded: the checkpoint is returned unchanged (row restarts at 0)
    pages[8][0][:] = 0
    assert TTKVConnectorHost._with_pooled(h, states, 0, 192, mi) is states and h.pooled_missing == 1
    conn.take_loaded_pooled = lambda ids: None
    assert TTKVConnectorHost._with_pooled(h, states, 0, 192, mi) is states and h.pooled_missing == 2
