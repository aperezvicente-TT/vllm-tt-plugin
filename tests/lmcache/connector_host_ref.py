# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Reference worker-side KV-connector host for a TT model runner (A3 harness; A1 implements the same sequence inside
TTModelRunner).

vLLM drives a KV connector from two places the TT runner does not have: the model-runner mixin
(vllm/v1/worker/kv_connector_model_runner_mixin.py: bind metadata -> start_load_kv -> ... -> wait_for_save ->
get_finished -> clear metadata) and the per-attention-layer decorator
(vllm/model_executor/layers/attention/kv_transfer_utils.py:51-57: wait_for_layer_load before the layer, save_kv_layer
after it).  GLM-5.3-Flash's layers are not vLLM Attention modules, so the host calls both itself, once per DSA layer,
around the layer's compute, in the order doc/ttnn/15-lmcache/page_contract.md fixes.

Everything here is host-only torch; the KV tensors stand in for the per-layer host transfer buffer of page_contract.md
section 6 (vLLM MLA convention [num_blocks, block_size, 512] bf16 = the device tensor's bytes)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import ExampleConnector
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

LATENT_DIM = 512
BLOCK_SIZE = 64
# page_contract.md section 1: the 11 sparse_dsa layers (i % 4 == 3) are the only ones with vLLM KV pages.
DSA_LAYERS = tuple(i for i in range(45) if i % 4 == 3)
LAYER_NAMES = tuple(f"model.layers.{i}.self_attn" for i in DSA_LAYERS)


def layer_index(layer_name: str) -> int:
    return int(layer_name.split(".")[2])


def fingerprint(layer_idx: int, pos: int, token: int) -> torch.Tensor:
    """What the mock 'model' writes for token `pos` of a request in layer `layer_idx`: integer-valued bf16 (exact
    below 256 per element) so a loaded page can be checked bit for bit."""
    row = torch.zeros(LATENT_DIM, dtype=torch.bfloat16)
    row[0] = pos % 256
    row[1] = pos // 256
    row[2] = layer_idx
    row[3] = token % 256
    return row


def mla_meta_stub() -> MLACommonMetadata:
    """ExampleConnector selects its MLA slot-mapping path by isinstance(attn_metadata, MLACommonMetadata); the
    instance's fields are never read."""
    return MLACommonMetadata.__new__(MLACommonMetadata)


@dataclass
class CallRecord:
    method: str
    layer_name: str | None = None
    bytes: int = 0
    t_s: float = 0.0


@dataclass
class StepReport:
    calls: list[CallRecord] = field(default_factory=list)
    load_bytes: int = 0
    save_bytes: int = 0
    load_s: float = 0.0
    save_s: float = 0.0
    scheduled: dict[str, tuple[int, int, list[int]]] = field(default_factory=dict)  # req -> (nc, n, block_ids)

    def names(self, method: str) -> list[str]:
        return [c.layer_name for c in self.calls if c.method == method]


class RecordingConnector(ExampleConnector):
    """ExampleConnector (worker role) that records every host-visible callback and the bytes each one moved."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.report = StepReport()

    def _row_bytes(self) -> int:
        return LATENT_DIM * 2

    def _meta_bytes(self, is_store: bool) -> int:
        meta = self._get_connector_metadata()
        return sum(len(r.slot_mapping) for r in meta.requests if r.is_store == is_store) * self._row_bytes()

    def start_load_kv(self, forward_context, **kwargs):
        t0 = time.perf_counter()
        super().start_load_kv(forward_context, **kwargs)
        dt = time.perf_counter() - t0
        n = self._meta_bytes(False) * len(forward_context.no_compile_layers)
        self.report.calls.append(CallRecord("start_load_kv", None, n, dt))
        self.report.load_bytes += n
        self.report.load_s += dt

    def wait_for_layer_load(self, layer_name):
        t0 = time.perf_counter()
        super().wait_for_layer_load(layer_name)
        self.report.calls.append(CallRecord("wait_for_layer_load", layer_name, 0, time.perf_counter() - t0))

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        t0 = time.perf_counter()
        super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
        dt = time.perf_counter() - t0
        n = self._meta_bytes(True)
        self.report.calls.append(CallRecord("save_kv_layer", layer_name, n, dt))
        self.report.save_bytes += n
        self.report.save_s += dt

    def wait_for_save(self):
        super().wait_for_save()
        self.report.calls.append(CallRecord("wait_for_save"))


class ConnectorHostRef:
    """The runner-side sequence, per scheduler step.  `compute` is the model stand-in: compute(layer_name, kv,
    slot_mapping, positions, token_ids) must write the layer's KV rows for the newly scheduled tokens."""

    def __init__(self, connector: RecordingConnector, kv_caches: dict[str, torch.Tensor], block_size: int = BLOCK_SIZE):
        self.connector = connector
        self.kv_caches = kv_caches
        self.block_size = block_size
        self.req_blocks: dict[str, list[int]] = {}
        self.req_tokens: dict[str, list[int]] = {}
        self.forward_context = SimpleNamespace(
            no_compile_layers={n: SimpleNamespace(kv_cache=t) for n, t in kv_caches.items()},
            attn_metadata={n: mla_meta_stub() for n in kv_caches},
            slot_mapping={},
        )

    # the runner's bookkeeping of block tables (TTModelRunner keeps these in its InputBatch)
    def _scheduled(self, so: SchedulerOutput) -> dict[str, tuple[int, int, list[int]]]:
        out = {}
        for nr in so.scheduled_new_reqs:
            self.req_blocks[nr.req_id] = list(nr.block_ids[0])
            self.req_tokens[nr.req_id] = list(nr.prompt_token_ids or [])
            out[nr.req_id] = (nr.num_computed_tokens, so.num_scheduled_tokens[nr.req_id], self.req_blocks[nr.req_id])
        cr = so.scheduled_cached_reqs
        for i, rid in enumerate(cr.req_ids):
            nb = cr.new_block_ids[i]
            if nb is not None:
                if rid in cr.resumed_req_ids:
                    self.req_blocks[rid] = list(nb[0])
                else:
                    self.req_blocks[rid].extend(nb[0])
            if i < len(cr.new_token_ids) and cr.new_token_ids[i]:
                self.req_tokens[rid].extend(cr.new_token_ids[i])
            out[rid] = (cr.num_computed_tokens[i], so.num_scheduled_tokens[rid], self.req_blocks[rid])
        return out

    def slots(self, nc: int, n: int, block_ids: list[int]) -> tuple[list[int], torch.Tensor]:
        positions = list(range(nc, nc + n))
        slot = torch.tensor([block_ids[p // self.block_size] * self.block_size + p % self.block_size for p in positions])
        return positions, slot

    def step(self, so: SchedulerOutput, compute: Callable[..., None]) -> StepReport:
        c = self.connector
        c.report = StepReport()
        c.report.scheduled = self._scheduled(so)
        if so.kv_connector_metadata is None:
            # vLLM: kv_connector_no_forward / no metadata -> the connector is not touched this step
            return c.report
        c.bind_connector_metadata(so.kv_connector_metadata)
        try:
            c.start_load_kv(self.forward_context)
            for name, kv in self.kv_caches.items():
                c.wait_for_layer_load(name)
                for rid, (nc, n, blocks) in c.report.scheduled.items():
                    positions, slot = self.slots(nc, n, blocks)
                    toks = self.req_tokens.get(rid, [])
                    compute(name, kv, slot, positions, [toks[p] if p < len(toks) else 0 for p in positions])
                c.save_kv_layer(name, kv, self.forward_context.attn_metadata[name])
            c.wait_for_save()
            c.get_finished(so.finished_req_ids)
        finally:
            c.clear_connector_metadata()
        return c.report


def mock_compute(layer_name: str, kv: torch.Tensor, slot: torch.Tensor, positions: list[int], tokens: list[int]) -> None:
    li = layer_index(layer_name)
    flat = kv.reshape(-1, kv.shape[-1])
    for s, p, t in zip(slot.tolist(), positions, tokens):
        flat[s] = fingerprint(li, p, t)
