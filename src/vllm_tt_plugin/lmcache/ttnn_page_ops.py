# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Reference device-side ``CopyBlocksOp`` for the latent pages (ttnn, no new kernels).

STATUS: written against page_contract.md §6 and the ttnn API; NOT run on a
device by A2 (host-only work item). The plugin-side connector host (A1) owns
the production op; this is the ttnn call sequence it can start from.

  d2h: ``ttnn.slice`` of a contiguous block range of the replicated latent tensor,
       ``ttnn.get_device_tensors(...)[0]`` (one chip suffices), ``ttnn.to_torch``
       into the host transfer buffer rows.
  h2d: stage the host rows as ``[1, 1, 64 n, 512]`` bf16 TILE on the mesh
       (replicated) and ``ttnn.experimental.paged_fill_cache`` with a one-row
       page table of the destination block ids; ``ttnn.synchronize_device``
       before the host buffer may be reused.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_tt_plugin.lmcache import page_contract as pc


def _runs(src_ids: list[int], dst_ids: list[int]) -> list[tuple[int, int, int]]:
    """Group (src, dst) pairs into (src0, dst0, n) runs where both advance by one."""
    runs: list[tuple[int, int, int]] = []
    for s, d in zip(src_ids, dst_ids, strict=True):
        if runs and runs[-1][0] + runs[-1][2] == s and runs[-1][1] + runs[-1][2] == d:
            runs[-1] = (runs[-1][0], runs[-1][1], runs[-1][2] + 1)
        else:
            runs.append((s, d, 1))
    return runs


def make_ttnn_copy_blocks_op(
    mesh_device: Any,
    tokens_per_page: int = pc.TOKENS_PER_PAGE,
    latent_dim: int = pc.LATENT_DIM,
):
    """Build a vLLM ``CopyBlocksOp`` over ttnn device latent tensors (untested)."""
    import ttnn

    T, D = tokens_per_page, latent_dim

    def copy_blocks(
        src: Any, dst: Any, src_ids: list[int], dst_ids: list[int], direction: str
    ) -> None:
        if direction == "d2h":
            for name, cache in src.items():
                host = dst[name]
                for s0, d0, n in _runs(src_ids, dst_ids):
                    part = ttnn.slice(cache, (s0, 0, 0, 0), (s0 + n, 1, T, D))
                    try:
                        host[d0 : d0 + n].copy_(
                            ttnn.to_torch(ttnn.get_device_tensors(part)[0])
                        )
                    finally:
                        ttnn.deallocate(part)  # no device buffer may outlive the copy (trace-safety guard)
        elif direction == "h2d":
            n = len(src_ids)
            table = ttnn.from_torch(
                torch.tensor([dst_ids], dtype=torch.int32),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            for name, host in src.items():
                rows = host.index_select(0, torch.as_tensor(src_ids)).reshape(
                    1, 1, n * T, D
                )
                stage = ttnn.from_torch(
                    rows,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
                ttnn.experimental.paged_fill_cache(dst[name], stage, table, batch_idx=0)
            ttnn.synchronize_device(mesh_device)
        else:
            raise ValueError(direction)

    return copy_blocks
