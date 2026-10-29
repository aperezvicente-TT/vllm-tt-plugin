# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only stand-in for the paged latent caches and the plugin's ``CopyBlocksOp``.

No Tenstorrent device is ever opened. The "device" tensors are either plain
torch tensors (``backend="torch"``) or ttnn HOST tensors in TILE layout
(``backend="ttnn"``: ``ttnn.from_torch`` without a device), so the page copies
through ttnn's host tensors are exercised without hardware. The page-range copy
on a ttnn host tensor goes through ``ttnn.to_torch`` / ``ttnn.from_torch`` (host
tensors cannot be sliced by device ops); the device path uses
``ttnn.slice`` / ``paged_fill_cache`` instead (see ``ttnn_page_ops``).
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_tt_plugin.lmcache import page_contract as pc


class MockTTPagedLatentCache:
    """Per layer: a host transfer buffer and a mock device latent tensor.

    Both are ``[num_blocks, 1, 64, 512]`` bf16 (the contract's per-layer shape);
    ``copy_blocks`` is a vLLM ``CopyBlocksOp`` moving page ranges between them.
    """

    def __init__(
        self,
        num_blocks: int,
        num_layers: int = pc.NUM_DSA_LAYERS,
        backend: str = "torch",
        seed: int | None = None,
        with_pooled_keys: bool = False,
        tokens_per_page: int = pc.TOKENS_PER_PAGE,
        latent_dim: int = pc.LATENT_DIM,
    ):
        if backend not in ("torch", "ttnn"):
            raise ValueError(backend)
        self.backend = backend
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.tokens_per_page = tokens_per_page
        self.latent_dim = latent_dim
        self.layer_names = pc.dsa_layer_names(num_layers)
        self.calls: list[tuple[str, list[int], list[int]]] = []
        if backend == "ttnn":
            import ttnn  # noqa: F401  (host-only use; no device is opened)

            self._ttnn = ttnn
        shape = (num_blocks, 1, tokens_per_page, latent_dim)
        self.host_buffers: dict[str, torch.Tensor] = {
            n: torch.zeros(shape, dtype=pc.LATENT_DTYPE) for n in self.layer_names
        }
        self.pooled_host: dict[str, torch.Tensor] | None = None
        if with_pooled_keys:
            pshape = (num_blocks, 1, pc.POOLED_KEY_ROWS_PER_PAGE, pc.POOLED_KEY_DIM)
            self.pooled_host = {
                n: torch.zeros(pshape, dtype=pc.LATENT_DTYPE) for n in self.layer_names
            }
        self.device_caches: dict[str, Any] = {}
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        for n in self.layer_names:
            dev = (
                torch.randn(shape, generator=gen).to(pc.LATENT_DTYPE)
                if gen is not None
                else torch.zeros(shape, dtype=pc.LATENT_DTYPE)
            )
            self.device_caches[n] = self._to_device(dev)

    # -------------------------------------------------------------- helpers
    def _to_device(self, t: torch.Tensor) -> Any:
        if self.backend == "torch":
            return t.clone()
        ttnn = self._ttnn
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def _to_host(self, d: Any) -> torch.Tensor:
        if self.backend == "torch":
            return d
        return self._ttnn.to_torch(d)

    @property
    def host_list(self) -> list[torch.Tensor]:
        """``kvcaches`` as the vLLM adapter passes them (registration order)."""
        return list(self.host_buffers.values())

    def device_pages(self, name: str) -> torch.Tensor:
        """Torch copy/view of a device cache ``[num_blocks, 1, 64, 512]``."""
        return self._to_host(self.device_caches[name])

    def device_tokens(self, name: str, slot_mapping: torch.Tensor) -> torch.Tensor:
        """Latent rows ``[n, 512]`` of the tokens addressed by ``slot_mapping``."""
        pages = self.device_pages(name).reshape(
            self.num_blocks * self.tokens_per_page, self.latent_dim
        )
        return pages[slot_mapping.to(torch.long)]

    # --------------------------------------------------------- CopyBlocksOp
    def copy_blocks(
        self, src: Any, dst: Any, src_ids: list[int], dst_ids: list[int], direction: str
    ) -> None:
        """vLLM ``CopyBlocksOp``: page-range copy host->device or device->host."""
        if direction not in ("h2d", "d2h"):
            raise ValueError(direction)
        if len(src_ids) != len(dst_ids):
            raise ValueError("src/dst block id lists differ in length")
        self.calls.append((direction, list(src_ids), list(dst_ids)))
        s_idx = torch.as_tensor(src_ids, dtype=torch.long)
        d_idx = torch.as_tensor(dst_ids, dtype=torch.long)
        for name in src:
            if direction == "h2d":
                host, dev = src[name], dst[name]
                dev_t = self._to_host(dev)
                dev_t = dev_t if self.backend == "torch" else dev_t.clone()
                dev_t.index_copy_(0, d_idx, host.index_select(0, s_idx))
                if self.backend == "ttnn":
                    dst[name] = self._to_device(
                        dev_t
                    )  # host ttnn tensors are immutable: replace
                    self.device_caches[name] = dst[name]
            else:
                dev, host = src[name], dst[name]
                host.index_copy_(0, d_idx, self._to_host(dev).index_select(0, s_idx))
