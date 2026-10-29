# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Shared helpers for the A2 LMCache TT connector tests (A3 harness conventions:
``fingerprint`` rows, ``_align``, ``ROW_BYTES``, evidence rows to
$A2_EVIDENCE_JSONL)."""

from __future__ import annotations

import json
import os
import time

import torch

from vllm_tt_plugin.lmcache import page_contract as pc

BLOCK_SIZE = pc.TOKENS_PER_PAGE
LATENT_DIM = pc.LATENT_DIM
LAYER_NAMES = tuple(pc.dsa_layer_names())
ROW_BYTES = (
    LATENT_DIM * 2
)  # 1,024 B per token per layer (latent only, as the A3 harness counts)


def layer_index(layer_name: str) -> int:
    return int(layer_name.split(".")[2])


def fingerprint(layer_idx: int, pos: int, token: int) -> torch.Tensor:
    """Integer-valued bf16 row (exact below 256 per element): bit-for-bit checkable."""
    row = torch.zeros(LATENT_DIM, dtype=torch.bfloat16)
    row[0] = pos % 256
    row[1] = pos // 256
    row[2] = layer_idx
    row[3] = token % 256
    return row


def fingerprint_rows(
    layer_idx: int, positions: list[int], tokens: list[int]
) -> torch.Tensor:
    return torch.stack(
        [fingerprint(layer_idx, p, t) for p, t in zip(positions, tokens, strict=True)]
    )


def align(n: int) -> int:
    """ExampleConnector / contract §5: the saved prefix is align64(len - 1)."""
    return (n - 1) // BLOCK_SIZE * BLOCK_SIZE


def prompt(n: int, seed: int = 7) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, 128, (n,), generator=g).tolist()


def slot_mapping(block_ids: list[int], n_tokens: int) -> torch.Tensor:
    """vLLM MLA slot mapping ``block_id * 64 + offset`` for the first ``n_tokens``."""
    return torch.tensor(
        [b * BLOCK_SIZE + o for b in block_ids for o in range(BLOCK_SIZE)][:n_tokens],
        dtype=torch.long,
    )


def pick_blocks(n_tokens: int, num_blocks: int, seed: int) -> list[int]:
    """Random, non-contiguous block ids; block 0 (vLLM's null block) is never used."""
    g = torch.Generator().manual_seed(seed)
    need = -(-n_tokens // BLOCK_SIZE)
    return (torch.randperm(num_blocks - 1, generator=g)[:need] + 1).tolist()


def evidence(row: dict) -> None:
    path = os.environ.get("A2_EVIDENCE_JSONL")
    if path:
        row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row)
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


def ttnn_available() -> bool:
    try:
        import ttnn  # noqa: F401
    except Exception:
        return False
    return True
