# SPDX-License-Identifier: Apache-2.0
"""Host memory for KV connector transfer buffers (Platform.alloc_kv_host_buffers).

A Tenstorrent chip whose IOMMU group is not translating (passthrough, or no IOMMU)
can only DMA into physically contiguous host memory, which torch.empty memory is
not. There the buffers come from 1 GiB huge pages, laid out so that no KV block
spans two pages; the page ops pin them per page. With the IOMMU translating,
regular memory is used.
"""

from __future__ import annotations

import glob
import math
import mmap

import torch

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

PAGE = 1 << 30
_MAP_HUGETLB = 0x40000  # Linux; not exported by the mmap module before Python 3.13
_MAP_HUGE_SHIFT = 26
_TT_DEVICES = "/sys/bus/pci/drivers/tenstorrent/0000:*"

# Mappings backing buffers handed to the connector: they live as long as the process.
_mappings: list[mmap.mmap] = []


def needs_contiguous_host_memory(devices: str = _TT_DEVICES) -> bool:
    """Whether some Tenstorrent chip's IOMMU group is not in translated (DMA) mode."""
    for dev in glob.glob(devices):
        try:
            with open(f"{dev}/iommu_group/type") as f:
                if not f.read().strip().startswith("DMA"):
                    return True
        except OSError:  # no IOMMU group: the chip sees physical addresses
            return True
    return False


def alloc_hugepage_buffers(
    shapes: dict[str, tuple[tuple[int, ...], torch.dtype]], page: int = PAGE
) -> tuple[mmap.mmap, dict[str, torch.Tensor]]:
    """Buffers by name from one anonymous hugetlb mapping of `page`-byte pages.

    A buffer whose rows (first-dimension entries, the KV blocks) divide the page follows
    the previous one, aligned so no row spans two pages; any other buffer that fits
    in a page never spans two. Raises OSError when the huge-page pool cannot back the
    mapping.
    """
    offsets: dict[str, tuple[int, int]] = {}
    end = 0
    for name, (shape, dtype) in shapes.items():
        nbytes = math.prod(shape) * dtype.itemsize
        row = nbytes // shape[0] if shape and shape[0] else nbytes
        if 0 < row <= page and page % row == 0:
            end = -(-end // max(row, 4096)) * max(row, 4096)
        else:
            end = -(-end // 4096) * 4096
            if nbytes <= page and end // page != (end + nbytes - 1) // page:
                end = -(-end // page) * page
        offsets[name] = (end, nbytes)
        end += nbytes
    total = max(page, -(-end // page) * page)
    flags = mmap.MAP_PRIVATE | _MAP_HUGETLB | (page.bit_length() - 1) << _MAP_HUGE_SHIFT
    try:
        buf = mmap.mmap(-1, total, flags=flags, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    except OSError as e:
        raise OSError(
            e.errno, f"{e.strerror}: {total // page} huge pages of {page} bytes"
        ) from e
    tensors = {
        name: torch.frombuffer(buf, dtype=torch.uint8, count=nbytes, offset=off)
        .view(shapes[name][1])
        .view(shapes[name][0])
        for name, (off, nbytes) in offsets.items()
    }
    return buf, tensors


def alloc_kv_host_buffers(
    shapes: dict[str, tuple[tuple[int, ...], torch.dtype]],
) -> dict[str, torch.Tensor] | None:
    if not needs_contiguous_host_memory():
        return None
    try:
        buf, tensors = alloc_hugepage_buffers(shapes)
    except OSError as e:
        logger.warning(
            "Tenstorrent chips with an untranslated IOMMU need KV host buffers in "
            "1 GiB huge pages, which are not available (%s); using regular memory, "
            "which they cannot DMA into. Reserve more 1 GiB huge pages, or put the "
            "chips' IOMMU groups in DMA mode.",
            e,
        )
        return None
    _mappings.append(buf)
    logger.info("KV host transfer buffers: %d bytes in 1 GiB huge pages", len(buf))
    return tensors
