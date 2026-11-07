# SPDX-License-Identifier: Apache-2.0
import mmap

import pytest
import torch

from vllm_tt_plugin import host_buffers


@pytest.fixture
def regular_mmap(monkeypatch):
    """Back the mapping with regular memory, recording the size and flags asked for."""
    calls = []
    real = mmap.mmap

    def fake(fileno, length, flags, prot):
        calls.append((length, flags))
        return real(fileno, length, flags=mmap.MAP_PRIVATE, prot=prot)

    monkeypatch.setattr(host_buffers.mmap, "mmap", fake)
    return calls


def _start(buf, t):
    return t.data_ptr() - torch.frombuffer(buf, dtype=torch.uint8).data_ptr()


def test_rows_dividing_the_page_are_packed_without_spanning_pages(regular_mmap):
    page, row = 2 << 20, 256 << 10
    shapes = {
        f"l{i}": ((3, 128, 1024), torch.bfloat16) for i in range(5)
    }  # 3 rows of 256 KiB
    buf, tensors = host_buffers.alloc_hugepage_buffers(shapes, page)
    for t in tensors.values():
        start = _start(buf, t)
        assert start % row == 0
        assert all(
            (start + r * row) // page == (start + r * row + row - 1) // page
            for r in range(3)
        )
    assert regular_mmap == [
        (
            2 * page,
            mmap.MAP_PRIVATE
            | host_buffers._MAP_HUGETLB
            | 21 << host_buffers._MAP_HUGE_SHIFT,
        )
    ]


def test_buffers_with_other_rows_that_fit_a_page_never_span_two(regular_mmap):
    page = 2 << 20
    shapes = {f"l{i}": ((3, 250_000), torch.uint8) for i in range(5)}
    buf, tensors = host_buffers.alloc_hugepage_buffers(shapes, page)
    for t in tensors.values():
        start = _start(buf, t)
        assert start // page == (start + t.numel() - 1) // page
    assert regular_mmap[0][0] == 3 * page


def test_buffers_have_the_requested_shape_and_dtype_and_are_disjoint(regular_mmap):
    shapes = {
        "a": ((4, 8, 16, 64), torch.bfloat16),
        "b": ((2, 3), torch.float32),
        "c": ((5000,), torch.uint8),
    }
    _, tensors = host_buffers.alloc_hugepage_buffers(shapes, 2 << 20)
    for i, (name, (shape, dtype)) in enumerate(shapes.items()):
        assert tensors[name].shape == shape and tensors[name].dtype == dtype
        tensors[name].view(torch.uint8).fill_(i + 1)
    for i, t in enumerate(tensors.values()):
        assert bool((t.view(torch.uint8) == i + 1).all())


def _devices(tmp_path, *types):
    for i, kind in enumerate(types):
        d = tmp_path / f"0000:0{i}:00.0"
        d.mkdir()
        if kind is not None:
            (d / "iommu_group").mkdir()
            (d / "iommu_group" / "type").write_text(kind + "\n")
    return str(tmp_path / "0000:*")


@pytest.mark.parametrize(
    "types, needed",
    [
        (("DMA", "DMA-FQ"), False),
        (("DMA", "identity"), True),
        ((None,), True),
        ((), False),
    ],
)
def test_contiguous_memory_is_needed_unless_every_chip_translates(
    tmp_path, types, needed
):
    assert (
        host_buffers.needs_contiguous_host_memory(_devices(tmp_path, *types)) is needed
    )


def test_translating_chips_get_regular_memory(monkeypatch):
    monkeypatch.setattr(host_buffers, "needs_contiguous_host_memory", lambda: False)
    assert host_buffers.alloc_kv_host_buffers({"a": ((2, 4), torch.uint8)}) is None


def test_without_huge_pages_the_connector_falls_back_to_regular_memory(monkeypatch):
    monkeypatch.setattr(host_buffers, "needs_contiguous_host_memory", lambda: True)

    def no_pages(*_args, **_kwargs):
        raise OSError(12, "Cannot allocate memory")

    monkeypatch.setattr(host_buffers.mmap, "mmap", no_pages)
    assert host_buffers.alloc_kv_host_buffers({"a": ((2, 4), torch.uint8)}) is None
