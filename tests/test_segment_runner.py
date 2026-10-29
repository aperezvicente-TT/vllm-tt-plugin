# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Segmented prefill traces, the runner half (doc/ttnn/14b-persistent-stack/segtrace/runner_design.md section 4, item 10).

Under the scheduler's set_tt_segment_step stamp:
- step 0 rounds the chunk [s, e - 1) back up to [s, e) and runs its segment 0 head-less;
- every hold step re-sends that same row with the next segment before the step's decode;
- the release replaces the 1-token tail with it and runs the remaining segments, the last one with the head.
The open chunk's slot is pinned in the decode gather, and a request leaving inside the hold drops its open chunk.
Host-only: TTModelRunner's methods on a stand-in runner and a fake model that records every prefill_forward."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_tt_plugin  # noqa: F401  (activates the tt platform)
from vllm_tt_plugin.model_input import TTModelInput, TTMixedModelInput
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import set_tt_segment_step

PROMPT = 40  # the request's prompt; the chunk [0, 32) is segmented (end 32), a final chunk when PROMPT == 32


class FakeModel:
    def __init__(self, segments=3):
        self.calls, self.aborted, self.segments, self.open = [], [], segments, None

    def prefill_segments(self, start, n):
        return self.segments

    def prefill_forward(self, **kw):
        self.calls.append(kw)
        return None

    def prefill_abort(self, slot):
        self.aborted.append(slot)

    def open_chunk(self):
        return self.open


def _runner(model=None, prompt=PROMPT, slot=5):
    notes = []
    return SimpleNamespace(
        model=model or FakeModel(), kv_caches=object(), trace_mode="all", request_specific_rope=False,
        async_decode=SimpleNamespace(note_prefill_submitted=lambda: notes.append(1)), notes=notes,
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}, token_ids_cpu_tensor=torch.arange(64).view(1, 64) + 100,
                                    num_tokens=np.array([prompt])),
        _req_state_slot={"r": slot}, tt_per_lane_max_num_seqs=8,
    )  # fmt: skip


def _input(start=0, end=31, rid="r", slot=5):
    """a one-row prefill build as _prepare_model_inputs makes it: the chunk [start, end) of rid's history"""
    base = {f.name: None for f in fields(TTModelInput)}
    base.update(input_tokens=torch.arange(end).view(1, end) + 100, input_positions=torch.tensor([start]),
                prompt_lens=np.array([end]), block_tables=torch.zeros((1, 1)), block_tables_per_group=[],
                unpadded_batch_size=1, multi_modal_kwargs={}, perform_device_sampling=False, grammar_bitmask=[None],
                logitsprocs_list=[None], bad_words_token_ids_list=[{}], allowed_token_ids_mask_list=[None],
                generators_list=[{}], max_num_logprobs=[None], row_req_ids=[rid], prefill_empty_slots=[slot],
                intermediate_prefill_mask=torch.tensor([True]), kda_snapshot_at=[16])  # fmt: skip
    return TTModelInput(**base)


def _so(segments, n_seg=3, end=32):
    so = SimpleNamespace(finished_req_ids=set(), preempted_req_ids=set())
    set_tt_segment_step(so, "r", segments, n_seg, end)
    return so


def _step0(runner):
    out = TTModelRunner._segment_prepare(runner, _so([0]), _input())
    TTModelRunner._segment_record(runner, out)  # (after the connector's load)
    return out


def test_step0_rounds_the_chunk_up_and_runs_segment_0_headless():
    r = _runner()
    out = _step0(r)
    assert list(out.prompt_lens) == [32] and out.input_tokens.shape == (1, 32)
    assert out.input_tokens[0, 31].item() == 131  # the token at e - 1, from the persistent batch
    assert out.segment_rows == [0] and out.intermediate_prefill_mask.tolist() == [True]
    assert r._segment_open["input"] is out and r._segment_open["slot"] == 5
    assert out.kda_snapshot_at == [16]  # the connector's mark rides along to every segment


def test_hold_steps_resend_the_same_row_before_the_decode_without_breaking_the_decode_chain():
    r = _runner()
    _step0(r)
    dec = SimpleNamespace(prompt_lens=None, row_req_ids=["d1"])
    assert TTModelRunner._segment_prepare(r, _so([1]), dec) is dec
    (kw,) = r.model.calls
    assert kw["segment_rows"] == [1] and list(kw["prompt_lens"]) == [32] and kw["kda_snapshot_at"] == [16]
    assert kw["empty_slots"] == [5] and not r.notes  # the decode rows' resident inputs stay valid


@pytest.mark.parametrize("prompt", [PROMPT, 32])
def test_the_release_runs_the_remaining_segments_the_last_with_the_head(prompt):
    r = _runner(prompt=prompt)
    _step0(r)
    tail = _input(start=31, end=32)
    final = TTModelRunner._segment_prepare(r, _so([1, 2]), tail)  # an early release: segments 1 and 2
    assert [kw["segment_rows"] for kw in r.model.calls] == [[1]]  # segment 1 sent, segment 2 is the step's forward
    assert final.segment_rows == [2] and list(final.prompt_lens) == [32] and int(final.input_positions[0]) == 0
    assert final.intermediate_prefill_mask.tolist() == [prompt > 32]  # the head samples only for a final chunk
    assert r._segment_open is None


def test_the_release_inside_a_mixed_step_replaces_the_prefill_part_only():
    r = _runner()
    _step0(r)
    dec = SimpleNamespace(prompt_lens=None, row_req_ids=["d1"])
    mixed = TTMixedModelInput(prefill=_input(start=31, end=32), decode=dec, req_ids=("d1", "r"))
    out = TTModelRunner._segment_prepare(r, _so([2]), mixed)
    assert isinstance(out, TTMixedModelInput) and out.decode is dec and out.prefill.segment_rows == [2]


def test_a_segment_count_the_model_does_not_capture_is_refused():
    r = _runner(model=FakeModel(segments=4))
    with pytest.raises(RuntimeError, match="into 3 segments, the model into 4"):
        _step0(r)


def test_a_model_without_segments_is_refused():
    r = _runner()
    r.model = SimpleNamespace(prefill_forward=r.model.prefill_forward)
    with pytest.raises(RuntimeError, match="no prefill_segments"):
        _step0(r)


def test_a_stamp_without_an_open_chunk_is_refused():
    with pytest.raises(RuntimeError, match="no recorded chunk"):
        TTModelRunner._segment_prepare(_runner(), _so([1]), SimpleNamespace(prompt_lens=None, row_req_ids=["d1"]))


def test_submit_prefill_forwards_segment_rows_and_refuses_a_whole_chunk_while_open():
    r = _runner()
    TTModelRunner.submit_prefill(r, TTModelRunner._segment_round_up(r, _input(), "r", 32).__class__(
        **{**{f.name: getattr(_input(), f.name) for f in fields(TTModelInput)}, "segment_rows": [0]}), [1])
    assert r.model.calls[-1]["segment_rows"] == [0] and r.notes == [1]
    r.model.open = 5
    with pytest.raises(RuntimeError, match="slot 5's segmented chunk is open"):
        TTModelRunner.submit_prefill(r, _input(rid="q"), [1])


def test_the_open_chunks_slot_stays_pinned_in_the_decode_gather():
    r = _runner()
    r._req_state_slot = {"d1": 0, "d2": 1, "d3": 7, "r": 3}
    r._segment_open = {"req_id": "r", "slot": 3}
    remap = TTModelRunner._decode_state_slot_remap(r, ["d1", "d2", "d3"])
    assert remap.tolist()[:3] == [0, 1, 7] and remap.tolist()[3] == 3  # unpinned, slot 3 would move to row 4
    assert sorted(remap.tolist()) == list(range(8))
    r._segment_open = None
    assert TTModelRunner._decode_state_slot_remap(r, ["d1", "d2", "d3"]).tolist()[4] == 3


def test_a_pinned_slot_inside_the_decode_rows_is_refused():
    r = _runner()
    r._req_state_slot = {"d1": 0, "d2": 4, "r": 1}
    r._segment_open = {"req_id": "r", "slot": 1}
    with pytest.raises(RuntimeError, match="inside the 2 decode rows"):
        TTModelRunner._decode_state_slot_remap(r, ["d1", "d2"])


@pytest.mark.parametrize("how", ["finished_req_ids", "preempted_req_ids"])
def test_a_request_leaving_inside_the_hold_drops_its_open_chunk(how):
    r = _runner()
    _step0(r)
    so = SimpleNamespace(finished_req_ids=set(), preempted_req_ids=set())
    getattr(so, how).add("r")
    TTModelRunner._segment_drop_if_gone(r, so)
    assert r.model.aborted == [5] and r._segment_open is None and r.segment_aborts == 1


def test_no_stamp_is_today():
    r = _runner()
    inp = _input()
    assert TTModelRunner._segment_prepare(r, SimpleNamespace(), inp) is inp and not r.model.calls
