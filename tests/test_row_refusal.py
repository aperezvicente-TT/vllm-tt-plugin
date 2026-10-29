# SPDX-License-Identifier: Apache-2.0
"""Stage 17 (the Galaxy blk2 engine death 2026-10-29T01:53Z): a prefill whose persistent-batch page-table row is shorter
than the positions it writes is a REQUEST error, not a latched device fault. The runner refuses the step before any
device work (an empty output per row, the requests recorded in vllm_tt_plugin.row_check); the scheduler finishes them
aborted at its next schedule (stop_reason 'page row: ...', the client notified), and the engine keeps stepping. A step
that mixes short and sound rows still raises. Fakes only, as test_plan_refusal.py."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.request import RequestStatus

from vllm_tt_plugin import plan_check, row_check
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import TTScheduler

BS = 64


def _row(n_blocks, width=128):
    """a block-table row with n_blocks live (ids 1..n) and vLLM's null block 0 as padding"""
    r = torch.zeros(width, dtype=torch.int32)
    r[:n_blocks] = torch.arange(1, n_blocks + 1, dtype=torch.int32)
    return r


def _runner():
    return SimpleNamespace(cache_config=SimpleNamespace(block_size=BS))


def _input(rows, lens, req_ids):
    return SimpleNamespace(block_tables=torch.stack([_row(n) for n in rows]), prompt_lens=torch.tensor(lens),
                           row_req_ids=list(req_ids))  # fmt: skip


@pytest.fixture(autouse=True)
def _clean():
    row_check.take()
    plan_check.register(None)
    yield
    row_check.take()


def test_the_blk2_row_is_found_short_and_a_covering_row_is_not():
    """65 blocks against a chunk ending at 6,144 (96 blocks): short; 96 blocks: sound; a partial last block counts"""
    mi = _input([65, 96, 97], [6144, 6144, 6145], ["blk2", "sound", "partial"])
    assert TTModelRunner._short_prefill_rows(_runner(), mi) == {"blk2": (65, 96, 6144)}


def test_a_step_of_only_short_rows_is_refused_without_device_work():
    runner = _runner()
    mi = _input([65], [6144], ["r"])
    out = TTModelRunner._refuse_prefill_step(runner, None, mi, {"r": (65, 96, 6144)}, None)
    assert out.req_ids == ["r"] and out.req_id_to_index == {"r": 0}
    assert out.sampled_token_ids == [[]], "no token for a refused prefill (as an intermediate chunk)"
    assert runner._num_refused_prefill_rows == 1
    assert row_check.take() == {"r": "page row: 65 blocks < the 96 positions [0, 6144) need"}


def test_a_step_mixing_short_and_sound_rows_still_raises():
    mi = _input([65, 96], [6144, 6144], ["short", "sound"])
    with pytest.raises(ValueError, match="cannot be refused"):
        TTModelRunner._refuse_prefill_step(_runner(), None, mi, {"short": (65, 96, 6144)}, None)
    assert row_check.take() == {}


class _Req:
    def __init__(self, rid, status=RequestStatus.RUNNING):
        self.request_id, self.status, self.stop_reason = rid, status, None

    def is_finished(self):
        return RequestStatus.is_finished(self.status)


def _sched(running, waiting=()):
    finished = []

    def finish_requests(ids, status):
        out = [r for r in list(running) + list(waiting) if r.request_id in set(ids) and not r.is_finished()]
        for r in out:
            r.status = status
        finished.append((sorted(ids), status))
        return out

    return SimpleNamespace(running=list(running), waiting=list(waiting), finish_requests=finish_requests,
                           _finished=finished)  # fmt: skip


def test_the_scheduler_finishes_a_refused_request_aborted_and_keeps_the_rest():
    a, b = _Req("a"), _Req("b")
    sched = _sched([a, b])
    row_check.refuse("a", "page row: 65 blocks < the 96 positions [0, 6144) need")
    TTScheduler._refuse_unplannable(sched)
    assert sched._finished == [(["a"], RequestStatus.FINISHED_ABORTED)]
    assert a.stop_reason.startswith("page row: 65 blocks") and not b.is_finished()
    assert sched._refused_outputs == [a], "the client hears it at the next update_from_output (_emit_refused)"


def test_a_refusal_for_a_request_already_gone_is_dropped():
    sched = _sched([_Req("b")])
    row_check.refuse("gone", "page row: 1 blocks < the 2 positions [0, 100) need")
    TTScheduler._refuse_unplannable(sched)
    assert sched._finished == [] and row_check.take() == {}
