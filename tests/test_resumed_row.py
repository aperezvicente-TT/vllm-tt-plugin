# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""v3b7 (program 2026-10-29T02:30:48Z (2)(B), the blk2 kill): a request RESUMED from preemption while still in the
persistent batch must leave it and come back with the resumed block ids -- upstream vLLM's rule (unscheduled = batched -
(scheduled - resumed)); the plugin's copy evicted only the unscheduled, so such a request kept its PRE-preemption row and
had the resumed blocks APPENDED to it: freed blocks (now another request's) first, its own after. The real
TTModelRunner._update_states over a real InputBatch."""

from types import SimpleNamespace

import numpy as np

from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner
from tests.test_mixed_steps import _request, _scheduler


def _runner():
    r = TTModelRunner.__new__(TTModelRunner)
    r.tt_per_lane_max_num_seqs = 2
    r._req_state_slot, r._pending_state_slot_settle, r.requests = {}, None, {}
    r.encoder_cache, r._decode_layout_changed_since_last_decode = {}, False
    r.model = SimpleNamespace(release_request=lambda *a, **k: None)
    r.input_batch = InputBatch(max_num_reqs=2, max_model_len=4096, max_num_batched_tokens=4096, vocab_size=64,
                               block_sizes=[64], kernel_block_sizes=[64])  # fmt: skip
    return r


def _row(r, rid):
    i = r.input_batch.req_id_to_index[rid]
    bt = r.input_batch.block_table.block_tables[0]
    return [int(x) for x in np.asarray(bt.get_cpu_tensor()[i, : int(bt.num_blocks_per_row[i])])]


def _out(new=(), cached=(), resumed=(), sched=None, preempted=None):
    """the SchedulerOutput fields _update_states reads"""
    return SimpleNamespace(
        finished_req_ids=set(), preempted_req_ids=preempted, free_encoder_mm_hashes=[],
        num_scheduled_tokens=sched or {}, scheduled_new_reqs=list(new),
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[c[0] for c in cached], resumed_req_ids=set(resumed), num_computed_tokens=[c[1] for c in cached],
            new_block_ids=[c[2] for c in cached], new_token_ids=[], all_token_ids={},
            num_output_tokens=[0 for _ in cached]))  # fmt: skip


def test_a_resumed_request_still_batched_gets_its_resumed_row():
    s = _scheduler(mixed=False)
    s.add_request(_request("R", 200, max_tokens=64))
    first = s.schedule()  # (the real scheduler's new-request record: R's prompt in one step, 4 blocks)
    r = _runner()
    TTModelRunner._update_states(r, first)
    held = _row(r, "R")
    assert held == list(s.kv_cache_manager.get_block_ids("R")[0]) and len(held) == 4
    fresh = [b + 100 for b in held][:3]
    # one output resumes R with fresh blocks while R is still batched (the TT scheduler's two passes in one step can
    # do that; upstream evicts a resumed request for its forced-preemption case)
    TTModelRunner._update_states(r, _out(cached=[("R", 0, (list(fresh),))], resumed={"R"}, sched={"R": 128}))
    assert _row(r, "R") == fresh, _row(r, "R")  # not held + fresh: the freed blocks first
    assert list(r.requests["R"].block_ids[0]) == fresh


def test_a_short_row_is_logged_once_with_what_the_runner_knew(monkeypatch):
    import vllm_tt_plugin.model_runner as MR

    logged = []
    monkeypatch.setattr(MR.logger, "error", lambda msg, *a: logged.append(msg % a))
    s = _scheduler(mixed=False)
    s.add_request(_request("R", 200, max_tokens=64))
    first = s.schedule()
    r = _runner()
    TTModelRunner._update_states(r, first)
    assert not getattr(r, "runner_rows_short", 0)  # 200 tokens on 4 blocks of 64
    # a chunk that needs blocks the row lacks (the runner's cached block ids never got them)
    r.input_batch.num_computed_tokens_cpu[r.input_batch.req_id_to_index["R"]] = 256
    TTModelRunner._check_scheduled_rows(r, _out(sched={"R": 128}), set())
    TTModelRunner._check_scheduled_rows(r, _out(sched={"R": 128}), set())
    assert r.runner_rows_short == 2
    msgs = [m for m in logged if "covers 4 blocks < the 6" in m]
    assert len(msgs) == 1 and "batched last step True" in msgs[0], logged
