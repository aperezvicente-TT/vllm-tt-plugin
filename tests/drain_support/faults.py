# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Generator factory for the drain repro: the GLM mock generator with
- TT_DRAIN_TEST_DECODE_SLEEP_S: per-decode-step sleep (long generations stay in flight across the drain)
- TT_DRAIN_TEST_EXIT_FLAG=<path>: once <path> exists, the next decode step ends the process with os._exit(7) -- the exit
  tt-umd's SIGBUS handler takes (silicon_tlb_window.cpp sigbus_handler: _exit(sig)) when a reset zapped the BAR
  mappings under a live process: no Python log, no exception, no core.
Every decode step appends nothing; the marker (GLM53_MOCK_MARKER) gets {"event": "decode_exit7"} before the exit."""
import json
import os
import time

from models.autoports.zai_org_glm_5_3_flash.tests.vllm import glm53_mock_stack as S

_make_backend = S.make_backend


def _wrapped_make_backend(*a, **kw):
    backend = _make_backend(*a, **kw)
    decode = backend.decode
    sleep_s = float(os.environ.get("TT_DRAIN_TEST_DECODE_SLEEP_S", "0") or 0)
    flag = os.environ.get("TT_DRAIN_TEST_EXIT_FLAG", "")

    def slow_decode(io, kv, width):
        if flag and os.path.exists(flag):
            marker = os.environ.get("GLM53_MOCK_MARKER")
            if marker:
                with open(marker, "a") as f:
                    f.write(json.dumps({"event": "decode_exit7", "pid": os.getpid()}) + "\n")
            os._exit(7)
        if sleep_s:
            time.sleep(sleep_s)
        return decode(io, kv, width)

    backend.decode = slow_decode
    return backend


def build_drain_generator(*a, **kw):
    S.make_backend = _wrapped_make_backend
    try:
        return S.build_mock_generator(*a, **kw)
    finally:
        S.make_backend = _make_backend
