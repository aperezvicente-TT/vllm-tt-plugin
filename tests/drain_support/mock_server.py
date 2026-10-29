# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Copy of tests/vllm/mock_server.py (s9i29h) with an overridable generator factory (TT_DRAIN_TEST_FACTORY) so the drain
repro can inject a silent engine-process exit / slow decode. CPU mock only; never opens a TT device."""
import os
import sys

from models.autoports.zai_org_glm_5_3_flash.tests.vllm import engine_harness as H

if __name__ == "__main__":
    os.environ["GLM53_VLLM_GENERATOR_FACTORY"] = os.environ.get("TT_DRAIN_TEST_FACTORY", H.MOCK_FACTORY)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    with H.fake_tt_mesh():
        from vllm.entrypoints.cli.main import main

        sys.argv = ["vllm", "serve", *sys.argv[1:]]
        main()
