# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
import pytest


@pytest.fixture(autouse=True)
def reset_tt_platform_class_state():
    """Neutralize the host-suite fixture of the same name for the LMCache tests.

    The A2 connector tests never touch ``TTPlatform`` (they import lmcache and the
    plugin's ``lmcache`` package only), and the root fixture imports vLLM's engine
    modules at setup time. Tests in this directory that do drive vLLM (A3's
    harness) patch the platform state they need explicitly.
    """
    yield
