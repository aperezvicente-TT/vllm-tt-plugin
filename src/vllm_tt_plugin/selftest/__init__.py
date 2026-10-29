# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""GLM-5.3-Flash serving self-test: strict teacher-forcing rank check + greedy
smoke + decode-after-prefill, run at startup before the endpoint is reachable
(``endpoint.py``) and as a one-shot CLI (``python -m vllm_tt_plugin.selftest``).
``score.py`` is the pure-python scorer both share."""
