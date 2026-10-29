# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""LMCache integration for the Tenstorrent plugin (GLM-5.3-Flash tiered latent cache).

``connector`` holds the LMCache device-side connector (``GPUConnectorInterface``)
that moves latent pages between LMCache memory objects and the plugin's paged KV
host transfer buffers; ``page_contract`` the page geometry it codes against;
``mock_paged_cache`` a host-only stand-in for the device caches and the plugin's
``CopyBlocksOp``; ``ttnn_page_ops`` the reference device-side page copy op.

Importing this package does not import ``lmcache``; the submodules that need it do.
"""
