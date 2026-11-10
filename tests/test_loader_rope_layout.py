# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import pytest

from vllm_tt_plugin import loader


class _Model:
    model_capabilities = {"supports_hf_rope_kv": True}
    seen: dict = {}

    @classmethod
    def initialize_vllm_model(cls, hf_config, device, max_batch_size, **kwargs):
        cls.seen = kwargs
        return cls()


def _config(kv_transfer_config):
    return SimpleNamespace(
        device_config=SimpleNamespace(device="mesh"),
        kv_transfer_config=kv_transfer_config,
    )


@pytest.fixture
def load(monkeypatch):
    monkeypatch.setattr(loader, "get_tt_config", lambda cfg: {})
    monkeypatch.setattr(loader, "get_tt_data_parallel_size", lambda cfg: 1)
    monkeypatch.setattr(loader, "get_tt_max_batch_size", lambda cfg: 1)

    def run(model_class, kv_transfer_config):
        monkeypatch.setattr(
            loader, "get_model_architecture", lambda mc: (model_class, "arch")
        )
        model_config = SimpleNamespace(hf_config=None, max_model_len=128)
        loader.TTModelLoader.__new__(loader.TTModelLoader).load_model(
            _config(kv_transfer_config), model_config
        )
        return model_class.seen

    return run


def test_hf_rope_requested_only_with_a_kv_connector(load):
    assert load(_Model, SimpleNamespace(kv_connector="NixlConnector"))["use_hf_rope"]
    assert "use_hf_rope" not in load(_Model, None)


def test_models_without_the_declaration_are_unchanged(load):
    class Plain(_Model):
        model_capabilities = {}

    assert "use_hf_rope" not in load(Plain, SimpleNamespace(kv_connector="X"))
