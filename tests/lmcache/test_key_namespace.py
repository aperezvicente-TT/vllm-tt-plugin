"""B2 (program 2026-10-26T16:12:35Z): GLM53_KV_TILE_OBJECTS=1 only under the "#tt-tile-v2" LMCache key namespace (the
LMCache fork's extra_config key_namespace, appended to the key's model_name on every tier); the connector host refuses
to start otherwise, and a row-major engine refuses a tile namespace. The miss itself (a row-major object is a MISS for
a tile-mode engine on every tier) is the fork's tests/v1/test_glm53_key_namespace.py."""

import pytest

from vllm_tt_plugin.lmcache import connector as C

M = "/models/zai-org/GLM-5.3-Flash"


def test_tile_mode_needs_the_namespace(monkeypatch):
    monkeypatch.setenv("GLM53_KV_TILE_OBJECTS", "1")
    with pytest.raises(RuntimeError, match="needs the LMCache key namespace"):
        C.check_key_namespace(M)
    C.check_key_namespace(M + C.TILE_KEY_NAMESPACE)


def test_row_major_mode_refuses_a_tile_namespace(monkeypatch):
    monkeypatch.delenv("GLM53_KV_TILE_OBJECTS", raising=False)
    C.check_key_namespace(M)
    with pytest.raises(RuntimeError, match="GLM53_KV_TILE_OBJECTS is off"):
        C.check_key_namespace(M + C.TILE_KEY_NAMESPACE)


def test_no_engine_on_this_side_is_not_checked(monkeypatch):
    monkeypatch.setenv("GLM53_KV_TILE_OBJECTS", "1")
    C.check_key_namespace(None)


def test_the_host_checks_before_the_warm_start():
    """the connector host's constructor calls the check before it starts the warm-start reader (source order)"""
    import inspect

    from vllm_tt_plugin import kv_connector as K

    src = inspect.getsource(K)
    i, j = src.find("check_key_namespace(getattr("), src.find("self.warm_start_thread = _warm.start(")
    assert 0 < i < j
