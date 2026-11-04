"""Recurrent-state (MambaSpec) KV cache groups: kernel block sizes, state slot indices,
connector shadows and decode-join of requests whose state arrived from a remote prefill.
"""

from types import SimpleNamespace

import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_tt_plugin import kv_transfer
from vllm_tt_plugin.model_runner import TTModelRunner

ATTN_DECLARED = FullAttentionSpec(
    block_size=64, num_kv_heads=4, head_size=256, dtype=torch.bfloat16
)
ATTN = FullAttentionSpec(
    block_size=384, num_kv_heads=4, head_size=256, dtype=torch.bfloat16
)
STATE = MambaSpec(
    block_size=4096,
    shapes=((8192, 32), (32, 128, 128)),
    dtypes=(torch.bfloat16, torch.bfloat16),
    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
)


def _groups():
    return [
        SimpleNamespace(
            layer_names=["model.layers.0.linear_attn", "model.layers.4.linear_attn"],
            kv_cache_spec=STATE,
        ),
        SimpleNamespace(
            layer_names=["model.layers.3.self_attn", "model.layers.7.self_attn"],
            kv_cache_spec=ATTN,
        ),
        SimpleNamespace(
            layer_names=["model.layers.1.linear_attn", "model.layers.5.linear_attn"],
            kv_cache_spec=STATE,
        ),
    ]


class _Model:
    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        return {
            "model.layers.3.self_attn": ATTN_DECLARED,
            "model.layers.7.self_attn": ATTN_DECLARED,
        }


def _runner():
    r = SimpleNamespace(
        model=_Model(),
        vllm_config=None,
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    return r


def test_attention_keeps_its_kernel_block_size():
    """vLLM raised the attention block to a state page (384 tokens); the kernel still
    reads 64-token blocks, so the attention group's block table maps each block to six
    kernel blocks."""
    r = _runner()
    sizes = TTModelRunner._init_state_groups(r, _groups(), [4096, 384, 4096])
    assert sizes == [4096, 64, 4096]
    assert r._primary_group == 1 and r._state_groups == [0, 2]
    assert r._state_layers == {0: 0, 4: 0, 1: 2, 5: 2}


def test_state_slot_of_each_layer_is_its_groups_block():
    r = _runner()
    TTModelRunner._init_state_groups(r, _groups(), [4096, 384, 4096])
    tables = [
        torch.tensor([[7, 0], [9, 0]]),
        torch.zeros(2, 4, dtype=torch.int64),
        torch.tensor([[8, 0], [3, 0]]),
    ]
    got = TTModelRunner._mamba_state_indices(r, tables, 2)
    assert got == [{0: 7, 4: 7, 1: 8, 5: 8}, {0: 9, 4: 9, 1: 3, 5: 3}]


def test_no_state_groups_no_indices():
    r = _runner()
    groups = [
        SimpleNamespace(layer_names=["model.layers.0.self_attn"], kv_cache_spec=ATTN)
    ]
    assert TTModelRunner._init_state_groups(r, groups, [384]) == [384]
    assert TTModelRunner._mamba_state_indices(r, [torch.zeros(1, 1)], 1) is None


def test_state_shadow_is_one_page_per_block():
    cfg = SimpleNamespace(num_blocks=5, kv_cache_groups=_groups())
    shadows = kv_transfer.shadow_kv_caches(cfg)
    s = shadows["model.layers.0.linear_attn"]
    assert s.shape == (5, 1, 1, STATE.page_size_bytes) and s.dtype == torch.uint8
    assert s.untyped_storage().nbytes() == 1
    a = shadows["model.layers.3.self_attn"]
    assert (
        a.numel() * a.element_size()
        == 5 * ATTN.page_size_bytes
        == 5 * STATE.page_size_bytes
    )
