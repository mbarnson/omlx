# SPDX-License-Identifier: Apache-2.0
"""mlx-vlm Nemotron Omni (Blaizzy/mlx-vlm#2528) overlay and its DFlash contract."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from omlx.patches.mlx_vlm_nemotron_omni_compat import (
    PR_MERGE_SHA,
    apply_mlx_vlm_nemotron_omni_compat_patch,
    upstream_supported,
)


@pytest.fixture(scope="module", autouse=True)
def applied():
    assert apply_mlx_vlm_nemotron_omni_compat_patch() or upstream_supported()


def test_model_types_resolve():
    from mlx_vlm.prompt_utils import MODEL_CONFIG
    from mlx_vlm.utils import MODEL_REMAPPING, get_model_and_args

    assert MODEL_REMAPPING["nemotron_h_omni"] == "nemotron_h_nano_omni"
    assert MODEL_REMAPPING["nemotron_h_puzzle"] == "nemotron_h"
    assert "nemotron_h_omni" in MODEL_CONFIG
    module, model_type = get_model_and_args({"model_type": "nemotron_h_omni"})
    assert model_type == "nemotron_h_nano_omni"
    assert hasattr(module, "Model")


def test_vendored_tree_is_the_upstream_change():
    import mlx_vlm.models.nemotron_h.config as config

    assert hasattr(config.ModelConfig, "for_layer")  # Puzzle block_configs
    if not upstream_supported():
        vendor = Path(__file__).resolve().parents[1] / "omlx/patches/mlx_vlm_nemotron_omni_compat/vendor"
        assert str(vendor.resolve()) in str(Path(config.__file__).resolve())
    assert len(PR_MERGE_SHA) == 40


def _tiny_language_model():
    from mlx_vlm.models.nemotron_h.config import ModelConfig
    from mlx_vlm.models.nemotron_h.language import LanguageModel

    cfg = ModelConfig.from_dict(
        {
            "model_type": "nemotron_h", "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "max_position_embeddings": 512, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 32, "attention_bias": False, "mamba_num_heads": 4, "mamba_head_dim": 16,
            "mamba_proj_bias": False, "ssm_state_size": 16, "conv_kernel": 4, "n_groups": 2,
            "mlp_bias": False, "layer_norm_epsilon": 1e-5, "use_bias": False, "use_conv_bias": True,
            "moe_intermediate_size": 32, "moe_shared_expert_intermediate_size": 32,
            "moe_latent_size": 32, "n_routed_experts": 4, "n_shared_experts": 1,
            "num_experts_per_tok": 2, "n_group": 1, "topk_group": 1, "norm_topk_prob": True,
            "routed_scaling_factor": 1.0,
            "block_configs": [
                {"block_type": "mamba"}, {"block_type": "moe", "n_routed_experts": 6},
                {"block_type": "attention"}, {"block_type": "mamba"},
            ],
        }
    )
    mx.random.seed(0)
    lm = LanguageModel(cfg)
    mx.eval(lm.parameters())
    return lm


def test_dflash_verify_contract_and_rollback():
    from omlx.patches.mlx_vlm_mtp import nemotron_omni_vlm_runtime

    assert nemotron_omni_vlm_runtime.apply()
    lm = _tiny_language_model()
    assert lm._omlx_drafter_single_row
    prompt = mx.array([[3, 9, 27, 81, 50, 7]])
    block = mx.array([[11, 12, 13, 14]])

    cache = lm.make_cache()
    lm(prompt, cache=cache)
    out = lm(block, cache=cache, return_hidden=True, n_confirmed=1, capture_layer_ids=[2, 0])
    assert len(out.hidden_states) == 3 and out.gdn_states
    sink = []
    lm.backbone(block, cache=lm.make_cache(), capture_layer_ids=[0, 2], hidden_sink=sink)
    lm.rollback_speculative_cache(cache, out.gdn_states, 1, 4)
    after = lm(mx.array([[5]]), cache=cache).logits

    direct = lm.make_cache()
    lm(prompt, cache=direct)
    lm(block[:, :2], cache=direct)
    expected = lm(mx.array([[5]]), cache=direct).logits
    assert mx.allclose(after, expected, atol=2e-2, rtol=2e-2).item()


def test_plain_decode_feeds_the_drafter():
    from omlx.patches.mlx_vlm_mtp import nemotron_omni_vlm_runtime

    nemotron_omni_vlm_runtime.apply()
    lm = _tiny_language_model()
    seen = []
    lm._omlx_drafter = type(
        "D", (), {"scope_uids": (7,), "target_layer_ids": [0, 2], "observe": lambda self, s, c: seen.append((s, len(c)))}
    )()
    cache = lm.make_cache()
    lm(mx.array([[3, 9]]), cache=cache)
    lm(mx.array([[4]]), cache=cache)
    assert seen == [((7,), 2)]
