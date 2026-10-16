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


def test_native_speculative_helpers_still_work():
    """mlx-vlm's own helpers pass speculative_verify themselves."""
    from omlx.patches.mlx_vlm_mtp import nemotron_omni_vlm_runtime

    nemotron_omni_vlm_runtime.apply()
    lm = _tiny_language_model()
    cache = lm.make_cache()
    lm(mx.array([[3, 9, 27]]), cache=cache)
    captured, hidden, states = lm.speculative_verify_dflash_hidden(mx.array([[1, 2]]), cache, [0, 2])
    assert len(captured) == 2 and states
    lm.rollback_speculative_cache(cache, states, 0, 2)
    hidden, shared, states = lm.speculative_verify_hidden(mx.array([[4, 5]]), cache)
    assert states


def test_bolt_loader_hooks_only_touch_their_own_load(tmp_path, monkeypatch):
    import json
    import threading

    import mlx_vlm.utils as vlm_utils

    from omlx.patches import nemotron_bolt
    from omlx.patches.nemotron_bolt import loader

    monkeypatch.setattr(nemotron_bolt, "is_supported_config", lambda c: True)
    monkeypatch.setattr(loader, "is_supported_config", lambda c: True)
    monkeypatch.setattr(loader, "validate_layout", lambda c: None)
    bolt_dir, other_dir = tmp_path / "bolt", tmp_path / "other"
    for d, cfg in ((bolt_dir, {"model_type": "nemotron_h_omni", "quantization_config": {"quant_method": "modelopt"}}),
                   (other_dir, {"model_type": "nemotron_h_omni", "quantization_config": {"quant_method": "fp8"}})):
        d.mkdir()
        (d / "config.json").write_text(json.dumps(cfg))
    seen = {}

    def fake_load(path, **kwargs):
        seen["bolt"] = vlm_utils.load_config(path)
        seen["bolt_class"] = vlm_utils.get_model_and_args(seen["bolt"], path)[1]
        seen["other_same_thread"] = vlm_utils.load_config(other_dir)
        result = {}

        def other_thread():
            result["config"] = vlm_utils.load_config(bolt_dir)
            result["class"] = vlm_utils.get_model_and_args({"model_type": "nemotron_h_omni"}, bolt_dir)[1]

        t = threading.Thread(target=other_thread)
        t.start()
        t.join()
        seen["other_thread"] = result
        return "model", "processor"

    monkeypatch.setattr(vlm_utils, "load", fake_load)
    assert loader.load(bolt_dir) == ("model", "processor")
    assert "quantization_config" not in seen["bolt"] and seen["bolt_class"] == "nemotron_h_omni"
    assert seen["other_same_thread"]["quantization_config"]["quant_method"] == "fp8"
    assert "quantization_config" in seen["other_thread"]["config"]
    assert seen["other_thread"]["class"] == "nemotron_h_nano_omni"
    assert vlm_utils.load is fake_load and vlm_utils.load_config.__module__ == "mlx_vlm.utils"


def test_captures_follow_requested_order_on_every_path():
    from omlx.patches.mlx_vlm_mtp import nemotron_omni_vlm_runtime

    nemotron_omni_vlm_runtime.apply()
    lm = _tiny_language_model()
    ids = mx.array([[3, 9, 27]])
    sink = []
    lm.backbone(ids, cache=lm.make_cache(), capture_layer_ids=[0, 2], hidden_sink=sink)
    # Prefill seeding (captures only).
    seed = lm(ids, cache=lm.make_cache(), capture_layer_ids=[2, 0]).hidden_states
    assert len(seed) == 2
    assert mx.allclose(seed[0], sink[1]).item() and mx.allclose(seed[1], sink[0]).item()
    # Observed plain decode.
    seen = []
    lm._omlx_drafter = type(
        "D", (), {"scope_uids": (1,), "target_layer_ids": [2, 0],
                  "observe": lambda self, scope, captured: seen.append(captured)}
    )()
    cache = lm.make_cache()
    lm(ids[:, :2], cache=cache)
    lm(ids[:, 2:], cache=cache)
    step = []
    lm.backbone(ids, cache=lm.make_cache(), capture_layer_ids=[0, 2], hidden_sink=step)

    def nearer(got, want, other):
        return float(mx.abs(got - want).max()) < float(mx.abs(got - other).max())

    layer0, layer2 = step[0][:, -1:], step[1][:, -1:]
    assert nearer(seen[0][0], layer2, layer0) and nearer(seen[0][1], layer0, layer2)
    del lm._omlx_drafter
    # Verify: requested order, then the final hidden state.
    out = lm(ids, cache=lm.make_cache(), return_hidden=True, capture_layer_ids=[2, 0])
    assert len(out.hidden_states) == 3
    assert nearer(out.hidden_states[0], sink[1], sink[0])


def test_concurrent_first_application(monkeypatch):
    import threading

    import omlx.patches.mlx_vlm_nemotron_omni_compat as compat

    if compat.upstream_supported():
        pytest.skip("installed mlx-vlm already has Nemotron Omni")
    monkeypatch.setattr(compat, "_APPLIED", False)
    errors, results = [], []
    barrier = threading.Barrier(4)

    def worker():
        try:
            barrier.wait()
            results.append(compat.apply_mlx_vlm_nemotron_omni_compat_patch())
            from mlx_vlm.utils import get_model_and_args

            get_model_and_args({"model_type": "nemotron_h_omni"})
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert results.count(True) == 1 and results.count(False) == 3
