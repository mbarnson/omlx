# SPDX-License-Identifier: Apache-2.0
"""Tests for the K2 Horizon MoVA mlx-lm compatibility patch.

Gate A covers registration and config validation, Gate B covers component
numerics against straightforward references, and Gate C covers a tiny
synthetic model end to end through mlx-lm's batch generator.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import json
import math
import sys
import types

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from omlx.patches import k2_horizon


def _minimal_k2_config(**overrides):
    cfg = dict(
        model_type="k2_horizon",
        hidden_size=64,
        num_hidden_layers=3,
        intermediate_size=96,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        rms_norm_eps=1e-6,
        layernorm_num_groups=2,
        mlp_only_layers=[0],
        decoder_sparse_step=1,
        num_experts=6,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        num_shared_experts=1,
        mova_num_experts=5,
        mova_num_experts_per_tok=2,
        moe_gate_bias=True,
        norm_topk_prob=True,
        router_score_func="sigmoid",
        router_scaling_factor=2.5,
        attention_gate_func="softplus",
        query_key_norm=False,
        attention_bias=False,
        rope_parameters={"rope_theta": 10000000.0, "rope_type": "default"},
        rope_head_dim=16,
        use_sliding_window=False,
        sliding_window=None,
        tie_word_embeddings=False,
    )
    cfg.update(overrides)
    return cfg


def _k2_module():
    k2_horizon.apply_k2_horizon_patch()
    return importlib.import_module("mlx_lm.models.k2_horizon")


def _tiny_model(seed: int = 0, **overrides):
    k2 = _k2_module()
    mx.random.seed(seed)
    args = k2.ModelArgs.from_dict(_minimal_k2_config(**overrides))
    model = k2.Model(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return k2, args, model


def _hf_style_weights(model, args) -> dict[str, mx.array]:
    """Unstack a model's parameters into the checkpoint's per-expert names."""
    weights = {}
    for key, value in tree_flatten(model.parameters()):
        if ".mlp.experts." in key:
            prefix, rest = key.split(".mlp.experts.")
            proj = rest.removesuffix(".weight")
            for e in range(value.shape[0]):
                weights[f"{prefix}.mlp.experts.{e}.{proj}.weight"] = value[e]
        elif key.endswith("self_attn.v_experts.weight"):
            for e in range(value.shape[0]):
                weights[key.replace("v_experts.weight", f"v_experts.{e}.weight")] = (
                    value[e]
                )
        elif key.endswith("mlp.expert_bias"):
            weights[key.replace("expert_bias", "gate.bias")] = value
        elif key.endswith("self_attn.v_expert_bias"):
            weights[key.replace("v_expert_bias", "v_router.bias")] = value
        else:
            weights[key] = value
    return weights


def _reset_patch_state():
    k2_horizon._APPLIED = False
    for name in ("mlx_lm.models.k2_horizon", "mlx_lm.tool_parsers.k2_horizon"):
        sys.modules.pop(name, None)
    for package, attr in (
        ("mlx_lm.models", "k2_horizon"),
        ("mlx_lm.tool_parsers", "k2_horizon"),
    ):
        pkg = importlib.import_module(package)
        if hasattr(pkg, attr):
            delattr(pkg, attr)


# ---------------------------------------------------------------------------
# Gate A: registration and config validation
# ---------------------------------------------------------------------------


def test_apply_registers_model_and_parser_modules():
    _reset_patch_state()

    assert k2_horizon.apply_k2_horizon_patch() is True

    model_module = importlib.import_module("mlx_lm.models.k2_horizon")
    parser_module = importlib.import_module("mlx_lm.tool_parsers.k2_horizon")
    assert model_module.__package__ == "mlx_lm.models"
    assert parser_module.tool_call_start == "<ifm|tool_calls>"
    assert parser_module.tool_call_end == "</ifm|tool_calls>"
    assert importlib.import_module("mlx_lm.models").k2_horizon is model_module
    assert importlib.import_module("mlx_lm.tool_parsers").k2_horizon is parser_module


def test_apply_is_idempotent():
    _reset_patch_state()

    first = k2_horizon.apply_k2_horizon_patch()
    second = k2_horizon.apply_k2_horizon_patch()

    assert first is True
    assert second is False
    assert k2_horizon.is_applied() is True


@pytest.mark.parametrize(
    ("upstream_model", "upstream_parser"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_registration_is_independent_per_module(
    monkeypatch, upstream_model, upstream_parser
):
    """Each vendored module is registered exactly when its upstream twin is absent."""
    _reset_patch_state()
    present = {
        "mlx_lm.models.k2_horizon": upstream_model,
        "mlx_lm.tool_parsers.k2_horizon": upstream_parser,
    }
    registered: list[str] = []
    original_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name in present:
            if present[name]:
                return types.SimpleNamespace(__name__=name)
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return original_import_module(name)

    monkeypatch.setattr(k2_horizon.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        k2_horizon,
        "_register_module",
        lambda qualname, filename, package: registered.append(qualname),
    )

    applied = k2_horizon.apply_k2_horizon_patch()

    expected = [name for name, is_present in present.items() if not is_present]
    assert registered == expected
    assert applied is bool(expected)


def test_nested_upstream_import_failure_propagates(monkeypatch):
    _reset_patch_state()
    original_import_module = importlib.import_module

    def broken_upstream(name: str):
        if name == "mlx_lm.models.k2_horizon":
            raise ModuleNotFoundError(
                "No module named 'missing_dependency'", name="missing_dependency"
            )
        return original_import_module(name)

    monkeypatch.setattr(k2_horizon.importlib, "import_module", broken_upstream)

    with pytest.raises(ModuleNotFoundError, match="missing_dependency"):
        k2_horizon.apply_k2_horizon_patch()


def test_module_registration_cleans_up_after_execution_failure(monkeypatch):
    module_name = "mlx_lm.models.k2_horizon_broken_test"

    class FailingLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            raise RuntimeError("simulated vendored module failure")

    monkeypatch.setattr(
        k2_horizon.importlib.util,
        "spec_from_file_location",
        lambda *_: importlib.machinery.ModuleSpec(module_name, FailingLoader()),
    )
    sys.modules.pop(module_name, None)

    with pytest.raises(RuntimeError, match="simulated vendored module failure"):
        k2_horizon._register_module(module_name, "not-used.py", "mlx_lm.models")

    assert module_name not in sys.modules


def test_get_classes_resolves_k2_horizon():
    _k2_module()
    from mlx_lm.utils import _get_classes

    model_cls, args_cls = _get_classes(_minimal_k2_config())

    assert model_cls.__name__ == "Model"
    assert args_cls.__name__ == "ModelArgs"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mova_num_experts", 0),
        ("num_experts", 0),
        ("num_shared_experts", 0),
        ("moe_gate_bias", False),
        ("norm_topk_prob", False),
        ("router_score_func", "softmax"),
        ("attention_gate_func", "silu"),
        ("attention_gate_func", None),
        ("layernorm_num_groups", 1),
        ("query_key_norm", True),
        ("rope_head_dim", 8),
        ("rope_parameters", {"rope_theta": 1e7, "rope_type": "yarn"}),
        ("use_sliding_window", True),
        ("sliding_window", 4096),
        ("mlp_only_layers", [0, 1, 2]),
        ("router_scaling_factor", None),
    ],
)
def test_unsupported_config_variants_fail_naming_the_field(field, value):
    k2 = _k2_module()
    reported = "rope_parameters.rope_type" if field == "rope_parameters" else field

    with pytest.raises(ValueError, match=reported):
        k2.ModelArgs.from_dict(_minimal_k2_config(**{field: value}))


def test_pre_load_dispatch_applies_k2_horizon_patch(tmp_path):
    _reset_patch_state()
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "k2_horizon"}))
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(str(tmp_path))

    assert k2_horizon.is_applied() is True
    assert "mlx_lm.models.k2_horizon" in sys.modules


def test_pre_load_dispatch_skips_other_model_types(tmp_path, monkeypatch):
    calls: list[None] = []
    monkeypatch.setattr(
        k2_horizon, "apply_k2_horizon_patch", lambda: calls.append(None) or True
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(str(tmp_path))

    assert calls == []


def test_distributed_planning_rejects_k2_horizon():
    _k2_module()
    from omlx.cluster import planner
    from omlx.cluster.tensor_strategies import supports_model_type

    assert supports_model_type("k2_horizon") is False
    assert planner._supports_pipeline({"model_type": "k2_horizon"}) is False


# ---------------------------------------------------------------------------
# Gate B: component numerics
# ---------------------------------------------------------------------------


def _reference_grouped_rms_norm(x, weight, groups, eps):
    x32 = x.astype(mx.float32)
    grouped = x32.reshape(*x.shape[:-1], groups, -1)
    variance = mx.mean(grouped * grouped, axis=-1, keepdims=True)
    normed = (grouped * mx.rsqrt(variance + eps)).reshape(x.shape)
    return (weight.astype(mx.float32) * normed).astype(x.dtype)


def test_grouped_rms_norm_matches_fp32_reference():
    k2 = _k2_module()
    mx.random.seed(1)
    norm = k2.GroupedRMSNorm(8, 2, 1e-6)
    norm.weight = mx.random.normal((8,)).astype(mx.bfloat16)
    x = (mx.random.normal((3, 8)) * mx.array([[1.0] * 4 + [50.0] * 4])).astype(
        mx.bfloat16
    )

    expected = _reference_grouped_rms_norm(x, norm.weight, 2, 1e-6)
    actual = norm(x)

    assert actual.dtype == mx.bfloat16
    assert mx.allclose(actual, expected, atol=1e-2, rtol=1e-2)


def test_grouped_rms_norm_differs_from_single_group_on_non_uniform_input():
    k2 = _k2_module()
    x = mx.array([[1.0, 1.0, 1.0, 1.0, 50.0, 50.0, 50.0, 50.0]])
    grouped = k2.GroupedRMSNorm(8, 2, 1e-6)(x)
    single = nn.RMSNorm(8, eps=1e-6)(x)

    assert not mx.allclose(grouped, single, atol=1e-3)
    assert mx.allclose(grouped[0, :4], grouped[0, 4:], atol=1e-5)


def test_router_logits_use_two_part_bf16_accumulation():
    k2 = _k2_module()
    mx.random.seed(2)
    x = mx.random.normal((16, 64)).astype(mx.bfloat16)
    w = mx.random.normal((5, 64)).astype(mx.bfloat16)
    left = (x[:, :32] @ w[:, :32].T).astype(mx.float32)
    right = (x[:, 32:] @ w[:, 32:].T).astype(mx.float32)

    logits = k2.router_logits(x, w)

    assert logits.dtype == mx.float32
    assert mx.array_equal(logits, left + right)
    with pytest.raises(ValueError, match="BF16"):
        k2.router_logits(x.astype(mx.float16), w)


def test_two_part_and_full_gemm_disagree_on_near_ties():
    k2 = _k2_module()
    for seed in range(200):
        mx.random.seed(seed)
        x = mx.random.normal((256, 64)).astype(mx.bfloat16)
        w = mx.random.normal((8, 64)).astype(mx.bfloat16)
        two_part = mx.argmax(k2.router_logits(x, w), axis=-1)
        full = mx.argmax((x @ w.T).astype(mx.float32), axis=-1)
        if not mx.array_equal(two_part, full):
            return
    pytest.fail("no near-tie where partition rounding changed the route")


def test_router_bias_changes_selection_but_not_mixture_weights():
    k2 = _k2_module()
    x = mx.zeros((1, 64), dtype=mx.bfloat16)
    x = x.at[0, 0].add(mx.array(1.0, dtype=mx.bfloat16))
    w = mx.zeros((3, 64), dtype=mx.bfloat16)
    w = w.at[0, 0].add(mx.array(2.0, dtype=mx.bfloat16))
    w = w.at[1, 0].add(mx.array(1.0, dtype=mx.bfloat16))
    scores = mx.sigmoid(mx.array([2.0, 1.0, 0.0]))
    no_bias = mx.zeros((3,))
    flip_bias = mx.array([0.0, 0.0, 0.9])

    inds, weights = k2.route(x, w, no_bias, top_k=2, scaling_factor=2.5)
    assert sorted(inds[0].tolist()) == [0, 1]

    inds, weights = k2.route(x, w, flip_bias, top_k=2, scaling_factor=2.5)
    assert sorted(inds[0].tolist()) == [0, 2]
    picked = mx.take_along_axis(scores[None], inds, axis=-1)
    expected = picked / mx.sum(picked, axis=-1, keepdims=True) * 2.5
    assert mx.allclose(weights, expected, atol=1e-6)
    assert math.isclose(float(mx.sum(weights)), 2.5, rel_tol=1e-6)


def test_softplus_beta_ln2_matches_torch_form_and_stays_finite():
    k2 = _k2_module()
    x = mx.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    expected = mx.log1p(mx.exp(x * math.log(2))) / math.log(2)
    assert mx.allclose(k2.softplus_beta_ln2(x), expected, atol=1e-6)

    large = k2.softplus_beta_ln2(mx.array([200.0]))
    assert bool(mx.isfinite(large).all())
    assert abs(float(large[0]) - 200.0) < 1e-3

    beta_one = nn.softplus(mx.array([0.5]))
    assert not mx.allclose(k2.softplus_beta_ln2(mx.array([0.5])), beta_one, atol=1e-3)


def _dense_mlp(x, gate_w, up_w, down_w):
    return (nn.silu(x @ gate_w.T) * (x @ up_w.T)) @ down_w.T


def test_sparse_moe_block_sums_routed_and_shared_outputs():
    k2, args, model = _tiny_model()
    block = model.layers[1].mlp
    mx.random.seed(3)
    x = mx.random.normal((1, 4, args.hidden_size)).astype(mx.bfloat16)

    inds, weights = k2.route(
        x, block.gate.weight, block.expert_bias, args.num_experts_per_tok, 2.5
    )
    x32 = x.astype(mx.float32)
    expected = _dense_mlp(
        x32,
        block.shared_experts.gate_proj.weight.astype(mx.float32),
        block.shared_experts.up_proj.weight.astype(mx.float32),
        block.shared_experts.down_proj.weight.astype(mx.float32),
    )
    for t in range(4):
        for slot in range(args.num_experts_per_tok):
            e = int(inds[0, t, slot])
            out = _dense_mlp(
                x32[0, t],
                block.experts.gate_proj.weight[e].astype(mx.float32),
                block.experts.up_proj.weight[e].astype(mx.float32),
                block.experts.down_proj.weight[e].astype(mx.float32),
            )
            expected = expected.at[0, t].add(out * weights[0, t, slot])

    actual = block(x).astype(mx.float32)
    assert mx.allclose(actual, expected, atol=5e-2, rtol=5e-2)


def test_mova_applies_silu_before_router_weighting():
    k2, args, model = _tiny_model()
    attn = model.layers[1].self_attn
    mx.random.seed(4)
    x = mx.random.normal((1, 3, args.hidden_size)).astype(mx.bfloat16)

    inds, weights = k2.route(
        x, attn.v_router.weight, attn.v_expert_bias, args.mova_num_experts_per_tok, 2.5
    )
    x32 = x.astype(mx.float32)
    expected = mx.zeros((1, 3, args.num_key_value_heads * args.head_dim))
    wrong_order = mx.zeros_like(expected)
    for t in range(3):
        for slot in range(args.mova_num_experts_per_tok):
            e = int(inds[0, t, slot])
            proj = x32[0, t] @ attn.v_experts.weight[e].astype(mx.float32).T
            expected = expected.at[0, t].add(nn.silu(proj) * weights[0, t, slot])
            wrong_order = wrong_order.at[0, t].add(nn.silu(proj * weights[0, t, slot]))

    actual = attn._values(x).astype(mx.float32)
    assert mx.allclose(actual, expected, atol=5e-2, rtol=5e-2)
    assert not mx.allclose(actual, wrong_order, atol=5e-2, rtol=5e-2)


def test_dense_prefix_and_sparse_layers_are_constructed_from_config():
    k2, args, model = _tiny_model()

    assert isinstance(model.layers[0].mlp, k2.MLP)
    assert "v_proj" in model.layers[0].self_attn
    assert "v_router" not in model.layers[0].self_attn
    for layer in model.layers[1:]:
        assert isinstance(layer.mlp, k2.SparseMoeBlock)
        assert "v_router" in layer.self_attn
        assert "v_proj" not in layer.self_attn
    assert isinstance(model.model.norm, k2.GroupedRMSNorm)
    assert model.model.norm.groups == 2


def test_sanitize_stacks_experts_in_order_and_renames_router_biases():
    k2, args, model = _tiny_model()
    hf_weights = _hf_style_weights(model, args)
    expected = dict(tree_flatten(model.parameters()))

    sanitized = model.sanitize(dict(hf_weights))

    assert set(sanitized) == set(expected)
    for key, value in expected.items():
        assert mx.array_equal(sanitized[key], value), key
    fresh = k2.Model(args)
    fresh.load_weights(list(sanitized.items()), strict=True)


def test_sanitize_rejects_missing_and_mixed_expert_groups():
    k2, args, model = _tiny_model()
    hf_weights = _hf_style_weights(model, args)

    missing = dict(hf_weights)
    missing.pop("model.layers.1.mlp.experts.3.up_proj.weight")
    with pytest.raises(ValueError, match="experts.3.up_proj"):
        model.sanitize(missing)

    mixed = dict(hf_weights)
    mixed["model.layers.1.self_attn.v_experts.weight"] = mx.zeros(
        (
            args.mova_num_experts,
            args.num_key_value_heads * args.head_dim,
            args.hidden_size,
        )
    )
    with pytest.raises(ValueError, match="v_experts.0.weight"):
        model.sanitize(mixed)

    both_biases = dict(hf_weights)
    both_biases["model.layers.1.mlp.expert_bias"] = mx.zeros((args.num_experts,))
    with pytest.raises(ValueError, match="expert_bias"):
        model.sanitize(both_biases)


def test_model_predicates_protect_routers_and_router_biases():
    _, _, model = _tiny_model()

    assert model.quant_predicate("model.layers.1.mlp.gate", None) is False
    assert model.quant_predicate("model.layers.1.self_attn.v_router", None) is False
    assert model.quant_predicate("model.layers.1.mlp.experts.gate_proj", None) is True
    assert model.cast_predicate("model.layers.1.mlp.expert_bias") is False
    assert model.cast_predicate("model.layers.1.self_attn.v_expert_bias") is False
    assert model.cast_predicate("model.layers.1.mlp.gate.weight") is True


# ---------------------------------------------------------------------------
# Gate C: tiny model end to end
# ---------------------------------------------------------------------------


def _greedy(model, prompt: list[int], steps: int) -> list[int]:
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    tokens = list(prompt)
    logits = model(mx.array([prompt]), cache=cache)
    for _ in range(steps):
        nxt = int(mx.argmax(logits[:, -1, :], axis=-1)[0])
        tokens.append(nxt)
        logits = model(mx.array([[nxt]]), cache=cache)
    return tokens[len(prompt) :]


def test_tiny_model_forward_and_cache_are_consistent():
    from mlx_lm.models.cache import KVCache, make_prompt_cache

    _, args, model = _tiny_model()
    prompt = mx.array([[1, 2, 3, 4, 5, 6, 7]])

    full = model(prompt)
    mx.eval(full)
    assert full.shape == (1, 7, args.vocab_size)
    assert bool(mx.isfinite(full.astype(mx.float32)).all())

    cache = make_prompt_cache(model)
    assert len(cache) == args.num_hidden_layers
    assert all(type(c) is KVCache for c in cache)
    model(prompt[:, :5], cache=cache)
    model(prompt[:, 5:6], cache=cache)
    decoded = model(prompt[:, 6:7], cache=cache)
    assert mx.allclose(
        decoded[:, 0].astype(mx.float32), full[:, 6].astype(mx.float32), atol=2e-2
    )


def test_batch_generator_matches_separate_greedy_requests():
    from mlx_lm.generate import BatchGenerator

    _, _, model = _tiny_model(seed=7)
    prompts = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12]]
    steps = 4
    expected = [_greedy(model, p, steps) for p in prompts]

    generator = BatchGenerator(model, max_tokens=steps)
    generator.insert(prompts, max_tokens=[steps, steps])
    generated: dict[int, list[int]] = {}
    finished: set[int] = set()
    while len(finished) < len(prompts):
        for response in generator.next_generated():
            generated.setdefault(response.uid, []).append(response.token)
            if response.finish_reason is not None:
                finished.add(response.uid)

    assert sorted(generated.values()) == sorted(expected)
