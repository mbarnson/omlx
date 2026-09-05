# SPDX-License-Identifier: Apache-2.0
"""Exercise released family topologies, rotary math, and indexed checkpoints."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from omlx.patches.k2_horizon import apply_k2_horizon_patch
from omlx.patches.k2_horizon.checkpoint import checkpoint_files

FIXTURES = Path(__file__).parent / "fixtures" / "k2_horizon"
RELEASES = {
    name: revision
    for name, revision in json.loads((FIXTURES / "revisions.json").read_text()).items()
    if not name.endswith("-Uno")
}


def module():
    apply_k2_horizon_patch()
    return importlib.import_module("mlx_lm.models.k2_horizon")


def test_released_09_adapter_header_matches_base_projection_dimensions():
    """Compare actual cached tensor-header metadata with the released base."""
    config = release("0.9B")
    adapter = json.loads((FIXTURES / "0.9B-Uno-adapter_config.json").read_text())
    header = json.loads((FIXTURES / "0.9B-Uno-tensor-layout.json").read_text())
    revisions = json.loads((FIXTURES / "revisions.json").read_text())
    assert header["revision"] == revisions["0.9B-Uno"]
    width, intermediate = config["hidden_size"], config["intermediate_size"]
    query = config["num_attention_heads"] * config["head_dim"]
    kv = config["num_key_value_heads"] * config["head_dim"]
    shapes = {
        "self_attn.q_proj": (query, width),
        "self_attn.k_proj": (kv, width),
        "self_attn.v_proj": (kv, width),
        "self_attn.o_proj": (width, query),
        "mlp.gate_proj": (intermediate, width),
        "mlp.up_proj": (intermediate, width),
        "mlp.down_proj": (width, intermediate),
    }
    expected = {}
    for layer in range(config["num_hidden_layers"]):
        for name, (out_features, in_features) in shapes.items():
            prefix = f"model.layers.{layer}.{name}"
            expected[prefix + ".lora_A.weight"] = {
                "shape": [adapter["r"], in_features],
                "dtype": "BF16",
            }
            expected[prefix + ".lora_B.weight"] = {
                "shape": [out_features, adapter["r"]],
                "dtype": "BF16",
            }
    assert header["tensor_count"] == 392
    assert header["tensors"] == expected


def test_375b_router_and_expert_block_match_released_torch_source():
    """Exercise 192 experts/top-8 at reduced widths with the pinned HF code."""
    import importlib.util

    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location(
        "k2_375_reference", FIXTURES / "reference_375b_moe.py"
    )
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    config = tiny_config("375B-A23B")
    config.update(num_experts=192, num_experts_per_tok=8)
    torch.manual_seed(41)
    reference = (
        reference_module.K2HorizonSparseMoeBlock(SimpleNamespace(**config))
        .to(torch.bfloat16)
        .eval()
    )
    reference.gate.bias = torch.nn.Parameter(torch.linspace(-0.05, 0.05, 192))
    k2 = module()
    actual = k2.SparseMoeBlock(k2.ModelArgs.from_dict(config))

    def convert(tensor):
        return mx.array(tensor.detach().float().numpy(), dtype=mx.bfloat16)

    actual.gate.weight = convert(reference.gate.weight)
    actual.expert_bias = mx.array(reference.gate.bias.detach().numpy())
    for name in ("gate_proj", "up_proj", "down_proj"):
        getattr(actual.experts, name).weight = mx.stack(
            [convert(getattr(expert, name).weight) for expert in reference.experts]
        )
        getattr(actual.shared_experts, name).weight = convert(
            getattr(reference.shared_experts, name).weight
        )
    x = torch.randn(1, 4, config["hidden_size"], dtype=torch.bfloat16)
    with torch.inference_mode():
        expected, logits = reference(x)
    mx_x = convert(x)
    native_logits = k2.router_logits(mx_x, actual.gate.weight, actual.router_partitions)
    assert actual.router_partitions == 1
    np.testing.assert_allclose(
        np.array(native_logits).reshape(logits.shape),
        logits.float().numpy(),
        rtol=0,
        atol=0.008,
    )
    assert not np.array_equal(
        np.array(native_logits), np.array(k2.router_logits(mx_x, actual.gate.weight, 2))
    )
    selected, _ = k2.route(
        mx_x,
        actual.gate.weight,
        actual.expert_bias,
        8,
        config["router_scaling_factor"],
        partitions=1,
    )
    expected_selected = torch.topk(
        torch.sigmoid(logits.float()) + reference.gate.bias.detach(), 8, dim=-1
    ).indices.numpy()
    np.testing.assert_array_equal(
        np.sort(np.array(selected).reshape(4, 8), axis=-1),
        np.sort(expected_selected, axis=-1),
    )
    output = np.array(actual(mx_x).astype(mx.float32))
    target = expected.float().numpy()
    relative_rms = np.sqrt(np.mean((output - target) ** 2)) / np.sqrt(
        np.mean(target**2)
    )
    assert np.isfinite(output).all() and relative_rms < 0.02
    mova = k2.SparseMoeBlock(k2.ModelArgs.from_dict(tiny_config("MoVA-36B-A4B")))
    assert mova.router_partitions == 2


def release(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def tiny_config(name):
    config = release(name)
    sparse = config["num_experts"] > 0
    config.update(
        hidden_size=64,
        num_hidden_layers=3,
        intermediate_size=96,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rope_head_dim=8 if name == "375B-A23B" else 16,
        vocab_size=128,
        mlp_only_layers=[0] if sparse else [0, 1, 2],
    )
    if sparse:
        config.update(num_experts=6, num_experts_per_tok=2, moe_intermediate_size=32)
    if config.get("mova_num_experts", 0):
        config.update(mova_num_experts=5, mova_num_experts_per_tok=2)
    return config


@pytest.mark.parametrize("name", RELEASES)
def test_released_configs_and_module_topology(name):
    k2 = module()
    args = k2.ModelArgs.from_dict(release(name))
    assert args.rope_theta > 0
    model = k2.Model(k2.ModelArgs.from_dict(tiny_config(name)))
    for i, layer in enumerate(model.layers):
        sparse = name in ("375B-A23B", "MoVA-36B-A4B") and i > 0
        assert isinstance(layer.mlp, k2.SparseMoeBlock) == sparse
        assert ("v_experts" in layer.self_attn) == (sparse and name == "MoVA-36B-A4B")
        assert ("gate_proj" in layer.self_attn) == (name == "MoVA-36B-A4B")


@pytest.mark.parametrize("name", RELEASES)
def test_family_cached_forward_matches_full_forward(name):
    from mlx_lm.models.cache import make_prompt_cache

    k2 = module()
    mx.random.seed(17)
    model = k2.Model(k2.ModelArgs.from_dict(tiny_config(name)))
    model.set_dtype(mx.bfloat16)
    ids = mx.array([[2, 3, 5, 7]])
    full = model(ids).astype(mx.float32)
    cache = make_prompt_cache(model)
    parts = [model(ids[:, :2], cache=cache)]
    parts += [model(ids[:, i : i + 1], cache=cache) for i in range(2, 4)]
    incremental = mx.concatenate(parts, axis=1).astype(mx.float32)
    np.testing.assert_allclose(
        np.array(incremental), np.array(full), atol=0.04, rtol=0.04
    )


def test_partial_rope_leaves_tail_unchanged():
    k2 = module()
    attention = k2.Attention(
        k2.ModelArgs.from_dict(tiny_config("375B-A23B")), mova=False
    )
    x = mx.random.normal((1, 4, 3, 16))
    actual = attention.rope(x, offset=100)
    for start, end in ((4, 8), (12, 16)):
        np.testing.assert_array_equal(
            np.array(actual[..., start:end]), np.array(x[..., start:end])
        )
    torch = pytest.importorskip("torch")
    tx = torch.tensor(np.array(x))
    interleaved = tx.reshape(*tx.shape[:-1], 2, -1).transpose(-1, -2).reshape(tx.shape)
    rotate = (
        interleaved[..., :8]
        .reshape(*tx.shape[:-1], -1, 2)
        .transpose(-1, -2)
        .reshape(*tx.shape[:-1], 8)
    )
    angles = torch.arange(100, 103)[:, None] / (1e7 ** (torch.arange(0, 8, 2) / 8))
    first, second = rotate.chunk(2, dim=-1)
    rotate = torch.cat(
        [
            first * angles.cos() - second * angles.sin(),
            second * angles.cos() + first * angles.sin(),
        ],
        dim=-1,
    )
    rotated_interleaved = (
        rotate.reshape(*tx.shape[:-1], 2, -1)
        .transpose(-1, -2)
        .reshape(*tx.shape[:-1], 8)
    )
    joined = torch.cat([rotated_interleaved, interleaved[..., 8:]], dim=-1)
    expected = joined.reshape(*tx.shape[:-1], -1, 2).transpose(-1, -2).reshape(tx.shape)
    np.testing.assert_allclose(np.array(actual), expected.numpy(), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("offset", [0, 8191, 8192, 131071])
def test_yarn_matches_transformers_at_context_boundaries(offset):
    torch = pytest.importorskip("torch")
    from transformers.modeling_rope_utils import _compute_yarn_parameters

    k2 = module()
    args = k2.ModelArgs.from_dict(release("0.9B"))
    rope = k2.YarnRoPE(args)
    config = SimpleNamespace(
        rope_parameters=args.rope_parameters,
        head_dim=args.head_dim,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        standardize_rope_params=lambda: None,
    )
    inv_freq, scale = _compute_yarn_parameters(config)
    np.testing.assert_allclose(
        np.array(rope._inv_freq), inv_freq.numpy(), rtol=2e-7, atol=1e-9
    )
    x = torch.linspace(-1, 1, 128).reshape(1, 2, 1, 64).to(torch.bfloat16)
    angles = torch.tensor([offset], dtype=torch.float32)[:, None] * inv_freq
    cos, sin = (angles.cos() * scale).to(x.dtype), (angles.sin() * scale).to(x.dtype)
    first, second = x.chunk(2, dim=-1)
    expected = torch.cat(
        [first * cos - second * sin, second * cos + first * sin], dim=-1
    )
    actual = rope(mx.array(x.float().numpy()).astype(mx.bfloat16), offset=offset)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        expected.float().numpy(),
        atol=0.016,
        rtol=0.016,
    )


def test_yarn_batched_offsets_match_individual_offsets():
    k2 = module()
    rope = k2.YarnRoPE(k2.ModelArgs.from_dict(release("0.9B")))
    x = mx.random.normal((2, 4, 3, 64)).astype(mx.bfloat16)
    batch = rope(x, offset=mx.array([4, 8192]))
    individual = mx.concatenate([rope(x[:1], offset=4), rope(x[1:], offset=8192)])
    np.testing.assert_array_equal(
        np.array(batch.astype(mx.float32)), np.array(individual.astype(mx.float32))
    )


def test_config_rejects_conflicting_theta_and_incomplete_yarn():
    k2 = module()
    config = release("0.9B")
    config["rope_parameters"]["rope_theta"] = 1e7
    with pytest.raises(ValueError, match="rope_theta"):
        k2.ModelArgs.from_dict(config)
    config = release("0.9B")
    del config["rope_parameters"]["attention_factor"]
    with pytest.raises(ValueError, match="attention_factor"):
        k2.ModelArgs.from_dict(config)


def save_checkpoint(path, name="3.7B"):
    k2 = module()
    config = tiny_config(name)
    model = k2.Model(k2.ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)
    weights = dict(tree_flatten(model.parameters()))
    path.mkdir(exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))
    filename = "pytorch_model-00001-of-00001.safetensors"
    mx.save_safetensors(str(path / filename), weights)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: filename for k in weights}})
    )
    return model, filename


@pytest.mark.parametrize("lazy", [False, True])
def test_indexed_pytorch_shards_load_strictly_and_survive_temporary_view(
    tmp_path, lazy
):
    model, filename = save_checkpoint(tmp_path)
    from mlx_lm.utils import load_model

    loaded, _ = load_model(tmp_path, lazy=lazy, strict=True)
    ids = mx.array([[2, 3, 4]])
    np.testing.assert_array_equal(
        np.array(model(ids).astype(mx.float32)),
        np.array(loaded(ids).astype(mx.float32)),
    )
    assert list(tmp_path.glob("*.safetensors")) == [tmp_path / filename]


def test_indexed_hf_discovery_and_missing_shard(tmp_path):
    from omlx.model_discovery import _is_hf_cache_mlx_compatible

    _, filename = save_checkpoint(tmp_path)
    assert _is_hf_cache_mlx_compatible(tmp_path, "IFM/K2-Horizon-3.7B")
    (tmp_path / filename).unlink()
    assert not _is_hf_cache_mlx_compatible(tmp_path, "IFM/K2-Horizon-3.7B")
    with pytest.raises(FileNotFoundError, match="Missing K2 checkpoint shard"):
        checkpoint_files(tmp_path)


def test_manifest_rejects_tensor_ownership_and_path_escape(tmp_path):
    save_checkpoint(tmp_path)
    index = tmp_path / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    data["weight_map"].pop(next(iter(data["weight_map"])))
    index.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not owned"):
        checkpoint_files(tmp_path)
    index.write_text(json.dumps({"weight_map": {"a": "../other.safetensors"}}))
    with pytest.raises(ValueError, match="filename"):
        checkpoint_files(tmp_path)
