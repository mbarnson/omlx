# SPDX-License-Identifier: Apache-2.0
"""Nemotron 3.5 Bolt in NVIDIA's ModelOpt mixed IQ2_XXS/Q8_0/NVFP4 form."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches import nemotron_bolt
from omlx.patches.nemotron_bolt import model as bolt

BLOCKS = ["mamba", "moe", "attention", "mamba", "moe"]
EXPERTS = {1: 8, 4: 12}  # Puzzle: per-layer expert counts


def _text_config() -> dict:
    return {
        "model_type": "nemotron_h_puzzle",
        "vocab_size": 512,
        "hidden_size": 256,
        "intermediate_size": 256,
        "num_hidden_layers": len(BLOCKS),
        "max_position_embeddings": 4096,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "attention_bias": False,
        "mamba_num_heads": 8,
        "mamba_head_dim": 32,
        "mamba_proj_bias": False,
        "ssm_state_size": 32,
        "conv_kernel": 4,
        "n_groups": 2,
        "mlp_bias": False,
        "layer_norm_epsilon": 1e-5,
        "use_bias": False,
        "use_conv_bias": True,
        "moe_intermediate_size": 256,
        "moe_shared_expert_intermediate_size": 256,
        "moe_latent_size": 256,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "time_step_limit": [0.0, {"__float__": "Infinity"}],
        "block_configs": [
            {"block_type": b, **({"n_routed_experts": EXPERTS[i]} if i in EXPERTS else {})}
            for i, b in enumerate(BLOCKS)
        ],
    }


def _quantized_layers() -> dict:
    layers = {"language_model.lm_head": {"quant_algo": "Q8_0"}}
    for i, block in enumerate(BLOCKS):
        p = f"language_model.backbone.layers.{i}.mixer"
        if block == "mamba":
            for proj in ("in_proj", "out_proj"):
                layers[f"{p}.{proj}"] = {"quant_algo": "W4A16_NVFP4"}
        elif block == "attention":
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                layers[f"{p}.{proj}"] = {"quant_algo": "Q8_0"}
        else:
            for proj in ("fc1_latent_proj", "fc2_latent_proj"):
                layers[f"{p}.{proj}"] = {"quant_algo": "Q8_0"}
            for e in range(EXPERTS[i]):
                for proj in ("up_proj", "down_proj"):
                    layers[f"{p}.experts.{e}.{proj}"] = {"quant_algo": "IQ2_XXS"}
            for proj in ("up_proj", "down_proj"):
                layers[f"{p}.shared_experts.{proj}"] = {"quant_algo": "IQ2_XXS"}
    return layers


def _config() -> dict:
    return {
        "model_type": "nemotron_h_omni",
        "architectures": ["NemotronH_Omni_Reasoning_V3"],
        "auto_map": {"AutoModel": "modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3"},
        "llm_config": _text_config(),
        "vision_config": {"model_type": "radio"},
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": _quantized_layers(),
        },
    }


# --- checkpoint-format tensors -----------------------------------------------------

_rng = np.random.default_rng(0)


def _fp16_bytes(value, shape):
    return np.full(shape, value, np.float16).view(np.uint8).reshape(*shape, 2)


def _q8_0(out_dims, in_dims, d=0.002):
    nb = in_dims // 32
    q = _rng.integers(-127, 128, (out_dims, nb, 32), dtype=np.int8).view(np.uint8)
    return mx.array(np.concatenate([_fp16_bytes(d, (out_dims, nb)), q], axis=-1))


def _iq2_xxs(out_dims, in_dims, d=0.0005):
    nb = in_dims // 256
    payload = _rng.integers(0, 256, (out_dims, nb, 64), dtype=np.uint8)
    return mx.array(np.concatenate([_fp16_bytes(d, (out_dims, nb)), payload], axis=-1))


def _nvfp4(prefix, out_dims, in_dims, weights):
    weights[f"{prefix}.weight"] = mx.array(_rng.integers(0, 256, (out_dims, in_dims // 2), dtype=np.uint8))
    weights[f"{prefix}.weight_scale"] = mx.full((out_dims, in_dims // 16), 0x38, mx.uint8)  # e4m3 1.0
    weights[f"{prefix}.weight_scale_2"] = mx.array(0.02, mx.float32)


def _bf16(*shape, scale=0.05):
    return (mx.random.normal(shape) * scale).astype(mx.bfloat16)


def _checkpoint(cfg: dict) -> dict:
    mx.random.seed(0)
    hidden, vocab = cfg["hidden_size"], cfg["vocab_size"]
    w = {
        "language_model.backbone.embeddings.weight": _bf16(vocab, hidden, scale=0.5),
        "language_model.backbone.norm_f.weight": mx.ones((hidden,), mx.bfloat16),
        "language_model.lm_head.weight": _q8_0(vocab, hidden),
    }
    heads, hd, ds, g = cfg["mamba_num_heads"], cfg["mamba_head_dim"], cfg["ssm_state_size"], cfg["n_groups"]
    inner = heads * hd
    conv_dim = inner + 2 * g * ds
    for i, block in enumerate(BLOCKS):
        p = f"language_model.backbone.layers.{i}"
        w[f"{p}.norm.weight"] = mx.ones((hidden,), mx.bfloat16)
        m = f"{p}.mixer"
        if block == "mamba":
            _nvfp4(f"{m}.in_proj", inner + conv_dim + heads, hidden, w)
            _nvfp4(f"{m}.out_proj", hidden, inner, w)
            w[f"{m}.conv1d.weight"] = _bf16(conv_dim, 1, cfg["conv_kernel"], scale=0.3)
            w[f"{m}.conv1d.bias"] = _bf16(conv_dim, scale=0.1)
            w[f"{m}.A_log"] = mx.zeros((heads,), mx.bfloat16)
            w[f"{m}.D"] = mx.ones((heads,), mx.bfloat16)
            w[f"{m}.dt_bias"] = _bf16(heads, scale=0.5)
            w[f"{m}.norm.weight"] = mx.ones((inner,), mx.bfloat16)
        elif block == "attention":
            q, kv = cfg["num_attention_heads"] * 128, cfg["num_key_value_heads"] * 128
            w[f"{m}.q_proj.weight"] = _q8_0(q, hidden)
            w[f"{m}.k_proj.weight"] = _q8_0(kv, hidden)
            w[f"{m}.v_proj.weight"] = _q8_0(kv, hidden)
            w[f"{m}.o_proj.weight"] = _q8_0(hidden, q)
        else:
            experts, latent, ffn = EXPERTS[i], cfg["moe_latent_size"], cfg["moe_intermediate_size"]
            shared = cfg["moe_shared_expert_intermediate_size"]
            w[f"{m}.gate.weight"] = _bf16(experts, hidden, scale=0.2)
            w[f"{m}.gate.e_score_correction_bias"] = mx.zeros((experts,), mx.float32)
            w[f"{m}.fc1_latent_proj.weight"] = _q8_0(latent, hidden)
            w[f"{m}.fc2_latent_proj.weight"] = _q8_0(hidden, latent)
            for e in range(experts):
                w[f"{m}.experts.{e}.up_proj.weight"] = _iq2_xxs(ffn, latent)
                w[f"{m}.experts.{e}.down_proj.weight"] = _iq2_xxs(latent, ffn)
            w[f"{m}.shared_experts.up_proj.weight"] = _iq2_xxs(shared, hidden)
            w[f"{m}.shared_experts.down_proj.weight"] = _iq2_xxs(hidden, shared)
    return w


@pytest.fixture(scope="module")
def language_model():
    cfg = bolt.TextConfig.from_dict(_text_config())
    weights = bolt.Model.sanitize(SimpleNamespace(config=SimpleNamespace(text_config=cfg)), _checkpoint(_text_config()))
    lm = bolt.LanguageModel(cfg)
    prefix = "language_model."
    lm.load_weights([(k[len(prefix):], v) for k, v in weights.items()], strict=True)
    mx.eval(lm.parameters())
    return lm


# --- configuration --------------------------------------------------------------------


def test_detects_nvidia_mixed_checkpoint():
    config = _config()
    assert nemotron_bolt.is_supported_config(config)
    nemotron_bolt.validate_layout(config)


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(model_file="nemotron_bolt.py"),
        lambda c: c.pop("quantization_config"),
        lambda c: c["quantization_config"].update(quant_method="compressed-tensors"),
        lambda c: c["llm_config"].pop("block_configs"),
        lambda c: c["quantization_config"]["quantized_layers"].update(x={"quant_algo": "INT4_AWQ"}),
    ],
)
def test_other_checkpoints_are_not_claimed(change):
    config = _config()
    change(config)
    assert not nemotron_bolt.is_supported_config(config)


def test_layout_mismatch_is_refused():
    config = _config()
    layers = config["quantization_config"]["quantized_layers"]
    layers["language_model.backbone.layers.0.mixer.in_proj"] = {"quant_algo": "Q8_0"}
    with pytest.raises(ValueError, match="quantization layout"):
        nemotron_bolt.validate_layout(config)


def test_checkpoint_files_requires_every_shard(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}})
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        nemotron_bolt.checkpoint_files(tmp_path)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"x")
    assert len(nemotron_bolt.checkpoint_files(tmp_path)) == 2


def test_custom_quantization_dispatches_to_bolt_loader(tmp_path, monkeypatch):
    from omlx.utils import model_loading

    (tmp_path / "config.json").write_text(json.dumps(_config()))
    calls = []
    monkeypatch.setattr(nemotron_bolt, "load", lambda name: calls.append(name) or ("model", "processor"))
    assert model_loading.maybe_load_custom_quantization(str(tmp_path), is_vlm=True) == ("model", "processor")
    assert calls == [str(tmp_path)]
    with pytest.raises(ValueError, match="VLM"):
        model_loading.maybe_load_custom_quantization(str(tmp_path), is_vlm=False)


def test_dflash_admits_nemotron_omni():
    from omlx.utils.model_loading import dflash_batched_supported

    assert dflash_batched_supported("nemotron_h_omni", "vlm")
    assert dflash_batched_supported("nemotron_h_nano_omni", "vlm")
    assert not dflash_batched_supported("nemotron_h_omni", "batched")


def test_hf_cache_accepts_complete_nvidia_checkpoint(tmp_path):
    from omlx.model_discovery import _is_hf_cache_mlx_compatible

    (tmp_path / "config.json").write_text(json.dumps(_config()))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}})
    )
    assert not _is_hf_cache_mlx_compatible(tmp_path, "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB")
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"x")
    assert _is_hf_cache_mlx_compatible(tmp_path, "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB")


def test_text_config_puzzle_layers():
    cfg = bolt.TextConfig.from_dict(_text_config())
    assert cfg.hybrid_override_pattern == ["M", "E", "*", "M", "E"]
    assert cfg.for_layer(4).n_routed_experts == 12
    assert cfg.time_step_limit == (0.0, float("inf"))


# --- arithmetic (Metal) ---------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["bf16", "q8", "iq2", "nvfp4"])
def test_mmv_matches_dense_and_rows_are_independent(fmt):
    out_dims, in_dims = 72, 512
    x = (mx.random.normal((6, in_dims)) * 0.5).astype(mx.bfloat16)
    if fmt == "bf16":
        w = _bf16(out_dims, in_dims, scale=0.1)
        dense, args = w.astype(mx.float32), (w,)
    elif fmt == "q8":
        q, s, b = bolt.q8_0_to_affine(_q8_0(out_dims, in_dims))
        dense = mx.dequantize(q, s, b, group_size=32, bits=8).astype(mx.float32)
        args = (q, s, b)
    elif fmt == "iq2":
        qs, d = bolt.iq2xxs_planes(_iq2_xxs(out_dims, in_dims)[None])
        dense = bolt.iq2xxs_dequantize(qs[0], d[0], mx.float32)
        args = (qs[0], d[0])
    else:
        w = {}
        _nvfp4("p", out_dims, in_dims, w)
        packed, scales = w["p.weight"].view(mx.uint32), w["p.weight_scale"]
        lin = bolt.NVFP4Linear(in_dims, out_dims)
        lin.weight, lin.scales, lin.global_scale = packed, scales, mx.array(1.0)
        dense = mx.concatenate([lin(x[i : i + 1].astype(mx.float32)) for i in range(6)])
        args = (packed, scales)
    y = bolt.mmv(x, fmt, args[0], out_dims, in_dims, *args[1:], out_dtype=mx.float32)
    ref = dense if fmt == "nvfp4" else x.astype(mx.float32) @ dense.T
    assert float(mx.abs(y - ref).max()) <= 2e-2 * float(mx.abs(ref).max()) + 1e-4
    y3 = bolt.mmv(x[:3], fmt, args[0], out_dims, in_dims, *args[1:], out_dtype=mx.float32)
    assert mx.array_equal(y3, y[:3]).item()



def test_mmv_cold_concurrent_first_use(monkeypatch):
    """Engines on separate threads may hit mmv's shared constants together."""
    import threading

    monkeypatch.setattr(bolt, "_CONST", {})
    w = _bf16(16, 64, scale=0.1)
    x = mx.ones((2, 64), mx.bfloat16)
    mx.eval(w, x)
    start, errors, outs = threading.Barrier(4), [], []

    def run():
        try:
            with mx.stream(mx.new_stream(mx.gpu)):  # as each engine's worker has
                start.wait()
                y = bolt.mmv(x, "bf16", w, 16, 64, out_dtype=mx.float32)
                mx.eval(y)
            outs.append(y)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(mx.array_equal(y, outs[0]).item() for y in outs)
    assert set(bolt._CONST) == {"mmv"}


def test_decode_agrees_with_prompt_processing(language_model):
    lm = language_model
    ids = mx.array([[5, 17, 300, 41, 99, 7, 256, 11, 3, 480, 64, 128]])
    full = lm(ids, cache=lm.make_cache()).logits[0, -1].astype(mx.float32)
    cache = lm.make_cache()
    lm(ids[:, :-1], cache=cache)
    step = lm(ids[:, -1:], cache=cache).logits[0, -1].astype(mx.float32)
    assert float(mx.abs(full - step).max()) <= 0.05 * float(mx.abs(full).max())
    assert int(full.argmax()) == int(step.argmax())


def test_verify_rollback_is_exact(language_model):
    """A 4-token verify rolled back to 2 tokens equals a 2-token verify."""
    lm = language_model
    prompt = mx.array([[5, 17, 300, 41, 99, 7, 256, 11, 3, 480]])
    block = mx.array([[64, 128, 9, 200]])
    nxt = mx.array([[33]])

    a = lm.make_cache()
    lm(prompt, cache=a)
    lm(block, cache=a)
    lm.spec_rollback(a, 2, 4)
    la = lm(nxt, cache=a).logits

    b = lm.make_cache()
    lm(prompt, cache=b)
    lm(block[:, :2], cache=b)
    lb = lm(nxt, cache=b).logits
    assert mx.array_equal(la, lb).item()


def test_rollback_speculative_cache_accepts_lists(language_model):
    lm = language_model
    prompt = mx.array([[5, 17, 300, 41, 99, 7, 256, 11]])
    c = lm.make_cache()
    lm(prompt, cache=c)
    lm(mx.array([[1, 2, 3]]), cache=c, return_hidden=True)
    lm.rollback_speculative_cache(c, [], [1], 3)
    attn = [x for x in c if type(x).__name__ == "KVCache"][0]
    assert attn.offset == prompt.shape[1] + 2


def test_captures_follow_requested_order(language_model):
    lm = language_model
    ids = mx.array([[5, 17, 300]])
    out = lm(ids, cache=lm.make_cache(), capture_layer_ids=[3, 0], return_hidden=True)
    sink = []
    final = lm.backbone(ids, cache=lm.make_cache(), capture_layer_ids=[0, 3], hidden_sink=sink)
    assert len(out.hidden_states) == 3
    assert mx.array_equal(out.hidden_states[0], sink[1]).item()
    assert mx.array_equal(out.hidden_states[1], sink[0]).item()
    assert mx.array_equal(out.hidden_states[2], final).item()
    assert out.gdn_states == []
