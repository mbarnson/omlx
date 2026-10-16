# SPDX-License-Identifier: Apache-2.0
"""NVIDIA LiLiCorr drafter: config, loading, the chain head and selection."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from omlx.speculative.dflash import lilicorr, selection
from omlx.speculative.dflash.drafter import DFlashDrafter


def _config(**dflash):
    head = {
        "mask_token_id": 3,
        "target_layer_ids": [0, 2],
        "projector_type": "lilicorr",
        "lilicorr_candidate_topk": 4,
        "lilicorr_hidden_size": 64,
        "lilicorr_num_layers": 1,
        "lilicorr_num_heads": 2,
        "lilicorr_mlp_ratio": 2.0,
        "lilicorr_factor_dim": 32,
        "lilicorr_vector_eps": 1e-4,
        "lilicorr_logit_scale": 8.0,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "swa_window_size": 64,
        "attention_sink_bias": True,
        "causal": False,
    }
    head.update(dflash)
    return {
        "architectures": ["LiLiCorrDraftModel"],
        "model_type": "qwen3",
        "block_size": 4,
        "dflash_config": head,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "intermediate_size": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "layer_types": ["sliding_attention"] * 2,
        "sliding_window": 64,
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "W4A16_NVFP4"},
    }


def _nvfp4(name, out_dims, in_dims, weights, rng, bias=False):
    weights[f"{name}.weight"] = mx.array(rng.integers(0, 256, (out_dims, in_dims // 2), dtype=np.uint8))
    weights[f"{name}.weight_scale"] = mx.full((out_dims, in_dims // 16), 0x30, mx.uint8)
    weights[f"{name}.weight_scale_2"] = mx.array(0.05, mx.float32)
    if bias:
        weights[f"{name}.bias"] = mx.zeros((out_dims,), mx.bfloat16)


def _checkpoint(tmp_path, config):
    """A random checkpoint in NVIDIA's published layout."""
    rng = np.random.default_rng(1)
    model = lilicorr.LiLiCorrDraftModel(lilicorr.LiLiCorrConfig.from_dict(config))
    nvfp4_dims = {}
    hidden, dims = config["hidden_size"], config["dflash_config"]["lilicorr_hidden_size"]
    nvfp4_dims["fc"] = (hidden, 2 * hidden, False)
    for i in range(config["num_hidden_layers"]):
        for proj, o, n in (("gate_proj", 128, hidden), ("up_proj", 128, hidden), ("down_proj", hidden, 128)):
            nvfp4_dims[f"layers.{i}.mlp.{proj}"] = (o, n, False)
    for name in ("token_proj", "pass_hidden_proj", "context_proj"):
        nvfp4_dims[f"lilicorr.{name}"] = (dims, hidden, True)
    nvfp4_dims["lilicorr.layers.0.attn.out_proj"] = (dims, dims, True)
    nvfp4_dims["lilicorr.layers.0.mlp.up_proj"] = (2 * dims, dims, True)
    nvfp4_dims["lilicorr.layers.0.mlp.down_proj"] = (dims, 2 * dims, True)
    weights = {}
    for name, (o, n, bias) in nvfp4_dims.items():
        _nvfp4(name, o, n, weights, rng, bias)
    skip = {f"{n}.{t}" for n in nvfp4_dims for t in ("packed", "block_scales", "global_scale", "bias")}
    for key, value in dict(__import__("mlx.utils", fromlist=["x"]).tree_flatten(model.parameters())).items():
        if key in skip:
            continue
        weights[key] = (mx.random.normal(value.shape) * 0.05).astype(mx.bfloat16)
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    return tmp_path


def test_config_requires_shape_free_head_fields():
    config = _config()
    del config["dflash_config"]["lilicorr_logit_scale"]
    with pytest.raises(ValueError, match="lilicorr_logit_scale"):
        lilicorr.LiLiCorrConfig.from_dict(config)


def test_detection():
    assert lilicorr.is_lilicorr_config(_config())
    plain = _config()
    plain["dflash_config"]["projector_type"] = "dflash2"
    plain["architectures"] = ["DFlash2DraftModel"]
    assert not lilicorr.is_lilicorr_config(plain)


def test_loads_published_layout_and_proposes(tmp_path):
    path = _checkpoint(tmp_path, _config())
    model = lilicorr.load_lilicorr_drafter(path)
    vocab = 256
    embed = mx.random.normal((vocab, 64)).astype(mx.bfloat16)
    model.embed_tokens = lambda ids: embed[ids]
    slots = mx.random.normal((2, 3, 64)).astype(mx.bfloat16)
    logits = mx.random.normal((2, 3, vocab))
    tokens, candidates, picks = model.propose(slots, logits, mx.random.normal((2, 64)))
    assert tokens.shape == (2, 3) and candidates.shape == (2, 3, 4) and picks.shape == (2, 3)
    # The committed path is made of each slot's candidates.
    assert mx.array_equal(mx.take_along_axis(candidates, picks[..., None], axis=-1)[..., 0], tokens).item()
    # Candidates are the per-slot top-k in rank order.
    top = mx.argsort(-logits, axis=-1)[..., :4]
    assert mx.array_equal(candidates, top).item()


def test_greedy_path_follows_transitions():
    start = mx.array([[0.0, 5.0, 1.0]])
    pair = mx.array(
        [[[[0, 0, 0], [0, 0, 9], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [7, 0, 0]]]], dtype=mx.float32
    )
    # slot 0 -> 1 (start), slot 1 -> 2 (row 1), slot 2 -> 0 (row 2)
    assert lilicorr.greedy_path(start, pair).tolist() == [[1, 2, 0]]


def test_sampled_rows_get_point_mass_q():
    tokens = mx.array([[7, 9]], dtype=mx.int32)
    candidates = mx.array([[[7, 1, 2], [3, 9, 4]]], dtype=mx.int32)
    picks = mx.array([[0, 1]])
    model = SimpleNamespace(propose=lambda hidden, logits, anchor: (tokens, candidates, picks))
    logits = mx.zeros((1, 2, 16))
    [(greedy_tokens, greedy_q)] = selection.select_lilicorr(model, None, logits, None, [None])
    assert greedy_q == [] and greedy_tokens.tolist() == [[7, 9]]
    [(_, q)] = selection.select_lilicorr(model, None, logits, None, [object()])
    assert len(q) == 2
    for position, token in enumerate((7, 9)):
        row = np.array(q[position])
        assert row[token] == 0.0
        assert np.isneginf(np.delete(row, token)).all()


def test_anchor_is_the_newest_committed_row():
    drafter = DFlashDrafter.__new__(DFlashDrafter)
    h_ctx = mx.arange(2 * 3 * 4, dtype=mx.float32).reshape(2, 3, 4)
    rows = [
        (SimpleNamespace(uid=0), SimpleNamespace(history=[]), None, None, None),
        (SimpleNamespace(uid=1), SimpleNamespace(history=[mx.full((1, 2, 4), -1.0)]), None, None, None),
    ]
    anchors = drafter._lilicorr_anchors(rows, h_ctx, [2, 0], None)
    assert anchors[0].tolist() == h_ctx[0, 1].tolist()  # last of two new rows
    assert anchors[1].tolist() == [-1.0] * 4  # no new rows: newest history row
    counts = [mx.array([3]), mx.array([1])]
    anchors = drafter._lilicorr_anchors(rows, h_ctx, [3, 3], counts)
    assert anchors[0].tolist() == h_ctx[0, 2].tolist()
    assert anchors[1].tolist() == h_ctx[1, 0].tolist()


def test_math_constants():
    # vector_eps / logit_scale reach the head unchanged.
    head = lilicorr.LiLiCorrDraftModel(lilicorr.LiLiCorrConfig.from_dict(_config())).lilicorr
    assert head.logit_scale == 8.0 and math.isclose(head.vector_eps, 1e-4)


def test_shorter_block_than_trained(tmp_path):
    """dflash_block_size below the checkpoint's block: fewer slots, same head."""
    import mlx.nn as nn

    from omlx.speculative.dflash import load_dflash_drafter

    path = _checkpoint(tmp_path, _config())
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            backbone=SimpleNamespace(embeddings=nn.Embedding(256, 64)),
            lm_head=nn.Linear(64, 256, bias=False),
        )
    )
    drafter = load_dflash_drafter(str(path), target, block_size=3)
    assert drafter.block_size == 3 and drafter.kind == "lilicorr"
    model = drafter.model
    slots = mx.random.normal((1, 2, 64)).astype(mx.bfloat16)
    tokens, candidates, _ = model.propose(slots, model._logits(slots), mx.zeros((1, 64)))
    assert tokens.shape == (1, 2) and candidates.shape == (1, 2, 4)
    with pytest.raises(ValueError, match="trained for 3 slots"):
        too_long = mx.random.normal((1, 4, 64)).astype(mx.bfloat16)
        model.propose(too_long, model._logits(too_long), mx.zeros((1, 64)))
