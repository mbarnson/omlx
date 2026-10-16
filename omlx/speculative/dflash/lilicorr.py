# SPDX-License-Identifier: Apache-2.0
"""NVIDIA LiLiCorr block drafter (DFlash backbone plus a lattice reranker).

LiLiCorr checkpoints (e.g. ``nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr``)
are DFlash drafters: five Qwen3-style layers with sliding-window attention,
attention sinks and DFlash2 grouped dynamic convolutions, conditioned on
captured target layers. Instead of the per-slot argmax, the LiLiCorr head scores
the drafter's top-k candidates per slot with an anchor-conditioned chain of
cosine factors and commits a path greedily, left to right.

The checkpoint is loaded as published: ModelOpt W4A16 NVFP4 matrices run on
their packed bytes (MLX ``nvfp4`` plus the per-tensor ``weight_scale_2``), every
other tensor as stored. The drafter has no embeddings or LM head; it borrows
the target's.

Reference: NVIDIA Model-Optimizer @ 6bf36980,
modelopt/torch/speculative/plugins/{modeling_dflash,modeling_dflash2,
modeling_lilicorr,hf_dflash}.py.
"""

from __future__ import annotations

import functools
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DecoderLayer
from mlx_vlm.speculative.drafters.qwen3_dflash.config import DFlashConfig
from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import DFlashDraftModel

_HEAD_KEYS = (
    "lilicorr_candidate_topk",
    "lilicorr_hidden_size",
    "lilicorr_num_layers",
    "lilicorr_num_heads",
    "lilicorr_mlp_ratio",
    "lilicorr_factor_dim",
    "lilicorr_vector_eps",
    "lilicorr_logit_scale",
    "conv_kernel_size",
    "conv_group_size",
    "swa_window_size",
)
# These change the scores without changing any tensor shape, so a default
# would load cleanly and score a different function: require them.
_REQUIRED = (
    "lilicorr_candidate_topk",
    "lilicorr_num_layers",
    "lilicorr_num_heads",
    "lilicorr_mlp_ratio",
    "lilicorr_factor_dim",
    "lilicorr_vector_eps",
    "lilicorr_logit_scale",
)


def is_lilicorr_config(config: dict) -> bool:
    dflash = config.get("dflash_config") or {}
    return dflash.get("projector_type") == "lilicorr" or "LiLiCorrDraftModel" in (
        config.get("architectures") or ()
    )


@dataclass
class LiLiCorrConfig(DFlashConfig):
    conv_kernel_size: int = 0
    conv_group_size: int = 0
    swa_window_size: int = 0
    lilicorr_candidate_topk: int = 0
    lilicorr_hidden_size: int = 0
    lilicorr_num_layers: int = 0
    lilicorr_num_heads: int = 0
    lilicorr_mlp_ratio: float = 0.0
    lilicorr_factor_dim: int = 0
    lilicorr_vector_eps: float = 0.0
    lilicorr_logit_scale: float = 0.0

    @classmethod
    def from_dict(cls, params: dict) -> LiLiCorrConfig:
        params = dict(params)
        dflash = dict(params.get("dflash_config") or {})
        missing = [k for k in _REQUIRED if k not in dflash]
        if missing:
            raise ValueError(f"LiLiCorr dflash_config is missing {missing}")
        for key in _HEAD_KEYS:
            if key in dflash:
                params[key] = dflash[key]
        base = DFlashConfig.from_dict(params)
        fields = {k: getattr(base, k) for k in inspect.signature(DFlashConfig).parameters}
        fields.update({k: params[k] for k in _HEAD_KEYS if k in params})
        config = cls(**fields)
        if not config.lilicorr_hidden_size:
            config.lilicorr_hidden_size = config.hidden_size
        # Training windows the context per query: query i of a block at
        # position L sees context positions > L + i - W.
        window = config.swa_window_size or config.sliding_window
        if window:
            config.sliding_window = int(window)
        return config


class NVFP4Linear(nn.Module):
    """ModelOpt NVFP4 linear on the packed bytes, with the per-tensor scale.

    The tensors are not called weight/scales: oMLX's drafter reads the
    activation dtype from ``fc.weight`` / ``fc.scales``.
    """

    def __init__(self, input_dims: int, output_dims: int, bias: bool = False):
        super().__init__()
        self.packed = mx.zeros((output_dims, input_dims // 8), mx.uint32)
        self.block_scales = mx.zeros((output_dims, input_dims // 16), mx.uint8)
        self.global_scale = mx.zeros((), mx.float32)
        if bias:
            self.bias = mx.zeros((output_dims,), mx.bfloat16)

    def __call__(self, x):
        rows = x.size // x.shape[-1]
        if rows <= 8:
            mmv = _small_batch_matmul()
            if mmv is not None:
                out_dims, in_dims = self.packed.shape[0], x.shape[-1]
                y = mmv(x.reshape(rows, in_dims), "nvfp4", self.packed, out_dims, in_dims,
                        self.block_scales, gscale=self.global_scale)
                y = y.reshape(*x.shape[:-1], out_dims)
                return y + self.bias if "bias" in self else y
        y = mx.quantized_matmul(
            x, self.packed, self.block_scales, transpose=True, group_size=16, bits=4, mode="nvfp4"
        )
        y = (y.astype(mx.float32) * self.global_scale).astype(x.dtype)
        return y + self.bias if "bias" in self else y


@functools.lru_cache(maxsize=1)
def _small_batch_matmul():
    """The verify-width NVFP4 matmul from the Nemotron Bolt kernels, if available."""
    try:
        from ...patches.nemotron_bolt.model import mmv
    except Exception:  # noqa: BLE001
        return None
    return mmv


def _normalize(x, eps):
    """torch.nn.functional.normalize: x / max(||x||, eps), in float32."""
    x = x.astype(mx.float32)
    norm = mx.sqrt(mx.sum(mx.square(x), axis=-1, keepdims=True))
    return x / mx.maximum(norm, eps)


class _MLP(nn.Module):
    def __init__(self, dims, hidden):
        super().__init__()
        self.up_proj = NVFP4Linear(dims, hidden, bias=True)
        self.down_proj = NVFP4Linear(hidden, dims, bias=True)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.up_proj(x)))


class _MHA(nn.Module):
    """torch.nn.MultiheadAttention (batch_first, fused in_proj) with an additive mask."""

    def __init__(self, dims, heads):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = mx.zeros((3 * dims, dims), mx.bfloat16)
        self.in_proj_bias = mx.zeros((3 * dims,), mx.bfloat16)
        self.out_proj = NVFP4Linear(dims, dims, bias=True)

    def __call__(self, x, bias):
        batch, length, dims = x.shape
        head_dim = dims // self.heads
        qkv = x @ self.in_proj_weight.T + self.in_proj_bias
        q, k, v = mx.split(qkv.reshape(batch, length, 3, self.heads, head_dim), 3, axis=2)
        q, k, v = (t.squeeze(2).transpose(0, 2, 1, 3) for t in (q, k, v))
        o = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=head_dim**-0.5, mask=bias.astype(q.dtype)
        )
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(batch, length, dims))


class _HeadLayer(nn.Module):
    def __init__(self, dims, heads, mlp_ratio, eps):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dims, eps=eps)
        self.attn = _MHA(dims, heads)
        self.mlp_norm = nn.RMSNorm(dims, eps=eps)
        self.mlp = _MLP(dims, int(dims * mlp_ratio))

    def __call__(self, x, bias):
        x = x + self.attn(self.attn_norm(x), bias)
        return x + self.mlp(self.mlp_norm(x))


class _FeatureMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.up_proj = nn.Linear(5, dims)
        self.down_proj = nn.Linear(dims, dims)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.up_proj(x)))


class LiLiCorrHead(nn.Module):
    """Anchor-conditioned chain scorer over the per-slot top-k candidates."""

    def __init__(self, config: LiLiCorrConfig):
        super().__init__()
        hidden, dims = config.hidden_size, config.lilicorr_hidden_size
        self.slots = config.block_size - 1
        self.topk = config.lilicorr_candidate_topk
        self.block_size = config.block_size
        self.num_heads = config.lilicorr_num_heads
        self.vector_eps = float(config.lilicorr_vector_eps)
        self.logit_scale = float(config.lilicorr_logit_scale)
        self.token_proj = NVFP4Linear(hidden, dims, bias=True)
        self.pass_hidden_proj = NVFP4Linear(hidden, dims, bias=True)
        self.context_proj = NVFP4Linear(hidden, dims, bias=True)
        self.feature_norm = nn.LayerNorm(5)
        self.feature_mlp = _FeatureMLP(dims)
        self.slot_embedding = mx.zeros((1, 1, self.slots, 1, dims))
        self.rank_embedding = mx.zeros((1, 1, 1, self.topk, dims))
        self.relative_slot_bias = mx.zeros((self.num_heads, 2 * self.block_size - 1))
        self.same_slot_bias = mx.zeros((self.num_heads,))
        eps = config.rms_norm_eps
        self.layers = [
            _HeadLayer(dims, self.num_heads, config.lilicorr_mlp_ratio, eps)
            for _ in range(config.lilicorr_num_layers)
        ]
        self.output_norm = nn.RMSNorm(dims, eps=eps)
        self.anchor_norm = nn.RMSNorm(dims, eps=eps)
        self.factor_input_proj = nn.Linear(3 * dims, dims)
        self.out_head = nn.Linear(dims, config.lilicorr_factor_dim)
        self.in_head = nn.Linear(dims, config.lilicorr_factor_dim)
        self.anchor_out_head = nn.Linear(dims, config.lilicorr_factor_dim)

    def attention_bias(self, topk):
        slots = mx.repeat(mx.arange(self.slots), topk)
        rel = slots[:, None] - slots[None, :] + self.block_size - 1
        bias = self.relative_slot_bias[:, rel]
        same = (slots[:, None] == slots[None, :]).astype(bias.dtype)
        return bias + same[None] * self.same_slot_bias[:, None, None]

    def __call__(self, ids, log_probs, pass_hidden, embed, anchor_hidden):
        """``(log_start [B, k], log_pair [B, slots-1, k, k])`` in float32.

        ids, log_probs: [B, slots, k] in rank order; pass_hidden: [B, slots, H]
        (the drafter's normed slot hidden states); anchor_hidden: [B, H], the
        projected target row before the block.
        """
        batch, slots, k = ids.shape
        dtype = pass_hidden.dtype
        tokens = self.token_proj(embed(ids).astype(dtype))
        passes = self.pass_hidden_proj(pass_hidden)[:, :, None, :]
        lp = log_probs.astype(mx.float32)
        rank = mx.broadcast_to(mx.arange(k, dtype=mx.float32) / max(k - 1, 1), lp.shape)
        top1 = mx.broadcast_to((mx.arange(k) == 0).astype(mx.float32), lp.shape)
        features = mx.stack(
            [lp, mx.exp(lp), lp - lp.max(axis=-1, keepdims=True), rank, top1], axis=-1
        ).astype(dtype)
        h = tokens + passes + self.feature_mlp(self.feature_norm(features))
        h = h + self.slot_embedding[0].astype(dtype) + self.rank_embedding[0, :, :, :k].astype(dtype)
        h = h.reshape(batch, slots * k, -1)
        bias = self.attention_bias(k)
        for layer in self.layers:
            h = layer(h, bias)
        h = self.output_norm(h).reshape(batch, slots, k, -1)
        anchor = self.anchor_norm(self.context_proj(anchor_hidden))
        a = mx.broadcast_to(anchor[:, None, None, :], h.shape)
        fh = nn.silu(self.factor_input_proj(mx.concatenate([h, a, h * a], axis=-1)))
        out_vec = _normalize(self.out_head(fh), self.vector_eps)
        in_vec = _normalize(self.in_head(fh), self.vector_eps)
        anchor_out = _normalize(self.anchor_out_head(anchor), self.vector_eps)
        start = (in_vec[:, 0] @ anchor_out[:, :, None])[..., 0]
        pair = out_vec[:, :-1] @ in_vec[:, 1:].transpose(0, 1, 3, 2)
        return self.logit_scale * start, self.logit_scale * pair


def greedy_path(log_start, log_pair):
    """Commit the chain left to right: argmax start, then argmax each transition.

    Greedy on purpose (NVIDIA's serving rule): speculative decoding accepts
    the longest correct prefix, so early slots matter most.
    """
    pick = mx.argmax(log_start, axis=-1)
    picks = [pick]
    for s in range(log_pair.shape[1]):
        row = mx.take_along_axis(log_pair[:, s], pick[:, None, None], axis=1)[:, 0]
        pick = mx.argmax(row, axis=-1)
        picks.append(pick)
    return mx.stack(picks, axis=1)


class LiLiCorrDraftModel(DFlashDraftModel):
    layer_class = DFlash2DecoderLayer

    def __init__(self, config: LiLiCorrConfig):
        super().__init__(config)
        hidden = config.hidden_size
        self.fc = NVFP4Linear(len(config.target_layer_ids) * hidden, hidden)
        for layer in self.layers:
            mlp = layer.mlp
            mlp.gate_proj = NVFP4Linear(hidden, config.intermediate_size)
            mlp.up_proj = NVFP4Linear(hidden, config.intermediate_size)
            mlp.down_proj = NVFP4Linear(config.intermediate_size, hidden)
        self.lilicorr = LiLiCorrHead(config)

    def sanitize(self, weights):
        out = {}
        for key, value in weights.items():
            key = key.removeprefix("model.")
            if key.endswith(".weight_scale_2"):
                key = key[: -len(".weight_scale_2")] + ".global_scale"
            elif key.endswith(".weight_scale"):
                key = key[: -len(".weight_scale")] + ".block_scales"
                value = value.view(mx.uint8)
            elif value.dtype == mx.uint8 and key.endswith(".weight"):
                key = key[: -len(".weight")] + ".packed"
                value = value.view(mx.uint32)
            out[key] = value
        return out

    def project_context(self, captured):
        """[B, n, K*H] layer captures -> [B, n, H] context rows."""
        return self.hidden_norm(self.fc(captured))

    def lattice(self, slots, logits, anchor_ctx):
        """Top-k candidates per slot, their log-probs and the chain factors."""
        logits = logits.astype(mx.float32)
        k = self.lilicorr.topk
        ids = mx.argsort(-logits, axis=-1)[..., :k]
        lse = mx.logsumexp(logits, axis=-1, keepdims=True)
        log_probs = mx.take_along_axis(logits, ids, axis=-1) - lse
        start, pair = self.lilicorr(ids, log_probs, slots, self.embed_tokens, anchor_ctx)
        return ids, log_probs, start, pair

    def propose(self, slots, logits, anchor_ctx):
        """Greedy LiLiCorr path ``[B, block-1]`` with the candidates and picks."""
        ids, _, start, pair = self.lattice(slots, logits, anchor_ctx.astype(slots.dtype))
        picks = greedy_path(start, pair)
        tokens = mx.take_along_axis(ids, picks[..., None], axis=-1)[..., 0]
        return tokens.astype(mx.int32), ids.astype(mx.int32), picks


def load_lilicorr_drafter(path: str | Path) -> LiLiCorrDraftModel:
    """Build the drafter from a LiLiCorr checkpoint directory as published."""
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    if not is_lilicorr_config(config):
        raise ValueError(f"{path} is not a LiLiCorr checkpoint")
    model = LiLiCorrDraftModel(LiLiCorrConfig.from_dict(config))
    weights = {}
    for shard in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    if not weights:
        raise FileNotFoundError(f"No safetensors in {path}")
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    mx.eval(model.parameters())
    return model
