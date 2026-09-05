# SPDX-License-Identifier: Apache-2.0
"""Implement the K2 Horizon MoVA architecture validated against the 36B-A4B checkpoint."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear

# The source checkpoint computed each router as two BF16 partial GEMMs.
SOURCE_ROUTER_GEMM_PARTITIONS = 2
_LN2 = math.log(2.0)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    layernorm_num_groups: int
    mlp_only_layers: list[int]
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    num_shared_experts: int
    mova_num_experts: int
    mova_num_experts_per_tok: int
    moe_gate_bias: bool
    norm_topk_prob: bool
    router_score_func: str
    router_scaling_factor: float
    attention_gate_func: str | None
    query_key_norm: bool
    rope_parameters: dict[str, Any]
    decoder_sparse_step: int = 1
    attention_bias: bool = False
    rope_head_dim: int | None = None
    use_sliding_window: bool = False
    sliding_window: int | None = None
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 524288

    def __post_init__(self):
        self.rope_theta = float(self.rope_parameters["rope_theta"])
        rope_type = self.rope_parameters.get("rope_type", "default")
        self._require(rope_type == "default", "rope_parameters.rope_type", rope_type)
        self._require(
            self.rope_head_dim in (None, self.head_dim),
            "rope_head_dim",
            self.rope_head_dim,
        )
        self._require(not self.use_sliding_window, "use_sliding_window", True)
        self._require(
            self.sliding_window is None, "sliding_window", self.sliding_window
        )
        self._require(not self.query_key_norm, "query_key_norm", True)
        self._require(
            self.attention_gate_func == "softplus",
            "attention_gate_func",
            self.attention_gate_func,
        )
        self._require(
            self.router_score_func == "sigmoid",
            "router_score_func",
            self.router_score_func,
        )
        self._require(self.norm_topk_prob, "norm_topk_prob", False)
        self._require(self.moe_gate_bias, "moe_gate_bias", False)
        self._require(
            self.layernorm_num_groups == 2,
            "layernorm_num_groups",
            self.layernorm_num_groups,
        )
        self._require(
            self.hidden_size
            % (self.layernorm_num_groups * SOURCE_ROUTER_GEMM_PARTITIONS)
            == 0,
            "hidden_size",
            self.hidden_size,
        )
        self._require(
            self.num_shared_experts == 1, "num_shared_experts", self.num_shared_experts
        )
        self._require(
            0 < self.num_experts_per_tok <= self.num_experts and self.num_experts > 0,
            "num_experts_per_tok",
            self.num_experts_per_tok,
        )
        self._require(
            0 < self.mova_num_experts_per_tok <= self.mova_num_experts
            and self.mova_num_experts > 0,
            "mova_num_experts_per_tok",
            self.mova_num_experts_per_tok,
        )
        self._require(
            self.decoder_sparse_step > 0,
            "decoder_sparse_step",
            self.decoder_sparse_step,
        )
        self._require(
            all(0 <= i < self.num_hidden_layers for i in self.mlp_only_layers),
            "mlp_only_layers",
            self.mlp_only_layers,
        )
        self._require(
            any(self.is_sparse_layer(i) for i in range(self.num_hidden_layers)),
            "mlp_only_layers",
            self.mlp_only_layers,
        )
        self._require(
            isinstance(self.router_scaling_factor, (int, float))
            and self.router_scaling_factor > 0,
            "router_scaling_factor",
            self.router_scaling_factor,
        )

    @staticmethod
    def _require(condition: bool, field: str, value: Any) -> None:
        if not condition:
            raise ValueError(
                f"Unsupported K2 Horizon config: {field}={value!r} is not the "
                "released K2-Horizon-MoVA-36B-A4B semantics"
            )

    def is_sparse_layer(self, layer_idx: int) -> bool:
        return (
            layer_idx not in self.mlp_only_layers
            and (layer_idx + 1) % self.decoder_sparse_step == 0
        )


class GroupedRMSNorm(nn.Module):
    """RMSNorm whose statistics are computed per contiguous feature group."""

    def __init__(self, dims: int, groups: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.groups = groups
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        grouped = x.astype(mx.float32).reshape(*x.shape[:-1], self.groups, -1)
        normed = mx.fast.rms_norm(grouped, None, self.eps).reshape(x.shape)
        return (self.weight * normed).astype(x.dtype)


def router_logits(x: mx.array, weight: mx.array) -> mx.array:
    """Sum FP32 casts of two BF16 partial GEMMs, matching the source checkpoint."""
    if x.dtype != mx.bfloat16 or weight.dtype != mx.bfloat16:
        raise ValueError(
            "K2 Horizon routers require BF16 activations and weights; got "
            f"{x.dtype} and {weight.dtype}"
        )
    x_parts = mx.split(x, SOURCE_ROUTER_GEMM_PARTITIONS, axis=-1)
    w_parts = mx.split(weight, SOURCE_ROUTER_GEMM_PARTITIONS, axis=-1)
    logits = (x_parts[0] @ w_parts[0].T).astype(mx.float32)
    for x_part, w_part in zip(x_parts[1:], w_parts[1:]):
        logits = logits + (x_part @ w_part.T).astype(mx.float32)
    return logits


def route(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    top_k: int,
    scaling_factor: float,
) -> tuple[mx.array, mx.array]:
    """Return selected expert indices and their normalized, scaled FP32 weights."""
    scores = mx.sigmoid(router_logits(x, weight))
    selection = scores + bias.astype(mx.float32)
    inds = mx.argpartition(-selection, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    return inds, weights * scaling_factor


def softplus_beta_ln2(x: mx.array) -> mx.array:
    """PyTorch ``softplus(x, beta=ln 2)`` computed in FP32 without overflow."""
    x32 = x.astype(mx.float32)
    return (mx.logaddexp(x32 * _LN2, 0.0) / _LN2).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, mova: bool):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.mova = mova
        self.top_k = args.mova_num_experts_per_tok
        self.scaling_factor = args.router_scaling_factor

        q_dims = self.n_heads * self.head_dim
        kv_dims = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(args.hidden_size, q_dims, bias=args.attention_bias)
        self.k_proj = nn.Linear(args.hidden_size, kv_dims, bias=args.attention_bias)
        self.o_proj = nn.Linear(q_dims, args.hidden_size, bias=args.attention_bias)
        self.gate_proj = nn.Linear(args.hidden_size, q_dims, bias=False)
        if mova:
            self.v_router = nn.Linear(
                args.hidden_size, args.mova_num_experts, bias=False
            )
            self.v_expert_bias = mx.zeros((args.mova_num_experts,))
            self.v_experts = SwitchLinear(
                args.hidden_size, kv_dims, args.mova_num_experts, bias=False
            )
        else:
            self.v_proj = nn.Linear(args.hidden_size, kv_dims, bias=args.attention_bias)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)

    def _values(self, x: mx.array) -> mx.array:
        if not self.mova:
            return self.v_proj(x)
        inds, weights = route(
            x, self.v_router.weight, self.v_expert_bias, self.top_k, self.scaling_factor
        )
        routed = self.v_experts(mx.expand_dims(x, (-2, -3)), inds).squeeze(-2)
        routed = nn.silu(routed) * weights.astype(routed.dtype)[..., None]
        return routed.sum(axis=-2)

    def __call__(
        self, x: mx.array, mask: mx.array | None = None, cache: Any = None
    ) -> mx.array:
        batch, length, _ = x.shape
        queries = (
            self.q_proj(x)
            .reshape(batch, length, self.n_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(x)
            .reshape(batch, length, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self._values(x)
            .reshape(batch, length, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3)
        gate = softplus_beta_ln2(self.gate_proj(x)).reshape(
            batch, length, self.n_heads, -1
        )
        return self.o_proj((output * gate).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, dims: int, hidden_dims: int):
        super().__init__()
        self.gate_proj = nn.Linear(dims, hidden_dims, bias=False)
        self.up_proj = nn.Linear(dims, hidden_dims, bias=False)
        self.down_proj = nn.Linear(hidden_dims, dims, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.scaling_factor = args.router_scaling_factor
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = mx.zeros((args.num_experts,))
        self.experts = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_experts = MLP(
            args.hidden_size, args.moe_intermediate_size * args.num_shared_experts
        )

    def __call__(self, x: mx.array) -> mx.array:
        inds, weights = route(
            x, self.gate.weight, self.expert_bias, self.top_k, self.scaling_factor
        )
        routed = self.experts(x, inds) * weights.astype(x.dtype)[..., None]
        return routed.sum(axis=-2) + self.shared_experts(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        sparse = args.is_sparse_layer(layer_idx)
        self.self_attn = Attention(args, mova=sparse)
        if sparse:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = GroupedRMSNorm(
            args.hidden_size, args.layernorm_num_groups, args.rms_norm_eps
        )
        self.post_attention_layernorm = GroupedRMSNorm(
            args.hidden_size, args.layernorm_num_groups, args.rms_norm_eps
        )

    def __call__(
        self, x: mx.array, mask: mx.array | None = None, cache: Any = None
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class K2HorizonModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = GroupedRMSNorm(
            args.hidden_size, args.layernorm_num_groups, args.rms_norm_eps
        )

    def __call__(self, inputs: mx.array, cache: Any = None) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = K2HorizonModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Any = None) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        for layer_idx in range(self.args.num_hidden_layers):
            if not self.args.is_sparse_layer(layer_idx):
                continue
            prefix = f"model.layers.{layer_idx}"
            for name in ("gate_proj", "up_proj", "down_proj"):
                _stack_experts(
                    weights,
                    f"{prefix}.mlp.experts.{name}.weight",
                    [
                        f"{prefix}.mlp.experts.{e}.{name}.weight"
                        for e in range(self.args.num_experts)
                    ],
                )
            _stack_experts(
                weights,
                f"{prefix}.self_attn.v_experts.weight",
                [
                    f"{prefix}.self_attn.v_experts.{e}.weight"
                    for e in range(self.args.mova_num_experts)
                ],
            )
            _rename(weights, f"{prefix}.mlp.gate.bias", f"{prefix}.mlp.expert_bias")
            _rename(
                weights,
                f"{prefix}.self_attn.v_router.bias",
                f"{prefix}.self_attn.v_expert_bias",
            )
        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            return not (
                path.endswith("mlp.gate") or path.endswith("self_attn.v_router")
            )

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k

        return predicate


def _stack_experts(
    weights: dict[str, mx.array], stacked_key: str, expert_keys: list[str]
) -> None:
    present = [k for k in expert_keys if k in weights]
    if stacked_key in weights:
        if present:
            raise ValueError(
                f"{stacked_key} is present alongside per-expert tensors such as {present[0]}"
            )
        return
    missing = [k for k in expert_keys if k not in weights]
    if missing:
        raise ValueError(
            f"Cannot stack {stacked_key}: {len(missing)} expert tensors missing, "
            f"first {missing[0]}"
        )
    weights[stacked_key] = mx.stack([weights.pop(k) for k in expert_keys])


def _rename(weights: dict[str, mx.array], old: str, new: str) -> None:
    if old not in weights:
        return
    if new in weights:
        raise ValueError(f"Both {old} and {new} are present")
    weights[new] = weights.pop(old)
