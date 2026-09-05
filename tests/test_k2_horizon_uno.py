# SPDX-License-Identifier: Apache-2.0
"""Verify conditional adapters, target sampling, and cache commit behavior."""

import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from test_k2_horizon_dense import module, tiny_config

from omlx.patches.k2_horizon.uno_adapter import (
    TARGETS,
    ConditionalLoRALinear,
    load_uno_adapter,
)
from omlx.patches.k2_horizon.uno_decode import (
    UnoDecoder,
    acceptance_and_residual,
    probabilities,
)


def adapted_model(tmp_path):
    k2 = module()
    mx.random.seed(13)
    model = k2.Model(k2.ModelArgs.from_dict(tiny_config("0.9B")))
    model.set_dtype(mx.bfloat16)
    config = dict(
        base_model_name_or_path="IFM/K2-Horizon-0.9B",
        peft_type="LORA",
        bias="none",
        fan_in_fan_out=False,
        r=2,
        lora_alpha=4,
        target_modules=[name for names in TARGETS.values() for name in names],
    )
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    weights = {}
    for i, layer in enumerate(model.layers):
        for scope, names in TARGETS.items():
            for name in names:
                base = getattr(getattr(layer, scope), name)
                out_dims, in_dims = base.weight.shape
                prefix = f"model.layers.{i}.{scope}.{name}"
                weights[prefix + ".lora_A.weight"] = (
                    mx.random.normal((2, in_dims)) * 0.05
                ).astype(mx.bfloat16)
                weights[prefix + ".lora_B.weight"] = (
                    mx.random.normal((out_dims, 2)) * 0.05
                ).astype(mx.bfloat16)
    mx.save_safetensors(str(tmp_path / "adapter_model.safetensors"), weights)
    return model, weights


def test_mixed_row_projection_matches_torch():
    torch = pytest.importorskip("torch")
    mx.random.seed(29)
    base = nn.Linear(64, 32, bias=False)
    base.set_dtype(mx.bfloat16)
    a = mx.random.normal((8, 64)).astype(mx.bfloat16)
    b = mx.random.normal((32, 8)).astype(mx.bfloat16)
    layer = ConditionalLoRALinear(base, a, b, 16)
    x = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)
    mask = mx.array([[0, 1, 0]])

    def tensor(value):
        return torch.tensor(np.array(value.astype(mx.float32)), dtype=torch.bfloat16)

    tx, ta, tb, tw = map(tensor, (x, a, b, base.weight))
    expected = (
        tx @ tw.T + (((tx @ ta.T) * torch.tensor([0, 1, 0])[None, :, None]) @ tb.T) * 16
    )
    actual = layer.conditional_forward(x, mask)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        expected.float().numpy(),
        atol=1.0,
        rtol=0.02,
    )
    np.testing.assert_array_equal(
        np.array(actual[:, ::2].astype(mx.float32)),
        np.array(base(x)[:, ::2].astype(mx.float32)),
    )


def test_all_adapter_pairs_are_consumed_and_base_rows_unchanged(tmp_path):
    model, weights = adapted_model(tmp_path)
    ids = mx.array([[3, 5, 7]])
    before = model(ids)
    mx.eval(before)
    report = load_uno_adapter(model, tmp_path, base_model_id="IFM/K2-Horizon-0.9B")
    assert report["tensors"] == len(weights) == 42
    assert report["scale"] == 2
    clean = model(ids, lora_mask=mx.zeros((1, 3)))
    mixed = model(ids, lora_mask=mx.array([[0, 1, 1]]))
    np.testing.assert_array_equal(
        np.array(clean.astype(mx.float32)), np.array(before.astype(mx.float32))
    )
    np.testing.assert_array_equal(
        np.array(mixed[:, :1].astype(mx.float32)),
        np.array(before[:, :1].astype(mx.float32)),
    )
    assert not mx.array_equal(mixed[:, 1:], before[:, 1:]).item()


@pytest.mark.parametrize(
    "damage", ["missing", "extra", "shape", "nan", "base", "rslora"]
)
def test_invalid_adapter_fails_before_mutating_model(tmp_path, damage):
    model, weights = adapted_model(tmp_path)
    key = next(iter(weights))
    if damage == "missing":
        del weights[key]
    elif damage == "extra":
        weights["unexpected"] = mx.ones((2,))
    elif damage == "shape":
        weights[key] = mx.ones((3, 64)).astype(mx.bfloat16)
    elif damage == "nan":
        weights[key] = mx.full(weights[key].shape, mx.nan, dtype=mx.bfloat16)
    else:
        path = tmp_path / "adapter_config.json"
        config = json.loads(path.read_text())
        config["base_model_name_or_path" if damage == "base" else "use_rslora"] = (
            "wrong" if damage == "base" else True
        )
        path.write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "adapter_model.safetensors"), weights)
    with pytest.raises(ValueError):
        load_uno_adapter(model, tmp_path, base_model_id="IFM/K2-Horizon-0.9B")
    assert isinstance(model.layers[0].self_attn.q_proj, nn.Linear)


def test_stochastic_acceptance_and_correction_preserve_exact_distribution():
    p = np.array([0.1, 0.3, 0.6], dtype=np.float32)
    q = np.array([0.5, 0.4, 0.1], dtype=np.float32)
    # Enumerate proposal probability mass rather than relying on a lucky sample.
    accept_mass = np.minimum(p, q)
    _, residual = acceptance_and_residual(
        mx.array(p)[None], mx.array(q)[None], mx.array([0]), mx.array([0.9])
    )
    actual = accept_mass + (1 - accept_mass.sum()) * np.array(residual)[0]
    np.testing.assert_allclose(actual, p, atol=1e-7)
    flags, _ = acceptance_and_residual(
        mx.array(np.stack([p, p, p])),
        mx.array(np.stack([q, q, q])),
        mx.array([0, 1, 2]),
        mx.array([0.3, 0.7, 0.99]),
    )
    assert flags.tolist() == [False, True, True]


def test_filtered_probabilities_use_reference_nucleus_boundary():
    p = probabilities(mx.log(mx.array([[0.6, 0.3, 0.1]])), 1.0, top_p=0.6)
    np.testing.assert_allclose(np.array(p), [[2 / 3, 1 / 3, 0]], atol=1e-6)
    p = probabilities(mx.log(mx.array([[0.6, 0.3, 0.1]])), 1.0, top_p=1, top_k=2)
    np.testing.assert_allclose(np.array(p), [[2 / 3, 1 / 3, 0]], atol=1e-6)


@pytest.mark.parametrize("block_size", [1, 2, 4, 8, 16])
def test_cached_and_recomputed_uno_match_target_greedy(tmp_path, block_size):
    from mlx_lm.models.cache import make_prompt_cache

    model, _ = adapted_model(tmp_path)
    load_uno_adapter(model, tmp_path, base_model_id="IFM/K2-Horizon-0.9B")
    prompt = [2, 3, 5]
    expected, cache = [], make_prompt_cache(model)
    inputs = mx.array([prompt])
    for _ in range(12):
        token = mx.argmax(model(inputs, cache=cache)[:, -1], -1).item()
        expected.append(token)
        inputs = mx.array([[token]])
    for use_cache in (True, False):
        traces = []
        decoder = UnoDecoder(
            model,
            eos_token_ids=[],
            temperature=0,
            block_size=block_size,
            noise_mode="deterministic_uniform",
            use_cache=use_cache,
        )
        cycles = list(decoder.generate(prompt, max_tokens=12, trace=traces.append))
        actual = [token for cycle in cycles for token in cycle.tokens]
        assert actual == expected
        assert cycles[-1].finish_reason == "length"
        assert cycles[-1].cache_length == len(prompt) + 12 - 1
        assert all(trace["row_mask"][0] == 0 for trace in traces)


def test_stop_and_cancel_never_publish_extra_tokens(tmp_path):
    model, _ = adapted_model(tmp_path)
    load_uno_adapter(model, tmp_path, base_model_id="IFM/K2-Horizon-0.9B")
    prompt = [2, 3]
    first = mx.argmax(model(mx.array([prompt]))[:, -1], -1).item()
    decoder = UnoDecoder(model, eos_token_ids=[first], temperature=0)
    cycles = list(decoder.generate(prompt, max_tokens=10))
    assert len(cycles) == 1 and cycles[0].tokens == (first,)
    assert cycles[0].finish_reason == "stop"
    assert list(decoder.generate(prompt, max_tokens=10, cancelled=lambda: True)) == []


def test_decoder_requires_adapter(tmp_path):
    model, _ = adapted_model(tmp_path)
    with pytest.raises(ValueError, match="validated conditional adapter"):
        UnoDecoder(model, eos_token_ids=[])


class _ScriptedModel:
    """A fixed categorical target with real MLX KV storage for rollback tests."""

    def __init__(self, reject_at=None):
        self.args = SimpleNamespace(vocab_size=128)
        self._uno_adapter_loaded = True
        self.reject_at = reject_at

    def make_cache(self):
        from mlx_lm.models.cache import KVCache

        self.cache = [KVCache()]
        return self.cache

    def __call__(self, inputs, cache=None, lora_mask=None):
        if cache is not None:
            values = inputs[:, None, :, None].astype(mx.float32)
            cache[0].update_and_fetch(values, values)
        length = inputs.shape[1]
        if lora_mask is not None:
            tokens = list(range(10, 10 + length))
        else:
            tokens = list(range(11, 11 + length))
            if self.reject_at is not None and self.reject_at < length - 1:
                tokens[self.reject_at] = 90
        return mx.where(
            mx.arange(128)[None, None, :] == mx.array(tokens)[None, :, None],
            0.0,
            -mx.inf,
        )


@pytest.mark.parametrize("reject_at", list(range(7)) + [None])
def test_each_rejection_frontier_and_all_accepted_preserve_real_kv(reject_at):
    model = _ScriptedModel(reject_at)
    decoder = UnoDecoder(model, eos_token_ids=[], block_size=8, temperature=0)
    iterator = decoder.generate([2, 3, 4], max_tokens=16)
    cycle = next(iterator)
    expected = (
        list(range(10, 19))
        if reject_at is None
        else list(range(10, 11 + reject_at)) + [90]
    )
    assert list(cycle.tokens) == expected
    assert cycle.accepted_proposals == (7 if reject_at is None else reject_at)
    assert cycle.cache_length == 3 + len(expected) - 1
    keys = model.cache[0].state[0]
    assert keys[0, 0, :, 0].tolist() == [2, 3, 4] + expected[:-1]
    iterator.close()


@pytest.mark.parametrize("eos_slot", range(9))
def test_eos_at_every_committed_slot_excludes_later_draft_tokens(eos_slot):
    model = _ScriptedModel()
    decoder = UnoDecoder(
        model, eos_token_ids=[10 + eos_slot], block_size=8, temperature=0
    )
    cycles = list(decoder.generate([2, 3, 4], max_tokens=16))
    assert len(cycles) == 1
    assert list(cycles[0].tokens) == list(range(10, 11 + eos_slot))
    assert cycles[0].accepted_proposals == min(7, eos_slot)
    assert cycles[0].finish_reason == "stop"
    assert model.cache[0].state[0][0, 0, :, 0].tolist() == [2, 3, 4] + list(
        range(10, 10 + eos_slot)
    )


@pytest.mark.parametrize("budget", range(9))
def test_budget_shorter_than_block_is_exact(budget):
    decoder = UnoDecoder(
        _ScriptedModel(), eos_token_ids=[], block_size=8, temperature=0
    )
    cycles = list(decoder.generate([2, 3], max_tokens=budget))
    assert [token for cycle in cycles for token in cycle.tokens] == list(
        range(10, 10 + budget)
    )
    if budget:
        assert cycles[-1].finish_reason == "length"


@pytest.mark.parametrize(
    "p,q",
    [
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]),
        ([1e-20, 0.0, 1.0], [0.0, 1e-20, 1.0]),
    ],
)
def test_residual_degenerate_disjoint_and_tiny_support(p, q):
    p, q = np.array(p, np.float32), np.array(q, np.float32)
    _, residual = acceptance_and_residual(
        mx.array(p)[None],
        mx.array(q)[None],
        mx.array([int(q.argmax())]),
        mx.array([0.5]),
    )
    actual = np.array(residual)[0]
    difference = np.maximum(p - q, 0)
    expected = difference / difference.sum() if difference.sum() > 0 else p
    np.testing.assert_allclose(actual, expected, atol=0, rtol=1e-6)
    assert np.isfinite(actual).all()


def test_stochastic_trace_captures_reproducible_decisions(tmp_path):
    model, _ = adapted_model(tmp_path)
    load_uno_adapter(model, tmp_path, base_model_id="IFM/K2-Horizon-0.9B")
    traces = []
    for _ in range(2):
        trace = []
        decoder = UnoDecoder(
            model, eos_token_ids=[], block_size=4, temperature=0.7, seed=42
        )
        list(decoder.generate([2, 3, 5], max_tokens=12, trace=trace.append))
        traces.append(trace)
    assert len(traces[0]) == len(traces[1])
    for left, right in zip(*traces):
        for key in ("draft_ids", "proposals", "committed", "cache_after"):
            assert left[key] == right[key]
        for key in (
            "proposal_probabilities",
            "target_probabilities",
            "rng_before",
            "rng_after",
        ):
            np.testing.assert_array_equal(np.array(left[key]), np.array(right[key]))
        if left["acceptance_uniforms"] is not None:
            flags, residual = acceptance_and_residual(
                left["target_probabilities"][:-1],
                left["proposal_probabilities"][1:],
                mx.array(left["proposals"][1:]),
                left["acceptance_uniforms"],
            )
            np.testing.assert_array_equal(
                np.array(flags), np.array(left["acceptance_flags"])
            )
            np.testing.assert_array_equal(
                np.array(residual), np.array(left["residual_probabilities"])
            )
