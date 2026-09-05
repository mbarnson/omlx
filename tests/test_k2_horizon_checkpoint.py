# SPDX-License-Identifier: Apache-2.0
"""Validate K2-Horizon-MoVA-36B-A4B against an explicitly selected checkpoint."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from omlx.api.utils import extract_k2_horizon_messages
from omlx.reasoning_effort import apply_chat_template_with_reasoning_effort_fallback

_SNAPSHOT = os.environ.get("OMLX_K2_HORIZON_SNAPSHOT")
_ORACLE_DIR = os.environ.get("OMLX_K2_HORIZON_ORACLE_DIR")
_FULL = os.environ.get("OMLX_K2_HORIZON_FULL") == "1"

pytestmark = pytest.mark.skipif(
    not _SNAPSHOT, reason="OMLX_K2_HORIZON_SNAPSHOT is not set"
)

_EXPECTED_SHARDS = 48
_EXPECTED_TENSORS = 16998
_EXPECTED_LOGICAL_BYTES = 74_889_584_040
_MAX_PEAK_METAL_GB = 100
# Absolute floors apply to dense layers; sparse layers are gated against the
# reference's own CPU-vs-MPS noise, which near-tie routing amplifies.
_ROUTER_LOGIT_ATOL = 1e-2
_COMPONENT_ATOL = 2e-2
_BASELINE_FACTOR = 1.5
_ARGMAX_SLACK = 0.05
_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "Get the weather.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"],
        },
    },
}


@pytest.fixture(scope="module")
def snapshot() -> Path:
    path = Path(_SNAPSHOT)
    assert (path / "config.json").exists(), path
    return path


@pytest.fixture(scope="module")
def tokenizer(snapshot):
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    from mlx_lm.utils import load_config

    from omlx.utils.model_loading import maybe_apply_pre_load_patches
    from omlx.utils.tokenizer import get_tokenizer_config

    maybe_apply_pre_load_patches(str(snapshot))
    config = load_config(snapshot)
    return load_tokenizer(
        snapshot,
        get_tokenizer_config(str(snapshot)),
        eos_token_ids=config.get("eos_token_id"),
    )


def _render(tokenizer, messages, **kwargs) -> str:
    return apply_chat_template_with_reasoning_effort_fallback(
        tokenizer,
        messages,
        {"tokenize": False, "add_generation_prompt": True, **kwargs},
    )


# ---------------------------------------------------------------------------
# Layout, tokenizer, and template contracts (fast)
# ---------------------------------------------------------------------------


def test_safetensors_layout_matches_release(snapshot):
    shards = sorted(snapshot.glob("model-*.safetensors"))
    assert len(shards) == _EXPECTED_SHARDS

    tensors = 0
    logical_bytes = 0
    for shard in shards:
        with open(shard, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            tensors += 1
            start, end = entry["data_offsets"]
            logical_bytes += end - start
            assert entry["dtype"] == "BF16", name
    assert tensors == _EXPECTED_TENSORS
    assert logical_bytes == _EXPECTED_LOGICAL_BYTES


def test_tokenizer_contract(tokenizer):
    assert {1, 250019} <= set(tokenizer.eos_token_ids)
    assert tokenizer.has_thinking is False
    assert tokenizer.tool_call_start == "<ifm|tool_calls>"
    assert tokenizer.tool_call_end == "</ifm|tool_calls>"
    for marker in (
        "<ifm|think>",
        "</ifm|think>",
        "<ifm|think_fast>",
        "</ifm|think_fast>",
        "<ifm|think_faster>",
        "</ifm|think_faster>",
        "<ifm|tool_calls>",
        "</ifm|tool_calls>",
    ):
        assert len(tokenizer.encode(marker, add_special_tokens=False)) == 1, marker


@pytest.mark.parametrize(
    ("effort", "opener"),
    [
        ("high", "<ifm|think>"),
        ("medium", "<ifm|think_fast>"),
        ("low", "<ifm|think_faster>"),
        ("xhigh", "<ifm|think>"),
        ("minimal", "<ifm|think_faster>"),
        (None, "<ifm|think>"),
    ],
)
def test_reasoning_effort_selects_opener(tokenizer, effort, opener):
    kwargs = {} if effort is None else {"reasoning_effort": effort}
    rendered = _render(tokenizer, [{"role": "user", "content": "hi"}], **kwargs)

    assert rendered.endswith(f"<|ifm|im_start|>assistant\n{opener}\n")


def test_content_only_assistant_history_renders_after_extraction(tokenizer):
    from omlx.api.openai_models import Message

    messages = [
        Message(role="user", content="Q1"),
        Message(role="assistant", content="A1"),
        Message(role="user", content="Q2"),
    ]
    raw = [m.model_dump(exclude_none=True) for m in messages]
    with pytest.raises(Exception, match="missing a thinking field"):
        _render(tokenizer, raw)

    rendered = _render(
        tokenizer, extract_k2_horizon_messages(messages, tokenizer=tokenizer)
    )

    assert "<ifm|think>\n</ifm|think>A1<|ifm|im_end|>" in rendered
    assert rendered.endswith("<ifm|think>\n")


@pytest.mark.parametrize("tool_call_format", ["xml", "json", "xml_typed"])
def test_reasoning_and_tool_call_history_renders(tokenizer, tool_call_format):
    from omlx.api.openai_models import Message

    messages = [
        Message(role="user", content="Weather in Paris for 3 days?"),
        Message(
            role="assistant",
            reasoning_content="I need the weather tool.",
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": '{"city": "Paris", "days": 3}',
                    },
                }
            ],
        ),
        Message(role="tool", tool_call_id="call_1", content="Sunny all week."),
    ]
    rendered = _render(
        tokenizer,
        extract_k2_horizon_messages(messages, tokenizer=tokenizer),
        tools=[_WEATHER_TOOL],
        tool_call_format=tool_call_format,
    )

    assert "<ifm|think>\nI need the weather tool.</ifm|think>" in rendered
    assert "<ifm|tool_calls>" in rendered and "</ifm|tool_calls>" in rendered
    assert "<|ifm|im_start|>tool\nSunny all week.<|ifm|im_end|>" in rendered
    if tool_call_format == "json":
        assert (
            '{"name": "weather", "arguments": {"city": "Paris", "days": 3}}' in rendered
        )
    else:
        assert "<ifm|arg_key>city</ifm|arg_key>" in rendered
        assert ("<ifm|arg_type>" in rendered) == (tool_call_format == "xml_typed")


# ---------------------------------------------------------------------------
# Full BF16 load, generation, and oracle parity (slow)
# ---------------------------------------------------------------------------


def _wired_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page_size = 16384
    for line in out.splitlines():
        if line.startswith("Mach Virtual Memory Statistics"):
            page_size = int(line.split("page size of")[1].split()[0])
        if line.startswith("Pages wired down"):
            return int(line.split(":")[1].strip().rstrip(".")) * page_size / 2**30
    raise RuntimeError("vm_stat did not report wired pages")


@pytest.fixture(scope="module")
def loaded_model(snapshot):
    if not (_FULL or _ORACLE_DIR):
        pytest.skip("set OMLX_K2_HORIZON_FULL=1 or OMLX_K2_HORIZON_ORACLE_DIR")
    from omlx.utils.model_loading import lm_load_compat, maybe_apply_pre_load_patches
    from omlx.utils.tokenizer import get_tokenizer_config

    maybe_apply_pre_load_patches(str(snapshot))
    mx.reset_peak_memory()
    wired_before = _wired_gb()
    started = time.time()
    model, tokenizer = lm_load_compat(
        str(snapshot),
        tokenizer_config=get_tokenizer_config(str(snapshot)),
        trust_remote_code=False,
    )
    stats = {
        "load_seconds": round(time.time() - started, 1),
        "peak_metal_gb": round(mx.get_peak_memory() / 2**30, 2),
        "active_metal_gb": round(mx.get_active_memory() / 2**30, 2),
        "wired_gb_after_load": round(_wired_gb(), 2),
        "wired_gb_before_load": round(wired_before, 2),
    }
    print(f"\nK2 load stats: {json.dumps(stats)}")
    assert model.model_type == "k2_horizon"
    assert stats["peak_metal_gb"] < _MAX_PEAK_METAL_GB, stats
    return model, tokenizer, stats


@pytest.mark.slow
def test_full_bf16_load_and_greedy_generation(loaded_model):
    from mlx_lm.generate import generate

    model, tokenizer, stats = loaded_model
    prompt = _render(
        tokenizer,
        [{"role": "user", "content": "Reply with the single word: ready"}],
        reasoning_effort="low",
    )
    started = time.time()
    text = generate(model, tokenizer, prompt=prompt, max_tokens=64, verbose=False)
    seconds = time.time() - started
    print(f"\nK2 greedy sample ({seconds:.1f}s): {text!r}")

    assert text.strip()
    closers = ("</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>")
    assert (
        any(closer in text for closer in closers) or len(tokenizer.encode(text)) >= 60
    )
    for layer in model.layers:
        assert hasattr(layer.self_attn, "v_proj") or hasattr(
            layer.self_attn, "v_experts"
        )


def _mlx_layer_capture(k2, model, tokens: list[int], layers: list[int]) -> dict:
    """Mirror the oracle's hooks by stepping the MLX model layer by layer."""
    from mlx_lm.models.base import create_attention_mask

    arrays = {}
    h = model.model.embed_tokens(mx.array([tokens]))
    mask = create_attention_mask(h, None)
    for layer_idx, layer in enumerate(model.layers):
        attn_in = layer.input_layernorm(h)
        attn_out = layer.self_attn(attn_in, mask, None)
        h2 = h + attn_out
        ffn_in = layer.post_attention_layernorm(h2)
        ffn_out = layer.mlp(ffn_in)
        h = h2 + ffn_out
        if layer_idx not in layers:
            continue
        store = {
            "attn_in": attn_in,
            "attn_out": attn_out,
            "ffn_in": ffn_in,
            "ffn_out": ffn_out,
            "layer_out": h,
        }
        attn = layer.self_attn
        if "v_router" in attn:
            store["v_router_logits"] = k2.router_logits(attn_in, attn.v_router.weight)
            store["mixed_v"] = attn._values(attn_in)
            store["mlp_gate_logits"] = k2.router_logits(ffn_in, layer.mlp.gate.weight)
        mx.eval(*store.values())
        for name, value in store.items():
            arrays[f"L{layer_idx}.{name}"] = np.asarray(value.astype(mx.float32))
    logits = model.lm_head(model.model.norm(h))
    mx.eval(logits)
    arrays["logits"] = np.asarray(logits.astype(mx.float32))
    return arrays


def _topk_sets(logits: np.ndarray, bias: np.ndarray, k: int) -> np.ndarray:
    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
    return np.sort(np.argpartition(-(scores + bias), k - 1, axis=-1)[..., :k], axis=-1)


@pytest.mark.slow
def test_layer_outputs_match_mps_oracle(loaded_model, snapshot, tmp_path):
    """Compare named tensors against the MPS oracle and its own CPU-vs-MPS noise."""
    if not _ORACLE_DIR:
        pytest.skip("OMLX_K2_HORIZON_ORACLE_DIR is not set")
    from mlx_lm.models import k2_horizon as k2

    oracle_dir = Path(_ORACLE_DIR)
    reference = np.load(oracle_dir / "oracle_capture.npz")
    meta = json.loads((oracle_dir / "oracle_capture.json").read_text())
    assert meta["revision"] == "05cab0a4d7150c1c460a000b37ff40cc1af2feaa"
    baseline_path = oracle_dir / "compare_report.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    model, _, _ = loaded_model
    tokens = reference["tokens"].tolist()
    layers = [int(v) for v in meta["layers"]]

    actual = _mlx_layer_capture(k2, model, tokens, layers)
    report = {}
    failures = []

    def gate(key: str, err: float, floor: float) -> None:
        sparse = model.args.is_sparse_layer(int(key[1:].split(".")[0]))
        if not sparse:
            limit = floor
        elif baseline is None:
            pytest.fail(
                "sparse-layer tolerances need the reference's CPU-vs-MPS baseline: "
                "run tools/k2_horizon_oracle.py capture --device cpu, then compare"
            )
        else:
            limit = max(floor, _BASELINE_FACTOR * baseline[key]["normalized_err"])
        report[key]["limit"] = limit
        if err > limit:
            failures.append((key, err, limit))

    for layer_idx in layers:
        layer = model.layers[layer_idx]
        prefix = f"L{layer_idx}."
        for name in (
            "attn_in",
            "attn_out",
            "ffn_in",
            "ffn_out",
            "layer_out",
            "mixed_v",
        ):
            key = prefix + name
            if key not in reference.files:
                continue
            ref = reference[key].reshape(actual[key].shape)
            err = float(np.max(np.abs(actual[key] - ref)))
            scale = float(np.max(np.abs(ref))) or 1.0
            report[key] = {"max_abs_err": err, "normalized_err": err / scale}
            gate(key, err / scale, _COMPONENT_ATOL)
        for kind, bias_attr, k, owner in (
            (
                "v_router",
                "v_expert_bias",
                model.args.mova_num_experts_per_tok,
                layer.self_attn,
            ),
            ("mlp_gate", "expert_bias", model.args.num_experts_per_tok, layer.mlp),
        ):
            key = f"{prefix}{kind}_logits"
            if key not in reference.files:
                continue
            ref = reference[key].reshape(actual[key].shape)
            err = float(np.max(np.abs(actual[key] - ref)))
            scale = float(np.max(np.abs(ref))) or 1.0
            bias = np.asarray(owner[bias_attr].astype(mx.float32))
            agreement = float(
                np.mean(
                    np.all(
                        _topk_sets(actual[key], bias, k) == _topk_sets(ref, bias, k),
                        axis=-1,
                    )
                )
            )
            report[key] = {
                "max_abs_err": err,
                "normalized_err": err / scale,
                "route_set_agreement": agreement,
            }
            gate(key, err / scale, _ROUTER_LOGIT_ATOL)
    logits_err = float(np.max(np.abs(actual["logits"] - reference["logits"])))
    argmax_agreement = float(
        np.mean(np.argmax(actual["logits"], -1) == np.argmax(reference["logits"], -1))
    )
    report["logits"] = {"max_abs_err": logits_err, "argmax_agreement": argmax_agreement}
    if baseline is not None:
        floor = baseline["logits"]["argmax_agreement"] - _ARGMAX_SLACK
        report["logits"]["limit"] = floor
        if argmax_agreement < floor:
            failures.append(("logits.argmax_agreement", argmax_agreement, floor))
    (oracle_dir / "mlx_parity_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nK2 parity report written to {oracle_dir / 'mlx_parity_report.json'}")
    assert report["L0.attn_in"]["max_abs_err"] == 0.0
    assert not failures, failures


@pytest.mark.slow
def test_dump_router_corpus_for_replay(loaded_model, tokenizer):
    """Capture real hidden states so the oracle can replay both routers on MPS."""
    if not _ORACLE_DIR:
        pytest.skip("OMLX_K2_HORIZON_ORACLE_DIR is not set")
    from mlx_lm.models import k2_horizon as k2
    from mlx_lm.models.base import create_attention_mask

    model, _, _ = loaded_model
    corpus_path = Path(__file__).with_name("test_k2_horizon_checkpoint.py")
    text = corpus_path.read_text() * 3
    tokens = tokenizer.encode(text, add_special_tokens=False)[:4096]
    assert len(tokens) >= 4096
    layers = [3, 24, 47]

    arrays = {
        "mova_top_k": np.asarray(model.args.mova_num_experts_per_tok),
        "moe_top_k": np.asarray(model.args.num_experts_per_tok),
    }
    h = model.model.embed_tokens(mx.array([tokens]))
    mask = create_attention_mask(h, None)
    for layer_idx, layer in enumerate(model.layers):
        attn_in = layer.input_layernorm(h)
        h2 = h + layer.self_attn(attn_in, mask, None)
        ffn_in = layer.post_attention_layernorm(h2)
        h = h2 + layer.mlp(ffn_in)
        mx.eval(h)
        if layer_idx not in layers:
            continue
        attn = layer.self_attn
        for kind, x, weight, bias, k in (
            (
                "v_router",
                attn_in,
                attn.v_router.weight,
                attn.v_expert_bias,
                arrays["mova_top_k"],
            ),
            (
                "mlp_gate",
                ffn_in,
                layer.mlp.gate.weight,
                layer.mlp.expert_bias,
                arrays["moe_top_k"],
            ),
        ):
            logits = k2.router_logits(x, weight)
            selected, _ = k2.route(x, weight, bias, int(k), 1.0)
            mx.eval(logits, selected)
            arrays[f"L{layer_idx}.{'attn_in' if kind == 'v_router' else 'ffn_in'}"] = (
                np.asarray(x.astype(mx.float32))[0]
            )
            arrays[f"L{layer_idx}.{kind}_logits"] = np.asarray(logits)[0]
            arrays[f"L{layer_idx}.{kind}_selected"] = np.asarray(selected)[0]
    np.savez(Path(_ORACLE_DIR) / "mlx_router_inputs.npz", **arrays)


@pytest.mark.slow
def test_greedy_agreement_length_against_oracle(loaded_model):
    """Record how many greedy tokens MLX and the MPS oracle agree on per prompt."""
    if not _ORACLE_DIR or not (Path(_ORACLE_DIR) / "oracle_greedy.json").exists():
        pytest.skip("no oracle_greedy.json in OMLX_K2_HORIZON_ORACLE_DIR")
    from mlx_lm.models.cache import make_prompt_cache

    model, _, _ = loaded_model
    oracle = json.loads((Path(_ORACLE_DIR) / "oracle_greedy.json").read_text())
    report = []
    for entry in oracle["results"]:
        prompt = entry["prompt_token_ids"]
        reference = entry["generated_token_ids"]
        cache = make_prompt_cache(model)
        logits = model(mx.array([prompt]), cache=cache)
        generated = []
        for _ in range(len(reference)):
            token = int(mx.argmax(logits[:, -1, :], axis=-1)[0])
            generated.append(token)
            logits = model(mx.array([[token]]), cache=cache)
        agreement = 0
        for ours, theirs in zip(generated, reference):
            if ours != theirs:
                break
            agreement += 1
        report.append(
            {
                "prompt": entry["prompt"],
                "oracle_tokens": len(reference),
                "agreement_length": agreement,
                "mlx_token_ids": generated,
            }
        )
        print(
            f"\nK2 greedy agreement {agreement}/{len(reference)}: {entry['prompt'][:40]!r}"
        )
    (Path(_ORACLE_DIR) / "mlx_greedy_report.json").write_text(
        json.dumps(report, indent=2)
    )
    assert all(item["agreement_length"] >= 1 for item in report), report


def test_router_replay_report_meets_agreement_gate():
    """Read the oracle replay report once ``tools/k2_horizon_oracle.py replay`` has run."""
    if not _ORACLE_DIR or not (Path(_ORACLE_DIR) / "replay_report.json").exists():
        pytest.skip("no replay_report.json in OMLX_K2_HORIZON_ORACLE_DIR")
    report = json.loads((Path(_ORACLE_DIR) / "replay_report.json").read_text())

    entries = [entry for layer in report["layers"].values() for entry in layer.values()]
    assert min(entry["vectors"] for entry in entries) >= 4096
    assert max(entry["max_abs_logit_error"] for entry in entries) == 0.0, report[
        "layers"
    ]
    assert (
        min(entry["route_set_agreement_excluding_exact_ties"] for entry in entries)
        >= 0.995
    ), report["layers"]
