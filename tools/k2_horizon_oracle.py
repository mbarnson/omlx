# SPDX-License-Identifier: Apache-2.0
"""Local PyTorch oracle for the K2 Horizon MoVA port.

Runs the checkpoint's own ``modeling_k2_horizon.py`` on the MPS backend with
both router GEMMs replaced by the source xLLM two-part BF16 contract, then
writes per-layer reference arrays for the oMLX parity test. Never run this
while an MLX copy of the model is resident; both wire Metal memory.

Modes::

    capture  --snapshot DIR --out DIR [--layers 0,3,24,47] [--tokens N]
    greedy   --snapshot DIR --out DIR [--steps N]
    replay   --snapshot DIR --out DIR   (needs DIR/mlx_router_inputs.npz)
    compare  --snapshot DIR --out DIR --other DIR2   (two capture dirs)
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

_REVISION = "05cab0a4d7150c1c460a000b37ff40cc1af2feaa"
_DEFAULT_LAYERS = (0, 3, 24, 47)
_FIXED_FILLER = (
    "Compare mixture-of-experts routing with dense feed-forward layers, then "
    "explain how routed value projections change attention, citing one concrete "
    "trade-off for memory, one for latency, and one for training stability."
)
_PROMPTS = (
    "Explain why long-context evaluation is difficult.",
    "Write a Python function that merges two sorted lists.",
    "What is the capital of Mongolia, and what is it known for?",
)


def two_part_router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Sum FP32 casts of two BF16 half-width GEMMs, as SGLang's ``_xllm_router_gemm``."""
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("router GEMM contract requires BF16 input and weight")
    x_parts = x.chunk(2, dim=-1)
    w_parts = weight.chunk(2, dim=-1)
    first = functional.linear(x_parts[0].contiguous(), w_parts[0].contiguous())
    second = functional.linear(x_parts[1].contiguous(), w_parts[1].contiguous())
    return first.float() + second.float()


class _RouterAwareFunctional:
    """Proxy for ``torch.nn.functional`` that reroutes the two inline router GEMMs."""

    def __init__(self, router_weights: dict[int, tuple[int, str]], captures: dict):
        self._router_weights = router_weights
        self._captures = captures
        self.current_layer = -1

    def __getattr__(self, name):
        return getattr(functional, name)

    def linear(self, x, weight, bias=None):
        tag = self._router_weights.get(id(weight))
        if bias is not None or tag is None:
            return functional.linear(x, weight, bias)
        logits = two_part_router_gemm(x, weight)
        layer_idx, kind = tag
        if layer_idx in self._captures:
            self._captures[layer_idx][f"{kind}_logits"] = _numpy(logits)
        return logits


def _cache_identity(snapshot: Path) -> tuple[str, str] | None:
    """Return ``(repo_id, revision)`` when the snapshot lives in a Hugging Face cache."""
    if snapshot.parent.name != "snapshots":
        return None
    repo_dir = snapshot.parent.parent.name
    if not repo_dir.startswith("models--"):
        return None
    return repo_dir[len("models--") :].replace("--", "/"), snapshot.name


def _load_model(snapshot: Path, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    identity = _cache_identity(snapshot)
    if identity is None:
        source, kwargs = str(snapshot), {}
    else:
        source, kwargs = (
            identity[0],
            {"revision": identity[1], "local_files_only": True},
        )
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        **kwargs,
    )
    # Move after loading so the tool needs no ``accelerate`` for ``device_map``.
    model.to(device)
    model.eval()
    return model, tokenizer


def _prompt_ids(tokenizer, content: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        reasoning_effort="high",
        tokenize=False,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def _install_router_contract(model, captures: dict) -> _RouterAwareFunctional:
    modeling = sys.modules[type(model).__module__]
    router_weights: dict[int, tuple[int, str]] = {}
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, "v_router"):
            router_weights[id(layer.self_attn.v_router.weight)] = (
                layer_idx,
                "v_router",
            )
        if hasattr(layer.mlp, "gate"):
            router_weights[id(layer.mlp.gate.weight)] = (layer_idx, "mlp_gate")
    if not router_weights:
        raise RuntimeError(
            "no MoVA or MoE routers found; is this the 36B-A4B checkpoint?"
        )
    proxy = _RouterAwareFunctional(router_weights, captures)
    modeling.F = proxy

    original_combine = modeling.combine_routed_experts

    def combine_and_capture(
        hidden_states, routing_weights, selected_indices, experts, activation=None
    ):
        out = original_combine(
            hidden_states, routing_weights, selected_indices, experts, activation
        )
        layer_idx = proxy.current_layer
        if layer_idx in captures:
            captures[layer_idx]["mixed_v"] = _numpy(out)
            captures[layer_idx]["v_router_selected"] = _numpy(selected_indices)
        return out

    modeling.combine_routed_experts = combine_and_capture
    return proxy


def _first(output):
    return output[0] if isinstance(output, tuple) else output


def _install_capture_hooks(
    model, captures: dict, proxy: _RouterAwareFunctional
) -> None:
    for layer_idx, layer in enumerate(model.model.layers):

        def set_layer(_module, _args, idx=layer_idx):
            proxy.current_layer = idx

        layer.register_forward_pre_hook(set_layer)
        if layer_idx not in captures:
            continue
        store = captures[layer_idx]
        layer.input_layernorm.register_forward_hook(
            lambda _m, _a, out, s=store: s.__setitem__("attn_in", _numpy(out))
        )
        layer.self_attn.register_forward_hook(
            lambda _m, _a, out, s=store: s.__setitem__("attn_out", _numpy(_first(out)))
        )
        layer.post_attention_layernorm.register_forward_hook(
            lambda _m, _a, out, s=store: s.__setitem__("ffn_in", _numpy(out))
        )
        layer.mlp.register_forward_hook(
            lambda _m, _a, out, s=store: s.__setitem__("ffn_out", _numpy(_first(out)))
        )
        layer.register_forward_hook(
            lambda _m, _a, out, s=store: s.__setitem__("layer_out", _numpy(_first(out)))
        )


def _fixed_tokens(tokenizer, count: int) -> list[int]:
    ids = _prompt_ids(tokenizer, " ".join(_PROMPTS) + " " + _FIXED_FILLER)
    if len(ids) < count:
        raise ValueError(f"fixed prompt has only {len(ids)} tokens, need {count}")
    return ids[:count]


def _run_capture(model, tokens: list[int], captures: dict, device: str) -> dict:
    for store in captures.values():
        store.clear()
    with torch.no_grad():
        out = model(torch.tensor([tokens], device=device), use_cache=False)
    if device == "mps":
        torch.mps.synchronize()
    arrays = {
        "tokens": np.asarray(tokens, dtype=np.int64),
        "logits": _numpy(out.logits),
    }
    for layer_idx, store in captures.items():
        for name, value in store.items():
            arrays[f"L{layer_idx}.{name}"] = value
    return arrays


def _arrays_equal(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    """Copy a tensor to numpy, widening BF16 to FP32 since numpy has no BF16."""
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.numpy()


def _save(arrays: dict, path: Path, meta: dict) -> None:
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def cmd_capture(args) -> None:
    layers = tuple(int(v) for v in args.layers.split(","))
    model, tokenizer = _load_model(args.snapshot, args.device)
    tokens = _fixed_tokens(tokenizer, args.tokens)
    captures = {idx: {} for idx in layers}
    proxy = _install_router_contract(model, captures)
    _install_capture_hooks(model, captures, proxy)
    started = time.time()
    first = _run_capture(model, tokens, captures, args.device)
    second = _run_capture(model, tokens, captures, args.device)
    if not _arrays_equal(first, second):
        diffs = [k for k in first if not np.array_equal(first[k], second[k])]
        raise SystemExit(f"oracle is not deterministic; differing arrays: {diffs}")
    meta = {
        "revision": _REVISION,
        "device": args.device,
        "torch": torch.__version__,
        "layers": list(layers),
        "num_tokens": len(tokens),
        "router_contract": "two_part_bf16",
        "seconds": round(time.time() - started, 1),
    }
    _save(second, args.out / "oracle_capture.npz", meta)
    print(
        f"wrote {args.out / 'oracle_capture.npz'} ({len(second)} arrays, {meta['seconds']}s)"
    )


def cmd_greedy(args) -> None:
    model, tokenizer = _load_model(args.snapshot, args.device)
    _install_router_contract(model, {})
    results = []
    for prompt in _PROMPTS:
        prompt_ids = _prompt_ids(tokenizer, prompt)
        started = time.time()
        with torch.no_grad():
            generated = model.generate(
                torch.tensor([prompt_ids], device=args.device),
                max_new_tokens=args.steps,
                do_sample=False,
            )
        generated_ids = generated[0].tolist()[len(prompt_ids) :]
        results.append(
            {
                "prompt": prompt,
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": generated_ids,
                "text": tokenizer.decode(generated_ids),
                "seconds": round(time.time() - started, 1),
            }
        )
        print(
            f"{prompt[:40]!r}: {len(generated_ids)} tokens in {results[-1]['seconds']}s"
        )
    (args.out / "oracle_greedy.json").write_text(
        json.dumps(
            {"revision": _REVISION, "device": args.device, "results": results}, indent=2
        )
    )


def _read_router_weights(snapshot: Path, layer_idx: int) -> dict[str, torch.Tensor]:
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    names = {
        "v_router.weight": f"model.layers.{layer_idx}.self_attn.v_router.weight",
        "v_router.bias": f"model.layers.{layer_idx}.self_attn.v_router.bias",
        "mlp_gate.weight": f"model.layers.{layer_idx}.mlp.gate.weight",
        "mlp_gate.bias": f"model.layers.{layer_idx}.mlp.gate.bias",
    }
    tensors = {}
    for short, full in names.items():
        shard = snapshot / index[full]
        with open(shard, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
            entry = header[full]
            start, end = entry["data_offsets"]
            fh.seek(8 + header_len + start)
            raw = fh.read(end - start)
        tensors[short] = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(
            entry["shape"]
        )
    return tensors


def _selection_scores(logits: np.ndarray, bias: np.ndarray) -> np.ndarray:
    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    return scores + bias.astype(np.float64)


def _topk_sets(selection: np.ndarray, k: int) -> np.ndarray:
    return np.sort(np.argsort(-selection, axis=-1, kind="stable")[..., :k], axis=-1)


def _exact_tie_at_boundary(selection: np.ndarray, k: int) -> np.ndarray:
    """True where the k-th and (k+1)-th selection scores are exactly equal."""
    ordered = -np.sort(-selection, axis=-1)
    return ordered[..., k - 1] == ordered[..., k]


def _route_agreement(
    logits: np.ndarray, bias: np.ndarray, k: int, other_sets: np.ndarray
) -> dict:
    selection = _selection_scores(logits, bias)
    sets = _topk_sets(selection, k)
    agree = np.all(sets == np.sort(other_sets, axis=-1), axis=-1)
    tied = _exact_tie_at_boundary(selection, k)
    untied = ~tied
    return {
        "route_set_agreement": float(agree.mean()),
        "exact_tie_fraction": float(tied.mean()),
        "route_set_agreement_excluding_exact_ties": float(agree[untied].mean()),
        "disagreements_without_exact_tie": int(np.sum(~agree & untied)),
    }


def cmd_replay(args) -> None:
    inputs = np.load(args.out / "mlx_router_inputs.npz")
    report = {"revision": _REVISION, "device": args.device, "layers": {}}
    for key in inputs.files:
        if not key.endswith(".attn_in"):
            continue
        layer_idx = int(key[1:].split(".")[0])
        weights = _read_router_weights(args.snapshot, layer_idx)
        layer_report = {}
        for kind, input_key, k in (
            ("v_router", f"L{layer_idx}.attn_in", int(inputs["mova_top_k"])),
            ("mlp_gate", f"L{layer_idx}.ffn_in", int(inputs["moe_top_k"])),
        ):
            x = torch.from_numpy(inputs[input_key]).to(torch.bfloat16).to(args.device)
            w = weights[f"{kind}.weight"].to(args.device)
            logits = two_part_router_gemm(x, w).cpu().numpy()
            bias = weights[f"{kind}.bias"].float().numpy()
            mlx_logits = inputs[f"L{layer_idx}.{kind}_logits"]
            layer_report[kind] = {
                "vectors": int(x.shape[0]),
                "max_abs_logit_error": float(np.max(np.abs(logits - mlx_logits))),
                "duplicate_bf16_bias_values": int(len(bias) - len(np.unique(bias))),
                **_route_agreement(
                    logits, bias, k, inputs[f"L{layer_idx}.{kind}_selected"]
                ),
            }
            entry = layer_report[kind]
            print(
                f"layer {layer_idx} {kind}: max|dlogit|={entry['max_abs_logit_error']:.4g} "
                f"agreement={entry['route_set_agreement']:.4f} "
                f"excluding exact ties={entry['route_set_agreement_excluding_exact_ties']:.4f} "
                f"(ties {entry['exact_tie_fraction']:.2%})"
            )
        report["layers"][str(layer_idx)] = layer_report
    (args.out / "replay_report.json").write_text(json.dumps(report, indent=2))


def cmd_compare(args) -> None:
    """Report per-tensor differences between two capture files, e.g. CPU vs MPS."""
    left = np.load(args.out / "oracle_capture.npz")
    right = np.load(args.other / "oracle_capture.npz")
    report = {}
    for key in left.files:
        if key == "tokens" or key.endswith("_selected"):
            continue
        a = left[key].astype(np.float32).reshape(-1)
        b = right[key].astype(np.float32).reshape(-1)
        err = float(np.max(np.abs(a - b)))
        scale = float(np.max(np.abs(b))) or 1.0
        report[key] = {"max_abs_err": err, "normalized_err": err / scale}
        if key.endswith("_logits"):
            layer_idx = int(key[1:].split(".")[0])
            kind = key.split(".")[1].removesuffix("_logits")
            k = 4 if kind == "v_router" else 8
            bias = _read_router_weights(args.snapshot, layer_idx)[f"{kind}.bias"]
            bias = bias.float().numpy()
            other = _topk_sets(
                _selection_scores(right[key].reshape(-1, len(bias)), bias), k
            )
            report[key].update(
                _route_agreement(left[key].reshape(-1, len(bias)), bias, k, other)
            )
    report["logits"]["argmax_agreement"] = float(
        np.mean(np.argmax(left["logits"], -1) == np.argmax(right["logits"], -1))
    )
    for key, entry in report.items():
        print(f"{key:24s} " + "  ".join(f"{k}={v:.4g}" for k, v in entry.items()))
    (args.out / "compare_report.json").write_text(json.dumps(report, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=("capture", "greedy", "replay", "compare"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--other", type=Path, help="second capture dir for compare")
    parser.add_argument("--layers", default=",".join(str(v) for v in _DEFAULT_LAYERS))
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if not (args.snapshot / "modeling_k2_horizon.py").exists():
        raise SystemExit(f"{args.snapshot} does not contain modeling_k2_horizon.py")
    {
        "capture": cmd_capture,
        "greedy": cmd_greedy,
        "replay": cmd_replay,
        "compare": cmd_compare,
    }[args.mode](args)


if __name__ == "__main__":
    main()
