# SPDX-License-Identifier: Apache-2.0
"""Capture independent HF dense-model references, then compare native MLX."""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def capture(snapshot, directory, device):
    import torch
    from k2_horizon_oracle import _load_model

    started = time.perf_counter()
    model, tokenizer = _load_model(snapshot, device)
    rendered = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "What is the capital of Australia? Answer in one sentence.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )
    ids = list(tokenizer.encode(rendered, add_special_tokens=False))
    selected = [0, len(model.model.layers) // 2, len(model.model.layers) - 1]
    arrays = {}
    handles = []

    def hook(name):
        def save(_module, _args, output):
            if isinstance(output, tuple):
                output = output[0]
            arrays[name] = output.detach().float().cpu().numpy()

        return save

    for i in selected:
        layer = model.model.layers[i]
        for attr in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"):
            handles.append(
                getattr(layer, attr).register_forward_hook(hook(f"layer_{i}_{attr}"))
            )
        for projection in ("q_proj", "k_proj", "v_proj"):
            handles.append(
                getattr(layer.self_attn, projection).register_forward_hook(
                    hook(f"layer_{i}_{projection}")
                )
            )
    with torch.inference_mode():
        output = model(torch.tensor([ids], device=device), use_cache=False)
        arrays["logits"] = output.logits.float().cpu().numpy()
    for handle in handles:
        handle.remove()
    generated = []
    with torch.inference_mode():
        cache = None
        inputs = torch.tensor([ids], device=device)
        for _ in range(32):
            output = model(inputs, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            token = int(output.logits[:, -1].argmax(-1).item())
            generated.append(token)
            inputs = torch.tensor([[token]], device=device)
            if token in (1, tokenizer.convert_tokens_to_ids("<|ifm|im_end|>")):
                break
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / f"{device}.npz", **arrays)
    metadata = {
        "snapshot": str(snapshot),
        "device": device,
        "ids": ids,
        "layers": selected,
        "greedy_ids": generated,
        "greedy_text": tokenizer.decode(generated),
        "seconds": time.perf_counter() - started,
    }
    (directory / f"{device}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


def compare(snapshot, directory):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import make_prompt_cache

    from omlx.patches.k2_horizon import apply_k2_horizon_patch

    apply_k2_horizon_patch()
    metadata = json.loads((directory / "mps.json").read_text())
    if metadata["snapshot"] != str(snapshot):
        raise ValueError("Oracle snapshot does not match the requested model")
    model, tokenizer = load(
        str(snapshot),
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    ids = mx.array([metadata["ids"]])
    h = model.model.embed_tokens(ids)
    mask = create_attention_mask(h, None)
    arrays = {}
    for i, layer in enumerate(model.layers):
        norm = layer.input_layernorm(h)
        attention = layer.self_attn(norm, mask)
        h = h + attention
        post = layer.post_attention_layernorm(h)
        mlp = layer.mlp(post)
        h = h + mlp
        if i in metadata["layers"]:
            for projection in ("q_proj", "k_proj", "v_proj"):
                value = getattr(layer.self_attn, projection)(norm)
                arrays[f"layer_{i}_{projection}"] = np.array(value.astype(mx.float32))
            for key, value in [
                ("input_layernorm", norm),
                ("self_attn", attention),
                ("post_attention_layernorm", post),
                ("mlp", mlp),
            ]:
                arrays[f"layer_{i}_{key}"] = np.array(value.astype(mx.float32))
    arrays["logits"] = np.array(model.lm_head(model.model.norm(h)).astype(mx.float32))
    np.savez(directory / "mlx.npz", **arrays)
    target = np.load(directory / "mps.npz")
    cpu = np.load(directory / "cpu.npz") if (directory / "cpu.npz").exists() else None
    report = {
        "components": {},
        "thresholds": {
            "relative_rms_floor": 0.05,
            "reference_multiplier": 2.0,
            "rounding_allowance": 0.005,
        },
    }
    for name, actual in arrays.items():
        expected = target[name]
        rms = float(np.sqrt(np.mean(expected**2)))
        item = {
            "max_abs": float(np.max(np.abs(actual - expected))),
            "relative_rms": float(
                np.sqrt(np.mean((actual - expected) ** 2)) / max(rms, 1e-8)
            ),
        }
        if cpu is not None:
            item["cpu_mps_relative_rms"] = float(
                np.sqrt(np.mean((cpu[name] - expected) ** 2)) / max(rms, 1e-8)
            )
        item["threshold"] = max(0.05, 2 * item.get("cpu_mps_relative_rms", 0.0) + 0.005)
        item["passed"] = bool(
            np.isfinite(actual).all() and item["relative_rms"] <= item["threshold"]
        )
        report["components"][name] = item
    report["full_logit_argmax_agreement"] = float(
        np.mean(arrays["logits"].argmax(-1) == target["logits"].argmax(-1))
    )
    cache = make_prompt_cache(model)
    generated = []
    inputs = ids
    for _ in metadata["greedy_ids"]:
        token = int(mx.argmax(model(inputs, cache=cache)[:, -1], -1).item())
        generated.append(token)
        inputs = mx.array([[token]])
    agreement = 0
    for left, right in zip(generated, metadata["greedy_ids"]):
        if left != right:
            break
        agreement += 1
    report.update(
        greedy_agreement_prefix=agreement,
        greedy_ids=generated,
        greedy_text=tokenizer.decode(generated),
    )
    report["passed"] = all(item["passed"] for item in report["components"].values())
    (directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["capture", "compare"])
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    args = parser.parse_args()
    if args.mode == "capture":
        capture(args.snapshot, args.out, args.device)
    else:
        compare(args.snapshot, args.out)
