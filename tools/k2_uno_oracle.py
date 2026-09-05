# SPDX-License-Identifier: Apache-2.0
"""Independent Torch conditional-LoRA captures and MLX comparisons.

The oracle loads the cached HF architecture, not the MLX implementation. It
uses plain torch.nn.functional.linear for all 196 pairs and explicit row masks.
Inputs are persisted so every backend sees identical noise and verification
tokens. No PEFT, CUDA engine, or production decoder dependency is required.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
PROMPTS = (
    "What is the capital of Australia? Answer in one sentence.",
    "Write a Python function is_palindrome(s) that ignores spaces and case.",
    "A train travels 60 miles in 45 minutes. What is its speed in miles per hour?",
)


def fingerprint(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rms_error(actual, reference):
    actual, reference = actual.astype(np.float64), reference.astype(np.float64)
    rms = float(np.sqrt(np.mean(reference**2)))
    return {
        "max_abs": float(np.max(np.abs(actual - reference))),
        "relative_rms": float(
            np.sqrt(np.mean((actual - reference) ** 2)) / max(rms, 1e-8)
        ),
        "reference_rms": rms,
    }


def capture(args):
    import torch
    import torch.nn.functional as functional
    from k2_horizon_oracle import _load_model
    from safetensors.torch import load_file

    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = _load_model(args.snapshot, args.device)
    config = json.loads((args.adapter / "adapter_config.json").read_text())
    weights = load_file(
        str(args.adapter / "adapter_model.safetensors"), device=args.device
    )
    scale = config["lora_alpha"] / config["r"]
    context = {"mask": None, "capture": False}
    arrays = {}
    pairs = []

    class Projection(torch.nn.Module):
        def __init__(self, linear, a, b, name):
            super().__init__()
            self.linear, self.a, self.b, self.name = linear, a, b, name

        def forward(self, x):
            base = self.linear(x)
            mask = context["mask"]
            if mask is None:
                return base
            hidden = functional.linear(x, self.a) * mask[..., None].to(x.dtype)
            # Unfused reference order: BF16 A, BF16 B, scale, BF16 addition.
            result = base + functional.linear(hidden, self.b) * scale
            if context["capture"]:
                arrays[f"input/{self.name}"] = x[:, -8:].float().cpu().numpy()
                arrays[f"output/{self.name}"] = result[:, -8:].float().cpu().numpy()
                # Record the public CUDA source's fused-B accumulation as a
                # separate reference, not as the same BF16 operation order.
                fused = torch.addmm(
                    base.reshape(-1, base.shape[-1]),
                    hidden.reshape(-1, hidden.shape[-1]),
                    self.b.T,
                    beta=1.0,
                    alpha=scale,
                ).reshape(base.shape)
                arrays[f"fused/{self.name}"] = fused[:, -8:].float().cpu().numpy()
            return result

    for index, layer in enumerate(model.model.layers):
        for target in TARGETS:
            scope, attribute = target.split(".")
            parent = getattr(layer, scope)
            name = f"model.layers.{index}.{target}"
            a_key, b_key = name + ".lora_A.weight", name + ".lora_B.weight"
            a, b = weights[a_key], weights[b_key]
            setattr(
                parent, attribute, Projection(getattr(parent, attribute), a, b, name)
            )
            pairs.extend((a_key, b_key))
    if set(pairs) != set(weights):
        raise ValueError("Reference adapter did not consume exactly every tensor")

    cases_path = args.out / "inputs.json"
    identity = {
        "snapshot": str(args.snapshot),
        "adapter": str(args.adapter),
        "adapter_sha256": fingerprint(args.adapter / "adapter_model.safetensors"),
    }
    if cases_path.exists():
        saved = json.loads(cases_path.read_text())
        if any(saved[key] != value for key, value in identity.items()):
            raise ValueError("Oracle inputs were captured from a different artifact")
        cases = saved["cases"]
    else:
        rng = random.Random(42)
        cases = []
        for prompt in PROMPTS:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="low",
            )
            ids = list(tokenizer.encode(rendered, add_special_tokens=False))
            noise = [rng.randrange(1, model.config.vocab_size) for _ in range(7)]
            cases.append(
                {
                    "prompt": prompt,
                    "ids": ids,
                    "draft_ids": ids + noise,
                    "row_mask": [0] * len(ids) + [1] * 7,
                }
            )

    metadata = {
        **identity,
        "device": args.device,
        "pairs": len(pairs) // 2,
        "scale": scale,
        "cases": [],
    }
    with torch.inference_mode():
        for index, case in enumerate(cases):
            draft = torch.tensor([case["draft_ids"]], device=args.device)
            mask = torch.tensor([case["row_mask"]], device=args.device)
            context.update(mask=None, capture=False)
            base = model(draft, use_cache=False).logits
            context["mask"] = torch.zeros_like(mask)
            zero = model(draft, use_cache=False).logits
            assert torch.equal(base, zero), "Zero-gated reference changed AR outputs"
            context.update(mask=mask, capture=index == 0)
            logits = model(draft, use_cache=False).logits
            arrays[f"case_{index}/draft_logits"] = logits[:, -8:].float().cpu().numpy()
            arrays[f"case_{index}/base_logits"] = base[:, -8:].float().cpu().numpy()
            assert torch.equal(
                base[:, : len(case["ids"])], logits[:, : len(case["ids"])]
            ), "Noisy rows changed earlier causal rows"
            if "verify_ids" not in case:
                proposals = logits[0, -8:].argmax(-1).cpu().tolist()
                case["verify_ids"] = case["ids"] + proposals
            context.update(mask=None, capture=False)
            verify = model(
                torch.tensor([case["verify_ids"]], device=args.device), use_cache=False
            ).logits
            arrays[f"case_{index}/verify_logits"] = verify[:, -8:].float().cpu().numpy()
            metadata["cases"].append(
                {"zero_gated_exact": True, "clean_prefix_exact": True}
            )
    metadata["seconds"] = time.perf_counter() - started
    cases_path.write_text(json.dumps({**identity, "cases": cases}, indent=2) + "\n")
    np.savez_compressed(args.out / f"{args.device}.npz", **arrays)
    (args.out / f"{args.device}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def compare(args):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from omlx.patches.k2_horizon import apply_k2_horizon_patch
    from omlx.patches.k2_horizon.uno_adapter import load_uno_adapter

    apply_k2_horizon_patch()
    from mlx_lm import load

    inputs = json.loads((args.out / "inputs.json").read_text())
    if inputs["snapshot"] != str(args.snapshot) or inputs[
        "adapter_sha256"
    ] != fingerprint(args.adapter / "adapter_model.safetensors"):
        raise ValueError("MLX comparison must use the captured artifacts")
    model, _ = load(
        str(args.snapshot),
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    base_id = json.loads((args.adapter / "adapter_config.json").read_text())[
        "base_model_name_or_path"
    ]
    adapter = load_uno_adapter(model, args.adapter, base_model_id=base_id)
    mps, cpu = np.load(args.out / "mps.npz"), np.load(args.out / "cpu.npz")
    report = {"adapter": adapter, "projections": {}, "cases": []}
    # Predeclared gates: individual projection RMS <= 1.5%; accumulated
    # logits <= max(5%, 2 * reference CPU/MPS variation + 0.5%). All finite.
    report["thresholds"] = {
        "projection_relative_rms": 0.015,
        "logit_floor": 0.05,
        "logit_reference_multiplier": 2.0,
        "logit_rounding_allowance": 0.005,
    }
    failed = []
    mask = mx.array([inputs["cases"][0]["row_mask"][-8:]], dtype=mx.bfloat16)
    for index, layer in enumerate(model.layers):
        for target in TARGETS:
            scope, attribute = target.split(".")
            projection = getattr(getattr(layer, scope), attribute)
            name = f"model.layers.{index}.{target}"
            x = mx.array(mps[f"input/{name}"], dtype=mx.bfloat16)
            output = np.array(
                projection.conditional_forward(x, mask).astype(mx.float32)
            )
            item = rms_error(output, mps[f"output/{name}"])
            item["fused_reference_relative_rms"] = rms_error(
                mps[f"fused/{name}"], mps[f"output/{name}"]
            )["relative_rms"]
            item["passed"] = bool(
                np.isfinite(output).all() and item["relative_rms"] <= 0.015
            )
            report["projections"][name] = item
            if not item["passed"]:
                failed.append(name)
    for index, case in enumerate(inputs["cases"]):
        draft = mx.array([case["draft_ids"]])
        mask = mx.array([case["row_mask"]])
        base = model(draft)
        zero = model(draft, lora_mask=mx.zeros_like(mask))
        mixed = model(draft, lora_mask=mask)
        verify = model(mx.array([case["verify_ids"]]))
        mx.eval(base, zero, mixed, verify)
        row = {
            "logits": {},
            "zero_gated_exact": bool(mx.array_equal(base, zero).item()),
            "clean_prefix_exact": bool(
                mx.array_equal(
                    base[:, : len(case["ids"])], mixed[:, : len(case["ids"])]
                ).item()
            ),
        }
        for kind, value in (("base", base), ("draft", mixed), ("verify", verify)):
            actual = np.array(value[:, -8:].astype(mx.float32))
            key = f"case_{index}/{kind}_logits"
            item = rms_error(actual, mps[key])
            reference_error = rms_error(cpu[key], mps[key])["relative_rms"]
            threshold = max(0.05, 2 * reference_error + 0.005)
            item.update(
                cpu_mps_relative_rms=reference_error,
                threshold=threshold,
                argmax_agreement=float(
                    np.mean(actual.argmax(-1) == mps[key].argmax(-1))
                ),
                passed=bool(
                    np.isfinite(actual).all() and item["relative_rms"] <= threshold
                ),
            )
            row["logits"][kind] = item
            if not item["passed"]:
                failed.append(f"case_{index}/{kind}")
        # Verify the runtime's exact causal seed/noise and rollback frontier.
        cache = make_prompt_cache(model)
        ids = case["ids"]
        model(mx.array([ids[:-1]]), cache=cache)
        cached_draft = model(
            mx.array([case["draft_ids"][len(ids) - 1 :]]),
            cache=cache,
            lora_mask=mx.array([[0] + [1] * 7]),
        )
        for layer in cache:
            layer.trim(layer.offset - len(ids))
        cached_verify = model(mx.array([case["verify_ids"][len(ids) :]]), cache=cache)
        for kind, full, cached in (
            ("draft", mixed, cached_draft),
            ("verify", verify, cached_verify),
        ):
            item = rms_error(
                np.array(cached.astype(mx.float32)),
                np.array(full[:, -8:].astype(mx.float32)),
            )
            item["passed"] = item["relative_rms"] <= 0.05
            row[f"cache_{kind}"] = item
            if not item["passed"]:
                failed.append(f"case_{index}/cache_{kind}")
        if not row["zero_gated_exact"] or not row["clean_prefix_exact"]:
            failed.append(f"case_{index}/gating")
        report["cases"].append(row)
    report.update(failed=failed, passed=not failed)
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": not failed,
                "failed": failed,
                "cases": report["cases"],
                "projection_max_relative_rms": max(
                    x["relative_rms"] for x in report["projections"].values()
                ),
            },
            indent=2,
        ),
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "compare"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    (capture if args.mode == "capture" else compare)(args)
