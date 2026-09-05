# SPDX-License-Identifier: Apache-2.0
"""Persist real Uno traces, replay decisions in torch, and run an HF sampler.

Run `capture` with MLX first, then `reference` in a separate process. The torch
reference uses full recomputation, independent conditional projection hooks,
and its own sampling loop. Random seeds are not a cross-framework token oracle.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from k2_uno_oracle import PROMPTS, TARGETS, fingerprint


def capture(args):
    import mlx.core as mx

    from omlx.patches.k2_horizon import apply_k2_horizon_patch
    from omlx.patches.k2_horizon.uno_adapter import load_uno_adapter
    from omlx.patches.k2_horizon.uno_decode import UnoDecoder, acceptance_and_residual

    apply_k2_horizon_patch()
    from mlx_lm import load

    model, tokenizer = load(
        str(args.snapshot),
        trust_remote_code=False,
        tokenizer_config={"trust_remote_code": False},
    )
    config = json.loads((args.adapter / "adapter_config.json").read_text())
    adapter = load_uno_adapter(
        model, args.adapter, base_model_id=config["base_model_name_or_path"]
    )
    report = {
        "snapshot": str(args.snapshot),
        "adapter": adapter,
        "runs": [],
        "passed": False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(PROMPTS):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            reasoning_effort="low",
        )
        for seed in (42, 43, 44):
            arrays, traces = {}, []

            def save(trace, arrays=arrays, traces=traces):
                number = len(traces)
                row = {}
                for key, value in trace.items():
                    if isinstance(value, mx.array):
                        arrays[f"{number}/{key}"] = np.array(
                            value.astype(mx.float32)
                            if value.dtype == mx.bfloat16
                            else value
                        )
                        row[key] = f"{number}/{key}"
                    else:
                        row[key] = value
                traces.append(row)

            decoder = UnoDecoder(
                model,
                eos_token_ids=tokenizer.eos_token_ids,
                block_size=8,
                temperature=1.0,
                top_p=0.95,
                seed=seed,
            )
            cycles = list(decoder.generate(ids, max_tokens=args.max_tokens, trace=save))
            tokens = [token for cycle in cycles for token in cycle.tokens]
            filename = f"prompt-{index}-seed-{seed}.npz"
            np.savez_compressed(args.out / filename, **arrays)
            run = {
                "prompt": prompt,
                "prompt_ids": ids,
                "seed": seed,
                "temperature": 1.0,
                "top_p": 0.95,
                "block_size": 8,
                "max_tokens": args.max_tokens,
                "eos": sorted(tokenizer.eos_token_ids),
                "output_ids": tokens,
                "text": tokenizer.decode(tokens),
                "finish_reason": cycles[-1].finish_reason,
                "arrays": filename,
                "traces": traces,
            }
            report["runs"].append(run)
            (args.out / "mlx.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps({k: v for k, v in run.items() if k != "traces"}), flush=True
            )

    # Fixed sample count and family-wise Hoeffding bound, declared independently
    # of observations. Include disjoint and partially filtered support.
    count, alpha = 65536, 1e-6
    cases = [
        ([0.1, 0.3, 0.6, 0], [0.5, 0.4, 0.1, 0]),
        ([0, 0, 0.4, 0.6], [0.7, 0.3, 0, 0]),
        ([0.6, 0.4, 0, 0], [0.6, 0.4, 0, 0]),
    ]
    bound = math.sqrt(math.log(2 * 4 * len(cases) / alpha) / (2 * count))
    statistics = {
        "samples_per_case": count,
        "family_wise_alpha": alpha,
        "absolute_error_bound": bound,
        "cases": [],
    }
    for index, (p, q) in enumerate(cases):
        keys = mx.random.split(mx.random.key(628 + index), num=3)
        target = mx.broadcast_to(mx.array(p), (count, 4))
        proposal = mx.broadcast_to(mx.array(q), (count, 4))
        tokens = mx.random.categorical(mx.log(proposal), key=keys[0])
        flags, residual = acceptance_and_residual(
            target, proposal, tokens, mx.random.uniform(shape=(count,), key=keys[1])
        )
        repaired = mx.random.categorical(mx.log(residual), key=keys[2])
        output = np.array(mx.where(flags, tokens, repaired))
        empirical = np.bincount(output, minlength=4) / count
        error = float(np.max(np.abs(empirical - p)))
        statistics["cases"].append(
            {
                "p": p,
                "q": q,
                "empirical": empirical.tolist(),
                "max_absolute_error": error,
                "passed": error <= bound,
            }
        )
    report["statistics"] = statistics
    report["passed"] = all(case["passed"] for case in statistics["cases"])
    (args.out / "mlx.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("Statistical sampler gate failed")


def reference(args):
    import torch
    import torch.nn.functional as functional
    from k2_horizon_oracle import _load_model
    from safetensors.torch import load_file

    captured = json.loads((args.out / "mlx.json").read_text())
    assert captured["snapshot"] == str(args.snapshot)
    assert captured["adapter"]["sha256"] == fingerprint(
        args.adapter / "adapter_model.safetensors"
    )
    report = {
        "adapter_sha256": fingerprint(args.adapter / "adapter_model.safetensors"),
        "snapshot": str(args.snapshot),
        "device": args.device,
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "replay": [],
        "hf_runs": [],
        "nucleus_support_checks": [],
        "passed": False,
    }

    def probabilities(logits, temperature, top_p):
        values, order = torch.sort(
            logits.float() / temperature, descending=True, dim=-1
        )
        probs = values.softmax(-1)
        probs = probs.masked_fill(probs.cumsum(-1) - probs > top_p, 0)
        probs = probs / probs.sum(-1, keepdim=True)
        return torch.zeros_like(probs).scatter(-1, order, probs)

    def residual_and_flags(p, q, proposed, uniforms):
        selected_p = p.gather(-1, proposed[:, None])[:, 0]
        selected_q = q.gather(-1, proposed[:, None])[:, 0]
        flags = uniforms < torch.where(
            selected_q > 0, selected_p / selected_q, 0
        ).clamp(max=1)
        mass = (p - q).clamp(min=0)
        total = mass.sum(-1, keepdim=True)
        residual = torch.where(
            total > 0, mass / total.clamp(min=torch.finfo(torch.float32).tiny), p
        )
        return flags, residual

    for run in captured["runs"]:
        arrays = np.load(args.out / run["arrays"])
        committed = list(run["prompt_ids"])
        for row in run["traces"]:

            def tensor(name, arrays=arrays, row=row):
                return torch.from_numpy(arrays[row[name]].astype(np.float32))

            q, p = tensor("proposal_probabilities"), tensor("target_probabilities")
            for key, expected in (("draft_logits", q), ("verify_logits", p)):
                logits = tensor(key) / run["temperature"]
                actual = probabilities(tensor(key), run["temperature"], run["top_p"])
                # Equal BF16 logits need not have the same sort order on Metal
                # and torch. Verify the actual support against the mathematical
                # nucleus contract, then compare probabilities ON that support.
                # Acceptance below still replays identical captured p/q exactly.
                support = expected > 0
                cutoff = (
                    logits.masked_fill(~support, torch.inf).min(-1, keepdim=True).values
                )
                assert torch.all(support | (logits <= cutoff))
                # Use float64 for the independent mass check: torch's CPU
                # float32 softmax over 64K tokens itself loses several ppm.
                raw = logits.double().softmax(-1)
                retained = raw.masked_fill(~support, 0)
                mass = retained.sum(-1, keepdim=True)
                final_mass = (
                    retained.masked_fill(~support, torch.inf)
                    .min(-1, keepdim=True)
                    .values
                )
                assert torch.all(mass >= run["top_p"] - 1e-6)
                assert torch.all(mass - final_mass <= run["top_p"] + 1e-6)
                torch.testing.assert_close(
                    retained / mass, expected.double(), rtol=2e-5, atol=2e-6
                )
                different = (actual > 0) != support
                report["nucleus_support_checks"].append(
                    {
                        "prompt": run["prompt"],
                        "seed": run["seed"],
                        "cycle": len(report["nucleus_support_checks"]) // 2,
                        "kind": key,
                        "torch_sort_support_disagreements": int(different.sum()),
                        "max_raw_probability_difference": float(
                            (actual - expected).abs().max()
                        ),
                        "nucleus_contract_passed": True,
                    }
                )
            proposals = torch.tensor(row["proposals"])
            if len(proposals) > 1:
                flags, residual = residual_and_flags(
                    p[:-1], q[1:], proposals[1:], tensor("acceptance_uniforms")
                )
                np.testing.assert_array_equal(
                    flags.numpy(), arrays[row["acceptance_flags"]]
                )
                torch.testing.assert_close(
                    residual, tensor("residual_probabilities"), rtol=2e-5, atol=2e-6
                )
                rejected = torch.nonzero(~flags).flatten().tolist()
                accepted = rejected[0] if rejected else len(proposals) - 1
            else:
                accepted = 0
            assert accepted == row["accepted"]
            output = row["proposals"][: accepted + 1]
            correction = arrays[row["correction_tokens"]]
            targets = arrays[row["target_samples"]]
            output.append(
                int(
                    correction[accepted]
                    if accepted < len(proposals) - 1
                    else targets[-1]
                )
            )
            output = output[
                : run["max_tokens"] - (len(committed) - len(run["prompt_ids"]))
            ]
            for i, token in enumerate(output):
                if token in run["eos"]:
                    output = output[: i + 1]
                    break
            assert row["cache_before"] == len(committed) - 1
            assert output == row["committed"]
            committed.extend(output)
            assert row["cache_after"] == len(committed) - 1
        assert committed[len(run["prompt_ids"]) :] == run["output_ids"]
        report["replay"].append(
            {
                "prompt": run["prompt"],
                "seed": run["seed"],
                "cycles": len(run["traces"]),
                "passed": True,
            }
        )

    # Independent real HF model and unfused conditional LoRA, not the MLX
    # decoder or the earlier fixed-block reference's projection wrappers.
    model, tokenizer = _load_model(args.snapshot, args.device)
    weights = load_file(
        str(args.adapter / "adapter_model.safetensors"), device=args.device
    )
    config = json.loads((args.adapter / "adapter_config.json").read_text())
    scale, context, consumed = config["lora_alpha"] / config["r"], {"mask": None}, set()

    def hook(a, b):
        def apply(_module, inputs, output):
            if context["mask"] is None:
                return output
            hidden = functional.linear(inputs[0], a) * context["mask"][..., None]
            return output + functional.linear(hidden, b) * scale

        return apply

    for index, layer in enumerate(model.model.layers):
        for target in TARGETS:
            owner, name = target.split(".")
            prefix = f"model.layers.{index}.{target}"
            a, b = prefix + ".lora_A.weight", prefix + ".lora_B.weight"
            getattr(getattr(layer, owner), name).register_forward_hook(
                hook(weights[a], weights[b])
            )
            consumed.update((a, b))
    assert consumed == set(weights)
    with torch.inference_mode():
        for run in captured["runs"]:
            generator = torch.Generator(device=args.device).manual_seed(run["seed"])
            committed = list(run["prompt_ids"])
            cycles = []
            while len(committed) - len(run["prompt_ids"]) < run["max_tokens"]:
                left = run["max_tokens"] - (len(committed) - len(run["prompt_ids"]))
                size = min(run["block_size"], left)
                noise = torch.randint(
                    1,
                    model.config.vocab_size,
                    (size - 1,),
                    generator=generator,
                    device=args.device,
                ).tolist()
                context["mask"] = torch.tensor(
                    [[0] * len(committed) + [1] * (size - 1)],
                    dtype=torch.bfloat16,
                    device=args.device,
                )
                logits = model(
                    torch.tensor([committed + noise], device=args.device),
                    use_cache=False,
                ).logits[0, -size:]
                q = probabilities(logits, run["temperature"], run["top_p"])
                proposed = torch.multinomial(q, 1, generator=generator)[:, 0]
                context["mask"] = None
                logits = model(
                    torch.tensor([committed + proposed.tolist()], device=args.device),
                    use_cache=False,
                ).logits[0, -size:]
                p = probabilities(logits, run["temperature"], run["top_p"])
                assert torch.isfinite(p).all() and torch.isfinite(q).all()
                accepted = 0
                for index in range(size - 1):
                    token = proposed[index + 1]
                    ratio = min(1.0, float(p[index, token] / q[index + 1, token]))
                    if (
                        float(torch.rand((), generator=generator, device=args.device))
                        >= ratio
                    ):
                        break
                    accepted += 1
                if accepted < size - 1:
                    residual = (p[accepted] - q[accepted + 1]).clamp(min=0)
                    distribution = (
                        residual / residual.sum() if residual.sum() > 0 else p[accepted]
                    )
                else:
                    distribution = p[-1]
                correction = int(
                    torch.multinomial(distribution, 1, generator=generator)
                )
                output = (proposed[: accepted + 1].tolist() + [correction])[:left]
                stop = next(
                    (i for i, token in enumerate(output) if token in run["eos"]), None
                )
                if stop is not None:
                    output = output[: stop + 1]
                committed.extend(output)
                cycles.append(
                    {
                        "noise": noise,
                        "proposals": proposed.tolist(),
                        "accepted": accepted,
                        "committed": output,
                    }
                )
                if stop is not None:
                    break
            tokens = committed[len(run["prompt_ids"]) :]
            result = {
                "prompt": run["prompt"],
                "seed": run["seed"],
                "output_ids": tokens,
                "text": tokenizer.decode(tokens),
                "cycles": cycles,
                "finish_reason": "stop" if tokens[-1] in run["eos"] else "length",
            }
            report["hf_runs"].append(result)
            (args.out / "reference.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                json.dumps({k: v for k, v in result.items() if k != "cycles"}),
                flush=True,
            )
    report["passed"] = True
    (args.out / "reference.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "reference"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    (capture if args.mode == "capture" else reference)(args)
