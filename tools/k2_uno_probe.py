# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual local Uno adapter and record AR/Uno output agreement."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from k2_family_probe import PROMPTS

from omlx.patches.k2_horizon import apply_k2_horizon_patch
from omlx.patches.k2_horizon.uno_adapter import load_uno_adapter
from omlx.patches.k2_horizon.uno_decode import UnoDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    apply_k2_horizon_patch()
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(
        str(args.snapshot),
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    report = {"snapshot": str(args.snapshot), "adapter": str(args.adapter), "runs": []}
    base_id = json.loads((args.adapter / "adapter_config.json").read_text())[
        "base_model_name_or_path"
    ]
    baselines = []
    for prompt in PROMPTS:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            reasoning_effort="low",
        )
        started = time.perf_counter()
        tokens = [
            r.token
            for r in stream_generate(
                model,
                tokenizer,
                ids,
                max_tokens=args.max_tokens,
                sampler=make_sampler(temp=0),
            )
        ]
        baselines.append((prompt, ids, tokens, time.perf_counter() - started))
    report["adapter_load"] = load_uno_adapter(
        model, args.adapter, base_model_id=base_id
    )
    for prompt, ids, expected, baseline_seconds in baselines:
        # stream_generate's EOS event may be excluded from its output iterator.
        expected = [t for t in expected if t not in tokenizer.eos_token_ids]
        for block_size in (1, 2, 4, 8, 16):
            decoder = UnoDecoder(
                model,
                eos_token_ids=tokenizer.eos_token_ids,
                temperature=0,
                block_size=block_size,
                noise_mode="deterministic_uniform",
            )
            started = time.perf_counter()
            cycles = list(decoder.generate(ids, max_tokens=args.max_tokens))
            elapsed = time.perf_counter() - started
            actual = [
                token
                for cycle in cycles
                for token in cycle.tokens
                if token not in tokenizer.eos_token_ids
            ]
            agreement = 0
            for left, right in zip(actual, expected):
                if left != right:
                    break
                agreement += 1
            item = {
                "prompt": prompt,
                "prompt_ids": ids,
                "block_size": block_size,
                "temperature": 0,
                "baseline_ids": expected,
                "output_ids": actual,
                "text": tokenizer.decode(actual),
                "exact_greedy_match": actual == expected,
                "agreement_prefix": agreement,
                "ar_seconds": baseline_seconds,
                "seconds": elapsed,
                "accepted_proposals": sum(c.accepted_proposals for c in cycles),
                "proposed_tokens": sum(c.proposed_tokens for c in cycles),
                "forwards": sum(c.forwards for c in cycles),
                "tokens_per_forward": len(actual) / sum(c.forwards for c in cycles),
            }
            report["runs"].append(item)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in item.items()
                        if k not in ("prompt_ids", "baseline_ids", "output_ids")
                    }
                ),
                flush=True,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
    decoder = UnoDecoder(
        model,
        eos_token_ids=tokenizer.eos_token_ids,
        temperature=1.0,
        top_p=0.95,
        block_size=8,
        seed=42,
    )
    cycles = list(decoder.generate(baselines[0][1], max_tokens=args.max_tokens))
    report["sampled"] = {
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": 42,
        "text": tokenizer.decode([t for c in cycles for t in c.tokens]),
    }
    report["peak_metal_bytes"] = mx.get_peak_memory()
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
