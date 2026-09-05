# SPDX-License-Identifier: Apache-2.0
"""Record strict load, cached-forward, and greedy proof for a local K2 base."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from omlx.patches.k2_horizon import apply_k2_horizon_patch
from omlx.patches.k2_horizon.checkpoint import checkpoint_files

PROMPTS = [
    "What is the capital of Australia? Answer in one sentence.",
    "Write a Python function is_palindrome(s) that ignores spaces and case.",
    "A train travels 60 miles in 45 minutes. What is its speed in miles per hour?",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Report discovery, shards and config without loading tensors",
    )
    args = parser.parse_args()
    report = {
        "snapshot": str(args.snapshot),
        "revision": args.snapshot.name,
        "platform": platform.platform(),
        "versions": {
            n: importlib.metadata.version(n) for n in ("mlx", "mlx-lm", "transformers")
        },
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "settings": {
            "temperature": 0,
            "reasoning_effort": "low",
            "max_tokens": args.max_tokens,
        },
        "stages": {},
    }
    current_stage = "discovery"
    try:
        from omlx.model_discovery import _is_hf_cache_mlx_compatible, _is_model_dir

        source = args.snapshot.parent.parent.name.removeprefix("models--").replace(
            "--", "/"
        )
        report["stages"]["discovery"] = {
            "model_directory": _is_model_dir(args.snapshot),
            "hf_cache_compatible": _is_hf_cache_mlx_compatible(args.snapshot, source),
        }
        if not report["stages"]["discovery"]["model_directory"]:
            raise ValueError("Snapshot is not discoverable as a model directory")
        current_stage = "shards"
        report["shards"] = [p.name for p in checkpoint_files(args.snapshot)]
        report["stages"]["shards"] = {"passed": True, "count": len(report["shards"])}
        current_stage = "configuration"
        apply_k2_horizon_patch()
        from mlx_lm.models.k2_horizon import ModelArgs

        config = ModelArgs.from_dict(
            json.loads((args.snapshot / "config.json").read_text())
        )
        report["stages"]["configuration"] = {
            "passed": True,
            "layers": config.num_hidden_layers,
            "rms_groups": config.layernorm_num_groups,
            "experts": config.num_experts,
            "value_experts": config.mova_num_experts,
            "attention_gate": config.attention_gate_func,
        }
        if args.inspect_only:
            report["passed"] = True
            report["scope"] = "metadata only; load and generation not attempted"
            return
        from mlx_lm import load, stream_generate
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        mx.reset_peak_memory()
        current_stage = "load"
        started = time.perf_counter()
        model, tokenizer = load(
            str(args.snapshot),
            tokenizer_config={"trust_remote_code": False},
            trust_remote_code=False,
        )
        report["load"] = {
            "seconds": time.perf_counter() - started,
            "tensors": len(tree_flatten(model.parameters())),
            "parameters": sum(x.size for _, x in tree_flatten(model.parameters())),
            "dtypes": sorted(
                {str(x.dtype) for _, x in tree_flatten(model.parameters())}
            ),
            "peak_metal_bytes": mx.get_peak_memory(),
        }
        report["stages"]["load"] = {"passed": True, "strict": True}
        print(json.dumps(report["load"]), flush=True)
        report["eos_token_ids"] = sorted(tokenizer.eos_token_ids)
        report["generations"] = []
        current_stage = "generation"
        for prompt in PROMPTS:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                reasoning_effort="low",
            )
            inputs = mx.array([ids])
            full = model(inputs)[:, -1].astype(mx.float32)
            cache = make_prompt_cache(model)
            model(inputs[:, :-1], cache=cache)
            cached = model(inputs[:, -1:], cache=cache)[:, -1].astype(mx.float32)
            error = mx.max(mx.abs(full - cached)).item()
            finite = mx.all(mx.isfinite(full)).item()
            same_argmax = mx.array_equal(
                mx.argmax(full, -1), mx.argmax(cached, -1)
            ).item()
            del cache, full, cached
            started = time.perf_counter()
            tokens, pieces = [], []
            for response in stream_generate(
                model,
                tokenizer,
                ids,
                max_tokens=args.max_tokens,
                sampler=make_sampler(temp=0),
            ):
                tokens.append(response.token)
                pieces.append(response.text)
            item = {
                "prompt": prompt,
                "prompt_ids": ids,
                "output_ids": tokens,
                "text": "".join(pieces),
                "seconds": time.perf_counter() - started,
                "cached_logit_max_error": error,
                "finite_logits": finite,
                "cached_argmax_agreement": same_argmax,
            }
            report["generations"].append(item)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in item.items()
                        if k not in ("prompt_ids", "output_ids")
                    }
                ),
                flush=True,
            )
            if not finite or not same_argmax or not item["text"].strip():
                raise AssertionError("Generation/finite/cache gate failed")
        report["peak_metal_bytes"] = mx.get_peak_memory()
        report["stages"]["generation"] = {"passed": True, "prompts": len(PROMPTS)}
        report["passed"] = True
    except Exception as error:
        report["passed"] = False
        report["error"] = repr(error)
        report["failed_stage"] = current_stage
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
