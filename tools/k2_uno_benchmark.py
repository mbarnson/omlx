# SPDX-License-Identifier: Apache-2.0
"""Matched single-request AR/Uno core throughput on an existing local checkpoint.

Exactly N tokens are evaluated, with EOS stopping disabled in BOTH lanes. This
isolates throughput from differing near-tie text/EOS decisions; the resulting
post-EOS text is not a quality evaluation. Timings exclude model loading and
tokenization. Profiled runs add synchronization and are reported separately.
"""

import argparse
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from k2_uno_oracle import PROMPTS, TARGETS
from mlx_lm.models.cache import make_prompt_cache

from omlx.patches.k2_horizon import apply_k2_horizon_patch
from omlx.patches.k2_horizon.uno_adapter import load_uno_adapter
from omlx.patches.k2_horizon.uno_decode import UnoDecoder


class TimedModel:
    def __init__(self, model, metrics):
        self.model, self.metrics, self.phase = model, metrics, "prefill"

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        if kwargs.get("lora_mask") is not None:
            self.phase = "draft"
        elif self.phase == "draft":
            self.phase = "verify"
        start = time.perf_counter()
        output = self.model(*args, **kwargs)
        submitted = time.perf_counter()
        if self.phase == "prefill" and kwargs.get("cache") is not None:
            # Prefill only materializes KV in the real generators. Forcing the
            # unused vocabulary logits here would profile extra work.
            mx.eval([layer.state for layer in kwargs["cache"]])
        else:
            mx.eval(output)
        finished = time.perf_counter()
        self.metrics[self.phase + "_seconds"] += finished - start
        self.metrics[self.phase + "_submit_seconds"] += submitted - start
        self.metrics[self.phase + "_eval_wait_seconds"] += finished - submitted
        self.metrics[self.phase + "_forwards"] += 1
        return output


class TimedDecoder(UnoDecoder):
    def _trim(self, cache, length):
        start = time.perf_counter()
        super()._trim(cache, length)
        self.model.metrics["rollback_seconds"] += time.perf_counter() - start


def run(model, ids, count, block_size, profile, bindings):
    # Retain adapter allocations for an equal resident-memory budget, but remove
    # its Python wrappers from the base AR path before starting the clock.
    for owner, name, wrapper in bindings:
        setattr(
            owner,
            name,
            wrapper.linear if block_size in (None, "ar_native") else wrapper,
        )
    metrics = defaultdict(float)
    selected = TimedModel(model, metrics) if profile else model
    mx.synchronize()
    mx.reset_peak_memory()
    start = time.perf_counter()
    first = None
    tokens, forwards, accepted, proposed = [], 0, 0, 0
    if block_size == "ar_native":
        from mlx_lm.generate import generate_step

        prefetched = None

        def progress(done, total):
            nonlocal prefetched
            if done >= total - 1 and prefetched is None:
                prefetched = time.perf_counter()
                if profile:
                    selected.phase = "decode"

        for token, _ in generate_step(
            mx.array(ids),
            selected,
            max_tokens=count,
            prefill_step_size=512,
            prompt_progress_callback=progress,
        ):
            tokens.append(token)
            if first is None:
                first = time.perf_counter()
        # The pinned generator computes one unused lookahead token before it
        # finishes; include that real forward in TPF instead of hiding it.
        forwards = count + 1
        prefill_seconds = prefetched - start
    elif block_size is None:
        cache = make_prompt_cache(selected)
        for offset in range(0, len(ids) - 1, 512):
            selected(
                mx.array([ids[offset : min(len(ids) - 1, offset + 512)]]), cache=cache
            )
            mx.eval([layer.state for layer in cache])
        prefetched = time.perf_counter()
        if profile:
            selected.phase = "decode"
        seed = ids[-1]
        for _ in range(count):
            logits = selected(mx.array([[seed]]), cache=cache)[0, -1]
            seed = int(mx.argmax(logits).item())
            tokens.append(seed)
            forwards += 1
            if first is None:
                first = time.perf_counter()
        prefill_seconds = prefetched - start
    else:
        decoder = (TimedDecoder if profile else UnoDecoder)(
            selected,
            eos_token_ids=[],
            block_size=block_size,
            temperature=0,
            seed=42,
            prefill_step_size=512,
        )
        for cycle in decoder.generate(ids, max_tokens=count):
            if first is None:
                first = time.perf_counter()
            tokens.extend(cycle.tokens)
            forwards += cycle.forwards
            accepted += cycle.accepted_proposals
            proposed += cycle.proposed_tokens
        prefill_seconds = metrics.get("prefill_seconds") if profile else None
    mx.synchronize()
    elapsed = time.perf_counter() - start
    assert len(tokens) == count
    return {
        "block_size": block_size,
        "profiled": profile,
        "seconds": elapsed,
        "first_commit_seconds": first - start,
        "prefill_seconds": prefill_seconds,
        "post_first_commit_seconds": elapsed - (first - start),
        "tokens_per_second_including_prefill": count / elapsed,
        "tokens_per_forward_excluding_prefill": count / forwards,
        "forwards_excluding_prefill": forwards,
        "accepted_proposals": accepted,
        "proposed_tokens": proposed,
        "acceptance": accepted / proposed if proposed else None,
        "peak_metal_bytes": mx.get_peak_memory(),
        "phase_profile": dict(metrics),
        "output_ids": tokens,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Refresh instrumented phases without rerunning throughput trials",
    )
    args = parser.parse_args()
    if args.tokens <= 0 or args.trials < 3:
        raise ValueError("Use positive tokens and at least three timed trials")
    apply_k2_horizon_patch()
    from mlx_lm import load

    started = time.perf_counter()
    model, tokenizer = load(
        str(args.snapshot),
        trust_remote_code=False,
        tokenizer_config={"trust_remote_code": False},
    )
    config = json.loads((args.adapter / "adapter_config.json").read_text())
    adapter = load_uno_adapter(
        model, args.adapter, base_model_id=config["base_model_name_or_path"]
    )
    bindings = []
    for layer in model.layers:
        for target in TARGETS:
            scope, name = target.split(".")
            owner = getattr(layer, scope)
            bindings.append((owner, name, getattr(owner, name)))
    mx.eval(model.parameters())
    if args.profile_only:
        report = json.loads(args.out.read_text())
        assert report["passed"] and report["snapshot"] == str(args.snapshot)
        assert report["adapter"]["sha256"] == adapter["sha256"]
        assert report["method"]["tokens"] == args.tokens
        for row in report["prompts"]:
            row["profiles"] = [
                run(model, row["prompt_ids"], args.tokens, lane, True, bindings)
                for lane in (None, "ar_native", 2, 4, 8, 16)
            ]
        report["method"]["profile_prefill_materializes_only_kv"] = True
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        return
    report = {
        "snapshot": str(args.snapshot),
        "adapter": adapter,
        "load_seconds": time.perf_counter() - started,
        "platform": platform.platform(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm")
        },
        "method": {
            "concurrency": 1,
            "temperature": 0,
            "seed": 42,
            "tokens": args.tokens,
            "eos_stopping": False,
            "trials": args.trials,
            "warmups_per_prompt_lane": 1,
            "prefill_chunk": 512,
            "resident_adapter_in_ar": True,
            "adapter_wrappers_removed_in_ar": True,
            "ar_baselines": "ar_native: pinned mlx-lm generate_step; ar: direct serial KV loop",
            "profile_has_extra_forward_eval_barriers": True,
            "profile_prefill_materializes_only_kv": True,
            "eval_wait_note": "GPU evaluation wait includes execution; it is not pure synchronization overhead",
            "timing_note": "Unprofiled core end-to-end is authoritative; phase profiles are separate instrumented runs",
        },
        "prompts": [],
        "passed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for prompt in PROMPTS:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            reasoning_effort="low",
        )
        row = {
            "prompt": prompt,
            "prompt_ids": ids,
            "runs": [],
            "profiles": [],
            "summary": {},
        }
        lanes = [None, "ar_native", 2, 4, 8, 16]
        for lane in lanes:
            run(model, ids, args.tokens, lane, False, bindings)
        for trial in range(args.trials):
            order = lanes[trial % len(lanes) :] + lanes[: trial % len(lanes)]
            for lane in order:
                item = run(model, ids, args.tokens, lane, False, bindings)
                item["trial"] = trial
                row["runs"].append(item)
        for lane in lanes:
            row["profiles"].append(run(model, ids, args.tokens, lane, True, bindings))
            selected = [r for r in row["runs"] if r["block_size"] == lane]
            assert all(r["output_ids"] == selected[0]["output_ids"] for r in selected)
            summary = {}
            for key in (
                "seconds",
                "first_commit_seconds",
                "tokens_per_second_including_prefill",
                "tokens_per_forward_excluding_prefill",
                "peak_metal_bytes",
            ):
                values = [r[key] for r in selected]
                summary[key] = {
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            row["summary"]["ar" if lane is None else str(lane)] = summary
        baseline = row["summary"]["ar_native"]["seconds"]["median"]
        for summary in row["summary"].values():
            summary["speedup_vs_native_ar"] = baseline / summary["seconds"]["median"]
        report["prompts"].append(row)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"prompt": prompt, "summary": row["summary"]}), flush=True)
    report["passed"] = True
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
