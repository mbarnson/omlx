# SPDX-License-Identifier: Apache-2.0
"""Compare full/chunked/incremental real K2 logits across YaRN's 8192 boundary."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from omlx.patches.k2_horizon import apply_k2_horizon_patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--length", type=int, default=8208)
    args = parser.parse_args()
    if args.length < 8208:
        parser.error("length must cross the original 8192-token context boundary")
    apply_k2_horizon_patch()
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = load(
        str(args.snapshot),
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    unit = tokenizer.encode(
        "The following is a short factual record. Paris is in France. Canberra is in Australia.\n",
        add_special_tokens=False,
    )
    ids = (unit * ((args.length + len(unit) - 1) // len(unit)))[: args.length]
    positions = [0, 1, 511, 512, 8190, 8191, 8192, 8193, args.length - 1]
    report = {
        "snapshot": str(args.snapshot),
        "dtype": "bfloat16",
        "length": len(ids),
        "prompt_ids": ids,
        "positions": positions,
        "modes": {},
        "threshold_relative_rms": 0.05,
    }
    arrays = {}
    mx.reset_peak_memory()
    started = time.perf_counter()
    logits = model(mx.array([ids]))
    arrays["full"] = np.array(logits[0, mx.array(positions)].astype(mx.float32))
    report["modes"]["full"] = {"seconds": time.perf_counter() - started}
    del logits
    mx.clear_cache()
    for mode, boundary in (("chunks_512", None), ("incremental_boundary", 8189)):
        started = time.perf_counter()
        cache = make_prompt_cache(model)
        selected = {}
        start = 0
        while start < len(ids):
            step = 1 if boundary is not None and start >= boundary else 512
            end = min(
                len(ids),
                start + step,
                boundary if boundary is not None and start < boundary else len(ids),
            )
            output = model(mx.array([ids[start:end]]), cache=cache)
            for position in positions:
                if start <= position < end:
                    selected[position] = np.array(
                        output[0, position - start].astype(mx.float32)
                    )
            mx.eval([layer.state for layer in cache])
            start = end
        arrays[mode] = np.stack([selected[position] for position in positions])
        reference = arrays["full"].astype(np.float64)
        actual = arrays[mode].astype(np.float64)
        rms = np.sqrt(np.mean((actual - reference) ** 2, axis=-1)) / np.maximum(
            np.sqrt(np.mean(reference**2, axis=-1)), 1e-8
        )
        report["modes"][mode] = {
            "seconds": time.perf_counter() - started,
            "relative_rms_by_position": rms.tolist(),
            "argmax_agreement_by_position": (
                actual.argmax(-1) == reference.argmax(-1)
            ).tolist(),
            "cache_offsets": [layer.offset for layer in cache],
            "passed": bool(
                np.isfinite(actual).all()
                and np.max(rms) <= 0.05
                and all(layer.offset == len(ids) for layer in cache)
            ),
        }
        del output, cache
        mx.clear_cache()
    report["peak_metal_bytes"] = mx.get_peak_memory()
    report["passed"] = all(row.get("passed", True) for row in report["modes"].values())
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "logits.npz", **arrays)
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "prompt_ids"},
            indent=2,
        ),
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
