# SPDX-License-Identifier: Apache-2.0
"""Exercise cached Uno through the production engine and record serving evidence."""

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

from omlx.engine.uno import UnoEngine
from omlx.model_discovery import discover_models


async def probe(args):
    report = {"adapter": str(args.adapter), "requests": []}
    engine = UnoEngine(str(args.adapter))
    try:
        started = time.perf_counter()
        await engine.start()
        report["load_seconds"] = time.perf_counter() - started
        report["stats"] = engine.get_stats()
        assert (
            engine._prefill_guard.memory_monitor.estimate_prefill_peak_bytes(1024, 512)
            > 0
        )
        for prompt in (
            "What is the capital of Australia? Answer in one sentence.",
            "Write a Python function is_palindrome(s) that ignores spaces and case.",
            "A train travels 60 miles in 45 minutes. What is its speed in miles per hour?",
        ):
            messages = [{"role": "user", "content": prompt}]
            options = dict(
                max_tokens=args.max_tokens,
                temperature=0.0,
                seed=42,
                chat_template_kwargs={"reasoning_effort": "low"},
            )
            await engine.preflight_chat(messages, **options)
            result = await engine.chat(messages, **options)
            chunks = [item async for item in engine.stream_chat(messages, **options)]
            streamed = "".join(item.new_text for item in chunks)
            assert streamed == result.text
            assert chunks[-1].tokens == result.tokens
            assert result.completion_tokens == len(result.tokens) <= args.max_tokens
            assert result.cached_tokens == 0
            row = {
                "prompt": prompt,
                "output": asdict(result),
                "stream_matches": True,
                "speculation": engine.get_stats()["speculation"],
            }
            report["requests"].append(row)
            print(json.dumps(row), flush=True)
        report["discovery"] = {
            name: asdict(model)
            for name, model in discover_models(args.adapter.parents[2]).items()
            if "K2-Horizon" in name
        }
        report["peak_metal_bytes"] = mx.get_peak_memory()
    finally:
        await engine.stop()
        report["stopped"] = engine.get_stats()["loaded"] is False
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=256)
    asyncio.run(probe(parser.parse_args()))


if __name__ == "__main__":
    main()
