"""Measure Qwen4 MTP ceilings and prefill chunks without changing weights.

Run one process at a time; compare repeated per-workload results, not the
aggregate median. Prefill includes first-token latency and bypasses prefix cache.
The OS file cache is neither flushed nor controlled.
"""

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME))

PROMPTS = {
    "systems": "Explain how you would investigate intermittent latency spikes in a local database-backed API. Give a concrete, detailed sequence of measurements and explain how each observation changes the next step. Discuss disk I/O, locks, CPU, and memory. Write at least 800 words.",
    "story": "Write an original 1000-word story about a mechanic who discovers that the lighthouse on an abandoned island is still being maintained. Use concrete details, natural dialogue, a consistent sequence of events, and a satisfying explanation. Begin the story immediately.",
    "structured": 'Return ONLY a JSON object with keys "ids" and "total". Given records [{"id":"b","active":true,"amount":7},{"id":"a","active":false,"amount":100},{"id":"d","active":true,"amount":3},{"id":"c","active":true,"amount":11}], include only active records, sort their ids ascending, and sum their amounts. Do not use markdown.',
    "code_canary": 'Write Python code only, without markdown or an explanation. Implement first_unique(text: str) -> str | None: return the first character occurring exactly once, preserving original order, or None. Handle empty strings and Unicode. Include exactly these assertions after the function: assert first_unique("swiss") == "w"; assert first_unique("aabb") is None; assert first_unique("") is None; assert first_unique("éé猫a猫") == "a".',
    "code": "Write a complete Python implementation of an in-memory LRU cache with optional per-entry expiration, an injectable monotonic clock, and a fixed capacity. Include at least six executable unittest tests covering edge cases. Explain the locking and eviction behavior after the code.",
}


async def main(args):
    experts = json.loads((args.model / "config.json").read_text())["text_config"][
        "num_experts_per_tok"
    ]
    import mlx.core as mx
    import psutil

    from omlx.custom_kernels.glm_moe_dsa import fast
    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.model_settings import ModelSettings
    from omlx.patches.mlx_lm_mtp import batch_generator as batch
    from omlx.request import SamplingParams
    from omlx.scheduler import SchedulerConfig

    assert not args.output.exists(), args.output
    required = (
        "qwen4_qsa_indexer_scores",
        "qwen4_qsa_topk_indices",
        "qwen4_qsa_sparse_gqa_attention",
    )
    if not fast.is_native_available() or fast.missing_symbols(required):
        raise RuntimeError(
            "Build this checkout's custom kernels before benchmarking: "
            "OMLX_WITH_CUSTOM_KERNEL=1 python setup.py build_ext --inplace"
        )
    mx.set_memory_limit(96 * 2**30)
    mx.set_cache_limit(256 * 2**20)
    mx.set_wired_limit(90 * 2**30)
    settings = ModelSettings(
        mtp_enabled=args.depth > 0,
        mtp_num_draft_tokens=args.depth or None,
        qwen4_ple_ssd_offload=True,
        enable_thinking=False,
    )
    config = SchedulerConfig(
        max_num_seqs=1,
        completion_batch_size=1,
        prefill_step_size=args.chunk,
        decode_fairness=False,
    )
    engine = VLMBatchedEngine(
        str(args.model), model_settings=settings, scheduler_config=config
    )
    segments = []
    original_stats = batch._log_mtp_stats

    def log_stats(uid, stats, finish_reason):
        segments.append(dataclasses.asdict(stats))
        return original_stats(uid, stats, finish_reason)

    batch._log_mtp_stats = log_stats
    report = {
        "model": str(args.model),
        "runtime": str(RUNTIME),
        "runtime_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=RUNTIME, text=True
        ).strip(),
        "depth": args.depth,
        "chunk": args.chunk,
        "prefill_output_tokens": args.prefill_output_tokens,
        "sampling": "greedy; thinking disabled",
        "gdn_extension": False,
        "experts_per_token": experts,
        "ple": "original BF16 SSD offload",
        "prefill_corpus": "README.md + scheduler.py, varied repository text",
        "background_activity": "user working, compiling, and YouTube; small differences are noise",
        "records": [],
        "environment": {
            name: os.environ.get(name)
            for name in (
                "OMLX_GDN_BLOCK_T",
                "OMLX_GDN_FUSED_G_BETA",
                "OMLX_QWEN4_GATHERED_MIN_QUERY",
                "OMLX_QWEN4_QSA_GATHERED_VERIFY",
                "OMLX_QWEN4_QSA_NATIVE_SCORE_MIN_ROWS",
                "OMLX_QWEN4_QSA_NATIVE_TOPK_MIN_ROWS",
                "OMLX_QWEN4_QSA_NATIVE_MAIN_MIN_ROWS",
                "OMLX_QWEN4_EAGER_DISPATCH",
                "OMLX_QWEN4_HC_FUSED",
                "OMLX_DECODE_BURST_MAX_STEPS",
                "OMLX_DECODE_BURST_BUDGET_SINGLE_S",
                "OMLX_DECODE_BURST_BUDGET_S",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    started = time.perf_counter()
    await engine.start()
    report["load_seconds"] = time.perf_counter() - started
    try:
        language = engine._vlm_model.language_model
        from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

        blocks = [
            module
            for _, module in engine._vlm_model.named_modules()
            if type(module) is Qwen3_5MoeSparseMoeBlock
        ]
        assert blocks and all(module.top_k == experts for module in blocks)
        ple = language.model.layers[1].ple.ple_embedding.ngram_embedding
        assert "DiskBacked" in type(ple).__name__, type(ple).__name__
        report["ple_runtime_class"] = type(ple).__name__
        report["mtp_runtime_enabled"] = getattr(
            language, "_omlx_mtp_decode_enabled", False
        )
        report["mtp_runtime_depth"] = getattr(language, "_omlx_mtp_depth", None)
        report["memory_after_load_gib"] = mx.get_active_memory() / 2**30
        save()
        print(
            json.dumps(
                {
                    "event": "loaded",
                    **{k: v for k, v in report.items() if k != "records"},
                }
            ),
            flush=True,
        )

        async def request(label, content, cap, repeat=-1):
            nonlocal segments
            segments = []
            prompt = engine.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            t0 = time.perf_counter()
            result = await engine._engine.generate(
                prompt=prompt,
                benchmark_trace=True,
                sampling_params=SamplingParams(
                    max_tokens=cap, temperature=0, top_p=1, top_k=0, seed=17
                ),
            )
            elapsed = time.perf_counter() - t0
            ids = result.output_token_ids
            assert len(ids) > 1 and len(ids) == result.completion_tokens
            assert result.cached_tokens == 0, "Unexpected prefix cache hit"
            decode_seconds = result.generated_until - result.first_token_at
            ttft = result.first_token_at - t0
            text = engine.tokenizer.decode(ids)
            triples = [tuple(ids[i : i + 3]) for i in range(len(ids) - 2)]
            row = {
                "label": label,
                "repeat": repeat,
                "prompt_tokens": result.prompt_tokens,
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                "output_tokens": len(ids),
                "decode_tps": (len(ids) - 1) / decode_seconds,
                "prefill_including_first_token_tps": result.prompt_tokens / ttft,
                "ttft_seconds": ttft,
                "wall_seconds": elapsed,
                "finish_reason": result.finish_reason,
                "unique_trigram_fraction": len(set(triples)) / max(1, len(triples)),
                "token_hash": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "text": text,
                "mtp_segments": segments,
                "mlx_active_gib": mx.get_active_memory() / 2**30,
                "system_available_gib": psutil.virtual_memory().available / 2**30,
            }
            report["records"].append(row)
            save()
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k not in ("text", "mtp_segments")}
                ),
                flush=True,
            )
            print(json.dumps({"label": label, "sample": text[:1000]}), flush=True)
            return row

        await request(
            "warmup",
            "Briefly explain why a write-ahead log helps a database recover after a crash.",
            48,
        )
        for repeat in range(args.repeats):
            for label in args.prompts:
                await request(label, PROMPTS[label], args.tokens, repeat)
        if args.prefill:
            corpus = "\n\n".join(
                (RUNTIME / path).read_text()
                for path in ("README.md", "omlx/scheduler.py")
            )
            corpus_ids = engine.tokenizer.encode(corpus, add_special_tokens=False)
            for repeat in range(args.prefill_repeats):
                for size in args.prefill:
                    assert len(corpus_ids) >= size
                    content = engine.tokenizer.decode(corpus_ids[:size])
                    if args.prefill_output_tokens > 48:
                        content += (
                            "\nUsing this repository as context, " + PROMPTS["systems"]
                        )
                    else:
                        content += "\nSummarize the architecture and scheduling behavior in this text in two sentences."
                    await request(
                        "prefill_" + str(size),
                        content,
                        args.prefill_output_tokens,
                        repeat,
                    )
        report["medians_by_workload"] = {}
        for label in dict.fromkeys(r["label"] for r in report["records"]):
            if label == "warmup":
                continue
            rows = [r for r in report["records"] if r["label"] == label]
            report["medians_by_workload"][label] = {
                key: statistics.median(r[key] for r in rows)
                for key in ("decode_tps", "prefill_including_first_token_tps")
            }
        report["mlx_peak_gib"] = mx.get_peak_memory() / 2**30
        from mlx_vlm.models.qwen4_exp import qsa_fast

        report["native_kernels"] = {
            "available": fast.is_native_available(),
            "symbols": fast.native_symbols(),
            "qsa_score_proven": qsa_fast._NATIVE_QSA_SCORE_PROVEN,
            "qsa_topk_proven": qsa_fast._NATIVE_QSA_TOPK_PROVEN,
            "qsa_main_proven": qsa_fast._NATIVE_QSA_MAIN_PROVEN,
        }
        report["complete"] = True
        save()
        print(
            json.dumps(
                {
                    "event": "complete",
                    "medians_by_workload": report["medians_by_workload"],
                    "mlx_peak_gib": report["mlx_peak_gib"],
                }
            ),
            flush=True,
        )
    finally:
        batch._log_mtp_stats = original_stats
        await engine.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--depth", type=int, choices=range(9), default=5)
    p.add_argument(
        "--chunk", type=int, choices=[256, 512, 1024, 2048, 4096], default=2048
    )
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument(
        "--prompts",
        nargs="*",
        choices=list(PROMPTS),
        default=["systems", "story", "code"],
    )
    p.add_argument("--prefill", nargs="*", type=int, default=[])
    p.add_argument("--prefill-repeats", type=int, default=2)
    p.add_argument("--prefill-output-tokens", type=int, default=48)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(main(p.parse_args()))
