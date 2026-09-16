"""Serve a Qwen3.8 speed profile on localhost with isolated server state.

Each invocation creates a run directory under --state-dir. This launcher
sets the existing SchedulerConfig prefill size without changing the serving
library or the checkpoint. Native kernels must be built in this checkout.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument(
        "--chunk", type=int, choices=[512, 1024, 2048, 4096], default=2048
    )
    parser.add_argument(
        "--burst", choices=["balanced", "aggressive"], default="balanced"
    )
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--mtp-depth", type=int, choices=range(1, 9), default=5)
    args = parser.parse_args()
    model = args.model.expanduser().resolve()
    name = model.name
    assert model.is_dir() and not (model / "BUILDING").exists(), model
    config = json.loads((model / "config.json").read_text())
    assert config["text_config"]["num_experts_per_tok"] == 4
    state_root = args.state_dir.expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    state = Path(tempfile.mkdtemp(prefix="run-", dir=state_root))
    print(f"Experimental server state: {state}", flush=True)
    served = state / "served-models"
    served.mkdir(exist_ok=True)
    link = served / name
    link.symlink_to(model, target_is_directory=True)
    (state / "settings.json").write_text(
        json.dumps(
            {
                "server": {"burst_decode_mode": args.burst},
                "scheduler": {"max_concurrent_requests": 1, "decode_fairness": False},
                "cache": {"enabled": False},
            },
            indent=2,
        )
        + "\n"
    )
    (state / "model_settings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    name: {
                        "qwen4_ple_ssd_offload": True,
                        "mtp_enabled": True,
                        "mtp_num_draft_tokens": args.mtp_depth,
                        "enable_thinking": False,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": 0,
                        "max_tokens": 4096,
                        "max_context_window": 32768,
                        "ttl_seconds": 300,
                    }
                },
            },
            indent=2,
        )
        + "\n"
    )
    sys.path.insert(0, str(RUNTIME))
    from omlx.custom_kernels.glm_moe_dsa import fast

    if not fast.is_native_available() or not fast.has_symbol(
        "qwen4_qsa_sparse_gqa_attention"
    ):
        raise SystemExit("Build this checkout with OMLX_WITH_CUSTOM_KERNEL=1 first.")
    from omlx.settings import GlobalSettings

    original = GlobalSettings.to_scheduler_config

    def scheduler_config(settings):
        config = original(settings)
        config.prefill_step_size = args.chunk
        return config

    # Experimental launcher only: the stock CLI does not expose this existing
    # SchedulerConfig field. Keep the override out of the serving library.
    GlobalSettings.to_scheduler_config = scheduler_config
    import mlx.core as mx

    mx.set_memory_limit(96 * 2**30)
    mx.set_cache_limit(256 * 2**20)
    mx.set_wired_limit(90 * 2**30)
    from omlx.cli import main as omlx_main

    sys.argv = [
        "omlx",
        "serve",
        "--model-dir",
        str(served),
        "--base-path",
        str(state),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--max-concurrent-requests",
        "1",
        "--memory-guard-gb",
        "96",
        "--no-cache",
    ]
    omlx_main()


if __name__ == "__main__":
    main()
