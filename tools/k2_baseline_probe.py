# SPDX-License-Identifier: Apache-2.0
"""Reproduce pre-family K2 failures from a selected Git revision, without weights.

Only config construction, safetensors filename discovery, and adapter metadata
loading are attempted. No checkout files or Hugging Face artifacts are changed.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", args.ref], text=True).strip()
    report = {"commit": commit, "source_hashes": {}, "cases": {}, "passed": False}

    def module(path, name):
        source = subprocess.check_output(["git", "show", f"{commit}:{path}"])
        report["source_hashes"][path] = hashlib.sha256(source).hexdigest()
        result = types.ModuleType(name)
        result.__file__ = str(Path(path).absolute())
        result.__package__ = "omlx"
        sys.modules[name] = result
        exec(compile(source, f"{commit}:{path}", "exec"), result.__dict__)
        return result

    model = module("omlx/patches/k2_horizon/k2_horizon_model.py", "omlx._baseline_k2")
    discovery = module("omlx/model_discovery.py", "omlx._baseline_discovery")
    revisions = {
        "0.9B": "ee770e713760cf6350e4322cdbbff91a163b7d70",
        "3.7B": "633f52ad28b17edeabd82afc61d2d13b4c59a561",
        "0.9B-Uno": "edb1c6072e5dfda2840bef38b11b79f08e1015ed",
    }
    for name, revision in revisions.items():
        repo_id = f"IFM/K2-Horizon-{name}"
        path = args.cache / repo_id.replace("/", "--")
        path = path.with_name("models--" + path.name) / "snapshots" / revision
        if not path.is_dir():
            raise FileNotFoundError(path)
        case = {
            "snapshot": str(path),
            "model_glob_shards": len(list(path.glob("model*.safetensors"))),
            "pytorch_glob_shards": len(list(path.glob("pytorch_model*.safetensors"))),
            "discovered_as_mlx": discovery._is_hf_cache_mlx_compatible(path, repo_id),
        }
        try:
            if name.endswith("Uno"):
                import mlx.nn as nn
                from mlx_lm.tuner.utils import load_adapters

                load_adapters(nn.Module(), str(path))
            else:
                model.ModelArgs.from_dict(
                    json.loads((path / "config.json").read_text())
                )
        except (TypeError, ValueError, AttributeError, KeyError) as error:
            case["exception"] = {"type": type(error).__name__, "message": str(error)}
        report["cases"][name] = case
    assert report["cases"]["0.9B"]["exception"]["type"] == "TypeError"
    assert report["cases"]["3.7B"]["exception"]["type"] == "ValueError"
    assert report["cases"]["3.7B"]["model_glob_shards"] == 0
    assert report["cases"]["3.7B"]["pytorch_glob_shards"] == 36
    assert "num_layers" in report["cases"]["0.9B-Uno"]["exception"]["message"]
    assert not any(case["discovered_as_mlx"] for case in report["cases"].values())
    report["passed"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
