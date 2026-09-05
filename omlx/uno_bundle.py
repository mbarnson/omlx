# SPDX-License-Identifier: Apache-2.0
"""Local-only resolution of a K2 Uno base and its conditional adapter.

PEFT's CAUSAL_LM metadata is insufficient to select a diffusion decoder.
Recognize official cache identities, or an explicit local uno_config.json.
This module intentionally does not import MLX or fetch Hub files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .patches.k2_horizon.checkpoint import checkpoint_files

RELEASED_BASES = {
    "IFM/K2-Horizon-0.9B": "ee770e713760cf6350e4322cdbbff91a163b7d70",
    "IFM/K2-Horizon-7B": "586b03f0fd1fbbf2f13eeafc33749e95ae34dd10",
}
RELEASED_ADAPTERS = {f"{base}-Uno": base for base in RELEASED_BASES}
RELEASED_CONTEXT_LIMITS = {"IFM/K2-Horizon-0.9B": 131072, "IFM/K2-Horizon-7B": 262144}


def _cache_identity(path: Path) -> tuple[str, str] | None:
    if path.parent.name != "snapshots":
        return None
    parts = path.parent.parent.name.split("--")
    if len(parts) == 3 and parts[0] == "models":
        return f"{parts[1]}/{parts[2]}", path.name
    return None


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def is_uno_candidate(path: Path, source_repo_id: str | None = None) -> bool:
    if (path / "uno_config.json").is_file():
        return True
    identity = _cache_identity(path.resolve())
    repo = source_repo_id or (identity[0] if identity else None)
    return repo in RELEASED_ADAPTERS


@dataclass(frozen=True)
class UnoBundle:
    base_path: Path
    adapter_path: Path
    base_model_id: str
    base_revision: str | None
    adapter_revision: str | None
    estimated_size: int
    context_length: int
    block_size: int
    config: dict


def resolve_uno_bundle(
    path: str | Path, *, source_repo_id: str | None = None
) -> UnoBundle:
    """Resolve only existing files; missing dependencies are actionable errors.

    Local registrations contain {format: "k2_uno", version: 1,
    base_model_id, base_path, adapter_path, block_size: 8}. Relative paths
    are relative to the registration directory. Official cache adapters use
    the pinned released base revision in the same Hub cache.
    """
    path = Path(path).expanduser().resolve()
    descriptor_path = path / "uno_config.json"
    if descriptor_path.is_file():
        descriptor = _read_object(descriptor_path)
        allowed = {
            "format",
            "version",
            "base_model_id",
            "base_path",
            "adapter_path",
            "block_size",
        }
        if set(descriptor) - allowed:
            raise ValueError(
                f"Unknown Uno registration fields: {set(descriptor) - allowed}"
            )
        if descriptor.get("format") != "k2_uno" or descriptor.get("version") != 1:
            raise ValueError("Uno registration requires format=k2_uno and version=1")
        base_id = descriptor.get("base_model_id")
        paths = []
        for name in ("base_path", "adapter_path"):
            value = descriptor.get(name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Uno registration requires {name}")
            paths.append((path / Path(value).expanduser()).resolve())
        base_path, adapter_path = paths
        block_size = descriptor.get("block_size", 8)
    else:
        identity = _cache_identity(path)
        repo = source_repo_id or (identity[0] if identity else None)
        if repo not in RELEASED_ADAPTERS or not identity or identity[0] != repo:
            raise ValueError(
                "Uno requires an official HF cache snapshot or uno_config.json"
            )
        base_id = RELEASED_ADAPTERS[repo]
        base_path = (
            path.parents[2]
            / ("models--" + base_id.replace("/", "--"))
            / "snapshots"
            / RELEASED_BASES[base_id]
        )
        adapter_path, block_size = path, 8
    if base_id not in RELEASED_BASES:
        raise ValueError(f"No released K2 Uno adapter for {base_id}")
    if type(block_size) is not int or not 1 <= block_size <= 64:
        raise ValueError("Uno block_size must be an integer in [1, 64]")
    if not base_path.is_dir():
        raise FileNotFoundError(
            f"Uno requires local base {base_id} at {base_path}; provide that cached "
            "revision or register an existing base with uno_config.json"
        )
    config = _read_object(base_path / "config.json")
    if (
        config.get("model_type") != "k2_horizon"
        or config.get("num_experts", 0)
        or config.get("attention_gate_func") is not None
        or config.get("model_file")
    ):
        raise ValueError("Uno requires a native dense, ungated K2 Horizon base")
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("Uno currently requires an unquantized BF16 base")
    identity = _cache_identity(base_path)
    if identity and identity[0] != base_id:
        raise ValueError(
            "Uno registered base identity does not match its cache repository"
        )
    adapter = _read_object(adapter_path / "adapter_config.json")
    if adapter.get("base_model_name_or_path") != base_id:
        raise ValueError("Uno adapter declares a different base_model_name_or_path")
    adapter_weights = adapter_path / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise FileNotFoundError(f"Missing Uno adapter weights: {adapter_weights}")
    shards = checkpoint_files(base_path)
    context = config.get("max_position_embeddings")
    if type(context) is not int or context <= 0:
        raise ValueError("Uno base requires a positive max_position_embeddings")
    context = min(context, RELEASED_CONTEXT_LIMITS[base_id])
    adapter_identity = _cache_identity(adapter_path)
    if adapter_identity and RELEASED_ADAPTERS.get(adapter_identity[0]) != base_id:
        raise ValueError("Uno registered adapter identity does not match its base")
    return UnoBundle(
        base_path,
        adapter_path,
        base_id,
        identity[1] if identity else None,
        adapter_identity[1] if adapter_identity else None,
        sum(shard.stat().st_size for shard in shards) + adapter_weights.stat().st_size,
        context,
        block_size,
        config,
    )
