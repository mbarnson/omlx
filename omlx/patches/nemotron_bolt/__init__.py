# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Nemotron 3.5 Bolt in its published ModelOpt mixed-precision form.

``nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`` stores the routed and shared
experts as IQ2_XXS, attention, latent projections and ``lm_head`` as Q8_0, and
the Mamba projections as NVFP4 (``quant_method: modelopt``,
``quant_algo: MIXED_PRECISION``). mlx-vlm has no IQ2_XXS or Q8_0 support, so
oMLX runs these tensors with the Metal kernels in :mod:`.model` and loads the
checkpoint unchanged. BF16 and MLX-converted Bolt checkpoints load through
mlx-vlm's own Nemotron Omni code (see ``mlx_vlm_nemotron_omni_compat``).
"""

from __future__ import annotations

from typing import Any

MODEL_TYPES = ("nemotron_h_omni", "nemotron_h_nano_omni")
_LAYER_FORMATS = {"IQ2_XXS", "Q8_0", "W4A16_NVFP4"}


def _text_config(config: dict) -> dict:
    return config.get("llm_config") or config.get("text_config") or {}


def is_supported_config(config: dict[str, Any]) -> bool:
    """True for a Nemotron Puzzle (Bolt) checkpoint in ModelOpt IQ2_XXS/Q8_0/NVFP4 form."""
    if config.get("model_type") not in MODEL_TYPES or config.get("model_file"):
        return False
    quant = config.get("quantization_config") or {}
    if quant.get("quant_method") != "modelopt" or quant.get("quant_algo") != "MIXED_PRECISION":
        return False
    layers = quant.get("quantized_layers") or {}
    formats = {spec.get("quant_algo") for spec in layers.values() if isinstance(spec, dict)}
    if "IQ2_XXS" not in formats or not formats <= _LAYER_FORMATS:
        return False
    return bool(_text_config(config).get("block_configs"))


def expected_format(name: str, block_type: str) -> str | None:
    """The storage format this module's layout assumes for a quantized layer."""
    leaf = name.rsplit(".", 1)[-1]
    if name.endswith("lm_head"):
        return "Q8_0"
    if block_type == "mamba":
        return "W4A16_NVFP4" if leaf in ("in_proj", "out_proj") else None
    if block_type == "attention":
        return "Q8_0" if leaf in ("q_proj", "k_proj", "v_proj", "o_proj") else None
    if block_type == "moe":
        if leaf in ("fc1_latent_proj", "fc2_latent_proj"):
            return "Q8_0"
        if leaf in ("up_proj", "down_proj"):
            return "IQ2_XXS"
    return None


def validate_layout(config: dict[str, Any]) -> None:
    """Refuse a checkpoint whose per-layer formats differ from the kernels' layout."""
    blocks = [b["block_type"] for b in _text_config(config)["block_configs"]]
    layers = config["quantization_config"]["quantized_layers"]
    mismatched = []
    for name, spec in layers.items():
        algo = spec.get("quant_algo")
        if name == "language_model.lm_head":
            block = None
        else:
            parts = name.split(".")
            try:
                index = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                mismatched.append((name, algo, None))
                continue
            block = blocks[index] if index < len(blocks) else None
        want = expected_format(name, block or "")
        if want != algo:
            mismatched.append((name, algo, want))
    if mismatched:
        raise ValueError(
            "Unsupported Nemotron Bolt quantization layout "
            f"(layer, stored, expected): {mismatched[:5]}"
        )


def checkpoint_files(path) -> list:
    """Every shard named by the checkpoint index, all present."""
    import json
    from pathlib import Path

    path = Path(path)
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid weight_map in {path}")
    shards = [path / name for name in sorted(set(weight_map.values()))]
    missing = [s.name for s in shards if not s.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Nemotron Bolt shards in {path}: {missing[:3]}")
    return shards


def load(model_name):
    """Load the published checkpoint with oMLX's Nemotron Bolt kernels."""
    from .loader import load as _load

    return _load(model_name)


__all__ = ["MODEL_TYPES", "checkpoint_files", "expected_format", "is_supported_config", "load", "validate_layout"]
