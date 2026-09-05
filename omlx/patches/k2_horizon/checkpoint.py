# SPDX-License-Identifier: Apache-2.0
"""Resolve K2 checkpoint manifests without changing the Hugging Face cache."""

from __future__ import annotations

import functools
import inspect
import json
import struct
from pathlib import Path
from tempfile import TemporaryDirectory


def checkpoint_files(path: Path) -> list[Path]:
    """Validate shard membership and tensor ownership before loading weights."""
    index_path = path / "model.safetensors.index.json"
    weight_map = None
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid K2 weight_map: {index_path}")
        if any(
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
            or name.startswith("adapter")
            for name in weight_map.values()
        ):
            raise ValueError(f"Invalid K2 shard filename: {index_path}")
        names = sorted(set(weight_map.values()))
        shards = [path / name for name in names]
    else:
        shards = sorted(path.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No K2 base safetensors in {path}")

    seen = set()
    for shard in shards:
        if not shard.is_file():
            raise FileNotFoundError(f"Missing K2 checkpoint shard: {shard}")
        with shard.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated K2 safetensors header: {shard}")
            length = struct.unpack("<Q", prefix)[0]
            if not 0 < length <= 64 * 1024 * 1024:
                raise ValueError(f"Invalid K2 safetensors header size: {shard}")
            header = json.loads(handle.read(length))
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            if key in seen:
                raise ValueError(f"Duplicate K2 tensor: {key}")
            if weight_map is not None and weight_map.get(key) != shard.name:
                raise ValueError(f"K2 tensor not owned by manifest shard: {key}")
            start, end = entry["data_offsets"]
            if not 0 <= start <= end <= shard.stat().st_size - 8 - length:
                raise ValueError(f"Truncated K2 tensor data: {key} in {shard}")
            seen.add(key)
    if weight_map is not None and seen != set(weight_map):
        raise ValueError(
            f"Missing K2 tensors from manifest: {sorted(set(weight_map) - seen)[:3]}"
        )
    return shards


def apply_checkpoint_patch() -> None:
    """Give the pinned mlx-lm loader a manifest-selected, normalized shard view.

    Only symlinks and the remapped index are temporary. The original loader
    still owns model construction, quantization, strict loading, and evaluation.
    Its mmap arrays retain the opened data after the temporary view is removed.
    Tokenizer loading continues against the caller's original model directory.
    """
    from mlx_lm import utils

    _patch_tokenizer(utils)
    original = utils.load_model
    if getattr(original, "_omlx_k2_checkpoint", False):
        return
    signature = inspect.signature(original)

    @functools.wraps(original)
    def load_model(model_path, *args, **kwargs):
        path = Path(model_path)
        config = utils.load_config(path)
        bound = signature.bind_partial(model_path, *args, **kwargs)
        config.update(bound.arguments.get("model_config") or {})
        if config.get("model_type") != "k2_horizon" or config.get("model_file"):
            return original(model_path, *args, **kwargs)
        shards = checkpoint_files(path)
        with TemporaryDirectory(prefix="omlx-k2-shards-") as directory:
            view = Path(directory)
            for source in path.iterdir():
                if (
                    source.suffix != ".safetensors"
                    and source.name != "model.safetensors.index.json"
                ):
                    (view / source.name).symlink_to(source.resolve())
            remapping = {}
            for i, shard in enumerate(shards):
                name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
                (view / name).symlink_to(shard.resolve())
                remapping[shard.name] = name
            index_path = path / "model.safetensors.index.json"
            if index_path.exists():
                index = json.loads(index_path.read_text())
                index["weight_map"] = {
                    k: remapping[v] for k, v in index["weight_map"].items()
                }
                (view / index_path.name).write_text(json.dumps(index))
            return original(view, *args, **kwargs)

    load_model._omlx_k2_checkpoint = True
    utils.load_model = load_model


def _patch_tokenizer(utils) -> None:
    original = utils.load_tokenizer
    if getattr(original, "_omlx_k2_tokens", False):
        return

    @functools.wraps(original)
    def load_tokenizer(model_path, *args, **kwargs):
        tokenizer = original(model_path, *args, **kwargs)
        config_path = Path(model_path) / "config.json"
        if (
            config_path.is_file()
            and json.loads(config_path.read_text()).get("model_type") == "k2_horizon"
        ):
            tokenizer.add_eos_token("<|ifm|im_end|>")
            generation_path = Path(model_path) / "generation_config.json"
            if generation_path.is_file():
                eos = json.loads(generation_path.read_text()).get("eos_token_id", [])
                tokenizer.eos_token_ids.update([eos] if isinstance(eos, int) else eos)
        return tokenizer

    load_tokenizer._omlx_k2_tokens = True
    utils.load_tokenizer = load_tokenizer
