# SPDX-License-Identifier: Apache-2.0
"""Load NVIDIA's ModelOpt mixed Nemotron Bolt checkpoint through mlx-vlm's public API."""

from __future__ import annotations

import copy
import json
import logging
import threading
from pathlib import Path

from . import is_supported_config, validate_layout

logger = logging.getLogger(__name__)
# Two Bolt loads must not stack their hooks (the inner one would restore the
# outer one's functions as the originals).
_HOOK_LOCK = threading.Lock()


def load(model_name: str | Path):
    """``(model, processor)`` for the unchanged NVIDIA checkpoint.

    mlx-vlm builds the model class from :mod:`.model` and loads the weights;
    the class's ``sanitize`` repacks IQ2_XXS / Q8_0 / NVFP4 bytes for the
    kernels. mlx-vlm's generic quantization handling is kept out of the way
    (it would rewrite the NVFP4 tensors and cannot place IQ2_XXS or Q8_0).
    """
    model_path = Path(model_name)
    config = json.loads((model_path / "config.json").read_text())
    if not is_supported_config(config):
        raise ValueError(f"{model_path} is not a ModelOpt mixed Nemotron Bolt checkpoint")
    validate_layout(config)

    from ..mlx_vlm_nemotron_omni_compat import apply_mlx_vlm_nemotron_omni_compat_patch

    # The Nemotron Omni processor, prompt format and model-type names.
    apply_mlx_vlm_nemotron_omni_compat_patch()

    import mlx_vlm.utils as vlm_utils

    from . import model as bolt_model

    owner = threading.get_ident()
    target = model_path.resolve()

    def this_load(path) -> bool:
        # Only this thread's load of this checkpoint: other loads (an oQ job,
        # another model) may run while the hooks are installed.
        if path is None or threading.get_ident() != owner:
            return False
        try:
            return Path(path).resolve() == target
        except (OSError, TypeError):
            return False

    def load_config(path, *args, **kwargs):
        loaded = originals["load_config"](path, *args, **kwargs)
        if not this_load(path):
            return loaded
        loaded = copy.deepcopy(loaded)
        loaded.pop("quantization", None)
        loaded.pop("quantization_config", None)
        return loaded

    def get_model_and_args(config, model_path=None):
        if this_load(model_path):
            return bolt_model, "nemotron_h_omni"
        return originals["get_model_and_args"](config, model_path)

    with _HOOK_LOCK:
        originals = {
            "load_config": vlm_utils.load_config,
            "get_model_and_args": vlm_utils.get_model_and_args,
        }
        vlm_utils.load_config = load_config
        vlm_utils.get_model_and_args = get_model_and_args
        try:
            logger.info("Loading %s with the Nemotron Bolt IQ2_XXS/Q8_0/NVFP4 kernels", model_path)
            return vlm_utils.load(model_path, trust_remote_code=False)
        finally:
            vlm_utils.get_model_and_args = originals["get_model_and_args"]
            vlm_utils.load_config = originals["load_config"]
