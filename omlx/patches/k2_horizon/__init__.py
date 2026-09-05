# SPDX-License-Identifier: Apache-2.0
"""Register K2 Horizon until mlx-lm provides native support."""

import importlib
import sys

_APPLIED = False


def apply_k2_horizon_patch() -> bool:
    global _APPLIED
    if _APPLIED:
        return False
    for target, source in (
        ("mlx_lm.models.k2_horizon", ".k2_horizon_model"),
        ("mlx_lm.tool_parsers.k2_horizon", ".tool_parser"),
    ):
        try:
            importlib.import_module(target)
        except ModuleNotFoundError as error:
            if error.name != target:
                raise
            module = importlib.import_module(source, __name__)
            sys.modules[target] = module
            package, name = target.rsplit(".", 1)
            setattr(importlib.import_module(package), name, module)
    from .checkpoint import apply_checkpoint_patch

    apply_checkpoint_patch()
    _APPLIED = True
    return True
