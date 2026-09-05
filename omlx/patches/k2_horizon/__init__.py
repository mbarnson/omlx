# SPDX-License-Identifier: Apache-2.0
"""Register K2 Horizon MoVA model and tool protocol support for pinned mlx-lm."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_MODEL_MODULE = "mlx_lm.models.k2_horizon"
_PARSER_MODULE = "mlx_lm.tool_parsers.k2_horizon"
_APPLIED = False


def _register_module(qualname: str, filename: str, package: str) -> None:
    if qualname in sys.modules:
        return

    file_path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(qualname, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {qualname} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    sys.modules[qualname] = module
    try:
        spec.loader.exec_module(module)
        parent_module = importlib.import_module(package)
        setattr(parent_module, qualname.rsplit(".", 1)[1], module)
    except BaseException:
        if sys.modules.get(qualname) is module:
            sys.modules.pop(qualname)
        raise

    logger.info("Registered %s from %s", qualname, filename)


def _register_if_missing(qualname: str, filename: str, package: str) -> bool:
    """Register the vendored module only when exactly that upstream module is absent."""
    try:
        importlib.import_module(qualname)
    except ModuleNotFoundError as error:
        if error.name != qualname:
            raise
        _register_module(qualname, filename, package)
        return True
    return False


def apply_k2_horizon_patch() -> bool:
    """Register K2 Horizon support before mlx-lm resolves model assets."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        importlib.import_module("mlx_lm")
    except ModuleNotFoundError:
        logger.debug("mlx_lm not importable - k2_horizon patch skipped")
        return False

    model_applied = _register_if_missing(
        _MODEL_MODULE, "k2_horizon_model.py", "mlx_lm.models"
    )
    parser_applied = _register_if_missing(
        _PARSER_MODULE, "tool_parser.py", "mlx_lm.tool_parsers"
    )

    _APPLIED = True
    if model_applied or parser_applied:
        logger.info(
            "K2 Horizon mlx-lm patch applied (model=%s, parser=%s)",
            model_applied,
            parser_applied,
        )
        return True

    logger.debug("mlx_lm k2_horizon model and parser already available upstream")
    return False


def is_applied() -> bool:
    return _APPLIED


__all__ = [
    "apply_k2_horizon_patch",
    "is_applied",
]
