# SPDX-License-Identifier: Apache-2.0
"""Register mlx-vlm's Nemotron Omni (Nemotron 3.5 Super VL and Bolt) until oMLX's pin has it.

The vendored ``nemotron_h`` and ``nemotron_h_nano_omni`` packages are mlx-vlm
at the merge of Blaizzy/mlx-vlm#2528, which adds the Puzzle backbone
(per-layer ``block_configs``) used by Nemotron 3.5 Bolt, and NVIDIA's
``nemotron_h_omni`` / ``nemotron_h_puzzle`` model types. Every other file in
that tree is identical to oMLX's pinned mlx-vlm, so the overlay is exactly the
upstream change. When the installed mlx-vlm already maps ``nemotron_h_omni``,
this patch does nothing.
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PR_URL = "https://github.com/Blaizzy/mlx-vlm/pull/2528"
PR_MERGE_SHA = "63fc97955c4e3dbe4eaad7c838d4d81701b783af"
MODEL_TYPES = ("nemotron_h_omni", "nemotron_h_nano_omni", "nemotronh_nano_omni_reasoning_v3")
_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"
_PACKAGES = ("nemotron_h", "nemotron_h_nano_omni")
_APPLIED = False
# Engine loads and oQ jobs can both apply this on a cold process; module
# replacement must not interleave with another thread's imports.
_LOCK = threading.RLock()


_UPSTREAM: bool | None = None


def upstream_supported() -> bool:
    """True when the installed mlx-vlm already includes #2528.

    Decided before this patch adds its own remaps, then remembered.
    """
    global _UPSTREAM
    if _UPSTREAM is None:
        try:
            from mlx_vlm.utils import MODEL_REMAPPING
        except Exception:  # noqa: BLE001
            return False
        _UPSTREAM = MODEL_REMAPPING.get("nemotron_h_omni") == "nemotron_h_nano_omni"
    return _UPSTREAM


def _append_package_path(package: Any, path: Path) -> None:
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    path_str = str(path)
    if path_str not in package_path:
        package_path.insert(0, path_str)


def apply_mlx_vlm_nemotron_omni_compat_patch() -> bool:
    """Expose the vendored Nemotron Omni packages through ``mlx_vlm.models``."""
    with _LOCK:
        return _apply()


def _apply() -> bool:
    global _APPLIED
    if _APPLIED:
        return False
    if upstream_supported():
        _APPLIED = True
        logger.info("mlx-vlm has native Nemotron Omni support; vendored copy unused")
        return True
    try:
        import mlx_vlm
        import mlx_vlm.models

        _append_package_path(mlx_vlm, _VENDOR_MLX_VLM)
        _append_package_path(mlx_vlm.models, _VENDOR_MLX_VLM / "models")
        # Discovery may have imported the pinned packages; reload from vendor.
        for name in list(sys.modules):
            if any(
                name == f"mlx_vlm.models.{p}" or name.startswith(f"mlx_vlm.models.{p}.")
                for p in _PACKAGES
            ):
                del sys.modules[name]
        for package in _PACKAGES:
            module = importlib.import_module(f"mlx_vlm.models.{package}.language")
            if not str(Path(module.__file__).resolve()).startswith(
                str(_VENDOR_MLX_VLM.resolve())
            ):
                logger.error(
                    "Nemotron Omni compat patch did not take effect: %s resolves to %s",
                    package,
                    module.__file__,
                )
                return False
        importlib.import_module("mlx_vlm.models.nemotron_h_nano_omni")

        from mlx_vlm import utils
        from mlx_vlm.prompt_utils import MODEL_CONFIG, MessageFormat

        utils.MODEL_REMAPPING.setdefault("nemotron_h_omni", "nemotron_h_nano_omni")
        utils.MODEL_REMAPPING.setdefault("nemotron_h_puzzle", "nemotron_h")
        MODEL_CONFIG.setdefault("nemotron_h_omni", MessageFormat.LIST_WITH_IMAGE_TYPE)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Nemotron Omni mlx-vlm registration failed: %s", exc)
        return False
    _APPLIED = True
    logger.info("Nemotron Omni mlx-vlm compatibility patch applied (%s)", PR_URL)
    return True


def is_applied() -> bool:
    return _APPLIED


__all__ = [
    "MODEL_TYPES",
    "PR_MERGE_SHA",
    "PR_URL",
    "apply_mlx_vlm_nemotron_omni_compat_patch",
    "is_applied",
    "upstream_supported",
]
