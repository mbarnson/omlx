# SPDX-License-Identifier: Apache-2.0
"""Block-drafter (DFlash) verify contract for mlx-vlm's Nemotron-H language model.

Covers Nemotron Omni checkpoints that mlx-vlm builds itself (Nemotron 3 Nano
Omni, Nemotron 3.5 Bolt in BF16 or MLX-converted form). NVIDIA's ModelOpt
mixed Bolt checkpoint runs on ``omlx.patches.nemotron_bolt.model``, whose
language model implements the contract itself and is not touched here.

The Lightning MTP verify path calls the language model with
``return_hidden=True``, ``n_confirmed`` and ``capture_layer_ids``. mlx-vlm's
exact verifier (``speculative_verify=True``) returns per-Mamba-layer state
histories in ``gdn_states``, which ``rollback_speculative_cache`` uses to
keep an accepted prefix. Captures come back in layer order; the drafter wants
them in the order it asked for, followed by the final hidden state.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply() -> bool:
    """Patch ``nemotron_h.language.LanguageModel.__call__``. Idempotent.

    Checks the live class rather than a module flag: the Nemotron Omni compat
    patch can replace the ``nemotron_h`` package after a first application.
    """
    try:
        from mlx_vlm.models.base import LanguageModelOutput
        from mlx_vlm.models.nemotron_h import language as nh_language
    except Exception as exc:  # noqa: BLE001
        logger.debug("mlx_vlm.nemotron_h not importable for DFlash runtime: %s", exc)
        return False

    cls = nh_language.LanguageModel
    if getattr(cls.__call__, "_omlx_dflash_contract", False):
        return True
    original_call = cls.__call__

    def call(self, inputs=None, cache=None, inputs_embeds=None, **kwargs):
        kwargs.pop("n_confirmed", None)  # rollback is after the fact
        return_hidden = kwargs.pop("return_hidden", False)
        requested = kwargs.pop("capture_layer_ids", None)
        if not return_hidden:
            drafter = getattr(self, "_omlx_drafter", None)
            scope = getattr(drafter, "scope_uids", None)
            if (
                scope
                and requested is None
                and inputs is not None
                and inputs.ndim == 2
                and inputs.shape == (len(scope), 1)
            ):
                # A plain decode step while a block drafter is attached: keep
                # the committed token in the drafter's context.
                out = original_call(
                    self,
                    inputs,
                    cache=cache,
                    inputs_embeds=inputs_embeds,
                    capture_layer_ids=list(drafter.target_layer_ids),
                    **kwargs,
                )
                drafter.observe(scope, out.hidden_states)
                return out
            if requested is not None:
                kwargs["capture_layer_ids"] = requested
            return original_call(
                self, inputs, cache=cache, inputs_embeds=inputs_embeds, **kwargs
            )

        requested = list(requested or [])
        layers = sorted(set(requested))
        out = original_call(
            self,
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            capture_layer_ids=layers,
            speculative_verify=True,
            return_hidden=True,
            **kwargs,
        )
        hidden = list(out.hidden_states or [])
        by_layer = dict(zip(layers, hidden[:-1]))
        return LanguageModelOutput(
            logits=out.logits,
            hidden_states=[by_layer[i] for i in requested] + hidden[-1:],
            gdn_states=out.gdn_states,
            shared_kv_states=out.shared_kv_states,
        )

    call._omlx_dflash_contract = True
    cls.__call__ = call
    # Its rollback needs uniform acceptance, so draft one row at a time.
    cls._omlx_drafter_single_row = True
    logger.info("Nemotron-H mlx-vlm DFlash verify contract applied")
    return True
