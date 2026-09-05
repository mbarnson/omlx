# SPDX-License-Identifier: Apache-2.0
"""Memory-enforcer integration for engines without a Scheduler."""

from ..memory_monitor import MemoryMonitor, raise_if_prefill_exceeds
from ..utils.proc_memory import get_phys_footprint


class SerialPrefillGuard:
    """Apply scheduler memory limits to serial Uno and DFlash prefill.

    ProcessMemoryEnforcer updates the same watermarks used by Scheduler.
    """

    def __init__(self, memory_monitor: MemoryMonitor, prefill_step_size: int):
        self.memory_monitor = memory_monitor
        self._prefill_step_size = prefill_step_size
        self._last_mlx_active_memory_bytes: int = 0
        # Written by ProcessMemoryEnforcer._propagate_memory_limit each tick.
        self._prefill_memory_guard: bool = False
        self._memory_hard_limit_bytes: int = 0
        self._memory_hot_cache_used_bytes: int = 0
        # Component breakdown behind the ceiling above, so a rejection can
        # name the binding constraint instead of generic tier advice.
        self._memory_static_ceiling_bytes: int = 0
        self._memory_dynamic_ceiling_bytes: int = 0
        self._memory_metal_cap_bytes: int = 0
        self._memory_guard_tier: str = ""

    def record_mlx_active_memory(self, active_bytes: int) -> None:
        self._last_mlx_active_memory_bytes = max(0, int(active_bytes))

    def _current_usage_bytes(self) -> int:
        # The enforcer already subtracts a hot-cache reservation from the
        # ``_memory_hard_limit_bytes`` it propagates here, so serialized
        # hot-cache CPU bytes must not ALSO be counted in usage via
        # phys_footprint — that charges them twice and over-rejects by the
        # hot-cache size. Mirrors ``Scheduler._current_usage_bytes``.
        phys = max(
            0, get_phys_footprint() - max(0, int(self._memory_hot_cache_used_bytes))
        )
        return max(self._last_mlx_active_memory_bytes, phys)

    def preflight_or_raise(
        self,
        *,
        num_prompt_tokens: int,
        request_id: str | None = None,
        extra_bytes: int = 0,
    ) -> None:
        # Deliberately no cached_tokens parameter: a DFlash prefix-cache hit
        # reconstructs the matched KV into active memory, so the full prompt
        # must always be charged (see DFlashEngine.preflight_chat).
        raise_if_prefill_exceeds(
            self.memory_monitor,
            prefill_memory_guard=self._prefill_memory_guard,
            hard_limit_bytes=self._memory_hard_limit_bytes,
            current_usage_bytes=self._current_usage_bytes() + max(0, extra_bytes),
            prefill_step_size=self._prefill_step_size,
            num_prompt_tokens=num_prompt_tokens,
            request_id=request_id,
            static_ceiling_bytes=self._memory_static_ceiling_bytes,
            dynamic_ceiling_bytes=self._memory_dynamic_ceiling_bytes,
            metal_cap_bytes=self._memory_metal_cap_bytes,
            memory_guard_tier=self._memory_guard_tier,
        )
