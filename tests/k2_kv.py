# SPDX-License-Identifier: Apache-2.0
"""KV cache helpers for the K2 tests that work across mlx-lm versions.

Newer mlx-lm returns the whole preallocated buffer (rows past ``offset`` are stale) plus the offset from
``KVCache.state`` and drops ``meta_state``; older versions return only the committed rows."""

import mlx.core as mx


def committed_kv(cache):
    """The committed (keys, values) rows of one layer's cache."""
    return cache.keys[..., : cache.offset, :], cache.values[..., : cache.offset, :]


def clone_cache(caches):
    """Independent copies of a prompt cache, holding only the committed rows."""
    clones = []
    for c in caches:
        keys, values = committed_kv(c)
        clone = type(c)()
        clone.update_and_fetch(mx.array(keys), mx.array(values))
        clones.append(clone)
    mx.eval([committed_kv(c) for c in clones])
    return clones
