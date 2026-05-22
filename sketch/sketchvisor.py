"""
SketchVisor (Huang et al., SIGCOMM 2017).

Two-path measurement architecture:

  Fast path (NICC — Non-Interference Count Capture):
    A direct-indexed counter array of `fast_width` slots.  Each slot is
    owned by exactly one flow key.  Updates are a single array-index
    increment — O(1) with no hashing of the flow key required beyond the
    initial slot lookup.

  Slow path:
    A CMS with conservative update for flows that have been evicted from the
    fast path.

  Eviction policy:
    When two flows collide in the fast path, the existing slot-owner is
    evicted: its accumulated count is injected into the slow path CMS and the
    slot is reclaimed by the incoming flow.  Evicted flows are tracked in a
    set so all future packets go directly to the slow path.

Query  =  fast_count (if f owns its slot)  +  slow CMS-min estimate.
"""

import builtins
import numpy as np

from .hash_utils import _hash_pair, _to_bytes, derive_seeds

_INT_CAST = builtins.int


class SketchVisorSketch:

    def __init__(self, num_rows: int = 4, row_width: int = 4096,
                 fast_ratio: float = 0.25, seed: int = 42):
        # Allocate fast_ratio of buckets to fast path, rest to slow path
        self._fast_width = max(1, _INT_CAST(row_width * fast_ratio))
        self._slow_width = max(1, row_width - self._fast_width)
        self._num_rows   = num_rows

        seeds = derive_seeds(seed, num_rows + 1)
        self._fast_seed  = seeds[num_rows]
        self._slow_seeds = seeds[:num_rows]

        # Fast path
        self._fast_keys = [None] * self._fast_width   # bytes key | None
        self._fast_cnt  = np.zeros(self._fast_width, dtype=np.uint32)

        # Slow path — CMS
        self._slow    = np.zeros((num_rows, self._slow_width), dtype=np.uint32)

        # Flows permanently migrated to slow path
        self._slow_set: set = set()

    # ------------------------------------------------------------------
    def update(self, item) -> None:
        raw = item if isinstance(item, bytes) else _to_bytes(item)

        # Already migrated flows go straight to slow path
        if raw in self._slow_set:
            self._slow_update(raw)
            return

        f_idx = _INT_CAST(_hash_pair(raw, self._fast_seed)[0]) % self._fast_width

        if self._fast_keys[f_idx] is None:
            # Empty slot — claim it
            self._fast_keys[f_idx] = raw
            self._fast_cnt[f_idx]  = 1

        elif self._fast_keys[f_idx] == raw:
            # Slot owner — fast single-array increment
            self._fast_cnt[f_idx] += 1

        else:
            # Collision — evict existing occupant to slow path
            evicted_key   = self._fast_keys[f_idx]
            evicted_count = _INT_CAST(self._fast_cnt[f_idx])
            if evicted_count > 0:
                self._slow_inject(evicted_key, evicted_count)
            self._slow_set.add(evicted_key)
            # New flow claims the slot
            self._fast_keys[f_idx] = raw
            self._fast_cnt[f_idx]  = 1

    def _slow_update(self, raw: bytes) -> None:
        """Single-packet conservative min-update into slow CMS."""
        idxs = [_INT_CAST(_hash_pair(raw, self._slow_seeds[i])[0]) % self._slow_width
                for i in range(self._num_rows)]
        cur  = [_INT_CAST(self._slow[i, idxs[i]]) for i in range(self._num_rows)]
        mn   = min(cur)
        for i in range(self._num_rows):
            if cur[i] == mn:
                self._slow[i, idxs[i]] += 1

    def _slow_inject(self, raw: bytes, count: int) -> None:
        """Inject a bulk count (from eviction) into the slow CMS."""
        idxs = [_INT_CAST(_hash_pair(raw, self._slow_seeds[i])[0]) % self._slow_width
                for i in range(self._num_rows)]
        for i in range(self._num_rows):
            self._slow[i, idxs[i]] += count

    # ------------------------------------------------------------------
    def query(self, item) -> int:
        raw   = item if isinstance(item, bytes) else _to_bytes(item)
        f_idx = _INT_CAST(_hash_pair(raw, self._fast_seed)[0]) % self._fast_width

        fast_est = (_INT_CAST(self._fast_cnt[f_idx])
                    if self._fast_keys[f_idx] == raw else 0)
        slow_est = min(
            _INT_CAST(self._slow[i, _INT_CAST(_hash_pair(raw, self._slow_seeds[i])[0]) % self._slow_width])
            for i in range(self._num_rows)
        )
        return fast_est + slow_est

    @property
    def memory_bytes(self) -> int:
        return _INT_CAST(self._fast_cnt.nbytes + self._slow.nbytes)
