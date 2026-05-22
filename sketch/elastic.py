"""
ElasticSketch (Yang et al., SIGCOMM 2018).

Two-level design:
  Heavy part : open-addressed hash table (one slot per hash bucket).
               Each slot owns one flow key + a vote counter.
               Elephants accumulate large vote counts and stay here.
  Light part : Count-Min Sketch with conservative update for mice and
               flows evicted from the heavy part.

On a collision in the heavy part the incoming packet "votes against" the
existing key (decrement). When the vote counter hits zero the cell is
reclaimed by the incoming flow. The colliding packet is also forwarded to
the light CMS so no count is silently dropped.

Query  =  heavy_count (if slot key matches) + light CMS min-estimate.
"""

import numpy as np

from .hash_utils import _hash_pair, _to_bytes, derive_seeds

_INT_CAST = int
_MIN = min


class ElasticSketch:

    def __init__(self, num_rows: int = 4, row_width: int = 4096,
                 seed: int = 42, num_heavy: int | None = None):
        # Backward-compatible aliases for older method paths that access
        # caster helpers as instance attributes.
        self._INT_CAST = _INT_CAST
        self._MIN = _MIN

        self._num_rows  = num_rows
        self._row_width = row_width
        self._num_heavy = num_heavy if num_heavy is not None else row_width

        seeds = derive_seeds(seed, num_rows + 1)
        self._heavy_seed  = seeds[num_rows]
        self._light_seeds = seeds[:num_rows]

        # Heavy part
        self._heavy_keys = [None] * self._num_heavy   # bytes key | None
        self._heavy_cnt  = np.zeros(self._num_heavy, dtype=np.uint32)

        # Light part — CMS with conservative update
        self._light   = np.zeros((num_rows, row_width), dtype=np.uint32)
        self._row_idx = list(range(num_rows))

    # ------------------------------------------------------------------
    def update(self, item) -> None:
        raw   = item if isinstance(item, bytes) else _to_bytes(item)
        h_idx = _INT_CAST(_hash_pair(raw, self._heavy_seed)[0]) % self._num_heavy

        if self._heavy_keys[h_idx] is None:
            # Empty slot — claim it
            self._heavy_keys[h_idx] = raw
            self._heavy_cnt[h_idx]  = 1

        elif self._heavy_keys[h_idx] == raw:
            # Same key — simply increment
            self._heavy_cnt[h_idx] += 1

        else:
            # Collision — decrement votes of existing occupant
            c = _INT_CAST(self._heavy_cnt[h_idx])
            if c > 1:
                self._heavy_cnt[h_idx] = c - 1
            else:
                # Slot neutralised — incoming flow takes over
                self._heavy_keys[h_idx] = raw
                self._heavy_cnt[h_idx]  = 1
            # Current packet also counted in light sketch (no data dropped)
            self._light_update(raw)

    def _light_update(self, raw: bytes) -> None:
        idxs = [_INT_CAST(_hash_pair(raw, self._light_seeds[i])[0]) % self._row_width
                for i in self._row_idx]
        cur  = [_INT_CAST(self._light[i, idxs[i]]) for i in self._row_idx]
        mn   = _MIN(cur)
        for i in self._row_idx:
            if cur[i] == mn:
                self._light[i, idxs[i]] += 1

    # ------------------------------------------------------------------
    def query(self, item) -> int:
        raw   = item if isinstance(item, bytes) else _to_bytes(item)
        h_idx = _INT_CAST(_hash_pair(raw, self._heavy_seed)[0]) % self._num_heavy

        heavy_est = (_INT_CAST(self._heavy_cnt[h_idx])
                     if self._heavy_keys[h_idx] == raw else 0)
        light_est = _MIN(
            _INT_CAST(
                self._light[
                    i,
                    _INT_CAST(_hash_pair(raw, self._light_seeds[i])[0]) % self._row_width,
                ]
            )
            for i in self._row_idx
        )
        return heavy_est + light_est

    @property
    def memory_bytes(self) -> int:
        return _INT_CAST(self._heavy_cnt.nbytes + self._light.nbytes)
