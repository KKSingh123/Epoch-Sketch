"""
UnivMon (Liu et al., SIGCOMM 2016).

Hierarchical Count-Sketch with level sampling.

Level l (0-indexed) includes flow f if the top l bits of a per-level hash
of f are all zero — probability 2^{-l}.  Level 0 includes every flow.
Each level maintains an independent Count-Sketch (num_rows × row_width,
signed ±1 updates).

Update: For l = 0, 1, …, num_levels-1: insert into level l iff sampled;
        stop at the first level where the flow is NOT sampled (every deeper
        level is also excluded by the nesting property).

Query:  Find the deepest level l_best where f was sampled.  Recover the
        Count-Sketch median estimate at that level and scale by 2^{l_best}
        to compensate for the 2^{-l_best} sampling rate.
"""

import builtins
import numpy as np

from .hash_utils import _hash_pair, _to_bytes, derive_seeds


class UnivMonSketch:

    def __init__(self, num_levels: int = 8, num_rows: int = 2,
                 row_width: int = 4096, seed: int = 42):
        self._num_levels = num_levels
        self._num_rows   = num_rows
        self._row_width  = row_width
        self._row_idx    = list(range(num_rows))

        # Seed layout per level: 1 level-sampling seed + num_rows bucket seeds
        #                        + num_rows sign seeds = 2*num_rows+1 per level
        stride    = 2 * num_rows + 1
        all_seeds = derive_seeds(seed, num_levels * stride)

        self._level_seeds  = [all_seeds[l * stride]                        for l in range(num_levels)]
        self._sketch_seeds = [all_seeds[l * stride + 1          : l * stride + 1 + num_rows]         for l in range(num_levels)]
        self._sign_seeds   = [all_seeds[l * stride + 1 + num_rows : l * stride + 1 + 2 * num_rows]   for l in range(num_levels)]

        # Count-Sketch arrays (signed int32)
        self._levels = [np.zeros((num_rows, row_width), dtype=np.int32)
                        for _ in range(num_levels)]

    def _sign(self, raw: bytes, seed: int) -> int:
        """Return +-1 from hash parity, robust to unexpected None values."""
        h1, h2 = _hash_pair(raw, seed)
        probe = h2 if h2 is not None else h1
        bit = builtins.int(probe if probe is not None else 0) & 1
        return 1 if bit else -1

    # ------------------------------------------------------------------
    def _sampled_at(self, raw: bytes, level: int) -> bool:
        """True if flow is included at this level (top `level` bits == 0)."""
        if level == 0:
            return True
        h = builtins.int(_hash_pair(raw, self._level_seeds[level])[0])
        return (h >> (64 - level)) == 0

    # ------------------------------------------------------------------
    def update(self, item) -> None:
        raw = item if isinstance(item, bytes) else _to_bytes(item)
        for l in range(self._num_levels):
            if not self._sampled_at(raw, l):
                break   # deeper levels also exclude this flow
            sketch  = self._levels[l]
            s_seeds = self._sketch_seeds[l]
            g_seeds = self._sign_seeds[l]
            w       = self._row_width
            for i in self._row_idx:
                idx  = builtins.int(_hash_pair(raw, s_seeds[i])[0]) % w
                sign = self._sign(raw, g_seeds[i])
                sketch[i, idx] += sign

    # ------------------------------------------------------------------
    def query(self, item) -> int:
        raw = item if isinstance(item, bytes) else _to_bytes(item)

        # Find deepest level that sampled this flow
        best = 0
        for l in range(self._num_levels):
            if self._sampled_at(raw, l):
                best = l
            else:
                break

        sketch  = self._levels[best]
        s_seeds = self._sketch_seeds[best]
        g_seeds = self._sign_seeds[best]
        w       = self._row_width

        estimates = []
        for i in self._row_idx:
            idx  = builtins.int(_hash_pair(raw, s_seeds[i])[0]) % w
            sign = self._sign(raw, g_seeds[i])
            estimates.append(builtins.int(sketch[i, idx]) * sign)

        raw_est = builtins.int(np.median(estimates))
        return max(0, raw_est * (1 << best))

    @property
    def memory_bytes(self) -> int:
        return builtins.int(sum(s.nbytes for s in self._levels))
