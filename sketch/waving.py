"""
WavingSketch (Li et al., KDD 2020).

Each cell stores:
  count : signed int32  — net "wave" amplitude for the owning flow.
  fp    : uint32        — fingerprint (short hash) of the owning flow.

Update rule for row i, flow f:
  idx  = h_i(f) % row_width
  sign = +1 if g_i(f) is odd else -1   (pairwise-independent sign function)
  fp_f = fingerprint(f)

  If cells_fp[i,idx] == fp_f  OR  cells_count[i,idx] == 0:
      Reinforce: cells_fp[i,idx] = fp_f;  cells_count[i,idx] += sign
  Else (contested cell):
      Wave: cells_count[i,idx] -= sign   (counter washed toward zero)
      If cells_count[i,idx] == 0:
          New owner: cells_fp[i,idx] = fp_f;  cells_count[i,idx] = sign

Query rule: for each row i, if fp matches → contribution = count × sign;
else → 0. Return max(0, median(contributions)).
Heavy flows own their cells with high probability and accumulate accurate
signed counts; the median removes residual interference from mice.
"""

import builtins
import numpy as np

from .hash_utils import _hash_pair, _to_bytes, derive_seeds

_INT_CAST = builtins.int


class WavingSketch:

    def __init__(self, num_rows: int = 4, row_width: int = 4096, seed: int = 42):
        self._num_rows  = num_rows
        self._row_width = row_width

        seeds = derive_seeds(seed, num_rows * 2 + 1)
        self._row_seeds  = seeds[:num_rows]             # bucket index seeds
        self._sign_seeds = seeds[num_rows: num_rows * 2]  # sign seeds
        self._fp_seed    = seeds[num_rows * 2]          # fingerprint seed

        self._counts = np.zeros((num_rows, row_width), dtype=np.int32)
        self._fps    = np.zeros((num_rows, row_width), dtype=np.uint32)

        self._row_idx = list(range(num_rows))

    # ------------------------------------------------------------------
    def update(self, item) -> None:
        raw  = item if isinstance(item, bytes) else _to_bytes(item)
        fp_f = _INT_CAST(_hash_pair(raw, self._fp_seed)[0]) & 0xFFFF_FFFF

        for i in self._row_idx:
            idx   = _INT_CAST(_hash_pair(raw, self._row_seeds[i])[0]) % self._row_width
            sign  = 1 if (_hash_pair(raw, self._sign_seeds[i])[1] & 1) else -1
            cur_c = _INT_CAST(self._counts[i, idx])
            cur_f = _INT_CAST(self._fps[i, idx])

            if cur_f == fp_f or cur_c == 0:
                # Cell free or owned by this flow — reinforce
                self._fps[i, idx]    = fp_f
                self._counts[i, idx] = cur_c + sign
            else:
                # Contested — wave
                new_c = cur_c - sign
                self._counts[i, idx] = new_c
                if new_c == 0:
                    # Cell neutralised — current flow takes ownership
                    self._fps[i, idx]    = fp_f
                    self._counts[i, idx] = sign

    # ------------------------------------------------------------------
    def query(self, item) -> int:
        raw  = item if isinstance(item, bytes) else _to_bytes(item)
        fp_f = _INT_CAST(_hash_pair(raw, self._fp_seed)[0]) & 0xFFFF_FFFF

        estimates = []
        for i in self._row_idx:
            idx  = _INT_CAST(_hash_pair(raw, self._row_seeds[i])[0]) % self._row_width
            sign = 1 if (_hash_pair(raw, self._sign_seeds[i])[1] & 1) else -1
            if _INT_CAST(self._fps[i, idx]) == fp_f:
                estimates.append(_INT_CAST(self._counts[i, idx]) * sign)
            else:
                estimates.append(0)

        return max(0, _INT_CAST(np.median(estimates)))

    @property
    def memory_bytes(self) -> int:
        return _INT_CAST(self._counts.nbytes + self._fps.nbytes)
