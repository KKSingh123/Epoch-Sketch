"""
sketch/stablesketch.py
======================
Stable-Sketch: A Versatile Sketch for Accurate, Fast, Web-Scale Data
Stream Processing.

Reference:
    Weihe Li, Paul Patras. "Stable-Sketch: A Versatile Sketch for
    Accurate, Fast, Web-Scale Data Stream Processing." In Proceedings
    of the ACM Web Conference 2024 (WWW '24), Best Student Paper Award.
    DOI: 10.1145/3589334.3645581
    GitHub: https://github.com/Mobile-Intelligence-Lab/Stable-Sketch

Algorithm Summary
-----------------
Each bucket stores three fields: a 32-bit key fingerprint, an integer
count, and a bucket-stability counter.

Update(item):
  For each row i, compute bucket index b = H_i(item) % width:
    - If bucket is empty  → insert item (fp, count=1, stability++)
    - If fp matches item  → count++, stability++
  If no row had a match:
    Select the bucket with the minimum count across all rows, and
    apply stochastic eviction:
      k ~ Uniform[1, stability*count + 1]
      if k > stability*count (prob = 1/(stability*count+1)):
          count -= 1
          if count <= 0: replace fp, count=1, stability = max(0, s-1)

Query(item):
  Sum counts from rows where the fingerprint matches.  In practice only
  one row will match (the row where the item was first placed), so this
  equals the stored count for the item.

Memory:  num_rows * row_width * 12  bytes
         (4B fingerprint + 4B count + 4B stability per cell)
"""

from __future__ import annotations

import numpy as np

from .hash_utils import _hash_pair, _normalize_seed, _to_bytes, derive_seeds


def _looks_like_stable_sketch(obj) -> bool:
    """Duck-typed receiver check so class reloads don't break method dispatch."""
    required = (
        "_fingerprint",
        "_bucket",
        "_fps",
        "_counts",
        "_stab",
        "_rng",
        "num_rows",
        "row_width",
    )
    return all(hasattr(obj, name) for name in required)


def _coerce_receiver(receiver, item, method_name: str):
    if _looks_like_stable_sketch(receiver):
        return receiver, item
    if _looks_like_stable_sketch(item):
        return item, receiver
    raise TypeError(f"StableSketch.{method_name} called with invalid receiver")


class StableSketch:
    """
    Stable-Sketch (Li & Patras, WWW 2024).

    Parameters
    ----------
    num_rows : int
        Sketch depth (number of independent hash rows).
    row_width : int
        Number of cells per row.
    seed : int
        Master RNG seed.
    """

    def __init__(
        self,
        num_rows: int = 4,
        row_width: int = 1365,
        seed: int = 42,
    ) -> None:
        base_seed = _normalize_seed(seed)
        self.num_rows  = num_rows
        self.row_width = row_width
        self.seed      = base_seed
        self._rng = np.random.default_rng(base_seed)

        # Per-row hash seeds for bucket-index computation
        self._seeds = derive_seeds(base_seed, num_rows)

        # A separate seed for stable fingerprints (global)
        self._fp_seed = derive_seeds(base_seed + 0xDEAD, 1)[0]

        # Core arrays: shape (num_rows, row_width)
        self._fps    = np.zeros((num_rows, row_width), dtype=np.uint32)   # fingerprints
        self._counts = np.zeros((num_rows, row_width), dtype=np.int32)    # counts
        self._stab   = np.zeros((num_rows, row_width), dtype=np.int32)    # stability

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------
    @property
    def memory_bytes(self) -> int:
        return int(self._fps.nbytes + self._counts.nbytes + self._stab.nbytes)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fingerprint(self, item_bytes: bytes) -> int:
        """32-bit fingerprint for an item (non-zero for non-empty check)."""
        h1, h2 = _hash_pair(item_bytes, self._fp_seed)
        fp = (h1 ^ h2) & 0xFFFFFFFF
        return fp if fp != 0 else 1   # reserve 0 for «empty»

    def _bucket(self, item_bytes: bytes, row: int) -> int:
        h1, _ = _hash_pair(item_bytes, self._seeds[row])
        return int(h1 % self.row_width)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def update(self, item, _to_bytes_fn=_to_bytes) -> None:
        self, item = _coerce_receiver(self, item, "update")
        raw  = _to_bytes_fn(item)
        fp   = self._fingerprint(raw)

        min_count = 2**30
        min_r = min_b = -1

        for i in range(self.num_rows):
            b = self._bucket(raw, i)
            cell_fp = int(self._fps[i, b])
            cell_c  = int(self._counts[i, b])

            if cell_fp == 0 and cell_c == 0:
                # Empty bucket — insert here
                self._fps[i, b]    = fp
                self._counts[i, b] = 1
                self._stab[i, b]  += 1
                return

            if cell_fp == fp:
                # Matching fingerprint — update
                self._counts[i, b] += 1
                self._stab[i, b]   += 1
                return

            # Track min-count bucket for potential eviction
            if cell_c < min_count:
                min_count = cell_c
                min_r, min_b = i, b

        # No match found — stochastic eviction at min-count bucket
        if min_r >= 0:
            s = int(self._stab[min_r, min_b])
            c = int(self._counts[min_r, min_b])
            # k uniform in [1, s*c + 1]; evict if k > s*c (prob 1/(s*c+1))
            sc = max(0, int(s) * int(c))
            k = self._rng.integers(1, sc + 1, endpoint=True)
            if k > sc:
                self._counts[min_r, min_b] -= 1
                if self._counts[min_r, min_b] <= 0:
                    self._fps[min_r, min_b]    = fp
                    self._counts[min_r, min_b] = 1
                    self._stab[min_r, min_b]   = max(0, s - 1)

    def query(self, item, _to_bytes_fn=_to_bytes) -> int:
        self, item = _coerce_receiver(self, item, "query")
        raw = _to_bytes_fn(item)
        fp  = self._fingerprint(raw)
        total = 0
        for i in range(self.num_rows):
            b = self._bucket(raw, i)
            if int(self._fps[i, b]) == fp:
                total += int(self._counts[i, b])
        return total
