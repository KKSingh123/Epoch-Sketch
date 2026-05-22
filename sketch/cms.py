"""Count-Min Sketch (Cormode & Muthukrishnan, 2005)."""

from __future__ import annotations

import numpy as np
from .hash_utils import bucket_index, derive_seeds


class CountMinSketch:
    """
    Standard Count-Min Sketch used as a baseline.

    Parameters
    ----------
    num_rows    : depth;  failure probability δ = e^{-d}
    num_buckets : width;  additive error ε = e / w
    seed        : master seed for row hash functions
    """

    def __init__(self, num_rows: int = 4, num_buckets: int = 1024, seed: int = 42) -> None:
        self.num_rows    = num_rows
        self.num_buckets = num_buckets
        self.seed        = seed
        self._seeds  = derive_seeds(seed, num_rows)
        self._table  = np.zeros((num_rows, num_buckets), dtype=np.int32)

    def update(self, item, count: int = 1) -> None:
        for r in range(self.num_rows):
            self._table[r, bucket_index(item, self._seeds[r], self.num_buckets)] += count

    def query(self, item) -> int:
        return int(min(
            self._table[r, bucket_index(item, self._seeds[r], self.num_buckets)]
            for r in range(self.num_rows)
        ))

    def reset(self) -> None:
        self._table[:] = 0

    @property
    def memory_bytes(self) -> int:
        return int(self._table.nbytes)
