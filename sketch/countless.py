"""CountLess Sketch.

This is implemented as a conservative-update Count-Min style sketch with
merge/reset/bulk-update support and a helper for constructing the sketch
from a fixed memory budget.
"""

from __future__ import annotations

import numpy as np

from .hash_utils import bucket_index, derive_seeds


class CountLessSketch:
    """CountLess-style sketch with conservative update.

    Parameters
    ----------
    num_rows : int
        Number of hash rows.
    row_width : int
        Number of counters per row.
    seed : int
        Master seed used to derive per-row hash seeds.
    """

    def __init__(self, num_rows: int = 4, row_width: int = 1024, seed: int = 42) -> None:
        if num_rows <= 0:
            raise ValueError("num_rows must be > 0")
        if row_width <= 0:
            raise ValueError("row_width must be > 0")

        self.num_rows = int(num_rows)
        self.row_width = int(row_width)
        self.seed = int(seed)

        self._seeds = derive_seeds(self.seed, self.num_rows)
        self._table = np.zeros((self.num_rows, self.row_width), dtype=np.int32)

    @classmethod
    def from_memory_budget(
        cls,
        memory_bytes: int,
        num_rows: int = 4,
        seed: int = 42,
    ) -> "CountLessSketch":
        if memory_bytes <= 0:
            raise ValueError("memory_bytes must be > 0")
        if num_rows <= 0:
            raise ValueError("num_rows must be > 0")

        row_width = max(1, int(memory_bytes) // (int(num_rows) * 4))
        return cls(num_rows=num_rows, row_width=row_width, seed=seed)

    def update(self, item, count: int = 1) -> None:
        if count <= 0:
            return

        idxs = [bucket_index(item, self._seeds[r], self.row_width) for r in range(self.num_rows)]
        cur = [int(self._table[r, idxs[r]]) for r in range(self.num_rows)]
        mn = min(cur)
        for r in range(self.num_rows):
            if cur[r] == mn:
                self._table[r, idxs[r]] += count

    def update_bulk(self, items) -> None:
        for item in items:
            self.update(item)

    def query(self, item) -> int:
        return int(min(
            self._table[r, bucket_index(item, self._seeds[r], self.row_width)]
            for r in range(self.num_rows)
        ))

    def reset(self) -> None:
        self._table[:] = 0

    def merge(self, other: "CountLessSketch") -> None:
        if not isinstance(other, CountLessSketch):
            raise TypeError("other must be a CountLessSketch")
        if self.num_rows != other.num_rows or self.row_width != other.row_width:
            raise ValueError("sketch shapes must match")
        if self.seed != other.seed or self._seeds != other._seeds:
            raise ValueError("sketch seeds must match")

        self._table += other._table

    @property
    def memory_bytes(self) -> int:
        return int(self._table.nbytes)


# Backward-friendly alias for alternate capitalization.
CountlessSketch = CountLessSketch