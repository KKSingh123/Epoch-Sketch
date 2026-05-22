"""HeavyGuardianSketch: hardware-friendly heavy table + conservative sketch.

This method pairs a compact set-associative heavy table with a light
conservative Count-Min sketch. Heavy entries are inserted only when the
light sketch believes a flow is strong enough, which keeps the system
fully bounded and hardware-friendly.
"""
# python3 memory_experiment.py --dat dataset/CAIDA/0.dat
# python3 uni_memory_experiment.py --csv dataset/UNI/univ1_pt0.csv
# python3 caida/code/caida_prf_metrics.py --dat dataset/CAIDA/0.dat
# python3 uni/code/uni_prf_metrics.py --csv dataset/UNI/univ1_pt0.csv

from __future__ import annotations

import builtins
import time
from .hash_utils import _to_bytes, bucket_index

_UINT32_MAX = 2**32 - 1


class AegisSketch:

    def __init__(
        self,
        memory_mb: int = 8,
        seed: int = 42,
        _capacity_cells: int | None = None,
        heavy_fraction: float = 0.5,
        sketch_rows: int = 4,
        assoc: int = 8,
        drill_threshold: int = 4,
        age_period_ms: int = 30,
        **_compat,
    ) -> None:
        self.seed = int(seed)
        total_bytes = (
            builtins.int(_capacity_cells) * 4
            if _capacity_cells
            else builtins.int(memory_mb) * 1024 * 1024
        )
        self._budget_bytes = max(1024, total_bytes)
        self._heavy_fraction = min(max(float(heavy_fraction), 0.01), 0.99)
        self._sketch_rows = max(1, int(sketch_rows))
        self._assoc = max(1, int(assoc))
        self.drill_threshold = max(1, int(drill_threshold))

        self._counter_bytes = 4
        self._heavy_entry_bytes = 8

        heavy_bytes = max(self._assoc * self._heavy_entry_bytes,
                          int(self._budget_bytes * self._heavy_fraction))
        sketch_bytes = self._budget_bytes - heavy_bytes

        heavy_slots = max(self._assoc, heavy_bytes // self._heavy_entry_bytes)
        sketch_counters = max(self._sketch_rows * self._assoc,
                              sketch_bytes // self._counter_bytes)

        self._heavy_sets = self._floor_power_of_two(max(1, heavy_slots // self._assoc))
        self._sketch_width = self._floor_power_of_two(max(4, sketch_counters // self._sketch_rows))
        self._heavy_mask = self._heavy_sets - 1
        self._sketch_mask = self._sketch_width - 1

        self._light = [[0] * self._sketch_width for _ in range(self._sketch_rows)]
        self._light_epoch = [[0] * self._sketch_width for _ in range(self._sketch_rows)]
        self._heavy = [[None] * self._assoc for _ in range(self._heavy_sets)]

        self._row_salt = [self.seed ^ (0x9E37_79B9 + i * 0x6A09_E667) & _UINT32_MAX
                          for i in range(self._sketch_rows)]
        self._heavy_seed = (self.seed ^ 0xC6A4_A793) & _UINT32_MAX

        self.total_packets = 0
        self.promotions = 0
        self.updates = 0
        self._age_period_seconds = max(1, int(age_period_ms)) / 1000.0
        self._last_age_time = time.perf_counter()
        self._age_epoch = 0

    @staticmethod
    def _floor_power_of_two(value: int) -> int:
        return 1 << (value.bit_length() - 1) if value > 0 else 1

    def _row_index(self, key: bytes, row: int) -> int:
        return bucket_index(key, self._row_salt[row], self._sketch_width)

    def _heavy_index(self, key: bytes) -> int:
        return bucket_index(key, self._heavy_seed, self._heavy_sets)

    def _advance_age_epoch(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_age_time
        periods = int(elapsed // self._age_period_seconds)
        if periods <= 0:
            return

        self._age_epoch += periods
        self._last_age_time += periods * self._age_period_seconds

    def _age_cell(self, row: int, idx: int) -> int:
        elapsed = self._age_epoch - self._light_epoch[row][idx]
        if elapsed <= 0:
            return self._light[row][idx]

        value = self._light[row][idx] >> elapsed if elapsed < 32 else 0
        self._light[row][idx] = value
        self._light_epoch[row][idx] = self._age_epoch
        return value

    def _sketch_update(self, key: bytes) -> int:
        idxs = [0] * self._sketch_rows
        vals = [0] * self._sketch_rows
        mn = _UINT32_MAX
        for r in range(self._sketch_rows):
            idx = self._row_index(key, r)
            idxs[r] = idx
            value = self._age_cell(r, idx)
            vals[r] = value
            if value < mn:
                mn = value
        new_val = mn + 1
        for r in range(self._sketch_rows):
            if vals[r] == mn:
                self._light[r][idxs[r]] = new_val
                self._light_epoch[r][idxs[r]] = self._age_epoch
        return new_val

    def _sketch_query(self, key: bytes) -> int:
        return min(self._age_cell(r, self._row_index(key, r)) for r in range(self._sketch_rows))

    def _heavy_lookup(self, key: bytes):
        idx = self._heavy_index(key)
        row = self._heavy[idx]
        for i in range(self._assoc):
            slot = row[i]
            if slot is not None and slot[1] == key:
                return idx, i, slot
        return None

    def _insert_heavy(self, key: bytes, count: int) -> None:
        idx = self._heavy_index(key)
        row = self._heavy[idx]
        empty_slot = None
        victim = 0
        min_count = _UINT32_MAX

        for i in range(self._assoc):
            slot = row[i]
            if slot is None:
                empty_slot = i
                break
            if slot[0] < min_count:
                min_count = slot[0]
                victim = i

        if empty_slot is not None:
            row[empty_slot] = [count, key]
            return

        if count > min_count:
            row[victim] = [count, key]

    def update(self, item, _to_bytes_fn=_to_bytes):
        x = item if isinstance(item, bytes) else _to_bytes_fn(item)
        self.total_packets += 1
        self.updates += 1
        self._advance_age_epoch()

        existing = self._heavy_lookup(x)
        if existing is not None:
            existing[2][0] += 1
            return

        current_est = self._sketch_update(x)
        if current_est >= self.drill_threshold:
            self._insert_heavy(x, current_est)
            self.promotions += 1

    def query(self, item, _to_bytes_fn=_to_bytes):
        self._advance_age_epoch()
        x = item if isinstance(item, bytes) else _to_bytes_fn(item)
        existing = self._heavy_lookup(x)
        if existing is not None:
            return existing[2][0]
        return self._sketch_query(x)

    def update_bulk(self, items):
        for item in items:
            self.update(item)

    @property
    def memory_bytes(self) -> int:
        return self._budget_bytes

    @property
    def estimated_used_bytes(self) -> int:
        return self._budget_bytes
