"""
NitroSketch (Liu et al., SIGCOMM 2019).

Packet-level Bernoulli sampling: each arriving packet is admitted with
probability 1/sampling_rate using an independent PRNG (not a flow hash, so
every flow accumulates samples proportional to its frequency).

Admitted packets update the underlying CMS by +1.
Query returns CMS-min × sampling_rate — an unbiased estimate of true count.

Effect on metrics:
  • Very high throughput (only 1/sampling_rate of packets do CMS work).
  • ARE grows roughly proportionally to 1/sqrt(expected_samples_per_flow).
  • Default sampling_rate=16 is a balanced choice; can be tuned via CLI.
"""

import numpy as np

from .hash_utils import _hash_pair, _to_bytes, derive_seeds


def _looks_like_nitro(obj) -> bool:
    required = (
        "_num_rows",
        "_row_width",
        "_sampling_rate",
        "_seeds",
        "_banks",
        "_rng",
        "_row_idx",
        "_update_impl",
        "_query_impl",
    )
    return all(hasattr(obj, name) for name in required)


def _coerce_receiver(receiver, item, method_name: str):
    if _looks_like_nitro(receiver):
        return receiver, item
    if _looks_like_nitro(item):
        return item, receiver
    raise TypeError(f"NitroSketch.{method_name} called with invalid receiver")


class NitroSketch:

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if name == "update" and _looks_like_nitro(value):
            return object.__getattribute__(value, "_update_impl")
        if name == "query" and _looks_like_nitro(value):
            return object.__getattribute__(value, "_query_impl")
        return value

    def __init__(self, num_rows: int = 4, row_width: int = 4096,
                 sampling_rate: int = 16, seed: int = 42):
        self._num_rows      = num_rows
        self._row_width     = row_width
        self._sampling_rate = sampling_rate

        self._seeds   = derive_seeds(seed, num_rows)
        self._banks   = np.zeros((num_rows, row_width), dtype=np.uint32)
        self._rng     = np.random.default_rng(seed ^ 0xDEAD_BEEF)
        self._row_idx = list(range(num_rows))

    # ------------------------------------------------------------------
    def update(self, item) -> None:
        self, item = _coerce_receiver(self, item, "update")
        self._update_impl(item)

    def _update_impl(self, item) -> None:
        # Independent per-packet Bernoulli gate
        if self._rng.integers(0, self._sampling_rate) != 0:
            return
        raw = item if isinstance(item, bytes) else _to_bytes(item)
        for i in self._row_idx:
            idx = int(_hash_pair(raw, self._seeds[i])[0]) % self._row_width
            self._banks[i, idx] += 1

    # ------------------------------------------------------------------
    def query(self, item) -> int:
        self, item = _coerce_receiver(self, item, "query")
        return self._query_impl(item)

    def _query_impl(self, item) -> int:
        raw = item if isinstance(item, bytes) else _to_bytes(item)
        est = min(
            int(self._banks[i, int(_hash_pair(raw, self._seeds[i])[0]) % self._row_width])
            for i in self._row_idx
        )
        return est * self._sampling_rate

    @property
    def memory_bytes(self) -> int:
        return self._banks.nbytes
