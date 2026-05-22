"""Hashing utilities shared by all sketch implementations."""

from __future__ import annotations

import builtins
import hashlib
import struct
from typing import Union

_INT_TYPE = type(0)
_INT_CAST = builtins.int

try:
    import mmh3
    _HAS_MMH3 = True
except ImportError:
    _HAS_MMH3 = False


def _to_bytes(item: Union[bytes, str, int, float]) -> bytes:
    if isinstance(item, bytes): return item
    if isinstance(item, str):   return item.encode()
    if isinstance(item, int):   return struct.pack("<q", item)
    if isinstance(item, float): return struct.pack("<d", item)
    return repr(item).encode()


def _seed_bytes_to_int(seed_bytes: bytes) -> int:
    return _INT_TYPE.from_bytes(seed_bytes[:8], "little", signed=False)


def _normalize_seed(seed) -> int:
    if isinstance(seed, _INT_TYPE):
        return _INT_CAST(seed)
    if isinstance(seed, bytes):
        if not seed:
            return 0
        return _seed_bytes_to_int(seed)
    if isinstance(seed, str):
        seed = seed.strip()
        if not seed:
            return 0
        try:
            return _INT_CAST(seed, 0)
        except ValueError:
            return _seed_bytes_to_int(seed.encode())
    try:
        return _INT_CAST(seed)
    except (TypeError, ValueError):
        return _seed_bytes_to_int(repr(seed).encode())


def _hash_pair(raw: bytes, seed: int) -> tuple[int, int]:
    """Two pseudo-independent 64-bit hashes via mmh3 (or SHA-256 fallback)."""
    raw = raw if isinstance(raw, bytes) else _to_bytes(raw)
    seed32 = _INT_CAST(_normalize_seed(seed)) & 0x7FFF_FFFF

    if _HAS_MMH3:
        h1, h2 = mmh3.hash64(raw, seed32, signed=False)
        return _INT_CAST(h1), _INT_CAST(h2)
    s1 = seed32 & 0xFFFF_FFFF
    s2 = (seed32 ^ 0x9E37_79B9) & 0xFFFF_FFFF
    h1 = _INT_TYPE.from_bytes(hashlib.sha256(raw + struct.pack("<I", s1)).digest()[:4], "little")
    h2 = _INT_TYPE.from_bytes(hashlib.sha256(raw + struct.pack("<I", s2)).digest()[:4], "little")
    return h1, h2


def bucket_index(item: Union[bytes, str, int, float], seed: int, num_buckets: int) -> int:
    if not isinstance(num_buckets, _INT_TYPE):
        try:
            num_buckets = _INT_CAST(num_buckets)
        except (TypeError, ValueError):
            raise TypeError("num_buckets must be an integer")
    if num_buckets <= 0:
        raise ValueError("num_buckets must be > 0")
    h1, _ = _hash_pair(_to_bytes(item), seed)
    return h1 % num_buckets


def derive_seeds(master_seed: int, n: int) -> list[int]:
    import numpy as np
    rng = np.random.default_rng(_normalize_seed(master_seed))
    return [_INT_CAST(seed) for seed in rng.integers(0, 2**31, size=n, dtype=np.int64).tolist()]
