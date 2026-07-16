"""Runtime-managed ultra-low-bit KV cache with a bounded context window."""

from __future__ import annotations

from dataclasses import dataclass

from .kv_cache import PackedKV, decode_kv, encode_kv, estimate_kv_bytes


@dataclass(frozen=True)
class KVSnapshot:
    tokens: int
    width: int
    mode: str
    packed_bytes: int
    evicted_tokens: int


class UltraKVCache:
    """A compact rolling KV cache for reference execution.

    Values are kept as a flat token-major matrix and repacked after append.
    The reference implementation favors deterministic correctness; optimized
    backends can replace the repack step with append-only bit buffers later.
    """

    def __init__(self, width: int, capacity_tokens: int, mode: str = "sign1", group_size: int = 128, sparse_threshold: float = 0.125, backend: str = "auto"):
        if width <= 0 or capacity_tokens <= 0:
            raise ValueError("width and capacity_tokens must be positive")
        if mode not in {"sign1", "ternary05", "sparse05"}:
            raise ValueError("mode must be 'sign1', 'ternary05' or 'sparse05'")
        if sparse_threshold < 0:
            raise ValueError("sparse_threshold must be non-negative")
        if backend not in {"auto", "python", "cuda"}:
            raise ValueError("backend must be 'auto', 'python' or 'cuda'")
        self.width = width
        self.capacity_tokens = capacity_tokens
        self.mode = mode
        self.group_size = group_size
        self.sparse_threshold = sparse_threshold
        self.backend = backend
        self._values: list[float] = []
        self._cache: PackedKV | None = None
        self.evicted_tokens = 0

    @property
    def token_count(self) -> int:
        return len(self._values) // self.width

    def append(self, values: list[float]) -> None:
        if len(values) % self.width != 0:
            raise ValueError("KV append length must be a multiple of cache width")
        self._values.extend(float(value) for value in values)
        overflow = max(0, self.token_count - self.capacity_tokens)
        if overflow:
            del self._values[:overflow * self.width]
            self.evicted_tokens += overflow
        self._cache = encode_kv(self._values, self.mode, self.group_size, self.sparse_threshold) if self._values else None

    def values(self) -> list[float]:
        if not self._cache:
            return []
        if self.backend in {"auto", "cuda"}:
            from .cuda_backend import cuda_available, decode_kv_cuda, decode_sparse_kv_cuda
            if self.backend == "cuda" or cuda_available():
                if self.mode == "sparse05":
                    return decode_sparse_kv_cuda(self._cache)
                if self.mode in {"sign1", "ternary05"}:
                    return decode_kv_cuda(self._cache)
        return decode_kv(self._cache)

    def snapshot(self) -> KVSnapshot:
        packed_bytes = self._cache.payload_bytes if self._cache else 0
        return KVSnapshot(self.token_count, self.width, self.mode, packed_bytes, self.evicted_tokens)

    def clear(self) -> None:
        self._values.clear()
        self._cache = None
        self.evicted_tokens = 0
