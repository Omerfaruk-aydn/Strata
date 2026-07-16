"""Bounded-memory layer paging for the Strata Ultra runtime.

The pager is backend-neutral: a layer can be represented by any Python object
or a future mmap/CUDA/Vulkan handle.  The policy keeps a bounded LRU window and
exposes explicit load/evict events for telemetry and asynchronous prefetch.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Optional, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass(frozen=True)
class PageEvent:
    action: str
    layer_id: str
    resident_pages: int
    resident_bytes: int


class LayerPager(Generic[K, V]):
    """LRU pager with a hard page-count and byte budget."""

    def __init__(
        self,
        max_pages: int,
        max_bytes: int,
        loader: Callable[[K], tuple[V, int]],
        disposer: Optional[Callable[[V], None]] = None,
    ) -> None:
        if max_pages <= 0 or max_bytes <= 0:
            raise ValueError("max_pages and max_bytes must be positive")
        self.max_pages = max_pages
        self.max_bytes = max_bytes
        self.loader = loader
        self.disposer = disposer or (lambda _: None)
        self._pages: OrderedDict[K, tuple[V, int]] = OrderedDict()
        self._bytes = 0
        self.events: list[PageEvent] = []

    @property
    def resident_pages(self) -> int:
        return len(self._pages)

    @property
    def resident_bytes(self) -> int:
        return self._bytes

    def _emit(self, action: str, layer_id: K) -> None:
        self.events.append(PageEvent(action, str(layer_id), len(self._pages), self._bytes))

    def _evict_one(self) -> bool:
        if not self._pages:
            return False
        key, (value, size) = self._pages.popitem(last=False)
        self._bytes -= size
        self.disposer(value)
        self._emit("evict", key)
        return True

    def get(self, layer_id: K) -> V:
        if layer_id in self._pages:
            value, size = self._pages.pop(layer_id)
            self._pages[layer_id] = (value, size)
            self._emit("hit", layer_id)
            return value
        value, size = self.loader(layer_id)
        if size <= 0 or size > self.max_bytes:
            self.disposer(value)
            raise MemoryError(f"layer {layer_id} exceeds pager budget")
        while self._pages and (len(self._pages) >= self.max_pages or self._bytes + size > self.max_bytes):
            self._evict_one()
        self._pages[layer_id] = (value, size)
        self._bytes += size
        self._emit("load", layer_id)
        return value

    def prefetch(self, layer_id: K) -> None:
        """Load a future layer through the same bounded policy."""
        self.get(layer_id)

    def clear(self) -> None:
        while self._pages:
            self._evict_one()
