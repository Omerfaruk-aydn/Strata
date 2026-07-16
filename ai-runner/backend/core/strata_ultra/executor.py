"""Reference CPU executor for packed Strata tensors.

This is a correctness-first runtime kernel.  It performs dequantization on the
fly and uses the bounded layer pager, providing the contract that optimized
NumPy/CUDA/Vulkan kernels can later replace without changing the container.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Callable

from .container import StrataContainerReader, TensorRecord
from .paging import LayerPager


def _tensor_values(record: TensorRecord) -> list[float]:
    if record.codec != "ternary-q05":
        raise ValueError(f"unsupported Strata tensor codec: {record.codec}")
    count = record.rows * record.cols
    scales_count = (count + record.group_size - 1) // record.group_size
    if len(record.scales) != scales_count * 4:
        raise ValueError(f"invalid scale table for tensor {record.name}")
    scales = struct.unpack(f"<{scales_count}f", record.scales)
    values: list[float] = []
    for index in range(count):
        code = (record.payload[index // 4] >> ((index % 4) * 2)) & 3
        scale = scales[index // record.group_size]
        values.append(0.0 if code == 0 else -scale if code == 1 else scale)
    return values


def matvec(record: TensorRecord, vector: list[float]) -> list[float]:
    """Compute ``record × vector`` using on-the-fly Q0.5 dequantization."""
    if len(vector) != record.cols:
        raise ValueError(f"vector length {len(vector)} != tensor columns {record.cols}")
    values = _tensor_values(record)
    return [
        sum(values[row * record.cols + col] * vector[col] for col in range(record.cols))
        for row in range(record.rows)
    ]


class StrataRuntime:
    """Minimal model runtime with a bounded resident tensor window."""

    def __init__(self, model_path: str | Path, memory_budget_bytes: int, resident_window: int = 2):
        self.reader = StrataContainerReader(model_path)
        records = {record.name: record for record in self.reader.read_tensors()}
        if not records:
            self.reader.close()
            raise ValueError("Strata model contains no tensors")
        self._records = records
        self.pager = LayerPager(
            max_pages=resident_window,
            max_bytes=memory_budget_bytes,
            loader=lambda name: (self._records[name], len(self._records[name].payload) + len(self._records[name].scales)),
        )

    def tensor_matvec(self, tensor_name: str, vector: list[float]) -> list[float]:
        return matvec(self.pager.get(tensor_name), vector)

    def close(self) -> None:
        self.pager.clear()
        self.reader.close()

    def __enter__(self) -> "StrataRuntime":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
