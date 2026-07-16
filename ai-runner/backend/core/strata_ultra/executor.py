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
from .sparse_codec import decode_sparse05


def _tensor_values(record: TensorRecord) -> list[float]:
    if record.codec == "sparse05":
        scales = struct.unpack(f"<{len(record.scales) // 4}f", record.scales)
        return decode_sparse05(record.payload, scales, record.rows * record.cols, record.group_size)
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
    if record.codec == "ternary-q05":
        return matvec_streaming(record, vector)
    values = _tensor_values(record)
    return [
        sum(values[row * record.cols + col] * vector[col] for col in range(record.cols))
        for row in range(record.rows)
    ]


def matvec_streaming(record: TensorRecord, vector: list[float]) -> list[float]:
    """Multiply ternary weights without materializing the full float tensor."""
    if record.codec != "ternary-q05":
        return matvec(record, vector)
    if len(vector) != record.cols:
        raise ValueError(f"vector length {len(vector)} != tensor columns {record.cols}")
    count = record.rows * record.cols
    scales_count = (count + record.group_size - 1) // record.group_size
    if len(record.scales) != scales_count * 4:
        raise ValueError(f"invalid scale table for tensor {record.name}")
    scales = struct.unpack(f"<{scales_count}f", record.scales)
    output = [0.0] * record.rows
    for row in range(record.rows):
        total = 0.0
        base = row * record.cols
        for col in range(record.cols):
            index = base + col
            code = (record.payload[index // 4] >> ((index % 4) * 2)) & 3
            if code:
                total += (-1.0 if code == 1 else 1.0) * scales[index // record.group_size] * vector[col]
        output[row] = total
    return output


def matmul(record: TensorRecord, matrix: list[list[float]]) -> list[list[float]]:
    """Compute multiple vectors while decoding the packed tensor once."""
    if any(len(vector) != record.cols for vector in matrix):
        raise ValueError(f"every vector must have {record.cols} columns")
    if record.codec == "ternary-q05":
        return matmul_streaming(record, matrix)
    values = _tensor_values(record)
    output = []
    for vector in matrix:
        output.append([
            sum(values[row * record.cols + col] * vector[col] for col in range(record.cols))
            for row in range(record.rows)
        ])
    return output


def matmul_streaming(record: TensorRecord, matrix: list[list[float]]) -> list[list[float]]:
    """Batch matmul over packed ternary weights with bounded temporary memory."""
    if record.codec != "ternary-q05":
        return matmul(record, matrix)
    if any(len(vector) != record.cols for vector in matrix):
        raise ValueError(f"every vector must have {record.cols} columns")
    return [matvec_streaming(record, vector) for vector in matrix]


class StrataRuntime:
    """Minimal model runtime with a bounded resident tensor window."""

    def __init__(self, model_path: str | Path, memory_budget_bytes: int, resident_window: int = 2, backend: str = "auto"):
        if backend not in {"auto", "python", "numpy", "cuda"}:
            raise ValueError("backend must be 'auto', 'python', 'numpy', or 'cuda'")
        self.reader = StrataContainerReader(model_path)
        records = {record.name: record for record in self.reader.read_tensors()}
        if not records:
            self.reader.close()
            raise ValueError("Strata model contains no tensors")
        self._records = records
        self.backend = backend
        self.pager = LayerPager(
            max_pages=resident_window,
            max_bytes=memory_budget_bytes,
            loader=lambda name: (self._records[name], len(self._records[name].payload) + len(self._records[name].scales)),
        )

    def tensor_matvec(self, tensor_name: str, vector: list[float]) -> list[float]:
        record = self.pager.get(tensor_name)
        if self.backend == "cuda":
            from .cuda_backend import matvec_cuda
            return matvec_cuda(record, vector)
        if self.backend == "python":
            return matvec(record, vector)
        return self.tensor_matmul(tensor_name, [vector])[0]

    def tensor_matmul(self, tensor_name: str, matrix: list[list[float]]) -> list[list[float]]:
        record = self.pager.get(tensor_name)
        if self.backend == "cuda":
            from .cuda_backend import matvec_cuda
            return [matvec_cuda(record, vector) for vector in matrix]
        if self.backend in {"auto", "numpy"}:
            from .numpy_backend import matmul_fast, numpy_available
            if self.backend == "numpy" and not numpy_available():
                raise RuntimeError("NumPy backend requested but NumPy is unavailable")
            if numpy_available():
                return matmul_fast(record, matrix)
        return matmul(record, matrix)

    def tensor_row(self, tensor_name: str, row: int) -> list[float]:
        record = self.pager.get(tensor_name)
        if row < 0 or row >= record.rows:
            raise IndexError(f"tensor row {row} outside [0, {record.rows})")
        values = _tensor_values(record)
        start = row * record.cols
        return values[start:start + record.cols]

    def close(self) -> None:
        self.pager.clear()
        self.reader.close()

    def __enter__(self) -> "StrataRuntime":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
