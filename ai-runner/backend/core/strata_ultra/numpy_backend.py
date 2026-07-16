"""Optional NumPy acceleration for Strata packed tensor execution."""

from __future__ import annotations

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only on minimal installs
    np = None

from .container import TensorRecord
from .executor import matmul


def numpy_available() -> bool:
    return np is not None


def matmul_fast(record: TensorRecord, matrix: list[list[float]]) -> list[list[float]]:
    """Vectorized Q0.5 matmul when NumPy is installed, with safe fallback."""
    if np is None:
        return matmul(record, matrix)
    count = record.rows * record.cols
    scales = np.frombuffer(record.scales, dtype="<f4")
    packed = np.frombuffer(record.payload, dtype=np.uint8)
    codes = np.empty(count, dtype=np.int8)
    for index in range(count):
        codes[index] = (packed[index // 4] >> ((index % 4) * 2)) & 3
    values = np.where(codes == 0, 0.0, np.where(codes == 1, -1.0, 1.0)).astype(np.float32)
    values *= np.repeat(scales, record.group_size)[:count]
    weights = values.reshape(record.rows, record.cols)
    result = weights @ np.asarray(matrix, dtype=np.float32).T
    return result.T.tolist()
