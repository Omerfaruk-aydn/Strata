import struct

from backend.core.strata_ultra import TensorRecord, matmul, matmul_fast, numpy_available


def test_numpy_backend_matches_reference_matmul():
    record = TensorRecord("x", 2, 2, 4, "ternary-q05", bytes([0b10_00_00_10]), struct.pack("<f", 1.0))
    matrix = [[1.0, 2.0], [3.0, 4.0]]
    assert matmul_fast(record, matrix) == matmul(record, matrix)
    assert isinstance(numpy_available(), bool)
