from types import SimpleNamespace

import pytest

from backend.core.strata_ultra import cuda_backend
from backend.core.strata_ultra.kv_cache import PackedKV


def _record(payload=b"\x00", scales=b"\x00" * 4):
    return SimpleNamespace(
        codec="ternary-q05", rows=2, cols=2, group_size=4,
        payload=payload, scales=scales, name="weights",
    )


def test_cuda_rejects_bad_packed_payload_before_loading_library(monkeypatch):
    monkeypatch.setattr(cuda_backend, "_load", lambda: None)
    with pytest.raises(ValueError, match="packed payload"):
        cuda_backend.matvec_cuda(_record(payload=b""), [1.0, 1.0])


def test_cuda_rejects_non_finite_input_before_loading_library(monkeypatch):
    monkeypatch.setattr(cuda_backend, "_load", lambda: None)
    with pytest.raises(ValueError, match="finite"):
        cuda_backend.matvec_cuda(_record(), [float("nan"), 1.0])


@pytest.mark.parametrize("field", ["rows", "cols", "group_size"])
def test_cuda_rejects_values_outside_abi_uint32(monkeypatch, field):
    monkeypatch.setattr(cuda_backend, "_load", lambda: None)
    record = _record()
    setattr(record, field, 0x1_0000_0000)
    with pytest.raises(ValueError, match=field):
        cuda_backend.matvec_cuda(record, [1.0, 1.0])


def test_cuda_rejects_non_finite_scales_before_loading_library(monkeypatch):
    monkeypatch.setattr(cuda_backend, "_load", lambda: object())
    with pytest.raises(ValueError, match="scale"):
        cuda_backend.matvec_cuda(_record(scales=bytes.fromhex("0000807f")), [1.0, 1.0])


def test_cuda_kv_rejects_sparse_profile_before_loading_library(monkeypatch):
    monkeypatch.setattr(cuda_backend, "_load", lambda: None)
    cache = PackedKV("sparse05", 8, 4, b"\x00", (1.0, 1.0))
    with pytest.raises(ValueError, match="sign1 and ternary05"):
        cuda_backend.decode_kv_cuda(cache)


def test_cuda_kv_validates_payload_and_scale_geometry(monkeypatch):
    monkeypatch.setattr(cuda_backend, "_load", lambda: None)
    cache = PackedKV("sign1", 8, 4, b"", (1.0, 1.0))
    with pytest.raises(ValueError, match="payload"):
        cuda_backend.decode_kv_cuda(cache)
    cache = PackedKV("ternary05", 8, 4, b"\x00\x00", (1.0,))
    with pytest.raises(ValueError, match="scale"):
        cuda_backend.decode_kv_cuda(cache)
