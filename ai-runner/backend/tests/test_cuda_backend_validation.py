from types import SimpleNamespace

import pytest

from backend.core.strata_ultra import cuda_backend


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
