import pytest

from backend.core.strata_ultra import tensor_quality


def test_tensor_quality_reports_error_and_cosine():
    report = tensor_quality([1.0, 0.0, -1.0], [0.5, 0.0, -1.0])
    assert report["mse"] > 0
    assert 0 < report["cosine_similarity"] <= 1
    assert report["max_abs_error"] == 0.5


def test_tensor_quality_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        tensor_quality([1.0], [1.0, 2.0])
