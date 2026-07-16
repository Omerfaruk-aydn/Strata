from backend.core.strata_ultra.iq_registry import BY_NAME, NATIVE_BRIDGE_TYPE_IDS, capability_report, source_codec_names


def test_iq_registry_exposes_verified_and_pending_codecs():
    report = {item["name"]: item for item in capability_report()}
    assert report["IQ4_NL"]["decodable"] is True
    assert report["IQ4_NL"]["decoder"] == "_decode_iq4_nl"
    assert report["IQ2_XXS"]["decodable"] is False


def test_native_capability_report_marks_bridge_codecs_decodable():
    report = {item["name"]: item for item in capability_report(native_bridge=True)}
    assert report["IQ2_XXS"]["decodable"] is True
    assert report["IQ2_XXS"]["decoder"] == "native-ggml"
    assert report["IQ4_XS"]["decodable"] is True
    assert report["IQ4_XS"]["decoder"] == "native-ggml"
    assert 23 in NATIVE_BRIDGE_TYPE_IDS


def test_source_codec_names_only_reports_active_decoders():
    assert source_codec_names() == ["IQ4_NL"]
    assert "IQ4_XS" in source_codec_names(native_bridge=True)
    assert BY_NAME["IQ3_S"].type_id == 21
