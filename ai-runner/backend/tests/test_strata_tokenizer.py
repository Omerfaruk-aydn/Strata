import pytest

from backend.core.strata_ultra import GGUFTokenizer


def test_gguf_bpe_tokenizer_adapter_round_trips_basic_text():
    pytest.importorskip("tokenizers")
    tokenizer = GGUFTokenizer.from_metadata({
        "tokenizer.ggml.tokens": ["<unk>", "hello", " world"],
        "tokenizer.ggml.merges": [],
    })
    encoded = tokenizer.encode("hello world")
    assert encoded
    assert isinstance(tokenizer.decode(encoded), str)
