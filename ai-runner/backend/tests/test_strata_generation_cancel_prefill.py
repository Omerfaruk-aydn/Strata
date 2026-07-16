from threading import Event

from backend.core.strata_ultra.generation import GenerationConfig, StrataGenerator


class _Tokenizer:
    def encode(self, value):
        return [1, 2, 3]

    def decode(self, value):
        return "".join(str(item) for item in value)


class _Runtime:
    def tensor_row(self, name, token):
        return [float(token)]


class _Transformer:
    def step(self, hidden):
        return hidden


def test_generation_cancelled_during_prefill_returns_prompt_without_error():
    event = Event()
    event.set()
    generator = StrataGenerator(_Runtime(), _Transformer(), "embedding", "output", tokenizer=_Tokenizer())

    result = generator.generate_with_metadata("hello", GenerationConfig(cancel_event=event))

    assert result == {"text": "hello", "generated_tokens": 0, "finish_reason": "cancelled"}
