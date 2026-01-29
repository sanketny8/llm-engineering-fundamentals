from __future__ import annotations

from llm_engineering_fundamentals.tokenization.bpe import BPEModel


def test_roundtrip_basic() -> None:
    model = BPEModel.train(
        [
            "hello world",
            "hello tokenization",
            "tokenization is weird",
            "hello hello hello",
        ],
        merges=50,
    )

    text = "hello tokenization world"
    ids = model.encode(text)
    decoded = model.decode(ids)
    assert decoded == text


def test_encode_is_deterministic() -> None:
    corpus = ["a a a b b c", "a b c", "a b b c"]
    model = BPEModel.train(corpus, merges=20)

    t = "a b c"
    assert model.encode(t) == model.encode(t)


