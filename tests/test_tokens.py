from pathlib import Path

import pytest

from sdxl_trainer.tokens import TokenCounter, TokenCounts, tokenizer_fingerprint


class Tokenizer:
    model_max_length = 77

    def __init__(self, multiplier=1):
        self.multiplier = multiplier
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append(kwargs)
        return {"input_ids": [0] + list(range(len(text.split()) * self.multiplier)) + [1]}


def test_two_tokenizers_count_special_tokens_without_truncation():
    first, second = Tokenizer(), Tokenizer(2)
    count = TokenCounter(first, second).count("one two three")
    assert count == TokenCounts(5, 8)
    assert count.valid
    assert first.calls == [{"add_special_tokens": True, "truncation": False, "padding": False}]
    assert second.calls == first.calls


def test_exact_limit_and_overflow():
    counter = TokenCounter(Tokenizer(), Tokenizer())
    assert counter.count(" ".join(["word"] * 75)).valid
    assert not counter.count(" ".join(["word"] * 76)).valid
    assert counter.count("").clip_l == 2


def test_both_encoder_limits_matter():
    assert not TokenCounter(Tokenizer(), Tokenizer(2)).count(" ".join(["word"] * 40)).valid


def test_reject_non_sdxl_context():
    tokenizer = Tokenizer()
    tokenizer.model_max_length = 128
    with pytest.raises(ValueError, match="77-token"):
        TokenCounter(tokenizer, Tokenizer())


def test_fingerprint_depends_on_contents_and_names_not_path(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    for root in (one, two):
        (root / "vocab.json").write_text('{"a":1}')
    assert tokenizer_fingerprint(one) == tokenizer_fingerprint(two)
    (two / "vocab.json").write_text('{"a":2}')
    assert tokenizer_fingerprint(one) != tokenizer_fingerprint(two)


def test_missing_local_model_rejected(tmp_path):
    with pytest.raises(ValueError, match="tokenizer"):
        TokenCounter.from_model(tmp_path)
