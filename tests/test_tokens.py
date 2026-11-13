import io
import json
import sys

import pytest

from sdxl_trainer import runtime
from sdxl_trainer.tokens import TokenCounter, TokenCounts, tokenizer_fingerprint


class Tokenizer:
    model_max_length = 77

    def __init__(self, multiplier=1):
        self.multiplier = multiplier
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append(kwargs)
        return {"input_ids": [0] + list(range(len(text.split()) * self.multiplier)) + [1]}


class CharacterTokenizer(Tokenizer):
    def __call__(self, text, **kwargs):
        return {"input_ids": [0] + [2] * len(text) + [1]}


def test_two_tokenizers_count_special_tokens_without_truncation():
    first, second = Tokenizer(), Tokenizer(2)
    count = TokenCounter(first, second).count("one two three")
    assert count == TokenCounts(5, 8)
    assert count.valid
    assert first.calls == [{"add_special_tokens": True, "truncation": False, "padding": False, "verbose": False}]
    assert second.calls == first.calls


def test_exact_limit_and_overflow():
    counter = TokenCounter(Tokenizer(), Tokenizer())
    assert counter.count(" ".join(["word"] * 75)).valid
    assert not counter.count(" ".join(["word"] * 76)).valid
    assert counter.count("").clip_l == 2


def test_both_encoder_limits_matter():
    assert not TokenCounter(Tokenizer(), Tokenizer(2)).count(" ".join(["word"] * 40)).valid


def test_truncate_leaves_caption_at_limit_unchanged():
    counter = TokenCounter(Tokenizer(), Tokenizer())
    caption = "  " + " ".join(["Label"] * 75) + ",\n"
    assert counter.count(caption) == TokenCounts(77, 77)
    assert counter.truncate(caption) == caption


def test_truncate_preserves_whole_words_that_fit_both_encoders():
    counter = TokenCounter(Tokenizer(), Tokenizer(2))
    words = [f"Word{i}" for i in range(40)]
    caption = " ".join(words)
    result = counter.truncate(caption)
    assert result == " ".join(words[:37])
    assert counter.count(result) == TokenCounts(39, 76)
    assert not counter.count(" ".join(words[:38])).valid


@pytest.mark.parametrize("comma_after", [3, 71])
def test_truncate_prefers_nearby_commas_and_keeps_text_after_distant_commas(comma_after):
    counter = TokenCounter(Tokenizer(), Tokenizer())
    phrase = " ".join(["Rose"] * comma_after)
    caption = phrase + ", " + " ".join(["Stem"] * 80)
    expected = phrase if comma_after == 71 else phrase + ", " + " ".join(["Stem"] * 72)
    assert counter.truncate(caption) == expected


def test_truncate_unbroken_caption_keeps_combining_characters_together():
    counter = TokenCounter(CharacterTokenizer(), CharacterTokenizer())
    caption = "e\u0301" * 80
    result = counter.truncate(caption)
    assert result == "e\u0301" * 37
    assert counter.count(result).valid


def test_truncate_keeps_nonempty_prefix_of_separator_only_caption():
    counter = TokenCounter(CharacterTokenizer(), CharacterTokenizer())
    result = counter.truncate("," * 76)
    assert result == "," * 75
    assert counter.count(result).valid


def test_token_runtime_can_count_and_truncate_in_the_same_session(monkeypatch, capsys):
    counter = TokenCounter(Tokenizer(), Tokenizer())
    monkeypatch.setattr(TokenCounter, "from_model", lambda _: counter)
    caption = " ".join(["Word"] * 76)
    requests = [{"text": caption}, {"action": "truncate", "text": caption}, {"text": "Caption"}]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(map(json.dumps, requests))))

    assert runtime.main(["tokens", "--model", "unused"]) == 0
    ready, before, shortened, after = map(json.loads, capsys.readouterr().out.splitlines())
    assert ready["ready"] is True
    assert before == TokenCounts(78, 78).to_dict()
    assert shortened == {"text": " ".join(["Word"] * 75)}
    assert after == TokenCounts(3, 3).to_dict()


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


@pytest.mark.parametrize(('count', 'chunks'), [(2,1),(77,1),(78,2),(152,2),(153,3),(227,3),(228,4)])
def test_common_sdxl_windows_include_all_content_tokens(count, chunks):
    assert TokenCounts(count, 2).chunks == chunks
    assert TokenCounts(2, count).chunks == chunks
