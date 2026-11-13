"""Exact, offline SDXL caption counts for its two local CLIP tokenizers."""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
from typing import Any
import unicodedata


_COMMAS = ",，、"
_WORD_SPAN = re.compile(rf"[^\s{_COMMAS}]+")
_COMMA_BOUNDARY_SLACK = 8


@dataclass(frozen=True)
class TokenCounts:
    clip_l: int
    clip_g: int
    max_length: int = 77

    @property
    def valid(self) -> bool:
        """Whether both streams fit in a single native CLIP window."""
        return self.clip_l <= self.max_length and self.clip_g <= self.max_length

    @property
    def chunks(self) -> int:
        """Common number of 77-token windows (75 content tokens each)."""
        content = max(self.clip_l, self.clip_g) - 2
        return max(1, (content + (self.max_length - 3)) // (self.max_length - 2))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def tokenizer_fingerprint(directory: Path) -> str:
    """Hash the tokenizer assets, not the model weights or their location."""
    digest = hashlib.sha256()
    files = sorted(p for p in directory.iterdir() if p.is_file())
    if not files:
        raise ValueError(f"No tokenizer files in {directory}")
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(b"\0")
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class TokenCounter:
    """Counts include BOS and EOS. Long captions are never silently truncated.

    Injecting tokenizers is useful for tests; the app uses ``from_model`` only.
    The training worker must use the same tokenizer assets and max_length.
    """

    def __init__(self, tokenizer_one: Any, tokenizer_two: Any, *,
                 metadata: dict[str, Any] | None = None):
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        lengths = [int(getattr(t, "model_max_length", 77))
                   for t in (tokenizer_one, tokenizer_two)]
        if lengths != [77, 77]:
            raise ValueError("SDXL training requires two CLIP tokenizers with a 77-token context.")
        self.max_length = 77
        self.metadata = metadata or {"max_length": 77, "special_tokens_included": True}

    @classmethod
    def from_model(cls, model_path: str | Path) -> "TokenCounter":
        model = Path(model_path).expanduser().resolve(strict=True)
        paths = [model / "tokenizer", model / "tokenizer_2"]
        if any(not p.is_dir() for p in paths):
            raise ValueError("Select a local Diffusers SDXL folder containing tokenizer and tokenizer_2.")
        try:
            from transformers import CLIPTokenizer
        except ImportError as exc:
            raise RuntimeError("Exact counting needs the local training runtime (transformers).") from exc
        tokenizers = [CLIPTokenizer.from_pretrained(str(path), local_files_only=True) for path in paths]
        metadata = {
            "max_length": 77,
            "special_tokens_included": True,
            "tokenizer_sha256": tokenizer_fingerprint(paths[0]),
            "tokenizer_2_sha256": tokenizer_fingerprint(paths[1]),
        }
        return cls(*tokenizers, metadata=metadata)

    def count(self, text: str) -> TokenCounts:
        if not isinstance(text, str):
            raise TypeError("Caption must be text.")
        counts = [len(tokenizer(text, add_special_tokens=True, truncation=False,
                                padding=False, verbose=False)["input_ids"])
                  for tokenizer in (self.tokenizer_one, self.tokenizer_two)]
        return TokenCounts(*counts, self.max_length)

    def truncate(self, text: str) -> str:
        """Fit both CLIP windows using a prefix of the original caption."""
        if self.count(text).valid:
            return text

        word_ends = [match.end() for match in _WORD_SPAN.finditer(text)]
        end = self._fitting_prefix_end(text, word_ends)
        if not end:
            # An unbroken first span can still contain many BPE tokens. Keep
            # the original characters, without decoding a partial byte token.
            limit = word_ends[0] if word_ends else len(text)
            character_ends = [offset for offset in range(1, limit + 1)
                              if _can_end_unicode_prefix(text, offset)]
            end = self._fitting_prefix_end(text, character_ends)

        prefix = _caption_prefix(text, end)
        counts = self.count(prefix)
        if not prefix.strip() or not counts.valid:
            # Separator cleanup can empty an otherwise valid punctuation
            # caption or change the tokenization near the cutoff.
            prefix = text[:end]
            counts = self.count(prefix)
        if not prefix.strip() or not counts.valid:
            raise ValueError("No nonempty caption prefix fits both 77-token windows.")

        # Prefer a complete comma-delimited phrase near the limit. A distant
        # comma in a long sentence should not discard most of the useful text.
        comma = max(text.rfind(separator, 0, len(prefix) + 1) for separator in _COMMAS)
        if comma >= 0:
            complete = _caption_prefix(text, comma)
            if complete:
                complete_counts = self.count(complete)
                tokens_lost = max(counts.clip_l, counts.clip_g) - max(complete_counts.clip_l, complete_counts.clip_g)
                if complete_counts.valid and tokens_lost <= _COMMA_BOUNDARY_SLACK:
                    return complete
        return prefix

    def _fitting_prefix_end(self, text: str, ends: list[int]) -> int:
        first_overflow = bisect_left(ends, True, key=lambda end: not self.count(text[:end]).valid)
        return ends[first_overflow - 1] if first_overflow else 0


def _can_end_unicode_prefix(text: str, offset: int) -> bool:
    if offset == len(text):
        return True
    following = text[offset]
    return (text[offset - 1] != "\u200d" and following != "\u200d"
            and not unicodedata.category(following).startswith("M")
            and not "\U0001f3fb" <= following <= "\U0001f3ff")


def _caption_prefix(text: str, end: int) -> str:
    while end and (text[end - 1].isspace() or text[end - 1] in _COMMAS):
        end -= 1
    return text[:end]
