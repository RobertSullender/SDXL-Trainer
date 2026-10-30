"""Exact, offline SDXL caption counts for its two local CLIP tokenizers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TokenCounts:
    clip_l: int
    clip_g: int
    max_length: int = 77

    @property
    def valid(self) -> bool:
        return self.clip_l <= self.max_length and self.clip_g <= self.max_length

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
                                padding=False)["input_ids"])
                  for tokenizer in (self.tokenizer_one, self.tokenizer_two)]
        return TokenCounts(*counts, self.max_length)
