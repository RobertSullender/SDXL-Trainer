"""Non-destructive image preparation and sealed, verifiable dataset snapshots."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable

from PIL import Image, ImageCms, ImageColor, ImageOps

from .tokens import TokenCounter

# Conventional SDXL ~one-megapixel aspect buckets. Both dimensions divide by 64.
BUCKETS: tuple[tuple[int, int], ...] = (
    (512, 2048), (576, 1792), (640, 1664), (704, 1472), (768, 1344),
    (832, 1216), (896, 1152), (960, 1088), (1024, 1024),
    (1088, 960), (1152, 896), (1216, 832), (1344, 768), (1472, 704),
    (1664, 640), (1792, 576), (2048, 512),
)
SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"})
_SCHEMA = 1
_NAME = re.compile(r"Img-([1-9][0-9]*)\.png\Z")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nearest_bucket(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    ratio = width / height
    return min(BUCKETS, key=lambda b: abs(math.log(ratio / (b[0] / b[1]))))


def _atomic_text(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _safe_child(root: Path, filename: str) -> Path:
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise ValueError(f"Unsafe dataset filename: {filename!r}")
    path = root / filename
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise ValueError(f"Dataset file must be a regular local file: {filename}")
    return path


def _prepare_image(source: Path, destination: Path, fit: str, background: str) -> dict[str, Any]:
    color = ImageColor.getrgb(background)
    if len(color) != 3:
        raise ValueError("Background must be an opaque RGB color.")
    with Image.open(source) as opened:
        if getattr(opened, "n_frames", 1) != 1:
            raise ValueError(f"Animated/multi-page input is not supported: {source.name}")
        opened.load()
        profile = opened.info.get("icc_profile")
        oriented = ImageOps.exif_transpose(opened)
        original_size = [oriented.width, oriented.height]
        alpha = oriented.getchannel("A") if "A" in oriented.getbands() else None
        if oriented.mode == "P" and "transparency" in oriented.info:
            alpha = oriented.convert("RGBA").getchannel("A")
        if profile:
            try:
                color_image = oriented.convert("RGB") if oriented.mode in {"RGBA", "P"} else oriented
                rgb = ImageCms.profileToProfile(color_image, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                                                ImageCms.createProfile("sRGB"), outputMode="RGB")
            except (ImageCms.PyCMSError, OSError, ValueError) as exc:
                raise ValueError(f"Cannot safely convert embedded color profile in {source.name}: {exc}") from exc
        else:
            rgb = oriented.convert("RGB")
        if alpha is not None:
            canvas = Image.new("RGB", rgb.size, color)
            canvas.paste(rgb, mask=alpha)
            rgb = canvas
        target = nearest_bucket(*rgb.size)
        if fit == "crop":
            scale = max(target[0] / rgb.width, target[1] / rgb.height)
            size = (max(target[0], round(rgb.width * scale)), max(target[1], round(rgb.height * scale)))
            resized = rgb.resize(size, Image.Resampling.LANCZOS)
            left, top = (size[0] - target[0]) // 2, (size[1] - target[1]) // 2
            processed = resized.crop((left, top, left + target[0], top + target[1]))
        elif fit == "pad":
            scale = min(target[0] / rgb.width, target[1] / rgb.height)
            size = (max(1, min(target[0], round(rgb.width * scale))),
                    max(1, min(target[1], round(rgb.height * scale))))
            resized = rgb.resize(size, Image.Resampling.LANCZOS)
            processed = Image.new("RGB", target, color)
            processed.paste(resized, ((target[0] - size[0]) // 2, (target[1] - size[1]) // 2))
            top, left = 0, 0
        else:
            raise ValueError("Fit must be 'crop' or 'pad'.")
        # A fresh pixel-only image prevents EXIF, text, ICC, GPS and other metadata leakage.
        clean = Image.frombytes("RGB", processed.size, processed.tobytes())
        clean.save(destination, "PNG", compress_level=6)
        return {"width": target[0], "height": target[1], "original_size": original_size,
                "crop_top_left": [top, left], "fit": fit, "background": background}


@dataclass
class DatasetEntry:
    id: int
    image: str
    caption: str
    width: int
    height: int
    original_size: list[int]
    crop_top_left: list[int]
    fit: str
    source: str
    image_sha256: str
    background: str = "#000000"


class DatasetProject:
    def __init__(self, root: Path, entries: list[DatasetEntry]):
        self.root = root
        self.entries = entries

    @property
    def image_dir(self) -> Path:
        return self.root / "images"

    @classmethod
    def create(cls, root: str | Path) -> "DatasetProject":
        root = Path(root).expanduser().resolve()
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise ValueError("Create a project in a new or empty folder; existing files are never replaced.")
        root.mkdir(parents=True, exist_ok=True)
        (root / "images").mkdir()
        project = cls(root, [])
        project.save()
        return project

    @classmethod
    def open(cls, root: str | Path) -> "DatasetProject":
        root = Path(root).expanduser().resolve(strict=True)
        data = json.loads((root / "project.json").read_text(encoding="utf-8"))
        if data.get("schema_version") != _SCHEMA or data.get("kind") != "sdxl-trainer-draft":
            raise ValueError("This folder is not a supported SDXL-Trainer draft project.")
        entries = [DatasetEntry(**item) for item in data["entries"]]
        project = cls(root, entries)
        project._validate_entries()
        return project

    def _validate_entries(self) -> None:
        if self.image_dir.is_symlink() or not self.image_dir.is_dir():
            raise ValueError("Project images folder must be a real directory.")
        ids = set()
        for entry in self.entries:
            if entry.id in ids or entry.id <= 0:
                raise ValueError("Duplicate or invalid image number in project.")
            ids.add(entry.id)
            if entry.image != f"Img-{entry.id}.png" or entry.caption != f"Img-{entry.id}.txt":
                raise ValueError("Image and caption names do not match their image number.")
            for filename in (entry.image, entry.caption):
                if not _safe_child(self.image_dir, filename).is_file():
                    raise ValueError(f"Missing project file: {filename}")

    def save(self) -> None:
        _write_json(self.root / "project.json", {"schema_version": _SCHEMA, "kind": "sdxl-trainer-draft",
                                               "entries": [asdict(entry) for entry in self.entries]})

    def import_images(self, paths: Iterable[str | Path], fit: str = "crop", background: str = "#000000",
                      recursive: bool = False) -> list[DatasetEntry]:
        if fit not in {"crop", "pad"}:
            raise ValueError("Fit must be 'crop' or 'pad'.")
        self._validate_entries()
        files: list[Path] = []
        for value in paths:
            path = Path(value).expanduser().resolve(strict=True)
            if path == self.root or self.root in path.parents:
                raise ValueError("Import original files from outside the working project.")
            if path.is_dir():
                iterator = path.rglob("*") if recursive else path.iterdir()
                files.extend(sorted(p.resolve() for p in iterator
                                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                                    and self.root not in p.resolve().parents))
            elif path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)
            else:
                raise ValueError(f"Unsupported image type: {path.name}")
        files = list(dict.fromkeys(files))
        if not files:
            raise ValueError("No supported image files were selected.")
        numbers = [entry.id for entry in self.entries]
        numbers.extend(int(match.group(1)) for p in self.image_dir.iterdir()
                       if (match := _NAME.fullmatch(p.name)))
        next_id = max(numbers, default=0) + 1
        imported: list[DatasetEntry] = []
        moved: list[Path] = []
        previous = list(self.entries)
        with tempfile.TemporaryDirectory(prefix=".import-", dir=self.root) as temporary:
            stage = Path(temporary)
            for offset, source in enumerate(files):
                entry_id = next_id + offset
                image, caption = f"Img-{entry_id}.png", f"Img-{entry_id}.txt"
                if (self.image_dir / image).exists() or (self.image_dir / caption).exists():
                    raise ValueError("Image numbering collision; nothing was replaced.")
                metadata = _prepare_image(source, stage / image, fit, background)
                (stage / caption).write_text("", encoding="utf-8")
                imported.append(DatasetEntry(id=entry_id, image=image, caption=caption,
                                             source=str(source), image_sha256=sha256(stage / image), **metadata))
            try:
                for entry in imported:
                    for name in (entry.image, entry.caption):
                        target = _safe_child(self.image_dir, name)
                        # Hard-link publication is atomic and refuses to replace an existing file.
                        os.link(stage / name, target)
                        moved.append(target)
                self.entries.extend(imported)
                self.save()
            except BaseException:
                self.entries = previous
                for path in moved:
                    path.unlink(missing_ok=True)
                raise
        return imported

    def _entry(self, entry_id: int) -> DatasetEntry:
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        raise ValueError(f"Unknown image number: {entry_id}")

    def image_path(self, entry_id: int) -> Path:
        return _safe_child(self.image_dir, self._entry(entry_id).image)

    def caption_text(self, entry_id: int) -> str:
        return _safe_child(self.image_dir, self._entry(entry_id).caption).read_text(encoding="utf-8")

    def save_caption(self, entry_id: int, text: str) -> None:
        if not isinstance(text, str) or "\0" in text:
            raise ValueError("Caption must be UTF-8 text without NUL characters.")
        _atomic_text(_safe_child(self.image_dir, self._entry(entry_id).caption), text)

    def finalize(self, destination: str | Path, token_counter: TokenCounter,
                 allow_long_captions: bool = False) -> Path:
        """Copy reviewed pixels and captions into a new, content-hashed snapshot.

        Long-caption mode is deliberately unsupported: the initial training pipeline
        uses CLIP's native 77-token window and must never silently discard words.
        """
        if token_counter is None:
            raise ValueError("Load the local SDXL tokenizers before finalizing a dataset.")
        if allow_long_captions:
            raise ValueError("This pipeline uses native 77-token CLIP captions; long captions must be edited.")
        self._validate_entries()
        if not self.entries:
            raise ValueError("Import and caption at least one image before finalizing.")
        destination = Path(destination).expanduser().resolve()
        if destination.exists():
            raise ValueError("Export requires a new folder; finalized datasets are never overwritten.")
        if destination == self.root or destination in self.root.parents or self.image_dir in destination.parents:
            raise ValueError("Export to a new dataset folder outside the working images folder.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            rows = []
            for entry in self.entries:
                image = self.image_path(entry.id)
                if sha256(image) != entry.image_sha256:
                    raise ValueError(f"{entry.image} changed after import. Reimport and recaption it.")
                caption = self.caption_text(entry.id)
                if not caption.strip():
                    raise ValueError(f"{entry.caption} is empty. Review every caption before finalizing.")
                counts = token_counter.count(caption)
                if not counts.valid:
                    raise ValueError(f"{entry.caption}: CLIP-L {counts.clip_l}/77; CLIP-G {counts.clip_g}/77. Shorten the caption.")
                shutil.copyfile(image, stage / entry.image)
                if sha256(stage / entry.image) != entry.image_sha256:
                    raise ValueError(f"{entry.image} changed during export. Try again after reviewing it.")
                (stage / entry.caption).write_text(caption, encoding="utf-8", newline="\n")
                row = asdict(entry)
                row.pop("source")  # A portable export must not disclose private source paths.
                row.update(image_sha256=sha256(stage / entry.image),
                           caption_sha256=sha256(stage / entry.caption), tokens=counts.to_dict())
                rows.append(row)
            manifest = {"schema_version": _SCHEMA, "kind": "sdxl-trainer-dataset", "finalized": True,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "tokenizers": token_counter.metadata, "entries": rows}
            _write_json(stage / "manifest.json", manifest)
            # rename refuses a nonempty destination; create the directory exclusively
            # first and publish individual files only after the complete stage is ready.
            destination.mkdir()
            try:
                # Publish the sealing manifest last. A partial/crashed export cannot
                # pass verify_finalized until every immutable file is present.
                for path in sorted(stage.iterdir(), key=lambda p: p.name == "manifest.json"):
                    os.replace(path, destination / path.name)
            except BaseException:
                shutil.rmtree(destination)
                raise
            return destination
        finally:
            shutil.rmtree(stage, ignore_errors=True)


def verify_finalized(root: str | Path) -> dict[str, Any]:
    """Validate every sealed file before training; do not trust a flag alone."""
    root = Path(root).expanduser().resolve(strict=True)
    manifest_path = _safe_child(root, "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != _SCHEMA or manifest.get("kind") != "sdxl-trainer-dataset"
            or manifest.get("finalized") is not True or not manifest.get("entries")):
        raise ValueError("Select a finalized SDXL-Trainer dataset containing manifest.json.")
    seen: set[int] = set()
    for entry in manifest["entries"]:
        number = entry["id"]
        if not isinstance(number, int) or number <= 0 or number in seen:
            raise ValueError("Finalized manifest contains duplicate or invalid image numbers.")
        seen.add(number)
        if entry["image"] != f"Img-{number}.png" or entry["caption"] != f"Img-{number}.txt":
            raise ValueError("Finalized manifest contains inconsistent filenames.")
        for kind in ("image", "caption"):
            path = _safe_child(root, entry[kind])
            if not path.is_file() or sha256(path) != entry[f"{kind}_sha256"]:
                raise ValueError(f"Dataset integrity check failed: {entry[kind]}")
        for key in ("original_size", "crop_top_left"):
            value = entry.get(key)
            minimum = 1 if key == "original_size" else 0
            if (not isinstance(value, list) or len(value) != 2
                    or any(type(v) is not int or v < minimum for v in value)):
                raise ValueError(f"Invalid SDXL conditioning metadata: {key}")
        if entry.get("fit") not in {"crop", "pad"}:
            raise ValueError("Invalid image preparation mode in dataset.")
        with Image.open(root / entry["image"]) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (entry["width"], entry["height"]):
                raise ValueError(f"Invalid training image: {entry['image']}")
            if image.size not in BUCKETS:
                raise ValueError("Training image is not a supported SDXL bucket.")
        caption = (root / entry["caption"]).read_text(encoding="utf-8")
        tokens = entry["tokens"]
        if not caption.strip() or any(not isinstance(tokens[k], int) or not 2 <= tokens[k] <= 77
                                      for k in ("clip_l", "clip_g")):
            raise ValueError("Dataset has empty or over-limit captions.")
    return manifest
