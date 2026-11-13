"""Non-destructive image preparation and sealed, verifiable dataset snapshots."""
from __future__ import annotations

from contextlib import suppress
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
import unicodedata
from typing import Any, Callable, Iterable

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
_REVISION_NAME = re.compile(r"Dataset-([1-9][0-9]*)\Z")
_BUCKET_SUFFIX = re.compile(r":([1-9][0-9]*)X([1-9][0-9]*)\Z")


def _named_buckets(revision: Path) -> list[tuple[int, int, Path]]:
    buckets = [(int(match.group(1)), int(match.group(2)), path) for path in revision.iterdir()
               if (match := _BUCKET_SUFFIX.search(path.name))]
    return sorted(buckets)


def validate_project_name(name: str) -> str:
    """Return a readable folder name without allowing paths or hidden controls."""
    if not isinstance(name, str):
        raise ValueError("Project name must be text.")
    if any(char in "/\\" or unicodedata.category(char).startswith("C") for char in name):
        raise ValueError("Project names cannot contain slashes, backslashes or control characters.")
    name = name.strip()
    if name in {"", ".", ".."}:
        raise ValueError("Enter a project name other than a dot or two dots.")
    if len(name) > 200 or len(name.encode("utf-8")) > 255:
        raise ValueError("Project name is too long. Use up to 200 characters and 255 UTF-8 bytes.")
    return name


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


def _verify_training_image(path: Path, width: int, height: int) -> None:
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (width, height):
            raise ValueError(f"Invalid training image: {path.name}")
        if image.size not in BUCKETS:
            raise ValueError("Training image is not a supported SDXL bucket.")


class _UpscalingRequired(Exception):
    """The selected fit would enlarge the image to its bucket."""


def _prepare_image(source: Path, destination: Path, fit: str, background: str,
                   *, allow_upscale: bool = True) -> dict[str, Any]:
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
        if fit not in {"crop", "pad"}:
            raise ValueError("Fit must be 'crop' or 'pad'.")
        scale = (max if fit == "crop" else min)(target[0] / rgb.width, target[1] / rgb.height)
        if not allow_upscale and scale > 1:
            raise _UpscalingRequired
        if fit == "crop":
            size = (max(target[0], round(rgb.width * scale)), max(target[1], round(rgb.height * scale)))
            resized = rgb.resize(size, Image.Resampling.LANCZOS)
            left, top = (size[0] - target[0]) // 2, (size[1] - target[1]) // 2
            processed = resized.crop((left, top, left + target[0], top + target[1]))
        else:
            size = (max(1, min(target[0], round(rgb.width * scale))),
                    max(1, min(target[1], round(rgb.height * scale))))
            resized = rgb.resize(size, Image.Resampling.LANCZOS)
            processed = Image.new("RGB", target, color)
            processed.paste(resized, ((target[0] - size[0]) // 2, (target[1] - size[1]) // 2))
            top, left = 0, 0
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
    caption_incomplete_reason: str = ""


class DatasetProject:
    def __init__(self, root: Path, entries: list[DatasetEntry], next_image_id: int | None = None):
        self.root = root
        self.entries = entries
        minimum = max((entry.id for entry in entries), default=0) + 1
        if next_image_id is not None and (type(next_image_id) is not int or next_image_id < minimum):
            raise ValueError("Invalid next image number in project.")
        self.next_image_id = next_image_id if next_image_id is not None else minimum

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
        project = cls(root, entries, data.get("next_image_id"))
        project._validate_entries()
        return project

    def _validate_entries(self) -> None:
        if self.image_dir.is_symlink() or not self.image_dir.is_dir():
            raise ValueError("Project images folder must be a real directory.")
        ids = set()
        for entry in self.entries:
            if not isinstance(entry.caption_incomplete_reason, str):
                raise ValueError("Invalid incomplete-caption status in project.")
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
                                               "next_image_id": self.next_image_id,
                                               "entries": [asdict(entry) for entry in self.entries]})

    def import_images(self, paths: Iterable[str | Path], fit: str = "crop", background: str = "#000000",
                      recursive: bool = False, *, allow_upscale: bool = True,
                      on_skip: Callable[[Path], None] | None = None) -> list[DatasetEntry]:
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
        next_id = max(self.next_image_id, max(numbers, default=0) + 1)
        imported: list[DatasetEntry] = []
        moved: list[Path] = []
        previous = list(self.entries)
        previous_next_id = self.next_image_id
        with tempfile.TemporaryDirectory(prefix=".import-", dir=self.root) as temporary:
            stage = Path(temporary)
            for source in files:
                entry_id = next_id + len(imported)
                image, caption = f"Img-{entry_id}.png", f"Img-{entry_id}.txt"
                if (self.image_dir / image).exists() or (self.image_dir / caption).exists():
                    raise ValueError("Image numbering collision; nothing was replaced.")
                try:
                    metadata = _prepare_image(source, stage / image, fit, background, allow_upscale=allow_upscale)
                except _UpscalingRequired:
                    if on_skip is not None:
                        on_skip(source)
                    continue
                (stage / caption).write_text("", encoding="utf-8")
                imported.append(DatasetEntry(id=entry_id, image=image, caption=caption,
                                             source=str(source), image_sha256=sha256(stage / image), **metadata))
            if not imported:
                return []
            try:
                for entry in imported:
                    for name in (entry.image, entry.caption):
                        target = _safe_child(self.image_dir, name)
                        # Hard-link publication is atomic and refuses to replace an existing file.
                        os.link(stage / name, target)
                        moved.append(target)
                self.entries.extend(imported)
                self.next_image_id = next_id + len(imported)
                self.save()
            except BaseException:
                self.entries = previous
                self.next_image_id = previous_next_id
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
        entry = self._entry(entry_id)
        path = _safe_child(self.image_dir, entry.caption)
        changed = path.read_text(encoding="utf-8") != text
        if changed:
            _atomic_text(path, text)
            self._set_caption_incomplete(entry, "")

    def _caption_entries(self, image_ids: Iterable[int] | None) -> list[DatasetEntry]:
        if image_ids is None:
            return self.entries
        selected = set()
        known = {entry.id for entry in self.entries}
        for image_id in image_ids:
            if type(image_id) is not int or image_id not in known:
                raise ValueError(f"Unknown image number: {image_id}")
            selected.add(image_id)
        return [entry for entry in self.entries if entry.id in selected]

    def count_empty_captions(self, *, image_ids: Iterable[int] | None = None) -> int:
        return sum(
            not _safe_child(self.image_dir, entry.caption).read_text(encoding="utf-8").strip()
            for entry in self._caption_entries(image_ids)
        )

    def apply_bulk_caption(self, text: str, *, prepend: bool = False, only_empty: bool = False,
                           append: bool = False, image_ids: Iterable[int] | None = None) -> int:
        """Apply manual text to all or selected draft captions."""
        if not isinstance(text, str) or "\0" in text:
            raise ValueError("Caption must be UTF-8 text without NUL characters.")
        if sum((prepend, only_empty, append)) > 1:
            raise ValueError("Choose one bulk caption operation.")
        text = text.strip()
        if prepend:
            text = text.rstrip(",").rstrip()
        if append:
            text = re.sub(r"^[,\s]+|[,\s]+$", "", text)
        if not text:
            raise ValueError("Enter caption text to apply to the draft.")

        def edit(entry, current):
            stripped = current.strip()
            if only_empty and stripped:
                return current, entry.caption_incomplete_reason
            if prepend and stripped:
                existing = current.lstrip()
                if stripped == text or existing.startswith(text + ","):
                    return current, entry.caption_incomplete_reason
                return f"{text}, {existing}", entry.caption_incomplete_reason
            if append and stripped:
                existing = re.sub(r"[,\s]+$", "", current)
                prefix = existing[:-len(text)].rstrip()
                if existing.endswith(text) and (not prefix or prefix.endswith(",")):
                    return current, entry.caption_incomplete_reason
                updated = f"{existing}, {text}" if existing else text
                return updated, entry.caption_incomplete_reason
            return text, ""

        return self._edit_captions(edit, image_ids)

    def truncate_captions(self, counter: TokenCounter, *, image_ids: Iterable[int] | None = None) -> int:
        """Trim oversized draft captions to one CLIP window."""
        def edit(entry, current):
            updated = counter.truncate(current) if current.strip() else current
            return updated, entry.caption_incomplete_reason

        return self._edit_captions(edit, image_ids)

    def _edit_captions(self, edit: Callable[[DatasetEntry, str], tuple[str, str]],
                       image_ids: Iterable[int] | None) -> int:
        entries = self._caption_entries(image_ids)
        if not entries:
            return 0
        self._validate_entries()
        updates = []
        for entry in entries:
            path = _safe_child(self.image_dir, entry.caption)
            current = path.read_text(encoding="utf-8")
            updated, reason = edit(entry, current)
            if updated != current or reason != entry.caption_incomplete_reason:
                updates.append((entry, path, updated if updated != current else None, reason))
        if not updates:
            return 0

        stage = Path(tempfile.mkdtemp(prefix=".bulk-caption-", dir=self.root))
        published = []
        previous_reasons = []
        keep_stage = False
        try:
            for _, path, updated, _ in updates:
                if updated is not None:
                    _atomic_text(stage / f"new-{path.name}", updated)
                    os.link(path, stage / path.name, follow_symlinks=False)
            for entry, path, updated, reason in updates:
                if updated is not None:
                    os.replace(stage / f"new-{path.name}", path)
                    published.append(path)
                if reason != entry.caption_incomplete_reason:
                    previous_reasons.append((entry, entry.caption_incomplete_reason))
                    entry.caption_incomplete_reason = reason
            if previous_reasons:
                self.save()
        except BaseException as exc:
            for entry, reason in previous_reasons:
                entry.caption_incomplete_reason = reason
            try:
                for path in reversed(published):
                    os.replace(stage / path.name, path)
            except BaseException as rollback_exc:
                keep_stage = True
                raise RuntimeError(f"Caption update failed and recovery files remain in {stage}: {rollback_exc}") from exc
            raise
        finally:
            if not keep_stage:
                shutil.rmtree(stage)
        return len(updates)

    def _set_caption_incomplete(self, entry: DatasetEntry, reason: str) -> None:
        old = entry.caption_incomplete_reason
        if old == reason:
            return
        entry.caption_incomplete_reason = reason
        try:
            self.save()
        except BaseException:
            entry.caption_incomplete_reason = old
            raise

    def save_generated_caption(self, entry_id: int, text: str, incomplete_reason: str = "") -> None:
        """Save generated text without ever silently losing its partial status."""
        if not isinstance(text, str) or "\0" in text or not isinstance(incomplete_reason, str):
            raise ValueError("Caption and status must be text without NUL characters.")
        entry = self._entry(entry_id)
        path = _safe_child(self.image_dir, entry.caption)
        # Publish the warning first. If writing the caption is interrupted,
        # an extra warning is safe; an unmarked incomplete caption is not.
        if incomplete_reason:
            self._set_caption_incomplete(entry, incomplete_reason)
        _atomic_text(path, text)
        if not incomplete_reason:
            self._set_caption_incomplete(entry, "")

    def incomplete_captions(self) -> list[DatasetEntry]:
        return [entry for entry in self.entries if entry.caption_incomplete_reason]

    def _finalized_folder(self) -> Path:
        folder = self.root / "finalized"
        if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
            raise ValueError("The project's finalized folder must be a real directory, not a symbolic link or file.")
        folder.mkdir(exist_ok=True)
        if folder.is_symlink() or not folder.is_dir() or folder.resolve().parent != self.root.resolve():
            raise ValueError("The project's finalized folder must be a real directory inside the project.")
        return folder

    @staticmethod
    def _next_revision(folder: Path) -> Path:
        entries = list(folder.iterdir())
        if any(path.is_symlink() for path in entries):
            raise ValueError("The finalized folder cannot contain symbolic links.")
        numbers = [int(match.group(1)) for path in entries if (match := _REVISION_NAME.fullmatch(path.name))]
        return folder / f"Dataset-{max(numbers, default=0) + 1}"

    def finalized_datasets(self, *, include_all: bool = False) -> tuple[Path, ...]:
        """List selectable snapshots and optional multi-bucket revisions, newest first."""
        folder = self.root / "finalized"
        if folder.is_symlink() or not folder.is_dir():
            return ()
        revisions = [(int(match.group(1)), path) for path in folder.iterdir()
                     if (match := _REVISION_NAME.fullmatch(path.name)) and not path.is_symlink() and path.is_dir()]
        datasets = []
        for _, revision in sorted(revisions, reverse=True):
            revision_manifest = revision / "manifest.json"
            if not revision_manifest.is_symlink() and revision_manifest.is_file():
                datasets.append(revision)
                continue
            named = _named_buckets(revision)
            buckets = []
            for _, _, path in named:
                if path.is_symlink() or not path.is_dir():
                    continue
                manifest = path / "manifest.json"
                if not manifest.is_symlink() and manifest.is_file():
                    buckets.append(path)
            if (include_all and len(buckets) >= 2 and len(buckets) == len(named)
                    and not revision_manifest.exists() and not revision_manifest.is_symlink()):
                datasets.append(revision)
            datasets.extend(buckets)
        return tuple(datasets)

    def finalize_in_project(self, token_counter: TokenCounter) -> Path:
        """Create the next sealed snapshot in this project's finalized folder.

        The editable draft remains available for further imports and edits. Each
        finalization publishes a new numbered snapshot so an existing training
        run can continue referring to the exact pixels and captions it used.
        """
        if token_counter is None:
            raise ValueError("Load the local SDXL tokenizers before finalizing a dataset.")
        self._validate_entries()
        if not self.entries:
            raise ValueError("Import and caption at least one image before finalizing.")
        folder = self._finalized_folder()

        while True:
            destination = self._next_revision(folder)
            try:
                return self.finalize(destination, token_counter)
            except FileExistsError as exc:
                # An independent finalization can claim this name after our scan.
                # Retry only that collision; never suppress unrelated I/O errors.
                if exc.filename is None or Path(exc.filename) != destination:
                    raise
            except ValueError as exc:
                # A competing finalizer can also win before finalize's initial
                # existence check. No existing snapshot is changed.
                if not str(exc).startswith("Export requires a new folder;") or not destination.exists():
                    raise

    def finalize_buckets_in_project(self, token_counter: TokenCounter) -> tuple[Path, ...]:
        """Publish one revision containing a sealed training dataset per populated bucket."""
        self._validate_finalization(token_counter)
        folder = self._finalized_folder()
        buckets: dict[tuple[int, int], list[DatasetEntry]] = {}
        for entry in self.entries:
            bucket = (entry.width, entry.height)
            if bucket not in BUCKETS:
                raise ValueError(f"{entry.image} has an unsupported SDXL bucket.")
            buckets.setdefault(bucket, []).append(entry)

        name_limit = os.pathconf(folder, "PC_NAME_MAX")
        names = []
        with tempfile.TemporaryDirectory(prefix=".dataset-", dir=folder) as temporary:
            stage = Path(temporary)
            for (width, height), entries in sorted(buckets.items()):
                suffix = f":{width}X{height}"
                stem = self.root.name
                # Leave the full resolution suffix intact on filesystems with
                # component limits, even for names with multibyte characters.
                while stem and 0 < name_limit < len(os.fsencode(stem + suffix)):
                    stem = stem[:-1]
                name = stem + suffix
                bucket_stage = stage / name
                bucket_stage.mkdir()
                self._write_snapshot(bucket_stage, entries, token_counter)
                names.append(name)

            while True:
                destination = self._next_revision(folder)
                try:
                    destination.mkdir()
                except FileExistsError:
                    continue
                try:
                    # Reserve an empty directory exclusively, then replace it in
                    # one rename so readers see all sealed buckets together.
                    stage.rename(destination)
                except BaseException:
                    with suppress(OSError):
                        destination.rmdir()
                    raise
                return tuple(destination / name for name in names)

    def _validate_finalization(self, token_counter: TokenCounter) -> None:
        if token_counter is None:
            raise ValueError("Load the local SDXL tokenizers before finalizing a dataset.")
        self._validate_entries()
        if not self.entries:
            raise ValueError("Import and caption at least one image before finalizing.")
        incomplete = self.incomplete_captions()
        if incomplete:
            names = "\n".join(f"{entry.image}: {entry.caption_incomplete_reason}" for entry in incomplete)
            raise ValueError("Review and edit or re-caption these incomplete captions before finalizing:\n\n" + names)

    def _write_snapshot(self, stage: Path, entries: Iterable[DatasetEntry], token_counter: TokenCounter) -> None:
        rows = []
        for entry in entries:
            image = _safe_child(self.image_dir, entry.image)
            if sha256(image) != entry.image_sha256:
                raise ValueError(f"{entry.image} changed after import. Reimport and recaption it.")
            caption = _safe_child(self.image_dir, entry.caption).read_text(encoding="utf-8")
            if not caption.strip():
                raise ValueError(f"{entry.caption} is empty. Review every caption before finalizing.")
            counts = token_counter.count(caption)
            copy = stage / entry.image
            shutil.copyfile(image, copy)
            image_hash = sha256(copy)
            if image_hash != entry.image_sha256:
                raise ValueError(f"{entry.image} changed during export. Try again after reviewing it.")
            _verify_training_image(copy, entry.width, entry.height)
            (stage / entry.caption).write_text(caption, encoding="utf-8", newline="\n")
            row = asdict(entry)
            row.pop("source")  # A portable export must not disclose private source paths.
            row.pop("caption_incomplete_reason")
            row.update(image_sha256=image_hash, caption_sha256=sha256(stage / entry.caption),
                       tokens=counts.to_dict())
            rows.append(row)
        manifest = {"schema_version": _SCHEMA, "kind": "sdxl-trainer-dataset", "finalized": True,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "tokenizers": token_counter.metadata, "entries": rows}
        _write_json(stage / "manifest.json", manifest)

    def finalize(self, destination: str | Path, token_counter: TokenCounter,
                 allow_long_captions: bool = True) -> Path:
        """Copy reviewed pixels and complete captions into a hashed snapshot.

        SDXL training encodes long captions in multiple native CLIP windows. The
        retained argument is accepted for compatibility with earlier callers.
        """
        self._validate_finalization(token_counter)
        destination = Path(destination).expanduser().resolve()
        if destination.exists():
            raise ValueError("Export requires a new folder; finalized datasets are never overwritten.")
        if destination == self.root or destination in self.root.parents or self.image_dir in destination.parents:
            raise ValueError("Export to a new dataset folder outside the working images folder.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            self._write_snapshot(stage, self.entries, token_counter)
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
    manifest, _ = _verify_snapshot(root)
    return manifest


def _verify_snapshot(root: Path) -> tuple[dict[str, Any], str]:
    manifest_path = _safe_child(root, "manifest.json")
    encoded = manifest_path.read_bytes()
    manifest = json.loads(encoded.decode("utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != _SCHEMA
            or manifest.get("kind") != "sdxl-trainer-dataset" or manifest.get("finalized") is not True
            or not isinstance(manifest.get("entries"), list) or not manifest["entries"]):
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
        _verify_training_image(root / entry["image"], entry["width"], entry["height"])
        caption = (root / entry["caption"]).read_text(encoding="utf-8")
        tokens = entry["tokens"]
        if not caption.strip() or not isinstance(tokens, dict) or any(
            type(tokens.get(k)) is not int or tokens[k] < 2 for k in ("clip_l", "clip_g")
        ):
            raise ValueError("Dataset has an empty caption or invalid token counts.")
    return manifest, hashlib.sha256(encoded).hexdigest()


def load_training_dataset(root: str | Path) -> tuple[dict[str, Any], str]:
    """Verify one sealed snapshot or read a revision's buckets as one dataset.

    The combined rows use paths relative to the revision and exist only in memory.
    A single snapshot keeps its manifest fingerprint for existing checkpoints.
    """
    root = Path(root).expanduser()
    manifest_path = root / "manifest.json"
    if not _REVISION_NAME.fullmatch(root.name) or manifest_path.exists() or manifest_path.is_symlink():
        return _verify_snapshot(root.resolve(strict=True))

    if root.is_symlink() or not root.is_dir():
        raise ValueError("The finalized dataset revision must be a real directory.")
    root = root.resolve(strict=True)
    buckets = _named_buckets(root)
    if not buckets:
        raise ValueError(f"No finalized aspect buckets in {root.name}. Select a sealed dataset or export revision.")

    tokenizers = None
    entries = []
    seen = set()
    identities = []
    for width, height, path in buckets:
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Aspect bucket must be a real directory: {path.name}")
        try:
            manifest, fingerprint = _verify_snapshot(_safe_child(root, path.name))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f"Invalid finalized aspect bucket {path.name}: {exc}") from exc
        recorded = manifest.get("tokenizers")
        if not isinstance(recorded, dict) or (tokenizers is not None and tokenizers != recorded):
            raise ValueError(f"Tokenizer metadata differs or is missing in aspect bucket {path.name}.")
        tokenizers = recorded
        identities.append((path.name, fingerprint))
        for entry in manifest["entries"]:
            if (entry["width"], entry["height"]) != (width, height):
                raise ValueError(f"Image dimensions do not match aspect bucket {path.name}.")
            if entry["id"] in seen:
                raise ValueError(f"Duplicate image number across aspect buckets: {entry['id']}")
            seen.add(entry["id"])
            entries.append({**entry, "image": str(Path(path.name) / entry["image"]),
                            "caption": str(Path(path.name) / entry["caption"])})

    digest = hashlib.sha256(b"sdxl-trainer-bucket-revision-v1\0")
    digest.update(json.dumps(identities, separators=(",", ":")).encode("ascii"))
    return {"schema_version": _SCHEMA, "kind": "sdxl-trainer-dataset", "finalized": True,
            "tokenizers": tokenizers, "entries": entries}, digest.hexdigest()
