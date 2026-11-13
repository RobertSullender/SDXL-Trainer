"""Explicit, opt-in Hugging Face downloads; the rest of the application stays offline.

Repository inspection pins a full commit before selected files are downloaded.
No Hub Python client, implicit credential discovery, telemetry, or executable code.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import threading
from urllib.parse import quote, urljoin, urlsplit


class DownloadError(RuntimeError):
    pass


class DownloadCancelled(DownloadError):
    pass


SDXL_TRAINING_FILES = (
    "model_index.json", "scheduler/scheduler_config.json",
    "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json", "text_encoder/model.safetensors",
    "text_encoder_2/config.json", "text_encoder_2/model.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
    "tokenizer/merges.txt", "tokenizer/vocab.json", "tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json",
    "tokenizer_2/merges.txt", "tokenizer_2/vocab.json", "tokenizer_2/tokenizer_config.json", "tokenizer_2/special_tokens_map.json",
)

# Diffusers stores a model variant beside its canonical safetensors using the
# component-specific `*.fp16.safetensors` names. Configuration and tokenizer
# files are shared with the canonical model.
SDXL_FP16_TRAINING_FILES = tuple(
    name.removesuffix(".safetensors") + ".fp16.safetensors"
    if name.endswith(".safetensors") else name
    for name in SDXL_TRAINING_FILES
)


PRESETS = [
    {"name": "HauhauCS · Qwen3.5-9B Uncensored · Q8_0", "kind": "caption",
     "repo_id": "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive",
     "model_filename": "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf",
     "mmproj_filename": "mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf"},
    {"name": "LM Studio Community · Qwen3.5-9B · Q8_0", "kind": "caption",
     "repo_id": "lmstudio-community/Qwen3.5-9B-GGUF",
     "model_filename": "Qwen3.5-9B-Q8_0.gguf",
     "mmproj_filename": "mmproj-Qwen3.5-9B-BF16.gguf"},
    {"name": "Stability AI · SDXL Base 1.0 · Diffusers FP32", "kind": "sdxl",
     "repo_id": "stabilityai/stable-diffusion-xl-base-1.0", "filenames": list(SDXL_TRAINING_FILES),
     "model_variant": "default"},
    {"name": "Stability AI · SDXL Base 1.0 · Diffusers FP16 variant", "kind": "sdxl",
     "repo_id": "stabilityai/stable-diffusion-xl-base-1.0", "filenames": list(SDXL_FP16_TRAINING_FILES),
     "destination_suffix": "--fp16", "model_variant": "fp16"},
]


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str | None = None
    git_blob_sha1: str | None = None


@dataclass(frozen=True)
class RepositorySnapshot:
    repo_id: str
    revision: str
    files: list[RemoteFile]


def model_library_destination(library: str | Path, repo_id: str, preset: dict | None = None) -> Path:
    """Keep preset variants apart so adding files cannot change an existing model fingerprint."""
    repo = normalize_repository(repo_id)
    suffix = preset.get("destination_suffix", "") if preset and preset.get("repo_id") == repo else ""
    if not re.fullmatch(r"(?:--[A-Za-z0-9_-]+)?", suffix):
        raise DownloadError("Invalid model preset destination suffix.")
    return Path(library).expanduser() / (repo.replace("/", "--") + suffix)


def selected_training_files(snapshot: RepositorySnapshot) -> list[str]:
    """Select canonical Diffusers FP32 SDXL files, with no duplicate weight formats.

    This intentionally requires the standard base SDXL layout. Missing components
    produce an actionable error instead of downloading a partial training model.
    """
    available = {entry.path for entry in snapshot.files}
    missing = set(SDXL_TRAINING_FILES) - available
    if missing:
        raise DownloadError("The repository lacks required canonical SDXL Diffusers files: " + ", ".join(sorted(missing)))
    selected = list(SDXL_TRAINING_FILES)
    if "LICENSE.md" in available:
        selected.append("LICENSE.md")
    return selected


def normalize_repository(value: str) -> str:
    value = value.strip()
    if "://" in value:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or parsed.hostname != "huggingface.co" or parsed.username or parsed.port:
            raise DownloadError("Use a https://huggingface.co/owner/model link or owner/model repository ID.")
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2 or (len(parts) > 2 and parts[2] not in ("tree", "blob", "resolve")):
            raise DownloadError("This is not a Hugging Face model repository URL.")
        value = "/".join(parts[:2])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value) or ".." in value:
        raise DownloadError("Repository ID must have the form owner/model.")
    return value


def _safe_filename(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise DownloadError("The repository contains an unsafe file path.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise DownloadError("The repository contains an unsafe file path.")
    return value


def _cancel(event):
    if event is not None and event.is_set():
        raise DownloadCancelled("Download cancelled. Partial files are kept for verified resume.")


class HuggingFaceDownloader:
    """Constructing this object makes no network request.

    Set allow_network=True only for the user's explicit model download action.
    The optional token stays in memory and goes only to huggingface.co, never CDN
    redirects. It is not read from environment/configuration or written to disk.
    """

    def __init__(self, allow_network: bool = False, token: str | None = None, timeout: int = 30):
        if token and ("\r" in token or "\n" in token):
            raise DownloadError("Invalid authentication token.")
        self.allow_network = allow_network
        self._token = token
        self.timeout = timeout

    def _require_network(self):
        if not self.allow_network:
            raise DownloadError("Network access is disabled. Use the explicit model download action to opt in.")

    @contextmanager
    def _open(self, url: str, headers: dict | None = None):
        self._require_network()
        # Direct TLS connections deliberately avoid proxy/Hub environment settings.
        for _ in range(8):
            parsed = urlsplit(url)
            host = parsed.hostname or ""
            if (parsed.scheme != "https" or parsed.username or parsed.port
                    or not (host == "huggingface.co" or host.endswith(".huggingface.co") or host.endswith(".hf.co"))):
                raise DownloadError("The download redirected outside Hugging Face's HTTPS storage hosts.")
            request_headers = {"User-Agent": "SDXL-Trainer/0.1", "Accept-Encoding": "identity"}
            request_headers.update(headers or {})
            if host == "huggingface.co" and self._token:
                request_headers["Authorization"] = "Bearer " + self._token
            connection = http.client.HTTPSConnection(host, timeout=self.timeout)
            try:
                target = parsed.path + (("?" + parsed.query) if parsed.query else "")
                connection.request("GET", target, headers=request_headers)
                response = connection.getresponse()
                if response.status in (301, 302, 303, 307, 308):
                    location = response.getheader("Location")
                    if not location:
                        raise DownloadError("The download returned an empty redirect.")
                    url = urljoin(url, location)
                    continue
                yield response
                return
            except (OSError, http.client.HTTPException) as exc:
                # Signed redirect URLs and tokens never appear in user-facing errors.
                raise DownloadError(f"Hugging Face network request failed ({type(exc).__name__}). Retry to resume.") from None
            finally:
                connection.close()
        raise DownloadError("Too many Hugging Face redirects.")

    def inspect(self, repository: str, revision: str = "main", cancel_event: threading.Event | None = None) -> RepositorySnapshot:
        self._require_network()
        _cancel(cancel_event)
        repo_id = normalize_repository(repository)
        if not isinstance(revision, str) or not revision.strip() or len(revision) > 256:
            raise DownloadError("Select a valid branch, tag, or full commit revision.")
        url = f"https://huggingface.co/api/models/{repo_id}/revision/{quote(revision, safe='')}?blobs=true"
        with self._open(url) as response:
            if response.status != 200:
                raise DownloadError(f"Hugging Face returned HTTP {response.status}. Check the repository, revision, and access token for gated/private models.")
            raw = response.read(16 * 1024 * 1024 + 1)
            _cancel(cancel_event)
            if len(raw) > 16 * 1024 * 1024:
                raise DownloadError("The repository listing is too large.")
            try:
                data = json.loads(raw)
            except (ValueError, UnicodeError):
                raise DownloadError("Hugging Face returned an invalid repository listing.") from None
        commit = data.get("sha")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise DownloadError("The repository API did not return a full commit hash; refusing an unpinned download.")
        files = []
        for entry in data.get("siblings", []):
            filename = _safe_filename(entry.get("rfilename"))
            lfs = entry.get("lfs") or {}
            size = entry.get("size", lfs.get("size"))
            if type(size) is not int or size < 0:
                raise DownloadError(f"No verified size was supplied for {filename}.")
            sha = lfs.get("sha256")
            blob = entry.get("blobId")
            sha = sha if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) else None
            blob = blob if isinstance(blob, str) and re.fullmatch(r"[0-9a-f]{40}", blob) else None
            # For LFS the Git blob hashes the pointer, not the downloaded data.
            files.append(RemoteFile(filename, size, sha, None if lfs else blob))
        if len({entry.path for entry in files}) != len(files):
            raise DownloadError("The repository contains duplicate file paths.")
        return RepositorySnapshot(repo_id, commit, sorted(files, key=lambda entry: entry.path))

    @staticmethod
    def _verify(path: Path, entry: RemoteFile, cancel_event=None) -> bool:
        if not path.is_file() or path.stat().st_size != entry.size:
            return False
        if entry.sha256:
            digest = hashlib.sha256()
            expected = entry.sha256
        elif entry.git_blob_sha1:
            digest = hashlib.sha1(b"blob " + str(entry.size).encode("ascii") + b"\0")
            expected = entry.git_blob_sha1
        else:
            raise DownloadError(f"No integrity hash is available for {entry.path}; refusing the download.")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                _cancel(cancel_event)
                digest.update(chunk)
        return digest.hexdigest() == expected

    def download(self, snapshot: RepositorySnapshot, filenames: list[str], destination: str | Path,
                 progress=None, cancel_event: threading.Event | None = None) -> list[Path]:
        self._require_network()
        repo_id = normalize_repository(snapshot.repo_id)
        if not re.fullmatch(r"[0-9a-f]{40}", snapshot.revision):
            raise DownloadError("Downloads require the full pinned commit returned by repository inspection.")
        if not filenames:
            raise DownloadError("Select at least one repository file to download.")
        entries = {entry.path: entry for entry in snapshot.files}
        if len(set(filenames)) != len(filenames):
            raise DownloadError("A file was selected more than once.")
        root = Path(destination).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        outputs = []
        for filename in filenames:
            _cancel(cancel_event)
            _safe_filename(filename)
            if filename not in entries:
                raise DownloadError(f"{filename} is not in the inspected repository revision.")
            entry = entries[filename]
            if not entry.sha256 and not entry.git_blob_sha1:
                raise DownloadError(f"{filename} has no content hash; refusing an unverifiable download.")
            target = root / filename
            # Do not follow repository-path symlinks outside the selected model folder.
            if not target.resolve().is_relative_to(root) or target.is_symlink():
                raise DownloadError("Download destination contains an unsafe symbolic link.")
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            meta = target.with_name(target.name + ".part.json")
            if part.is_symlink() or meta.is_symlink():
                raise DownloadError("Download temporary files must not be symbolic links.")
            if target.exists():
                if self._verify(target, entry, cancel_event):
                    outputs.append(target)
                    if progress:
                        progress(filename, entry.size, entry.size)
                    continue
                raise DownloadError(f"{target.name} already exists with different content. Choose an empty destination or rename the existing file.")
            identity = {"repo_id": repo_id, "revision": snapshot.revision, "filename": filename,
                        "size": entry.size, "sha256": entry.sha256, "git_blob_sha1": entry.git_blob_sha1}
            previous = None
            try:
                previous = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            if previous != identity or (part.exists() and part.stat().st_size > entry.size):
                if part.exists():
                    raise DownloadError(f"{part.name} belongs to a different or unknown download. Move it aside before downloading this revision.")
            meta.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
            offset = part.stat().st_size if part.exists() else 0
            if entry.size == 0 and not part.exists():
                part.touch(exist_ok=False)
            url = f"https://huggingface.co/{repo_id}/resolve/{snapshot.revision}/{quote(filename, safe='/')}"
            if offset != entry.size:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with self._open(url, headers) as response:
                    if response.status not in (200, 206):
                        raise DownloadError(f"Hugging Face returned HTTP {response.status} while downloading {filename}. Retry to resume.")
                    if response.status == 206:
                        expected_range = f"bytes {offset}-"
                        content_range = response.getheader("Content-Range", "")
                        if not content_range.startswith(expected_range) or not content_range.endswith(f"/{entry.size}"):
                            raise DownloadError("Server returned an inconsistent partial-download range.")
                    else:
                        offset = 0  # A server can ignore Range: restart safely.
                    with part.open("ab" if offset else "wb") as stream:
                        if progress:
                            progress(filename, offset, entry.size)
                        while True:
                            _cancel(cancel_event)
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            if offset + len(chunk) > entry.size:
                                raise DownloadError("Downloaded data exceeds the pinned repository file size.")
                            stream.write(chunk)
                            offset += len(chunk)
                            if progress:
                                progress(filename, offset, entry.size)
                        stream.flush()
                        os.fsync(stream.fileno())
            _cancel(cancel_event)
            if offset != entry.size:
                raise DownloadError(f"Download of {filename} was interrupted at {offset}/{entry.size} bytes. Retry to resume.")
            if not self._verify(part, entry, cancel_event):
                part.unlink(missing_ok=True)
                meta.unlink(missing_ok=True)
                raise DownloadError(f"Integrity verification failed for {filename}. The corrupt partial file was removed; retry the download.")
            # Atomic link provides no-clobber behavior even if a file appears meanwhile.
            try:
                os.link(part, target)
            except FileExistsError:
                raise DownloadError(f"{target.name} appeared during the download; the existing file was preserved.") from None
            part.unlink()
            meta.unlink(missing_ok=True)
            outputs.append(target)
        return outputs
