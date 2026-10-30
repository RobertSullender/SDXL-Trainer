from contextlib import contextmanager
from dataclasses import replace
import hashlib
import io
import json
import threading

import pytest

from sdxl_trainer.downloader import (DownloadCancelled, DownloadError, HuggingFaceDownloader,
                                     PRESETS, RemoteFile, RepositorySnapshot, SDXL_TRAINING_FILES,
                                     normalize_repository, selected_training_files)


COMMIT = "a" * 40
DATA = b"GGUF" + bytes(range(256)) * 32


class Reply(io.BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content)
        self.status = status
        self.headers = headers or {}

    def getheader(self, name, default=None):
        return self.headers.get(name, default)


def snapshot(content=DATA, filename="folder/model.gguf"):
    return RepositorySnapshot("owner/model", COMMIT,
                              [RemoteFile(filename, len(content), hashlib.sha256(content).hexdigest())])


def fake_open(client, replies, calls=None):
    @contextmanager
    def opener(url, headers=None):
        if calls is not None:
            calls.append((url, headers))
        yield replies.pop(0)
    client._open = opener


def test_network_is_opt_in_and_has_no_construction_side_effects(tmp_path):
    client = HuggingFaceDownloader()
    with pytest.raises(DownloadError, match="Network access is disabled"):
        client.inspect("owner/model")
    with pytest.raises(DownloadError, match="Network access is disabled"):
        client.download(snapshot(), ["folder/model.gguf"], tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_inspection_pins_revision_and_hashes():
    client = HuggingFaceDownloader(allow_network=True)
    calls = []
    data = {"sha": COMMIT, "siblings": [
        {"rfilename": "model.gguf", "size": len(DATA), "blobId": "b" * 40,
         "lfs": {"sha256": hashlib.sha256(DATA).hexdigest(), "size": len(DATA)}},
        {"rfilename": "config.json", "size": 2, "blobId": "c" * 40}]}
    fake_open(client, [Reply(json.dumps(data).encode())], calls)
    result = client.inspect("https://huggingface.co/owner/model/blob/main/model.gguf", revision="tag/one")
    assert result.revision == COMMIT and result.repo_id == "owner/model"
    assert result.files[1].git_blob_sha1 is None  # Git LFS pointer is not file checksum.
    assert result.files[1].sha256 == hashlib.sha256(DATA).hexdigest()
    assert "tag%2Fone?blobs=true" in calls[0][0]


def test_no_unpinned_listing():
    client = HuggingFaceDownloader(allow_network=True)
    fake_open(client, [Reply(b'{"sha":"main","siblings":[]}')])
    with pytest.raises(DownloadError, match="full commit hash"):
        client.inspect("owner/model")


@pytest.mark.parametrize("repository", ["http://huggingface.co/a/b", "https://example.com/a/b", "../model",
                                         "https://huggingface.co:443/a/b", "https://user@huggingface.co/a/b"])
def test_invalid_repositories(repository):
    with pytest.raises(DownloadError):
        normalize_repository(repository)


def test_verified_download_preserves_path_and_no_temporary_files(tmp_path):
    client = HuggingFaceDownloader(allow_network=True)
    calls = []
    fake_open(client, [Reply(DATA)], calls)
    progress = []
    output = client.download(snapshot(), ["folder/model.gguf"], tmp_path,
                             progress=lambda *args: progress.append(args))
    assert output == [tmp_path / "folder/model.gguf"]
    assert output[0].read_bytes() == DATA
    assert sorted(p.name for p in output[0].parent.iterdir()) == ["model.gguf"]
    assert f"/resolve/{COMMIT}/folder/model.gguf" in calls[0][0]
    assert progress[-1] == ("folder/model.gguf", len(DATA), len(DATA))


def test_interrupted_download_resumes_with_range_then_verifies(tmp_path):
    client = HuggingFaceDownloader(allow_network=True)
    calls = []
    part_length = len(DATA) // 2
    fake_open(client, [Reply(DATA[:part_length])], calls)
    with pytest.raises(DownloadError, match="interrupted"):
        client.download(snapshot(), ["folder/model.gguf"], tmp_path)
    assert (tmp_path / "folder/model.gguf.part").stat().st_size == part_length
    fake_open(client, [Reply(DATA[part_length:], 206,
                   {"Content-Range": f"bytes {part_length}-{len(DATA)-1}/{len(DATA)}"})], calls)
    output = client.download(snapshot(), ["folder/model.gguf"], tmp_path)
    assert calls[-1][1] == {"Range": f"bytes={part_length}-"}
    assert output[0].read_bytes() == DATA


def test_server_ignoring_range_restarts_instead_of_appending(tmp_path):
    client = HuggingFaceDownloader(allow_network=True)
    fake_open(client, [Reply(DATA[:100])])
    with pytest.raises(DownloadError):
        client.download(snapshot(), ["folder/model.gguf"], tmp_path)
    fake_open(client, [Reply(DATA)])
    assert client.download(snapshot(), ["folder/model.gguf"], tmp_path)[0].read_bytes() == DATA


def test_integrity_failure_removes_corrupt_partial_and_preserves_original(tmp_path):
    client = HuggingFaceDownloader(allow_network=True)
    fake_open(client, [Reply(b"x" * len(DATA))])
    with pytest.raises(DownloadError, match="Integrity verification failed"):
        client.download(snapshot(), ["folder/model.gguf"], tmp_path)
    assert list((tmp_path / "folder").iterdir()) == []
    (tmp_path / "folder/model.gguf").write_bytes(b"my existing model")
    with pytest.raises(DownloadError, match="already exists with different content"):
        client.download(snapshot(), ["folder/model.gguf"], tmp_path)
    assert (tmp_path / "folder/model.gguf").read_bytes() == b"my existing model"


def test_changed_revision_never_resumes_old_partial(tmp_path):
    client = HuggingFaceDownloader(allow_network=True)
    fake_open(client, [Reply(DATA[:100])])
    with pytest.raises(DownloadError):
        client.download(snapshot(), ["folder/model.gguf"], tmp_path)
    with pytest.raises(DownloadError, match="different or unknown download"):
        client.download(replace(snapshot(), revision="b" * 40), ["folder/model.gguf"], tmp_path)


def test_cancellation_keeps_recoverable_partial(tmp_path):
    client = HuggingFaceDownloader(allow_network=True)
    fake_open(client, [Reply(DATA)])
    event = threading.Event()

    def progress(filename, done, total):
        if done:
            event.set()

    with pytest.raises(DownloadCancelled):
        client.download(snapshot(), ["folder/model.gguf"], tmp_path, progress=progress, cancel_event=event)
    assert not (tmp_path / "folder/model.gguf").exists()
    assert (tmp_path / "folder/model.gguf.part").exists()


def test_zero_byte_git_blob_verified(tmp_path):
    entry = RemoteFile("empty", 0, git_blob_sha1=hashlib.sha1(b"blob 0\0").hexdigest())
    client = HuggingFaceDownloader(allow_network=True)
    result = client.download(RepositorySnapshot("owner/model", COMMIT, [entry]), ["empty"], tmp_path)
    assert result[0].read_bytes() == b""


@pytest.mark.parametrize("name", ["../model", "/absolute", "a/../model", "a\\model", "a//model"])
def test_path_traversal_cannot_escape_destination(tmp_path, name):
    client = HuggingFaceDownloader(allow_network=True)
    with pytest.raises(DownloadError, match="unsafe file path"):
        client.download(snapshot(filename=name), [name], tmp_path)


def test_symlink_cannot_escape_destination(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "models"
    root.mkdir()
    (root / "folder").symlink_to(outside, target_is_directory=True)
    client = HuggingFaceDownloader(allow_network=True)
    with pytest.raises(DownloadError, match="unsafe symbolic link"):
        client.download(snapshot(), ["folder/model.gguf"], root)
    assert list(outside.iterdir()) == []


def test_redirect_strips_auth_and_rejects_non_hf_hosts(monkeypatch):
    calls = []
    replies = [Reply(b"", 302, {"Location": "https://cas-bridge.xethub.hf.co/signed?secret=test"}), Reply(DATA)]

    class Connection:
        def __init__(self, host, timeout):
            self.host = host
        def request(self, method, target, headers):
            calls.append((self.host, headers))
        def getresponse(self):
            return replies.pop(0)
        def close(self):
            pass

    monkeypatch.setattr("sdxl_trainer.downloader.http.client.HTTPSConnection", Connection)
    client = HuggingFaceDownloader(allow_network=True, token="secret-token")
    with client._open("https://huggingface.co/owner/model/resolve/main/model.gguf") as response:
        assert response.read() == DATA
    assert calls[0][1]["Authorization"] == "Bearer secret-token"
    assert "Authorization" not in calls[1][1]
    with pytest.raises(DownloadError, match="outside Hugging Face"):
        with client._open("https://localhost/private"):
            pass


def test_user_variants_include_matching_projector():
    caption_presets = [preset for preset in PRESETS if preset["kind"] == "caption"]
    assert len(caption_presets) == 2
    assert all(preset["model_filename"].endswith("Q8_0.gguf") for preset in caption_presets)
    assert all(preset["mmproj_filename"].startswith("mmproj-Qwen3.5-9B") for preset in caption_presets)


def test_training_selection_requires_all_components_and_excludes_duplicate_formats():
    all_files = list(SDXL_TRAINING_FILES) + ["unet/diffusion_pytorch_model.fp16.safetensors", "vae/model.onnx", "model.bin"]
    snap = RepositorySnapshot("stabilityai/stable-diffusion-xl-base-1.0", COMMIT,
                              [RemoteFile(path, 1) for path in all_files])
    assert selected_training_files(snap) == list(SDXL_TRAINING_FILES)
    assert len([name for name in selected_training_files(snap) if name.endswith(".safetensors")]) == 4
    with pytest.raises(DownloadError, match="lacks required canonical"):
        selected_training_files(snapshot())
