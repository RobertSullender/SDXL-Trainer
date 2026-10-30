"""Security/integrity checks for build-time runtime bundling; no network needed."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_llama")
builder = load_script("build_deb")


def archive(tmp_path, name, content=b"test", *, link=None):
    file = tmp_path / "runtime.tar.gz"
    with tarfile.open(file, "w:gz") as tar:
        info = tarfile.TarInfo(name)
        if link is not None:
            info.type = tarfile.SYMTYPE
            info.linkname = link
            tar.addfile(info)
        else:
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return file


def test_runtime_archive_traversal_cannot_write_outside_destination(tmp_path):
    file = archive(tmp_path, "../escaped.so")
    with pytest.raises(ValueError, match="Unsafe"):
        prepare.unpack_runtime(file, tmp_path / "out")
    assert not (tmp_path / "escaped.so").exists()


def test_runtime_archive_rejects_unsafe_symlink(tmp_path):
    file = archive(tmp_path, "release/libggml.so", link="../../outside")
    with pytest.raises(ValueError, match="Unsafe archive link"):
        prepare.unpack_runtime(file, tmp_path / "out")


def test_runtime_archive_keeps_only_supported_files(tmp_path):
    file = tmp_path / "runtime.tar.gz"
    with tarfile.open(file, "w:gz") as tar:
        for name in ("release/llama-server", "release/libggml.so", "release/LICENSE", "release/unneeded-tool"):
            info = tarfile.TarInfo(name)
            info.size = 4
            tar.addfile(info, io.BytesIO(b"data"))
    prepare.unpack_runtime(file, tmp_path / "out")
    assert set(p.name for p in (tmp_path / "out").iterdir()) == {"llama-server", "libggml.so", "LICENSE"}


def test_download_digest_mismatch_fails_closed(tmp_path):
    file = tmp_path / "download.tar.gz"
    file.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare.verify_archive(file, "0" * 64)


def test_package_refuses_wrong_runtime_release(tmp_path):
    (tmp_path / "runtime.json").write_text(json.dumps({"lock": {"version": "unverified"}, "files": {}}))
    with pytest.raises(ValueError, match="pinned release"):
        builder.validate_llama(tmp_path)


def test_package_runtime_manifest_cannot_read_outside_runtime(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    manifest = {"lock": json.loads((ROOT / "packaging/llama.lock.json").read_text()),
                "files": {"../private.txt": "0" * 64}}
    (runtime / "runtime.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe runtime"):
        builder.validate_llama(runtime)
