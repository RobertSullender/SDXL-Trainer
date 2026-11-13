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


def prepared_fixture_runtime(tmp_path, monkeypatch):
    """Exercise actual verified extraction/publication using a tiny local archive."""
    import os
    cache = tmp_path / "cache"
    cache.mkdir()
    packed = cache / "runtime-fixture.tar.gz"
    with tarfile.open(packed, "w:gz") as tar:
        names = ("llama-server", "libggml-cuda.so", "libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "LICENSE")
        for name in names:
            content = b"#!/bin/sh\nprintf 'fixture server version\\n'\n" if name == "llama-server" else b"local test fixture\n"
            member = tarfile.TarInfo("release/" + name)
            member.size = len(content)
            member.mode = 0o600
            tar.addfile(member, io.BytesIO(content))
    lock = {"version": "fixture", "assets": [{"name": packed.name, "sha256": prepare.digest(packed), "url": "https://invalid.example/never-requested"}]}
    lock_file = tmp_path / "fixture.lock.json"
    lock_file.write_text(json.dumps(lock))
    monkeypatch.setattr(prepare, "LOCK", lock_file)
    original_umask = os.umask(0o077)
    try:
        runtime = prepare.prepare(tmp_path / "published-runtime", cache, offline=True)
    finally:
        os.umask(original_umask)
    return runtime, lock


def test_prepare_publishes_traversable_runtime_after_private_staging(tmp_path, monkeypatch):
    import stat
    before_modes = []
    normalize = prepare.normalize_runtime_permissions

    def observe_private_stage(stage):
        before_modes.append(stat.S_IMODE(stage.stat().st_mode))
        normalize(stage)

    monkeypatch.setattr(prepare, "normalize_runtime_permissions", observe_private_stage)
    runtime, _ = prepared_fixture_runtime(tmp_path, monkeypatch)
    assert before_modes == [0o700]
    for path in [runtime, *runtime.rglob("*")]:
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            assert mode == 0o755, path
        else:
            expected = 0o755 if path.name == "llama-server" else 0o644
            assert mode == expected, path


def test_built_deb_normalizes_copied_private_directories(tmp_path, monkeypatch):
    """Inspect a real dpkg data tar, including copytree's inherited 0700 roots."""
    import os
    import shutil
    import stat
    import subprocess
    if not shutil.which("dpkg-deb") or not shutil.which("dpkg"):
        pytest.skip("Debian packaging tools are unavailable")
    if subprocess.check_output(["dpkg", "--print-architecture"], text=True).strip() != "amd64":
        pytest.skip("Pinned package targets amd64")
    runtime, lock = prepared_fixture_runtime(tmp_path, monkeypatch)
    # Reproduce an already-prepared 0.1.0 runtime with the historical private root.
    runtime.chmod(0o700)
    (runtime / "libexec").chmod(0o700)
    checkout = tmp_path / "checkout"
    source = checkout / "src/sdxl_trainer"
    source.mkdir(parents=True, mode=0o700)
    (source / "__init__.py").write_text('__version__ = "0.1.1"\n')
    (source / "assets").mkdir(mode=0o700)
    (source / "assets/sdxl-trainer.svg").write_text("<svg/>")
    packaging = checkout / "packaging"
    packaging.mkdir()
    for filename in ("launcher", "sdxl-trainer.desktop", "copyright", "postinst"):
        shutil.copyfile(ROOT / "packaging" / filename, packaging / filename)
    (packaging / "licenses").mkdir(mode=0o700)
    (packaging / "licenses/fixture.txt").write_text("fixture license")
    (packaging / "llama.lock.json").write_text(json.dumps(lock))
    for filename in ("LICENSE", "README.md", "VALIDATION.md"):
        (checkout / filename).write_text("fixture documentation")
    qt = tmp_path / "local-qt"
    (qt / "PySide6/private-nested").mkdir(parents=True, mode=0o700)
    (qt / "PySide6/private-nested/module.so").write_bytes(b"shared-library fixture")
    (qt / "PySide6/private-nested/module.so").chmod(0o600)

    def bundle_fixture_qt(site_packages, destination):
        # The production builder copies upstream file modes. Keep this small
        # fixture focused on actual staging, normalization and dpkg serialization.
        shutil.copytree(site_packages, destination)
        return sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())

    monkeypatch.setattr(builder, "ROOT", checkout)
    monkeypatch.setattr(builder, "bundle_qt", bundle_fixture_qt)
    original_umask = os.umask(0o077)
    try:
        package = builder.build(runtime, qt, tmp_path / "fixture.deb")
    finally:
        os.umask(original_umask)
    assert subprocess.check_output(["dpkg-deb", "--field", str(package), "Version"], text=True).strip() == "0.1.1"
    control_tar = subprocess.check_output(["dpkg-deb", "--ctrl-tarfile", str(package)])
    with tarfile.open(fileobj=io.BytesIO(control_tar), mode="r:") as tar:
        postinst = next(member for member in tar.getmembers() if member.name.removeprefix("./") == "postinst")
        assert postinst.mode == 0o755
        assert tar.extractfile(postinst).read() == (ROOT / "packaging/postinst").read_bytes()
    data_tar = subprocess.check_output(["dpkg-deb", "--fsys-tarfile", str(package)])
    with tarfile.open(fileobj=io.BytesIO(data_tar), mode="r:") as tar:
        members = {member.name.removeprefix("./").rstrip("/"): member for member in tar.getmembers()}
    assert members["opt/sdxl-trainer/runtime/llama"].mode == 0o755
    assert members["opt/sdxl-trainer/src/sdxl_trainer"].mode == 0o755
    assert members["opt/sdxl-trainer/lib/PySide6/private-nested"].mode == 0o755
    executables = {"usr/bin/sdxl-trainer", "opt/sdxl-trainer/runtime/llama/llama-server", "opt/sdxl-trainer/runtime/llama/libexec/llama-server"}
    for name, member in members.items():
        if member.isdir():
            assert member.mode == 0o755, name
        elif member.isfile():
            assert member.mode == (0o755 if name in executables else 0o644), name
    assert not stat.S_IMODE(source.stat().st_mode) & stat.S_IROTH
    assert stat.S_IMODE(runtime.stat().st_mode) == 0o700  # No chmod of input trees.


def test_package_cli_defaults_follow_literal_source_version(tmp_path, monkeypatch):
    source = tmp_path / "src/sdxl_trainer"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text('raise RuntimeError("must never import application")\n__version__ = "7.8.9"\n')
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    calls = []

    def capture(llama, qt, output, version):
        calls.append((output, version))
        return output

    monkeypatch.setattr(builder, "build", capture)
    monkeypatch.setattr(builder, "default_qt_path", lambda: tmp_path / "qt")
    assert builder.main([]) == 0
    assert calls == [(tmp_path / "dist/sdxl-trainer_7.8.9_amd64.deb", "7.8.9")]


@pytest.mark.parametrize("action", ["configure", "abort-upgrade", "abort-remove", "abort-deconfigure"])
def test_postinst_repairs_only_owned_runtime_paths_on_configure(tmp_path, action):
    """Run the real script with a fake chmod; never access the real /opt tree."""
    import os
    import stat
    import subprocess
    targets = [
        "/opt/sdxl-trainer/runtime/llama",
        "/opt/sdxl-trainer/runtime/llama/libexec",
        "/opt/sdxl-trainer/runtime/llama/llama-server",
        "/opt/sdxl-trainer/runtime/llama/libexec/llama-server",
    ]
    fake_root = tmp_path / "fake-installation"
    for filename in targets[:2]:
        (fake_root / filename.lstrip("/")).mkdir(parents=True, exist_ok=True, mode=0o700)
    for filename in targets[2:]:
        target = fake_root / filename.lstrip("/")
        target.write_text("fixture executable")
        target.chmod(0o600)
    sentinel = fake_root / "home/user/.local/share/sdxl-trainer/models/model.gguf"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"untouched model")
    sentinel.chmod(0o600)
    binary_dir = tmp_path / "fake-bin"
    binary_dir.mkdir()
    capture_log = tmp_path / "chmod-calls.jsonl"
    command = binary_dir / "chmod"
    command.write_text(
        "#!/usr/bin/python3\n"
        "import json, os, pathlib, sys\n"
        f"allowed = {targets!r}\n"
        "args = sys.argv[1:]\n"
        "assert args[:2] == ['0755', '--'], args\n"
        "assert args[2:] and all(path in allowed for path in args[2:]), args\n"
        "with open(os.environ['CAPTURE_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "for path in args[2:]:\n"
        "    (pathlib.Path(os.environ['FAKE_INSTALL_ROOT']) / path.lstrip('/')).chmod(0o755)\n"
    )
    command.chmod(0o755)
    environment = {**os.environ, "PATH": str(binary_dir), "CAPTURE_LOG": str(capture_log),
                   "FAKE_INSTALL_ROOT": str(fake_root)}
    arguments = ["/bin/sh", str(ROOT / "packaging/postinst"), action, "0.1.0"]
    subprocess.run(arguments, env=environment, check=True)
    if action == "configure":
        calls = [json.loads(line) for line in capture_log.read_text().splitlines()]
        assert calls == [["0755", "--", *targets[:2]], ["0755", "--", *targets[2:]]]
        assert all(stat.S_IMODE((fake_root / name.lstrip("/")).stat().st_mode) == 0o755 for name in targets)
        # Reconfiguration is safe and repairs retained historical directory modes.
        subprocess.run(arguments, env=environment, check=True)
        assert all(stat.S_IMODE((fake_root / name.lstrip("/")).stat().st_mode) == 0o755 for name in targets)
    else:
        assert not capture_log.exists()
        assert all(stat.S_IMODE((fake_root / name.lstrip("/")).stat().st_mode) == 0o700 for name in targets[:2])
        assert all(stat.S_IMODE((fake_root / name.lstrip("/")).stat().st_mode) == 0o600 for name in targets[2:])
    assert sentinel.read_bytes() == b"untouched model"
    assert stat.S_IMODE(sentinel.stat().st_mode) == 0o600
