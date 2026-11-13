#!/usr/bin/env python3
"""Build-time only: verify and assemble the pinned offline llama.cpp CUDA runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "packaging" / "llama.lock.json"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_archive(path: Path, expected: str) -> None:
    if digest(path) != expected:
        raise ValueError(f"SHA-256 verification failed for {path.name}; refusing to package it.")


def unpack_runtime(archive: Path, destination: Path) -> None:
    """Extract only flat runtime files, refusing traversal, devices and unsafe links."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if PurePosixPath(member.name).is_absolute() or ".." in parts or len(parts) > 2:
                raise ValueError(f"Unsafe archive member: {member.name}")
            if member.isdir():
                continue
            name = parts[-1]
            if not (name == "llama-server" or name == "LICENSE" or (name.startswith("lib") and ".so" in name)):
                continue
            target = destination / name
            if target.exists() or target.is_symlink():
                raise ValueError(f"Duplicate archive member: {name}")
            if member.issym():
                if PurePosixPath(member.linkname).name != member.linkname or member.linkname in {"", ".", ".."}:
                    raise ValueError(f"Unsafe archive link: {member.name}")
                target.symlink_to(member.linkname)
            elif member.isfile():
                with tar.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o755 if name == "llama-server" else 0o644)
            else:
                raise ValueError(f"Unsupported archive member: {member.name}")


def normalize_runtime_permissions(stage: Path) -> None:
    """Publish a readable runtime while leaving its temporary root private until ready."""
    root = stage.resolve()
    for path in stage.rglob("*"):
        if path.is_symlink():
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Runtime link escapes its installation: {path}")
            continue
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            executable = path.relative_to(stage).as_posix() in {"llama-server", "libexec/llama-server"}
            path.chmod(0o755 if executable else 0o644)
        else:
            raise ValueError(f"Unsupported runtime file type: {path}")
    # mkdtemp starts at 0700. Change the root only after all content is ready.
    stage.chmod(0o755)


def prepare(output: Path, cache: Path, offline: bool = False) -> Path:
    lock = json.loads(LOCK.read_text())
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Runtime output already exists: {output}. Choose a new folder.")
    cache.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".llama-", dir=output.parent))
    try:
        libexec = stage / "libexec"
        for asset in lock["assets"]:
            archive = cache / asset["name"]
            if not archive.exists():
                if offline:
                    raise FileNotFoundError(f"Offline build needs cached archive: {archive}")
                temporary = archive.with_suffix(archive.suffix + ".part")
                try:
                    print(f"Downloading pinned {asset['name']}", flush=True)
                    urllib.request.urlretrieve(asset["url"], temporary)
                    verify_archive(temporary, asset["sha256"])
                    os.replace(temporary, archive)
                finally:
                    temporary.unlink(missing_ok=True)
            verify_archive(archive, asset["sha256"])
            unpack_runtime(archive, libexec)
        required = ["llama-server", "libggml-cuda.so", "libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "LICENSE"]
        for name in required:
            if not (libexec / name).is_file():
                raise ValueError(f"Pinned archive is missing required runtime file: {name}")
        wrapper = stage / "llama-server"
        wrapper.write_text('#!/bin/sh\nset -eu\nHERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\nexport LD_LIBRARY_PATH="$HERE/libexec${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\nexec "$HERE/libexec/llama-server" "$@"\n')
        wrapper.chmod(0o755)
        shutil.copyfile(LOCK, stage / "llama.lock.json")
        files = {str(path.relative_to(stage)): digest(path) for path in stage.rglob("*") if path.is_file()}
        (stage / "runtime.json").write_text(json.dumps({"lock": lock, "files": files}, indent=2) + "\n")
        result = subprocess.run([str(wrapper), "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=60, check=True)
        print(result.stdout.strip())
        normalize_runtime_permissions(stage)
        stage.rename(output)
        return output
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/llama-runtime")
    parser.add_argument("--cache", type=Path, default=ROOT / "build/downloads")
    parser.add_argument("--offline", action="store_true", help="Use verified cached assets; perform no downloads")
    args = parser.parse_args(argv)
    print(prepare(args.output, args.cache, args.offline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
