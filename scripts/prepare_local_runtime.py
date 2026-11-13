#!/usr/bin/env python3
"""Clone a local ML virtual environment into SDXL-Trainer's independent runtime.

No downloads, pip installation, source modifications, settings changes or model
operations. Requires Linux coreutils cp and renameat2(RENAME_NOREPLACE).
"""
from __future__ import annotations
import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

REQUIRED_MODULES = ("torch", "diffusers", "transformers", "peft", "accelerate", "safetensors", "numpy", "PIL")


def check_links(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            target = os.readlink(path)
            if os.path.isabs(target) or not path.resolve(strict=True).is_relative_to(root.resolve()):
                raise ValueError(f"Runtime contains an absolute or outside symbolic link: {path}")


def validate_runtime(root: Path, modules=REQUIRED_MODULES) -> dict:
    code = '''import importlib,json,sys
from pathlib import Path
expected=Path(sys.argv[1]).resolve()
assert Path(sys.prefix).resolve()==expected, f"Unexpected Python prefix: {sys.prefix}"
locations={}
for name in json.loads(sys.argv[2]):
 module=importlib.import_module(name)
 location=Path(module.__file__).resolve()
 assert location.is_relative_to(expected), f"{name} imports outside the cloned runtime: {location}"
 locations[name]=str(location)
print(json.dumps({"prefix":str(expected),"modules":locations}))
'''
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([str(root / "bin/python"), "-I", "-B", "-c", code, str(root), json.dumps(list(modules))], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def publish_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise OSError(errno.ENOSYS, "Linux renameat2 is required for safe runtime publication")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        failure = ctypes.get_errno()
        raise OSError(failure, os.strerror(failure), str(destination))


def prepare_runtime(source: Path, destination: Path, *, modules=REQUIRED_MODULES) -> Path:
    source = source.expanduser().resolve(strict=True)
    destination = destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Runtime destination already exists; nothing was replaced: {destination}")
    if source == destination.resolve() or source in destination.resolve().parents or destination.resolve() in source.parents:
        raise ValueError("Source and destination runtime folders must be separate.")
    if not (source / "pyvenv.cfg").is_file() or not (source / "bin/python").is_file():
        raise ValueError("Source must be an existing Python virtual environment with bin/python.")
    print("Checking local source runtime links…", flush=True)
    check_links(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage_container = Path(tempfile.mkdtemp(prefix=".runtime-copy-", dir=destination.parent))
    stage = stage_container / "runtime"
    published = False
    try:
        print(f"Copying independent runtime to {destination} (reflink when supported)…", flush=True)
        subprocess.run(["cp", "-a", "--reflink=auto", "--", str(source), str(stage)], check=True)
        check_links(stage)
        # Console scripts and activation helpers may embed the old venv path.
        # The app invokes Python -m, but make local text launchers relocatable too.
        for path in (stage / "bin").iterdir():
            if not path.is_file() or path.is_symlink():
                continue
            with path.open("rb") as stream:
                prefix = stream.read(2)
            if prefix != b"#!" and not path.name.startswith("activate"):
                continue
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            rewritten = text.replace(str(source), str(destination))
            if rewritten != text:
                path.write_text(rewritten)
        print("Verifying staged Python prefix and required local imports…", flush=True)
        validate_runtime(stage, modules)
        print("Publishing runtime atomically without replacing existing files…", flush=True)
        publish_noreplace(stage, destination)
        published = True
        print("Verifying final runtime location…", flush=True)
        validate_runtime(destination, modules)
        print(f"Runtime ready: {destination / 'bin/python'}", flush=True)
        return destination
    except BaseException:
        if published:
            # Only the directory published by this invocation is removed on failure.
            shutil.rmtree(destination)
        raise
    finally:
        shutil.rmtree(stage_container, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    prepare_runtime(args.source, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
