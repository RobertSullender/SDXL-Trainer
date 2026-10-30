#!/usr/bin/env python3
"""Build an offline desktop .deb with pinned Qt and llama.cpp CUDA runtimes.

Run prepare_llama.py once before this script. Building the package itself never
uses the network. Large training libraries and user model weights remain in the
separate user-selected ML runtime and are not silently downloaded at launch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
QT_VERSION = "6.10.2"
DEBIAN_DEPENDS = (
    "python3 (>= 3.10), python3-pil, libenchant-2-2, hunspell-en-us, "
    "libc6 (>= 2.35), libstdc++6, libgcc-s1, libgomp1, libgl1, libegl1, "
    "libglib2.0-0t64, libfontconfig1, libfreetype6, libdbus-1-3, "
    "libxkbcommon0, libxkbcommon-x11-0, libxcb-cursor0, libxcb-icccm4, "
    "libxcb-image0, libxcb-keysyms1, libxcb-randr0, libxcb-render-util0, "
    "libxcb-shm0, libxcb-sync1, libxcb-xfixes0, libxcb-xkb1, libx11-xcb1, "
    "libxcb1, libx11-6, libwayland-client0, libwayland-cursor0, libwayland-egl1, "
    "libgtk-3-0t64"
)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_llama(runtime: Path) -> dict:
    manifest = json.loads((runtime / "runtime.json").read_text())
    expected = json.loads((ROOT / "packaging/llama.lock.json").read_text())
    if manifest.get("lock") != expected:
        raise ValueError("llama.cpp runtime does not match the pinned release lock.")
    if not manifest.get("files"):
        raise ValueError("llama.cpp runtime has no integrity manifest.")
    for name, expected_hash in manifest["files"].items():
        path = runtime / name
        if path.resolve() != runtime.resolve() and runtime.resolve() not in path.resolve().parents:
            raise ValueError("Unsafe runtime manifest path.")
        if digest(path) != expected_hash:
            raise ValueError(f"Bundled llama.cpp runtime changed: {name}")
    if not os.access(runtime / "llama-server", os.X_OK):
        raise ValueError("Bundled llama-server is missing or not executable.")
    return manifest


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def bundle_qt(site_packages: Path, destination: Path) -> int:
    """Bundle only desktop modules and the transitive Qt-library closure.

    No Qt WebEngine, QML runtime, development headers, pip or ML packages are
    copied. X11, Wayland and desktop portal native file dialogs are retained.
    """
    site_packages = site_packages.resolve()
    pyside = site_packages / "PySide6"
    init = (pyside / "__init__.py").read_text()
    if f'__version__ = "{QT_VERSION}"' not in init:
        raise ValueError(f"Build requires pinned PySide6=={QT_VERSION}.")
    sources: set[Path] = set()
    sources.update(pyside.glob("*.py"))
    sources.update(pyside.glob("libpyside6.abi3.so*"))
    for module in ("QtCore", "QtGui", "QtWidgets", "QtSvg", "QtDBus", "QtNetwork"):
        binding = pyside / f"{module}.abi3.so"
        if not binding.exists():
            raise ValueError(f"Missing pinned Qt binding: {binding}")
        sources.add(binding)
    sources.update(p for p in (site_packages / "shiboken6").rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts)
    sources.update(p for p in (pyside / "support").rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts)
    plugins = pyside / "Qt/plugins"
    folders = ("platformthemes", "iconengines", "xcbglintegrations", "wayland-shell-integration",
               "wayland-graphics-integration-client", "wayland-decoration-client")
    for folder in folders:
        sources.update((plugins / folder).glob("*.so"))
    for name in ("qxcb", "qwayland-generic", "qwayland-egl", "qoffscreen", "qminimal"):
        plugin = plugins / "platforms" / f"lib{name}.so"
        if plugin.exists():
            sources.add(plugin)
    for name in ("qjpeg", "qsvg", "qgif", "qwebp", "qico", "qtiff"):
        plugin = plugins / "imageformats" / f"lib{name}.so"
        if plugin.exists():
            sources.add(plugin)
    # ldd returns the complete transitive library closure for each dynamic module.
    for source in list(sources):
        if ".so" not in source.name:
            continue
        result = subprocess.run(["ldd", str(source)], text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=True)
        for dependency in re.findall(r"=>\s+(/[^\n ]+)", result.stdout):
            path = Path(dependency)
            if site_packages in path.resolve().parents:
                sources.add(path)
    for source in sorted(sources):
        _copy_file(source, destination / source.relative_to(site_packages))
    # Keep distribution metadata, version and dependency licenses available.
    for pattern in ("pyside6_essentials-*.dist-info", "shiboken6-*.dist-info"):
        for metadata in site_packages.glob(pattern):
            for source in metadata.rglob("*"):
                if source.is_file() and source.name != "RECORD":
                    _copy_file(source, destination / source.relative_to(site_packages))
    return sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())


def build(llama_dir: Path, qt_site_packages: Path, output: Path, version: str = "0.1.0") -> Path:
    if not re.fullmatch(r"[0-9][0-9A-Za-z.+~:-]*", version):
        raise ValueError("Invalid Debian package version.")
    if subprocess.check_output(["dpkg", "--print-architecture"], text=True).strip() != "amd64":
        raise ValueError("This pinned llama.cpp release targets Ubuntu amd64 only.")
    validate_llama(llama_dir)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sdxl-deb-", dir=ROOT / "build") as temporary:
        stage = Path(temporary)
        app = stage / "opt/sdxl-trainer"
        shutil.copytree(ROOT / "src/sdxl_trainer", app / "src/sdxl_trainer",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        qt_bytes = bundle_qt(qt_site_packages, app / "lib")
        shutil.copytree(llama_dir, app / "runtime/llama", symlinks=True)
        _copy_file(ROOT / "packaging/launcher", stage / "usr/bin/sdxl-trainer")
        (stage / "usr/bin/sdxl-trainer").chmod(0o755)
        _copy_file(ROOT / "packaging/sdxl-trainer.desktop", stage / "usr/share/applications/sdxl-trainer.desktop")
        _copy_file(ROOT / "src/sdxl_trainer/assets/sdxl-trainer.svg",
                   stage / "usr/share/icons/hicolor/scalable/apps/sdxl-trainer.svg")
        docs = stage / "usr/share/doc/sdxl-trainer"
        _copy_file(ROOT / "packaging/copyright", docs / "copyright")
        for filename in ("LICENSE", "README.md", "VALIDATION.md"):
            _copy_file(ROOT / filename, docs / filename)
        if (ROOT / "docs/images").is_dir():
            shutil.copytree(ROOT / "docs/images", docs / "docs/images")
        shutil.copytree(ROOT / "packaging/licenses", docs / "licenses")
        shutil.copyfile(ROOT / "packaging/llama.lock.json", app / "llama.lock.json")
        installed_size = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file()) // 1024
        control = stage / "DEBIAN/control"
        control.parent.mkdir()
        control.write_text(
            f"Package: sdxl-trainer\nVersion: {version}\nSection: graphics\nPriority: optional\n"
            f"Architecture: amd64\nMaintainer: SDXL-Trainer contributors\nInstalled-Size: {installed_size}\n"
            f"Depends: {DEBIAN_DEPENDS}\n"
            "Recommends: xdg-desktop-portal, xdg-desktop-portal-gnome\n"
            "Description: Private offline SDXL dataset preparation and training desktop\n"
            " Native Qt desktop with image preparation, caption review, offline wiki,\n"
            " local training controls, and a pinned bundled llama.cpp CUDA 12.8 engine.\n"
            " Model weights and a compatible ML Python runtime are supplied locally.\n")
        (app / "build-info.json").write_text(json.dumps({"version": version, "qt": QT_VERSION,
              "llama": "0.8.0/b11960", "cuda": "12.8", "qt_bytes": qt_bytes,
              "training_runtime_bundled": False}, indent=2) + "\n")
        subprocess.run(["dpkg-deb", "--root-owner-group", "-Zzstd", "-z6", "--build", str(stage), str(output)], check=True)
    (output.with_suffix(output.suffix + ".sha256")).write_text(f"{digest(output)}  {output.name}\n")
    return output


def default_qt_path() -> Path:
    choices = sorted((ROOT / ".venv/lib").glob("python*/site-packages"))
    if choices:
        return choices[0]
    spec = importlib.util.find_spec("PySide6")
    if spec and spec.origin:
        return Path(spec.origin).parent.parent
    raise ValueError("Pass --qt-site-packages pointing to PySide6==6.10.2 and Shiboken6==6.10.2.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-dir", type=Path, default=ROOT / "build/llama-runtime")
    parser.add_argument("--qt-site-packages", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/sdxl-trainer_0.1.0_amd64.deb")
    parser.add_argument("--version", default="0.1.0")
    args = parser.parse_args(argv)
    print(build(args.llama_dir.resolve(), args.qt_site_packages or default_qt_path(), args.output, args.version))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
