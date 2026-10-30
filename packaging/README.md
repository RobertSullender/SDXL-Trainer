# Offline installer build

The bundled caption engine makes no runtime downloads. Model files can be
supplied locally or downloaded through the application's explicit model-library
actions. The separate training Python environment is supplied locally.
The installer includes a pinned native llama.cpp server, CPU backends, CUDA 12.8
backend, CUDA runtime, cuBLAS, and the Qt desktop libraries needed by this app.
The installed NVIDIA driver remains a system prerequisite.

Build prerequisites: Ubuntu amd64, Python 3.10+, `dpkg-deb`, `ldd`, Pillow,
and a local PySide6/Shiboken6 6.10.2 installation, usually the project `.venv`.

```bash
python3 scripts/prepare_llama.py
python3 scripts/build_deb.py
```

`prepare_llama.py` is a **build-time download step**. It uses only the two URLs
and SHA-256 digests in `llama.lock.json`. After those archives are cached in
`build/downloads`, the entire build can run offline:

```bash
python3 scripts/prepare_llama.py --offline --output build/llama-runtime-new
python3 scripts/build_deb.py --llama-dir build/llama-runtime-new
```

The second command outputs `dist/sdxl-trainer_0.1.0_amd64.deb` and its SHA-256
checksum. Package assembly never downloads anything. It refuses an engine with
a mismatched lock or changed files. Existing runtime output directories are
never replaced. The `.deb` itself may be rebuilt as source changes.

Installed application source is under `/opt/sdxl-trainer/src`, its Qt runtime
under `/opt/sdxl-trainer/lib`, and its engine launcher is
`/opt/sdxl-trainer/runtime/llama/llama-server`. It loads its shared libraries
from its private `libexec` folder. These files and relevant licenses are
included in the package. Models, projects, datasets and training environments
are not installed into `/opt` or removed with the application.

The initial package is approximately 680 MiB (about 1.16 GiB installed).
Most of that size is CUDA/cuBLAS. Qt WebEngine, browser runtimes, QML, development
headers, unused executables, model weights and ML Python libraries are excluded.

To verify a package without installing or changing the computer:

```bash
dpkg-deb --info dist/sdxl-trainer_0.1.0_amd64.deb
dpkg-deb --extract dist/sdxl-trainer_0.1.0_amd64.deb build/package-smoke
build/package-smoke/opt/sdxl-trainer/runtime/llama/llama-server --version
```

This validates package structure and engine loading. Caption accuracy,
Qwen3.5 model compatibility and actual training must additionally be tested
with the user's real local model files and target GPU.
