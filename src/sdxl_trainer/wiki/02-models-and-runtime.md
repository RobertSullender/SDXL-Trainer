# Models and runtime

## Downloading models

**Models & runtime → Model downloads** is the only application area that intentionally connects to the internet. Choose a preset, inspect its repository, review selected filenames and sizes, then click **Download selected files**.

Inspection resolves a branch to an exact repository commit. Downloads use that commit and verify the published content hashes before accepting a file. Interrupted transfers keep a partial file and its metadata so the same revision can resume. Finished files are never silently replaced. Keep a library folder with adequate free space.

The presets include:

- HauhauCS Qwen3.5-9B Uncensored Aggressive Q8_0 with its BF16 vision projector.
- LM Studio Community Qwen3.5-9B Q8_0 with its BF16 vision projector.
- Stability AI SDXL Base 1.0, using the canonical Diffusers fp32 safetensors files and required configurations/tokenizers.

Each Qwen pair needs about 10.45 GB of disk space. The selected SDXL Diffusers files need about 13.9 GB. These figures exclude partial transfers, run output, and caches. Q8_0 is the requested quantization; memory use also includes the vision projector, image/context tokens, and runtime buffers.

A Hugging Face token can be entered for a gated model. It stays in memory, is not saved in application settings, and is sent only to Hugging Face rather than its redirected storage hosts. A gated repository may require you to accept its terms separately.

Models already on disk can be selected directly. The app does not require another download.

## The bundled caption engine

The Debian package contains a pinned llama.cpp server with CUDA 12.8 libraries and CPU backends. It is version 0.8.0 / build 11960, commit 8642541b255bfb5dc39cd351a178be09013a55d6. The application starts its own authenticated server on a random loopback port and unloads it after captioning. An upgrade requires an explicit package update.

Your NVIDIA driver must support the bundled CUDA runtime. Check that `nvidia-smi` works on your computer. The installer does not modify the driver.

## The training Python runtime

PyTorch is installed in a separate local Python environment. Choose its Python executable under **Local runtime**. This keeps the desktop independent of GPU-library changes and lets you use an existing compatible environment.

**Inspect runtime** checks Python, torch, Diffusers, Transformers, PEFT, safetensors, Pillow, CUDA availability, and compiled GPU architectures. RTX 5080 requires a Blackwell-capable torch build, normally CUDA 12.8 or later. CPU-only torch cannot train on your NVIDIA GPU. Training also requires NumPy and Accelerate; install the complete dependencies listed in the repository README.

Use an official PyTorch build that supports the selected Python version. See the repository README for setup commands. An offline machine can use a prepared environment or a local wheel directory. No package installation happens during training.

After downloads and runtime setup, ordinary use is offline. See [Captioning](04-captioning.md).
