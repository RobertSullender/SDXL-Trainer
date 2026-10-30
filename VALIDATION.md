# Validation record

The application is a development build. These checks establish implementation behavior, not real-model image quality or a 16 GB VRAM guarantee.

## Automated coverage

- Pixel preparation, original preservation, metadata stripping, ICC/EXIF behavior, crop/pad bucketing, stable naming, failed-import rollback, export safety, and tamper detection.
- Exact dual-tokenizer counts with special tokens and no truncation.
- Managed authenticated loopback caption protocol with exact image bytes, generation settings, cancellation, and rejection of truncated output. Tests use a synthetic local server, not downloaded 9B weights.
- Explicit-network-only downloader, revision pinning, content verification, resume, safe redirects, no-clobber completion, and canonical SDXL file selection.
- Native Qt UI workflow: all pages, import/edit/switch/reopen, caption recovery, actual finalization, tokenizer subprocess communication, and dictionary-unavailable behavior.
- Disk AdamW numerical comparison against ordinary torch AdamW, configuration checks, SDXL checkpoint key conversion, and actual tiny SDXL LoRA/full training.
- Tiny all-component training with checkpointing/accumulation, stopped and resumed execution, merged Diffusers reload, and bitwise equality of resumed versus uninterrupted resulting weights.
- Package runtime hashes, archive traversal/link rejection, pinned release enforcement, extracted Qt imports, SVG rendering, and bundled llama-server version check.

## Tested local ML environment

The existing local runtime was read without modification. Versions observed:

| Library | Version |
| --- | --- |
| Python | 3.14 |
| torch | 2.14.0.dev20260729+cu132 |
| Diffusers | 0.39.0 |
| Transformers | 5.14.1 |
| PEFT | 0.19.1 |
| Accelerate | 1.14.0 |
| safetensors | 0.8.0 |
| NumPy | 2.4.6 |
| Pillow | 12.3.0 |

This is an observation of the test environment, not a recommendation to install a nightly torch build. Use an official release compatible with your Python version and Blackwell GPU, then run the included checks.

## Still requires target-machine validation

- Real Qwen3.5-9B Q8_0 vision captioning using each selected variant and matching projector.
- Real SDXL Base forward/backward on RTX 5080, with measured VRAM, RAM, throughput, and disk usage.
- Reconstruction and fixed-prompt sample comparisons for VAE adaptation.
- Real checkpoint conversion and loading in the intended external SDXL application.
- Desktop installation and launch on the user's Ubuntu 26.04.1 system.

The development host's NVIDIA driver is unavailable. No large model weights were downloaded during implementation. The `.deb` was extracted and smoke-tested without installing it or changing system drivers.
