# Start here

SDXL-Trainer keeps your dataset preparation, captions, training, and model outputs on your computer. It does not require an account. This wiki is bundled inside the application and works offline.

## Your first project

1. Open **Models & runtime**. Either select existing local model files or use the explicit download controls. Set your local training Python runtime and inspect it.
2. In **Training → Files**, select the Diffusers SDXL base-model folder. Its two tokenizers provide the exact counts used by the caption editor and finalization.
3. In **Caption settings**, select the Qwen GGUF and its matching mmproj file. The server executable is bundled with the installed application.
4. In **Dataset**, create a project in an empty folder. Import image files or a folder. Crop-to-fill is the default.
5. Select an image and caption it, or caption all images with empty captions. Review each image, caption, spelling, and both token counts.
6. Finalize by choosing a parent folder and entering a new folder name. This creates a verified snapshot and selects it for training; existing destinations are never replaced.
7. Choose a run folder and training settings, check the configuration, and start a short first run.
8. Use the resulting Diffusers model, or create a combined checkpoint from **Checkpoint export**.

## What is tested

This development version includes automated tests for dataset integrity, downloader behavior, the desktop workflow, optimizer math, resumable tiny SDXL training, and packaging. A real RTX 5080 training run, peak memory, the chosen Qwen 9B weights, and output image quality still need validation on target hardware.

Joint VAE adaptation is experimental. Full fine-tuning may exceed your 16 GB GPU or 32 GB system RAM. A memory estimate is not a measured guarantee.

Continue with [Models and runtime](02-models-and-runtime.md), [Dataset preparation](03-dataset-preparation.md), or [Training](05-training.md).
