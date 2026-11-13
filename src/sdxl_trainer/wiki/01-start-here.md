# Start here

SDXL-Trainer keeps your dataset preparation, captions, training, and model outputs on your computer. It does not require an account. This wiki is bundled inside the application and works offline.

## Your first project

1. Open **Models & runtime**. Either select existing local model files or use the explicit download controls. Set your local training Python runtime and inspect it.
2. In **Models & runtime → Local runtime**, select the shared Diffusers SDXL base model and its file variant, then click **Save settings**. The model's two tokenizers provide the exact counts used by the caption editor and finalization. You can choose a different model for the open project under **Training → Files**.
3. In **Models & runtime → Shared caption paths**, select the Qwen GGUF and its matching vision projector (mmproj). The installed application includes a llama.cpp server. Click **Save shared caption paths** to use these selections for new projects.
4. In **Dataset**, click **New project** and enter a name. The app creates it under `~/SDXL-Trainer/projects` by default. Every new project starts with **Caption settings → Trim generated captions to 77 tokens** enabled. Uncheck it for longer captions. Import image files or a folder. Crop-to-fill is the default. Clear **Allow upscaling** to skip images that need enlargement.
5. Select images with Ctrl-click or Shift-click and click **Caption selected** to generate or replace their captions. Use **Caption all empty** to caption blank entries. In **Bulk caption…**, choose selected images or the entire draft. You can set captions, prepend or append shared text, fill empty captions, or trim to one 77-token window. Review each image, caption, spelling, token counts, and any continuation warnings. Click **Save caption settings** to store all caption controls with the open project.
6. Click **Finalize dataset**. The app creates a dataset folder for each occupied resolution bucket under `finalized/Dataset-1`, such as `Photo-Realistic:1024X1024`. Finalizing again creates the next numbered revision.
7. The first bucket is selected automatically. Open **Training → Files → Choose project dataset…** to select another bucket or **Dataset-N / All aspect ratios** for all the buckets in that revision. Review the run folder and training settings, check the configuration, and start a short first run. Use **Save training settings** to store the Training form with the open project.
8. After a LoRA run completes, use `adapters/lora_comfyui.safetensors` in Artificial-It or ComfyUI, or `adapters/lora_diffusers.safetensors` in Diffusers. To use the complete model, load the Diffusers pipeline or create a combined checkpoint from **Checkpoint export**. [Checkpoints and export](07-checkpoints-and-export.md)

To remove a project completely, open it and choose **Fully purge project…**. The confirmation shows what will be deleted and asks you to type the project name. This permanently removes the project folder, including datasets, project-contained runs, and logs. Original images and model downloads stored outside the project are preserved. See [Dataset preparation](03-dataset-preparation.md) for details.

## What is tested

This development version includes automated tests for dataset integrity, downloader behavior, the desktop workflow, optimizer math, resumable tiny SDXL training, and packaging. Short RTX 5080 tests have completed, including all-component BF16 training with a 306-token caption under a simulated 8 GiB PyTorch allocation cap. Other hardware, longer runs, and output image quality still need validation.

Joint VAE adaptation is experimental. Full fine-tuning may exceed your 16 GB GPU or 32 GB system RAM. A memory estimate is not a measured guarantee.

Continue with [Models and runtime](02-models-and-runtime.md), [Dataset preparation](03-dataset-preparation.md), or [Training](05-training.md).
