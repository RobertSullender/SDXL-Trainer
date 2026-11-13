# Changelog

## 0.1.34

- Show the training method and latest completed checkpoint step in each **Saved runs** entry.

## 0.1.33

- Add **Saved runs** in **Training → Files** to load the latest completed checkpoint from a project run. Use **Checkpoint step** to choose an older step or **Resume checkpoint → Browse…** to open a run elsewhere.

## 0.1.32

- Add **Checkpoint step** to switch between completed checkpoints after selecting a run folder. Selecting an older step suggests a new output when the original run is selected. Custom outputs carry across step changes.

## 0.1.31

- Select a run folder in **Resume checkpoint** to load its highest completed checkpoint step and saved training settings. You can still select an older checkpoint directly and choose a new output folder.

## 0.1.30

- Selecting a resume checkpoint loads the owning run's saved training settings. **Run output** starts at the owning run folder and remains editable.

## 0.1.29

- Fill **Run output** with the owning run folder when selecting a standard resume checkpoint. Choose a new empty output folder afterward to resume there.

## 0.1.28

- Remember a separate run output folder for LoRA and full fine-tuning. Switching methods restores the saved output and resume checkpoint together. Both methods share the model, Diffusers file variant, and dataset.

## 0.1.27

- Remember LoRA and full fine-tuning settings separately in each project. Switching **Training method** restores that method's settings or starts it with its defaults on first use. **Reset all training settings** resets the chosen method.

## 0.1.26

- Choose **LoRA defaults** or **Full fine-tune defaults** when resetting training settings. Each uses the fast profile for its method and sets the UNet learning rate to `1e-4` for LoRA or `1e-6` for full fine-tuning.
- Show the selected training method and optimizer when applying **Fast training defaults**.

## 0.1.25

- Add **Reset all training settings** to restore built-in defaults and save the reset settings to the open project. The reset keeps the model path and file variant, dataset path, and output path, and clears the resume checkpoint selection.

## 0.1.24

- Save and restore the full Training form for each project in `training-settings.json`. **Save training settings** is available with a project open.
- Start new projects with built-in training defaults and the shared SDXL model and Diffusers file variant from **Models & runtime → Local runtime**. Carry previous application-wide training settings into the last-open project.

## 0.1.23

- Save the whole Caption settings form to the open project. **Save caption settings** is disabled until a project is open.
- Add **Models & runtime → Shared caption paths** for the Qwen GGUF, vision projector, and llama.cpp server. Previously saved shared paths carry over and new projects start with those paths and 77-token trimming enabled.

## 0.1.22

- Keep the selected Qwen model, projector, and server paths when resetting all caption settings.
- Update the default system prompt for SDXL captioning and tagging.

## 0.1.21

- Add **Reset all caption settings** to restore and save the open project's caption defaults.

## 0.1.20

- Save the entire Caption settings form per project and restore it on reopening. New projects start with automatic 77-token trimming enabled.

## 0.1.19

- Start new projects with **Trim generated captions to 77 tokens** enabled, including when a project was purged in the same session.

## 0.1.18

- Default **Trim generated captions to 77 tokens** to on when there is no saved preference.

## 0.1.17

- Generate captions for every highlighted image in draft order with **Caption selected**. The 77-token setting applies to each result.

## 0.1.16

- Add **Trim generated captions to 77 tokens** in Caption settings. Enable it to fit generated captions into one window using both local SDXL tokenizers.

## 0.1.15

- Add **Trim to 77 tokens** using both local SDXL tokenizers. The edit keeps the start of the caption and looks for a clean stopping point within one CLIP window.
- Let **Bulk caption…** target the selected draft images or the entire draft, and add **Append to captions**.

## 0.1.14

- Add **Fill empty captions** to **Bulk caption…** alongside **Set every caption** and **Prepend to captions**.
- Close gaps in draft numbering after image removal, keeping each image paired with its caption. The next import continues the sequence, and finalized exports keep their existing filenames.

## 0.1.13

- Add **Allow upscaling** to file and folder imports. Clear it to skip images that need enlargement for the selected fit. The import summary reports the skipped count.
- Add **Bulk caption…** to set every draft caption or prepend shared text across the dataset.

## 0.1.12

- Add **All aspect ratios** to the project dataset chooser. One training run reads all the existing bucket folders in the selected export revision directly, with matching image dimensions within each micro batch.

## 0.1.11

- Export `adapters/lora_comfyui.safetensors` for Artificial-It and ComfyUI and `adapters/lora_diffusers.safetensors` for Diffusers automatically after each completed LoRA run. Keep `pytorch_lora_weights.safetensors` as a compatibility alias for the Diffusers adapter.

## 0.1.10

- Finalize each occupied resolution bucket as an independent training dataset, named `Project:WIDTHXHEIGHT` inside a numbered export revision.
- Choose a bucket or an earlier dataset from **Training → Files → Choose project dataset…**. Older mixed-resolution snapshots remain available.

## 0.1.9

- Make fast training selections application-wide defaults: frozen text encoders/VAE, BF16 UNet/text weights and forward, FP32 VAE, GPU saved activations, and gradient checkpointing.
- Select paged 8-bit AdamW with the local `130` override for full fine-tuning, and ordinary AdamW with optional fused CUDA updates for LoRA; retain every previous optimizer and offload path.
- Prepare frozen text and VAE caches on GPU before trainable components occupy VRAM, and add bounded text caching with 4 GiB text and 1 GiB VAE RAM limits allocated only as used.
- Default to one deterministic sampled latent per image; retain the posterior cache with fresh sampling as an explicit choice.
- Keep complete-caption chunking as the default; add optional native-window conditioning without editing stored captions or captioning prompts.
- Add explicit TF32 acceleration for reduced-precision forward, while FP32 forward retains strict TF32-disabled behavior.
- Add a performance-only **Fast training defaults** action and one-time saved-settings migration that preserve paths, learning rates, duration, seed, and captioning settings.
- Release disposable frozen-cache RAM entries before checkpoint serialization while retaining their validated disk files.
- Load visible dataset thumbnails lazily to reduce large-project startup work.

## 0.1.8

- Add selectable BF16 and FP16 model/forward precision, fp32 optimizer masters, dynamic FP16 loss scaling, and Diffusers weight-file variants while preserving the fp32 path.
- Add optional fp32 precision for both trainable CPU text encoders independently of UNet weight precision.
- Add opt-in CUDA paged AdamW with 8-bit moments, explicit direct-weight precision warnings, native runtime probing, and resumable paged state.
- Add GPU-text UNet staging and optional CUDA updates for fp32 optimizer masters when all trainable weights share one GPU; automatic updates for reduced-precision trainable weights remain on CPU.
- Train and finalize long captions with chunked dual-CLIP conditioning and automatic local-caption continuation; persist warnings for incomplete captions until reviewed.
- Add draft image removal and project-specific caption prompts with individual reset buttons.
- Add an official FP16 Diffusers model download preset in a separate library folder.
- Reuse unchanged staged UNet snapshots, cache verified frozen-VAE posteriors in bounded RAM, and stream checkpoint tensors and optimizer state to reduce redundant work and temporary memory.
- Reserve additional RAM headroom for fp32 masters, release RAM gradients during checkpoint saving, and recheck memory before reallocation.

## 0.1.7

- Add CPU/GPU component placement and staged UNet offloading for memory-constrained SDXL training.
- Add bounded disk-backed gradient writeback for staged training and document resource limits.
- Fix CLIP-L key normalization during single-file SDXL checkpoint export.
- Validate short multi-image full and LoRA runs on the RTX 5080 Laptop GPU.

## 0.1.6

- Improve checkbox visibility in the dark theme.
- Record full-size all-component GPU memory test results and limitations.

## 0.1.5

- Add adaptive RAM or disk gradient storage and bounded GPU-streamed fp32 updates to Disk AdamW, with selectable CPU updates. Keep moments disk-backed, avoid per-tensor forced flushes, and retain checkpoint backend compatibility.
- Cache frozen-VAE posterior parameters while sampling fresh latents on each image visit; VAE training bypasses the cache.
- Add optional phase timings, throughput, and local live NVIDIA GPU, VRAM, system memory, and swap monitoring.
- Keep UNet, both text encoders, and VAE training controls available for LoRA and full training.

## 0.1.4

- Finalize datasets inside the current project as numbered, verified snapshots. Continue editing and add images before creating another snapshot.
- Keep new training runs and launch records inside the project by default.
- Add confirmed full project purge for project-contained datasets, runs, and logs, while preserving external files.
- Strengthen Git ignore rules for generated datasets, model weights, and logs.
- Remove the project rename control; project names are chosen when creating a project.

## 0.1.3

- Enable minimize, maximize, and restore controls for themed file browsers.

## 0.1.2

- Repair access to the bundled caption server when upgrading from older installs with restricted directory permissions.
- Automatically select an existing app-managed training runtime when upgrading from the default system Python; preserve explicitly selected custom runtimes.

## 0.1.1

- Create named projects under `~/SDXL-Trainer/projects` by default, with a configurable parent folder.
- Add safe project renaming without changing imported pixels, captions, or manifests.
- Use matching dark Qt file browsers throughout the application.
- Remember the selected model-library folder and project location.
- Correct installed runtime permissions so regular users can start the bundled caption server.
- Recover the bundled caption-server path when upgrading an unresolved legacy default.
- Derive Debian build versions and filenames from the application version.
- Preserve existing model downloads, custom model paths, settings, and datasets during upgrades.

## 0.1.0

- Initial desktop implementation, dataset preparation, Qwen captioning, model downloader, SDXL training, checkpoint export, offline wiki, and Debian packaging.
