# Troubleshooting

## Token counts are unavailable

Select the Diffusers base-model folder in **Training → Files**, then inspect the chosen training Python. It needs Transformers and both local tokenizer folders. The application refuses approximate counts. Finalization needs this runtime even if captions were written manually.

## Caption model fails to load

Check that the selected files are GGUF files, that the mmproj belongs to that model family, and that the NVIDIA driver works. Both presets include the correct projector filename. Reduce GPU layers, disable projector offload, or reduce context/micro-batch if VRAM is exhausted. CPU captioning is possible but slower.

The pinned engine is bundled with the installed package. Development source runs need the prepared runtime under `build/llama-runtime`. Do not select an unrelated old llama.cpp binary and assume it supports Qwen3.5 vision.

## Caption generation ends early

The app automatically continues a response that reaches its per-response token limit, up to the configured continuation count. If it cannot finish, the available caption is saved and the warning identifies affected images. Review or complete those captions; raise **Maximum tokens**, **Maximum continuations**, or **Context size** as appropriate. Disable thinking if the model returns unfinished reasoning without a usable caption. The context also contains the image and previous responses. A fluent caption can still be inaccurate; review it against the preview.

## Spelling is unavailable

Install the English Hunspell dictionary and Enchant2 system library. The Debian package declares these dependencies. Spell checking remains offline and never silently edits text.

## A dataset fails verification

Under **Training → Files → Choose project dataset…**, select a single bucket or **Dataset-N / All aspect ratios**. For a manually selected single bucket, check for `manifest.json` in the bucket folder, such as `finalized/Dataset-1/Photo-Realistic:1024X1024`. For all aspect ratios, select the revision folder and check that its bucket subfolders each contain their own `manifest.json`.

If a file fails its hash check, make corrections in the working project and finalize again. A changed tokenizer also requires finalizing with the intended model. To resume a run, select the same revision and the same scope it originally used, either one bucket or all aspect ratios.

## A purged project left files elsewhere

The purge removes the complete project folder, including its datasets, runs and logs. Output folders and exported checkpoints created outside that folder are not tracked or deleted. Remove those separately if you no longer need them. Original images and model downloads stored outside the project are preserved.

## CUDA is unavailable

Run **Inspect runtime**. A working driver, a compatible Python version, and a CUDA-enabled torch wheel supporting Blackwell are all required. System CUDA toolkit installation alone does not make a CPU-only torch wheel use CUDA.

## A bitsandbytes CUDA warning appears

Fast full-training defaults use paged 8-bit AdamW and require bitsandbytes. Fast LoRA's ordinary AdamW, and the retained Disk AdamW path, do not require that optimizer integration. An optional library in the Python runtime may still print a warning about an unavailable bitsandbytes CUDA binary when another optimizer is selected. Check the rest of the worker log and actual training progress; a warning alone does not mean that optimizer failed.

**Paged AdamW · 8-bit moments (direct weights)** requires a working bitsandbytes CUDA binary. This integration currently accepts version 0.49.2. The installed package on this machine lacks a CUDA 13.2 binary, so fast defaults select `130` for **8-bit CUDA binary override**; the installed CUDA 13.0 binary passed a native probe and short training test here. The override is experimental and may not work elsewhere. Run **Check configuration** with the remaining settings valid; runtime/native checks run when the worker starts. Do not assume an import or a warning alone proves the optimizer can run. For requirements and remedies, see the [bitsandbytes installation guide](https://huggingface.co/docs/bitsandbytes/installation) and [troubleshooting guide](https://huggingface.co/docs/bitsandbytes/errors). The app does not silently replace the local runtime. Use Disk AdamW if no compatible runtime is available.

## Training spends time preparing caches

Fast defaults prepare frozen VAE and text caches before optimizer steps. With a large dataset, that first preparation pass can take time even while the optimizer step counter is zero. Read the preparation status and progress rather than treating it as a hung first step. Long captions still use all their CLIP windows by default and take more encoding time than short captions.

Subsequent steps use validated local cache files and bounded RAM entries. If disk reads remain frequent, compare the cache RAM limits with available system RAM; do not increase them until the rest of the process has headroom. Training the text encoders or VAE bypasses its corresponding cache and adds repeated computation. **Frozen VAE cache sampling → posterior** also draws fresh latent noise on each visit; **sample** reuses one deterministic latent per image.

## The upgrade changed training selections

Version 0.1.9 applies fast performance selections once to saved application settings. It freezes both text encoders and VAE, selects BF16 weights/forward with an FP32 VAE, and selects the fast optimizer for LoRA or full training. All other component, optimizer, precision, and offload choices remain available. Caption prompts, caption-model settings, image files, caption files, model paths, learning rates, duration, and seed are preserved.

Later manual changes are kept. Use **Fast training defaults** to reapply the performance selections deliberately. Review the learning rate when changing between LoRA and full training: this action preserves existing rates rather than choosing a new experiment. Load the original recipe when resuming an older run; applying new numerical settings to an existing checkpoint is rejected.

## Training fails or runs out of memory

Read the worker log. Preflight is a lower-bound estimate, not a guarantee. Check free disk, VRAM, and RAM. Use the memory controls described in [Memory and optimizer](06-memory-and-optimizer.md). Prefer a short validation run to guessing at a long run's peak usage.

Fast defaults keep saved activations on GPU for speed. Enable **Offload saved activations** to reduce VRAM pressure at the cost of CPU transfers and RAM. Keep micro batch 1 and gradient checkpointing. Full-caption chunking can still need substantial activation memory. Smaller cache RAM limits reduce CPU RAM pressure; they do not shorten the caption or lower image resolution.

Paged AdamW cannot use trainable CPU components or staged UNet/VAE training. If you enable both text encoders and the VAE and need component offload, select **Disk AdamW** with the appropriate CPU/staged placement. Paged optimizer states can move to system RAM, but model weights and gradients still occupy GPU memory. BF16 weights do not guarantee that every all-component workload fits.

If the VAE produces non-finite latents, check **VAE weight and compute precision**. The tested SDXL Base VAE produced non-finite latents during fp16 computation. Try the default fp32 VAE, evaluate bf16, or use a validated fp16-safe VAE in a compatible Diffusers pipeline. Reduced-precision source files remain rounded when loaded as fp32; validate stability and image quality for the selected model and settings.

## A caption's later details seem unused

Check **Training caption context**. **Chunked** is the default, including the fast profile, and includes every caption token in multiple CLIP windows. Optional **native** uses only the first 75 content tokens per tokenizer during training. Both choices preserve the full saved text and the editor's exact complete token counts. Changing this policy requires a new run; it cannot change an existing checkpoint's conditioning history.

## Downloads are interrupted

Inspect the same repository revision and download to the same library folder. Matching partial files can resume. If the repository revision changes, verification prevents combining different files. Token permission errors may require granting model access on Hugging Face.

## Quality is poor

Review the actual processed crops and captions first. Then compare a short UNet-only run to more complex runs using fixed prompts and seeds. Learning rates, data variety, step count, and overfitting affect results. VAE adaptation is experimental and needs separate reconstruction and generation review. The app does not guarantee model quality or automatically choose a good training recipe.
