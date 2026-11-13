# Training

## Start with a short run

Select a local Diffusers SDXL base and a finalized dataset. In **Training → Files → Choose project dataset…**, select a bucket to train it on its own or **Dataset-N / All aspect ratios** to train across that revision's buckets in one run. The all-aspect-ratios option reads the existing bucket subfolders directly. To browse manually, choose the revision folder, such as `finalized/Dataset-1`, or a single bucket folder containing `manifest.json`, such as `finalized/Dataset-1/Photo-Realistic:1024X1024`.

With a project open, a method without a saved run folder suggests the next available numbered folder under `runs/`. The suggestion skips folders already on disk or selected by the other method. You can select another output folder. Use **Check configuration** to verify files and see memory/disk estimates. Actual runtime and GPU checks happen when the worker starts.

Project-contained runs, checkpoints, and launch records are deleted if you choose **Fully purge project…**. Runs or exports you explicitly store outside the project are not included in that purge.

A small run confirms that the model, captions, memory settings, and output all work before a longer experiment. No setting can guarantee image quality. Keep original model weights and your finalized dataset so results can be compared.

The learning rate shown during warmup is the current rate, not necessarily the configured final rate. For example, a configured `1e-4` with 100 warmup steps begins at `1e-6` and rises to `1e-4`. A small dataset can be revisited many times: four images with micro batch 1 and accumulation 4 correspond to approximately one pass over that dataset per optimizer step. Review the configured rate and step count before a longer full fine-tune.

## Saving training settings

Click **Save training settings** to store both methods' settings and the selected method in `training-settings.json` inside the open project's folder. Switching projects, closing the app, or starting training also saves the open project's training settings. Reopening the project restores its saved settings.

Changing **Training method** remembers the current method's settings and restores the selected method's settings, including its run output folder and resume checkpoint. A method starts with its preset the first time it is selected. Both methods share the base model path, Diffusers file variant, and dataset path.

In **Training → Files → Saved runs**, choose a run from the open project by name, method, and latest completed checkpoint step. Use **Resume checkpoint → Browse…** for a run outside the project or a specific checkpoint folder. Selecting a run loads its saved training settings and selects its latest completed checkpoint. Use **Checkpoint step** to choose an older step. If you pick an older step while using the original run's output, the app suggests a new folder. You can edit **Run output** to choose where training will save. The app restores your saved output choice when switching methods, reopening a project, or loading a recipe. [Resuming a checkpoint](07-checkpoints-and-export.md)

When opening a project saved before 0.1.28, the previously selected method keeps the saved output folder. If the other method has a checkpoint saved under a standard `checkpoints/step-N` path, the app uses the run folder that contains it. Any explicitly saved output folder takes precedence.

**Save training settings** is available with a project open. For a one-off run with an external dataset, fill out the Training form and start the run directly. Changes made with no project open are temporary. Use **Save recipe…** to export the current form as a JSON recipe for a single run. **Load recipe…** restores the method and settings in that recipe.

New projects start with built-in training defaults and the shared SDXL base model and Diffusers file variant set under **Models & runtime → Local runtime**. On upgrade to 0.1.24, the app brings previously saved training settings into the last-open project and uses its base model and file variant to seed the shared choices.

## Reset all training settings

With a project open, click **Reset all training settings** beside **Save training settings**, then choose the preset you want to apply.

| Preset | UNet learning rate | Optimizer |
| --- | --- | --- |
| **LoRA defaults** | `1e-4` | **AdamW · device memory** with **Fused GPU AdamW** enabled |
| **Full fine-tune defaults** | `1e-6` | **Paged AdamW · 8-bit moments (direct weights)** with CUDA binary override `130` |

The preset selects the chosen method and restores its step counts and other controls, including its fast performance settings. It keeps that method's output folder and the shared base model path, Diffusers file variant, and dataset path. It clears the chosen method's resume checkpoint selection and immediately saves to `training-settings.json`. Choose **Cancel** to leave the form as it is.

## Fast training defaults

The fast LoRA and full fine-tune profiles train the UNet and freeze both text encoders and the VAE. Both use BF16 UNet and text weights and forward precision, an FP32 VAE, gradient checkpointing, and GPU saved activations. **Training caption context** stays **chunked**, so complete captions remain in use.

Before optimizer steps begin, the worker prepares frozen image and text caches on GPU while the trainable model is still off GPU. Frozen components then move off the training GPU. Steps reuse cached conditioning instead of repeatedly decoding PNGs and running frozen models. **Frozen VAE cache sampling** defaults to **sample**, reusing one deterministic sampled latent per image. Choose **posterior** for a fresh latent on every visit. These sampling choices change training randomness.

The cache RAM limits default to **4096 MiB for text** and **1024 MiB for VAE**, allocated only as entries are used. Validated local cache files remain available when an entry is not in RAM. Larger limits can reduce disk reads but compete with other application and system memory.

**Fast training defaults** applies the performance selections for the current training method and shows the selected method and optimizer in the status line. It keeps the chosen paths, learning rates, duration, seed, and captioning settings. With a project open, it saves the updated training settings to that project. With no project open, it updates the current form for a one-off run.

The fast selections retain all existing controls. Re-enable trainable text encoders or VAE, select FP32 weights, use Disk AdamW, or enable CPU activation offload when appropriate. Paged AdamW cannot combine trainable CPU components or staged UNet/VAE offload; use Disk AdamW for those placements. All-component training is substantially heavier than the fast UNet-only defaults.

## Components and method

**LoRA** learns low-rank attention adapters. **Full** updates all parameters in each selected component. The UNet always trains; enable both text encoders to adapt CLIP-L and CLIP-G, and optionally enable the VAE.

VAE adaptation is experimental. It uses a separate pixel reconstruction, KL, and frozen-base latent-anchor objective. Diffusion latents are detached from that objective, so diffusion loss does not directly update the VAE. A frozen original VAE anchors its posterior means to reduce latent drift. This has not established equal or better quality on real training runs. Without component offload, a 1024×1024 test on a 16 GB RTX 5080 Laptop GPU ran out of VRAM before the first optimizer update in both full and LoRA modes with all components enabled; other configurations have not been ruled out.

The two tokenizers always remain fixed. The trainer uses both encoders' penultimate hidden states and CLIP-G pooled conditioning, plus recorded size/crop conditioning. When training all aspect ratios, a run draws micro batches from across the selected buckets. Each micro batch contains images with the same dimensions.

The default **chunked** caption context uses multiple independent native 77-position CLIP windows, each containing up to 75 caption tokens. The UNet receives concatenated CLIP-L and CLIP-G penultimate hidden states for every window; CLIP-G's first-window pooled output supplies SDXL pooled conditioning. Batch-only padding windows are masked. This preserves every token but does not give CLIP cross-window attention; compatible chunking is needed in inference software to use long prompts similarly. More windows increase memory and time. Optional **native** context uses just the first 75 content tokens per CLIP for a run. Saved caption files remain complete in either mode. See [Captioning and token counts](04-captioning.md).

## Main controls

| Control | Practical effect |
| --- | --- |
| Micro batch | Images in each forward pass. Start at 1 for 16 GB. A bucket's final batch can be smaller. |
| Gradient accumulation | Micro batches averaged into each optimizer update. Gives a larger effective batch without holding them all at once. |
| Optimizer steps | Number of parameter updates, including resumed steps. This differs from image count and epochs. |
| Learning rates | Separate rates for UNet, both text encoders, and VAE. Full updates generally need a much smaller rate than LoRA. |
| LoRA rank | Adapter capacity and memory/disk cost. |
| LoRA alpha | Adapter scaling numerator; scale is alpha divided by rank. |
| LoRA dropout | Randomly drops adapter inputs during training. |
| Optimizer | Fast full uses paged 8-bit AdamW; fast LoRA uses ordinary AdamW. Paged AdamW directly updates weights without fp32 masters and requires all trainable components on one GPU. Disk AdamW and SGD remain available. Review [Memory and optimizer](06-memory-and-optimizer.md) for tradeoffs. |
| 8-bit CUDA binary override | Local bitsandbytes CUDA binary version. On this machine's current CUDA 13.2 runtime and bitsandbytes 0.49.2 installation, `130` is required and passed the native probe. Other installations may differ. |
| Fused GPU AdamW | Uses fused CUDA kernels for ordinary AdamW when all trainable weights share one GPU. CPU or mixed placement uses ordinary updates. Ignored by other optimizers; arithmetic may differ slightly. |
| Component offload | Select CPU text encoders, optionally a CPU VAE, or disk-backed UNet staging with text encoders on CPU or GPU. Trades GPU memory for system RAM, disk space and time. See [Memory and optimizer](06-memory-and-optimizer.md). |
| Disk AdamW update device | Auto uses CPU for reduced-precision trainable weights with fp32 masters; explicit CUDA can stream the updates when all trainable weights are on one GPU and sufficient VRAM is free. Auto may use GPU for fp32 weights. The saved backend is retained on resume. |
| Gradient checkpointing | Recomputes activations during backward to reduce memory. Slower. |
| Offload saved activations | Moves saved backward tensors to system RAM, adding transfers and RAM use. Off in fast defaults; enable to reduce VRAM pressure. |
| Allow TF32 acceleration | Allows NVIDIA TF32 matmul/cuDNN paths with BF16 or FP16 forward. FP32 forward always disables TF32. This can change numerical results. |
| Training caption context | `chunked` uses the complete caption and is the default. `native` uses the first 75 content tokens per tokenizer for training only. |
| Cache frozen text | Reuses embeddings when text encoders do not train; validated entries can remain in bounded CPU RAM. |
| Cache frozen VAE | Enables the selected frozen-VAE cache. Bypassed when training the VAE. Posterior mode requires FP32 VAE computation. |
| Frozen VAE cache sampling | `sample` reuses one deterministic sampled latent per image. `posterior` stores FP32 mean/log-variance and samples afresh on every visit. |
| Prepare frozen caches before training | Encodes frozen components on GPU before placing trainable components there. CPU training prepares on CPU. Disable for lazy CPU cache preparation. Trainable components are not cached. |
| Frozen text/VAE cache RAM limits | Maximum CPU RAM for verified cache entries: 4096/1024 MiB by default, allocated only as used. Zero uses the corresponding disk cache without a RAM layer. |
| Show phase timings | Shows elapsed step time, image throughput, and phase timings. Profiling adds one CUDA synchronization at the end of each step. |
| Monitor NVIDIA GPU | Samples utilization and VRAM locally for the selected NVIDIA device about every two seconds. |
| Diffusers file variant | Under **Files**, choose `default`, `fp16`, or `bf16` to match the local safetensors filenames. The corresponding files must exist for all components. |
| Final Diffusers shard size | Under **Files**, select a target size for each final output shard. Smaller targets can reduce temporary save memory but add files; an individual tensor can exceed the target. See [Checkpoints and export](07-checkpoints-and-export.md). |
| UNet and text weight precision | `fp32`, `bf16`, or `fp16` model weights. Disk AdamW uses fp32 master weights for reduced-precision full training; paged 8-bit AdamW updates live weights directly. LoRA adapters stay fp32. |
| Trainable CPU text precision | `match` uses model precision; opt-in `fp32` runs both trainable CPU text encoders in fp32. Requires both encoders enabled and on CPU, and can require about 1.5 GiB more RAM than 16-bit text weights. Source rounding and mixed-precision numerics still matter. |
| VAE weight and compute precision | Defaults to fp32. The tested SDXL Base VAE produced non-finite latents in fp16; evaluate bf16 or a validated fp16-safe VAE in a compatible pipeline if needed. Check stability and quality for your model. |
| Forward precision | FP32, BF16, or FP16 autocast. Reduced model weights require matching forward precision. FP16 uses dynamic loss scaling. |
| FP16 loss scale controls | Set the initial scale and growth interval. On non-finite gradients the same batch is retried with a reduced scale, with scaler state saved in checkpoints. |
| Device | CUDA or CPU. CPU can be extremely slow and still needs sufficient RAM. |
| Schedule / warmup | Shape learning rates over optimizer steps. |
| Gradient norm limit | Clips large gradients. Zero disables clipping. |
| Min-SNR / noise offset | Optional changes to the diffusion objective. Zero keeps these disabled. |
| VAE loss weights | Balance reconstruction, KL, and latent anchoring. Relevant only when training VAE. |
| Seed | Controls initialization, image ordering, noise, and timestep sampling. Hardware/library differences can still affect results. |

The tabs expose every implemented backend field. Hover a control to read its explanation.

## Reading performance information

The live GPU line reports the selected device's utilization and VRAM usage. These are sampled totals for the entire GPU, including other programs, not process-specific readings. Brief bursts can fall between samples. The monitor uses a local NVIDIA utility and may report unavailable if the utility, driver, or selected-device identification is unavailable; training can still proceed.

After each optimizer step, the app reports image throughput and phase wall times for image loading, text and VAE encoding, diffusion preparation, UNet, backward, and optimizer work. CPU wall intervals measure the calling thread and may overlap asynchronous GPU work; do not add them as exclusive components of total step time. CUDA stream elapsed intervals also may include idle time between GPU operations. They are not GPU utilization percentages. GPU events synchronize once at the step boundary when phase profiling is enabled, which adds some overhead. Checkpoint saving occurs after the step and is timed separately when present. Live monitoring also reports available system RAM and swap, plus the training worker's swap when available; system totals include other processes.

Cache preparation appears before optimizer-step progress. Its initial work depends on image count, bucket sizes, caption windows, and existing validated entries; it is not a stalled first optimizer step. With preparation disabled, the first visit encodes each uncached image/caption lazily. A run with **Train VAE** enabled does not reuse the frozen-VAE cache, and trainable text encoders bypass the text cache. Training those components can raise memory use and change speed considerably.

Cache RAM limits, monitoring controls, gradient storage and final Diffusers shard target are performance options, so changing them does not by itself change a checkpoint's training configuration identity. Earlier compatible checkpoints can be resumed with their original model, dataset, learning settings, and total step count; loading an older recipe retains its legacy numerical behavior. Numerical selections such as caption context, fixed-sample versus posterior sampling, TF32, fused AdamW, precision, and optimizer are recorded for resume. Use the saved recipe when resuming; changing these selections requires a separate run. Cache preparation placement is also retained in run identity. Disk AdamW records its update backend and automatically preserves it when resuming; older checkpoints use their original CPU backend.

See [Memory and optimizer](06-memory-and-optimizer.md) and [Checkpoints and export](07-checkpoints-and-export.md).
