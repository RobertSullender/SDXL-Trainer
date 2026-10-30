# Training

## Start with a short run

Select a local Diffusers SDXL base, a finalized dataset, and a new empty output folder. Use **Check configuration** to verify files and see memory/disk estimates. Actual runtime and GPU checks happen when the worker starts.

A small run confirms that the model, captions, memory settings, and output all work before a longer experiment. No setting can guarantee image quality. Keep original model weights and your finalized dataset so results can be compared.

## Components and method

**LoRA** learns low-rank attention adapters. **Full** updates all parameters in each selected component. The UNet always trains; enable both text encoders to adapt CLIP-L and CLIP-G, and optionally enable the VAE.

VAE adaptation is experimental. It uses a separate pixel reconstruction, KL, and frozen-base latent-anchor objective. Diffusion latents are detached from that objective, so diffusion loss does not directly update the VAE. A frozen original VAE anchors its posterior means to reduce latent drift. This has not established equal or better quality on real training runs.

The two tokenizers always remain fixed. The trainer uses both encoders' penultimate hidden states and CLIP-G pooled conditioning, plus recorded size/crop conditioning. Batches contain images from one bucket, so differently sized images are not stretched to match.

## Main controls

| Control | Practical effect |
| --- | --- |
| Micro batch | Images in each forward pass. Start at 1 for16 GB. A bucket's final batch can be smaller. |
| Gradient accumulation | Micro batches averaged into each optimizer update. Gives a larger effective batch without holding them all at once. |
| Optimizer steps | Number of parameter updates, including resumed steps. This differs from image count and epochs. |
| Learning rates | Separate rates for UNet, both text encoders, and VAE. Full updates generally need a much smaller rate than LoRA. |
| LoRA rank | Adapter capacity and memory/disk cost. |
| LoRA alpha | Adapter scaling numerator; scale is alpha divided by rank. |
| LoRA dropout | Randomly drops adapter inputs during training. |
| Gradient checkpointing | Recomputes activations during backward to reduce memory. Slower. |
| Offload saved activations | Moves saved backward tensors to system RAM, adding transfers and RAM use. |
| Cache frozen text | Caches embeddings on disk when text encoders do not train. |
| Forward precision | bf16 autocast or fp32. Trainable weights and optimizer math remain fp32; VAE computation is fp32. |
| Device | CUDA or CPU. CPU can be extremely slow and still needs sufficient RAM. |
| Schedule / warmup | Shape learning rates over optimizer steps. |
| Gradient norm limit | Clips large gradients. Zero disables clipping. |
| Min-SNR / noise offset | Optional changes to the diffusion objective. Zero keeps these disabled. |
| VAE loss weights | Balance reconstruction, KL, and latent anchoring. Relevant only when training VAE. |
| Seed | Controls initialization, image ordering, noise, and timestep sampling. Hardware/library differences can still affect results. |

The tabs expose every implemented backend field. Hover a control to read its explanation. Save a recipe to preserve the complete configuration as JSON, or load one for a later run.

See [Memory and optimizer](06-memory-and-optimizer.md) and [Checkpoints and export](07-checkpoints-and-export.md).
