# Memory and optimizer

## Your RTX 5080 and32 GB RAM

SDXL is large. The fp32 UNet alone is roughly 10 GiB. Training adds gradients, optimizer state, activations, text encoders, and temporary allocations. Gradient accumulation reduces the size of a forward pass; it does not remove model weights or optimizer state.

Preflight reports a conservative device-memory floor and a disk estimate. Neither is a measured peak. A configuration above available memory is rejected before loading. A configuration below the estimate can still run out of memory because of activations, drivers, allocator fragmentation, and other applications.

Full training of all components on 16 GB is experimental. The application exposes it without promising that it fits. 32 GB of RAM can also become a limit with activation offload, CPU components, or export. Close other GPU workloads and monitor both RAM and VRAM during initial validation.

## Optimizers

**Disk AdamW** is the default memory-oriented optimizer. Gradients are spilled after accumulation, and fp32 optimizer moments are stored in mapped files. It applies the AdamW update without 8-bit state quantization. Gradients are globally clipped before updates. This trades disk traffic and speed for memory, not lower optimizer precision. Numerical tests compare it to ordinary torch AdamW.

An SSD with substantial free space is appropriate. Full-model runs can need hundreds of gigabytes for gradients, moments, checkpoints, and output. With all components and two retained checkpoints, the conservative estimate is around 176 GiB. Keeping one checkpoint reduces the estimate. Do not place optimizer state on a small RAM filesystem. Disk failure or lack of space can interrupt a run.

**AdamW** keeps ordinary optimizer state in memory and can be faster when it fits. **SGD** uses a different optimization method; choosing it changes training behavior and is not a quality-equivalent substitute for AdamW.

## Precision

Trainable model weights and optimizer math use fp32. bf16 forward mode uses autocasting for supported operations and preserves fp32 master weights. VAE operations run in fp32. No 8-bit optimizer state or quantized training weights are used.

Reduced-precision forward arithmetic can still change results. Select full fp32 forward computation when that matters and the larger memory use is acceptable. This version does not offer fp16 training or fp16 loss scaling.

## If a run does not fit

Use micro batch1, gradient checkpointing, disk AdamW, and activation offload. Disable training components that are unnecessary for your experiment. CPU mode remains an option at a major speed cost. Keep the native bucket resolution unless you deliberately redesign and validate the dataset pipeline; this version does not silently lower resolution to make a job fit.
