# Checkpoints and export

## Saving and stopping

**Save interval** counts optimizer updates. Checkpoints contain trainable weights/adapters, optimizer state, scheduler progress, random-number state, dataset identity, and sampler position. **Checkpoints to keep** limits completed checkpoints from that run; older checkpoints are removed only after a new one commits.

Click **Cancel task** during training to request a safe stop. The worker finishes the current optimizer update and saves a resumable checkpoint. This can take time during a slow full-model step. Let the worker finish before closing the application.

A sudden power loss or process kill can lose work since the last completed checkpoint. A partially written checkpoint is not a valid resume source.

## Resuming

Select a completed checkpoint in **Training → Files → Resume checkpoint** and use the original run output folder. Keep the same base model, finalized dataset, method, components, optimization settings, and total step count. The worker validates checkpoint compatibility. This version requires the original schedule horizon for exact resume; it does not extend a completed run by changing its total steps.

Keep checkpoint files from your own trusted runs. The application uses safetensors and structured state files rather than loading arbitrary model pickle files.

## Final model

On normal completion, LoRA training saves reusable adapters and merges them into a complete Diffusers pipeline. Full training also saves a complete Diffusers pipeline. The pipeline includes UNet, both text encoders, VAE, tokenizers, scheduler, and model configuration.

The conventional `pytorch_lora_weights.safetensors` file covers UNet and any trained text encoders. It preserves the configured alpha/rank strength for ordinary loaders. To load it with Diffusers offline, specify `weight_name="pytorch_lora_weights.safetensors"` explicitly. VAE adapters are saved separately with their configuration; they are also included in the complete merged Diffusers output. The raw per-component adapters retain their original alpha metadata and are separate from the portable conventional file.

The completed Diffusers folder is the source for checkpoint conversion. A stopped intermediate checkpoint is a resume source, not a complete pipeline.

## A single checkpoint file

Open **Checkpoint export**, select the completed Diffusers folder, and choose a new `.safetensors` file outside the model folder. Conversion combines UNet, both text encoders, and VAE into the conventional SDXL-base checkpoint key layout. It reads safetensors rather than pickle and does not overwrite an existing output.

Export preserves fp32 source weights, so the file can be larger than common fp16 shared checkpoints. Keep sufficient RAM and disk space. Validate a converted model in the application where you intend to use it before deleting any intermediate artifacts.

The converter targets standard SDXL Base architecture. Refiner, SD3, Flux, and other architectures are outside this version's scope.
