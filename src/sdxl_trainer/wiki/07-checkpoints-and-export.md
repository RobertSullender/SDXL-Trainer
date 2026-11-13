# Checkpoints and export

## Saving and stopping

**Save interval** counts optimizer updates. Checkpoints contain trainable weights/adapters, optimizer state, scheduler progress, random-number state, dataset identity, and sampler position. **Checkpoints to keep** limits completed checkpoints from that run; older checkpoints are removed only after a new one commits. New runs default to `runs/Run-1` and subsequent numbered folders inside the open project.

Fully purging a project permanently deletes its contained runs, checkpoints, outputs, and launch records. Exported checkpoints or runs stored outside that project folder are not deleted; move any results you want to keep out of the project before purging.

Click **Cancel task** during training to request a safe stop. The worker finishes the current optimizer update and saves a resumable checkpoint. This can take time during a slow full-model step. Let the worker finish before closing the application.

Resume a stopped LoRA run to finish training and produce its two standalone adapters.

A sudden power loss or process kill can lose work since the last completed checkpoint. A partially written checkpoint is not a valid resume source.

Resumable checkpoint component weights are written as standard Safetensors in bounded chunks. This reduces temporary RAM compared with materializing a complete component at once; existing checkpoint filenames and loading remain compatible. Optimizer state still requires substantial disk space and I/O.

## Resuming

In **Training → Files → Saved runs**, choose a run from the open project. The dropdown lists folders directly under the project's `runs/` directory that contain at least one completed checkpoint. Each entry shows its training method and latest completed checkpoint step, such as **Run-10 · Full Fine-Tune · Step 2,500**. If the saved metadata has no recognized method, the method is shown as unknown. The list refreshes when you open **Training** and after a run ends. To open a run outside the project or a specific `checkpoints/step-N` folder, use **Resume checkpoint → Browse…**, or enter its path and press Enter.

When you choose a run folder, **Checkpoint step** lists completed checkpoints from newest to oldest, with the latest selected. Temporary folders and steps without checkpoint metadata are skipped. If you browse to a run with no completed checkpoints, the app keeps your previous selection.

The app loads the run's `training-config.json` into the Training form, including its method, model, dataset, and training controls. If **Run output** still points to the original run when you choose an older step in the dropdown, the app suggests a new folder so the original checkpoints stay intact. Switching back to the latest step restores the original run folder if you kept the suggestion, including after saving and reopening the project. If you choose your own output folder, it carries across step changes. A suggested folder is created when training starts.

Review the settings, then click **Start training**. The usual checks apply, including the requirement for a new empty output when resuming from an older checkpoint. If you selected the older checkpoint folder directly, you can set **Run output** to a new empty folder yourself.

If the saved training settings cannot be loaded, the app fills **Run output**, displays a message, and keeps the other training controls. For a checkpoint stored in a different folder layout, set **Run output** manually.

Each training method remembers its output folder and resume checkpoint together. Keep the same base model and file variant, finalized dataset, method, components, precision and placement, optimization settings, and total step count. FP16 loss-scaler state and any fp32 disk master weights are part of the resumable checkpoint. The worker validates checkpoint compatibility and requires the original schedule horizon for exact resume.

Keep checkpoint files from your own trusted runs. The application uses safetensors and structured state files rather than loading arbitrary model pickle files.

## LoRA files

Each completed LoRA run saves both standalone adapters in its `adapters/` folder.

| File | How to load it |
| --- | --- |
| `lora_comfyui.safetensors` | Add it to your LoRA library and select it in Artificial-It’s **LoRA** slot or ComfyUI’s **Load LoRA** node. Use the SDXL base checkpoint in the model slot. |
| `lora_diffusers.safetensors` | Load the `adapters/` folder with Diffusers `load_lora_weights` and set `weight_name="lora_diffusers.safetensors"` explicitly. |

The conventional `pytorch_lora_weights.safetensors` name also appears as a compatibility alias with the same weights as `lora_diffusers.safetensors`. Use the conventional filename as `weight_name` when loading an earlier run.

Both formats contain adapters for the UNet and any trained text encoders and preserve the configured alpha/rank scale. The raw per-component adapters are saved separately with their original alpha metadata. Any trained VAE adapter is saved as a separate component file with its configuration, and its weights are also included in the merged model.

## Final model

On completion, both full training and LoRA runs save a complete Diffusers pipeline under `diffusers/`. The LoRA run merges its trained adapters into this model. The pipeline includes UNet, both text encoders, VAE, tokenizers, scheduler, and model configuration.

Under **Training → Files**, **Final Diffusers shard size (MiB)** defaults to 512 MiB. Smaller shards can reduce temporary memory needed when saving model tensors, at the cost of more files. The limit is a target: one tensor larger than the chosen size remains whole and creates a larger shard. Keep all shard and index files together. Changing this export setting does not change training weights or prevent resuming a checkpoint, and it does not control the size of resumable checkpoints or a later single-file export.

The completed Diffusers folder is the source for checkpoint conversion. A stopped intermediate checkpoint is a resume source, not a complete pipeline. App-generated Diffusers outputs use canonical filenames even when their tensor dtypes are bf16 or fp16. To start a new run from one, select file variant `default` and set the desired UNet/text and VAE precisions separately.

## A single checkpoint file

Open **Checkpoint export**, select the completed Diffusers folder, and choose a new `.safetensors` file outside the model folder. Conversion combines UNet, both text encoders, and VAE into the conventional SDXL-base checkpoint key layout. It reads safetensors rather than pickle and does not overwrite an existing output.

Export preserves the saved component weight dtypes. A model can contain mixed dtypes, such as reduced-precision UNet/text and an fp32 VAE; fp32 output can be larger than common fp16 shared checkpoints. Keep sufficient RAM and disk space. Validate a converted model in the application where you intend to use it before deleting any intermediate artifacts.

The converter targets standard SDXL Base architecture. Refiner, SD3, Flux, and other architectures are outside this version's scope.


Optimizer moments and fp32 master files are also copied in bounded, synchronized ranges during checkpoint save and resume, with advisory cache eviction where supported. This reduces uncontrolled cache buildup but cannot guarantee that the system will avoid swap.
