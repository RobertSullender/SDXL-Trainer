# Troubleshooting

## Token counts are unavailable

Select the Diffusers base-model folder in **Training → Files**, then inspect the chosen training Python. It needs Transformers and both local tokenizer folders. The application refuses approximate counts. Finalization needs this runtime even if captions were written manually.

## Caption model fails to load

Check that the selected files are GGUF files, that the mmproj belongs to that model family, and that the NVIDIA driver works. Both presets include the correct projector filename. Reduce GPU layers, disable projector offload, or reduce context/micro-batch if VRAM is exhausted. CPU captioning is possible but slower.

The pinned engine is bundled with the installed package. Development source runs need the prepared runtime under `build/llama-runtime`. Do not select an unrelated old llama.cpp binary and assume it supports Qwen3.5 vision.

## Caption generation ends early

The app rejects truncated output and unfinished reasoning. Disable thinking, raise the generation budget, and ensure the context has room for image tokens plus instructions. A fluent caption can still be inaccurate; review it against the preview.

## Spelling is unavailable

Install the English Hunspell dictionary and Enchant2 system library. The Debian package declares these dependencies. Spell checking remains offline and never silently edits text.

## A dataset fails verification

Working pixels or finalized files changed after their hashes were recorded. Use the original working project and finalize again. Do not edit a sealed export in place. A changed tokenizer also requires revalidation with the intended model.

## CUDA is unavailable

Run **Inspect runtime**. A working driver, a compatible Python version, and a CUDA-enabled torch wheel supporting Blackwell are all required. System CUDA toolkit installation alone does not make a CPU-only torch wheel use CUDA.

## Training fails or runs out of memory

Read the worker log. Preflight is a lower-bound estimate, not a guarantee. Check free disk, VRAM, and RAM. Use the memory controls described in [Memory and optimizer](06-memory-and-optimizer.md). Prefer a short validation run to guessing at a long run's peak usage.

## Downloads are interrupted

Inspect the same repository revision and download to the same library folder. Matching partial files can resume. If the repository revision changes, verification prevents combining different files. Token permission errors may require granting model access on Hugging Face.

## Quality is poor

Review the actual processed crops and captions first. Then compare a short UNet-only run to more complex runs using fixed prompts and seeds. Learning rates, data variety, step count, and overfitting affect results. VAE adaptation is experimental and needs separate reconstruction and generation review. The app does not guarantee model quality or automatically choose a good training recipe.
