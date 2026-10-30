# Captioning and token counts

## Describe the image that will train

Captioning happens after conversion, cropping, and padding. The managed llama.cpp server receives the exact processed PNG through a local loopback connection. Images are not sent to Hugging Face or any external caption service.

**Caption selected** replaces the selected caption. **Caption all empty** only fills blank captions. A batch saves each completed caption as it goes. Review generated captions: a vision model can misunderstand an image or invent detail.

The editor saves changes automatically. The offline English spell checker underlines words it does not recognize. Right-click to choose a suggestion. Names, tags, and trigger words may be flagged even when intentional; spelling never automatically changes your text. If the dictionary is unavailable, the UI reports that state.

## Both SDXL tokenizers

SDXL uses CLIP-L and CLIP-G. Counts are computed using the two tokenizers from the selected local Diffusers base model, not the Qwen tokenizer, a word count, or a web service.

Each encoder has 77 positions including the start and end tokens: usually75 positions remain for caption content. Text is tokenized without truncation, so the UI shows the true count even when it exceeds the limit. A caption fits only if both counts are at most77.

This version blocks overlong captions at finalization and training. It does not silently drop words or implement long-caption chunking. Shorten the caption while preserving useful visible detail. Selecting a different base model restarts the token counter; finalization records its tokenizer fingerprints.

## Caption controls

| Control | Meaning |
| --- | --- |
| Model / mmproj | The local GGUF language model and matching vision projector. Both are required. |
| System prompt | Instructions shared by every image. Describe what you want captions to include. |
| User prompt | The request sent with the image. |
| Temperature | Randomness. Lower values usually give more consistent descriptions. |
| Top p / top k / min p / typical p | Filters that limit candidate tokens before sampling. |
| Repeat, presence, frequency penalties | Controls for discouraging repeated content. Excessive penalties can harm captions. |
| Maximum tokens | Generation budget. Includes reasoning when thinking is enabled. A truncated answer is rejected. |
| Thinking | Requests reasoning from the chat template. Off by default for concise captions. |
| Context size | Space for image tokens, instructions, and the answer. Too small can reject the request. |
| GPU layers | How much of the language model runs on GPU. Lower to use more CPU and less VRAM. |
| Projector offload | Run vision processing on GPU; disabling saves VRAM at a speed cost. |
| Batch / micro-batch | Prompt processing sizes. Lower values can save memory. |
| Threads / flash attention | CPU concurrency and supported attention implementation. |
| Seed | Repeatable sampling seed; -1 requests random. |
| Timeouts | How long model loading and generation may take before reporting an error. |

All backend caption settings are visible in **Caption settings** with hover explanations. The server unloads when the batch ends or is cancelled, freeing its VRAM before training.
