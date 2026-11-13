# Captioning and token counts

## Describe the image that will train

Captioning happens after conversion, cropping, and padding. The managed llama.cpp server receives the exact processed PNG through a local loopback connection. Images are not sent to Hugging Face or any external caption service.

On the **Dataset** page, Ctrl-click to select individual images or Shift-click to select a range. **Caption selected** generates captions for every highlighted image in draft order and replaces any existing captions on those images. **Caption all empty** fills blank captions across the draft. A batch saves each completed caption as it goes. Review generated captions for accuracy.

**New project** turns on **Caption settings → Trim generated captions to 77 tokens**. It shortens each overlong result from either generation action before saving it. It checks both tokenizers from the local SDXL base model selected in **Training → Files**. The limit includes the start and end tokens. Review any incomplete-caption warnings after generation. Uncheck it to generate longer captions for windowed training.

The editor saves changes automatically. The offline English spell checker underlines words it does not recognize. Right-click to choose a suggestion. Names, tags, and trigger words may be flagged even when intentional; spelling never automatically changes your text. If the dictionary is unavailable, the UI reports that state.

## Edit selected captions or the whole draft

Open **Bulk caption…** on the **Dataset** page and choose **Selected images** or **All draft images**. The Apply button shows how many images the action targets.

- **Set every caption** replaces the caption on each targeted image with your text.
- **Prepend to captions** and **Append to captions** add your text before or after the caption, separated by a comma. Empty captions receive the text on its own.
- **Fill empty captions** applies your text only to blank captions, including captions containing only whitespace.
- **Trim to 77 tokens** keeps the opening text of captions that exceed one CLIP window. It prefers a nearby comma or a whole-word ending and checks both tokenizers, including the start and end tokens. Use the SDXL base model you plan to train. Review trimmed captions before finalizing, especially when a trigger word or important detail is near the end.

Use **Trim to 77…** next to **Bulk caption…** to open the dialog directly in trim mode. With one image selected, the shortcut defaults to that image.

Choose **Finalize dataset** to export the updated captions in a new revision.

## Both SDXL tokenizers and long captions

The editor counts with both tokenizers from the selected local Diffusers base model, not the Qwen tokenizer or a word count. Each total includes one start and end token. The window count shows how many native 77-position windows are required for the longer of the two token streams: each window holds up to 75 content tokens plus start/end and padding. For example, a maximum count of 152 means 150 content tokens across two windows; 153 means three windows. Each window adds its own start/end and padding, so the total encoded positions are 77 times the window count. Training uses the same tokenizer assets. Selecting a different model reloads the tokenizers; finalization records their fingerprints and full counts.

**Training → Memory → Training caption context** defaults to **chunked**, including with fast training defaults. Every caption token is used. Optional **native** mode uses only the first 75 content tokens from each tokenizer in one 77-position window during training. This can reduce conditioning work, but loses the later description for that run. It never shortens saved captions, changes caption-generation controls, or changes the editor's complete token counts. Select it deliberately; the fast-default action keeps complete-caption chunking.

For a long caption, the two token streams are split independently into 75-content-token windows. Each encoder processes its windows separately. Their penultimate hidden states are aligned and concatenated for UNet cross-attention; CLIP-G's pooled vector comes from its first window. When a batch contains captions of different lengths, extra windows added only for batch alignment are masked from the UNet. Each CLIP still has its native context: information in one window cannot attend directly to another within CLIP. This is a long-prompt conditioning strategy, not unlimited-context CLIP. Long captions increase training memory and compute. At inference, use a program and long-prompt settings with compatible chunking; a program that truncates at 77 tokens cannot reproduce the full conditioning.

## Caption controls

| Control | Meaning |
| --- | --- |
| Model / mmproj | The local GGUF language model and matching vision projector. Both are required. |
| System prompt | Instructions sent with each image in the current project. Edit them for that dataset. |
| User prompt | The request sent with each image in the current project. |
| Temperature | Randomness. Lower values usually give more consistent descriptions. |
| Top p / top k / min p / typical p | Filters that limit candidate tokens before sampling. |
| Repeat, presence, frequency penalties | Controls for discouraging repeated content. Excessive penalties can harm captions. |
| Maximum tokens | Per-response generation budget, including reasoning when thinking is enabled. |
| Maximum continuations | Number of automatic follow-up responses when a response reaches its token limit. Zero saves the available partial caption with a warning. |
| Thinking | Requests reasoning from the chat template. Off by default for concise captions. |
| Context size | Shared space for image tokens, instructions, previous responses, and the answer. Large images and long continuations need more context and memory. |
| GPU layers | How much of the language model runs on GPU. Lower to use more CPU and less VRAM. |
| Projector offload | Run vision processing on GPU; disabling saves VRAM at a speed cost. |
| Batch / micro-batch | Prompt processing sizes. Lower values can save memory. |
| Threads / flash attention | CPU concurrency and supported attention implementation. |
| Seed | Repeatable sampling seed; -1 requests random. |
| Timeouts | How long model loading and generation may take before reporting an error. |

With a project open, click **Save caption settings** to save the whole form for that project, including its trim preference, prompts, generation controls, and local paths. Switching projects or closing the app also saves your edits. Reopening the project restores its saved settings.

Set the Qwen GGUF, vision projector (mmproj), and llama.cpp server in **Models & runtime → Shared caption paths**, then click **Save shared caption paths**. New projects start with those paths, the built-in prompts and generation controls, and 77-token trimming enabled. You can select different paths in a project's **Caption settings** and save them with that project. Use **Reset system prompt** or **Reset user prompt** to restore that prompt's default.

Use **Reset all caption settings** beside the prompt buttons to restore both prompts and generation controls, enable 77-token trimming, and save the open project's settings immediately. The reset keeps the selected Qwen GGUF, mmproj, and server paths.

When a response reaches its per-response token limit, the app automatically continues the accumulated assistant response, up to **Maximum continuations**. The text returned by each response is joined at its exact boundary, but the model may repeat, omit, or format text unexpectedly; review it. If a continuation cannot finish because the limit or context is exhausted, the usable text is saved and a warning identifies every affected image and reason. The project remembers incomplete captions even if a later image fails, the batch is cancelled, or the app is closed. They are marked ⚠ in the image list and cannot be finalized until you edit the caption or successfully re-caption it. An unchanged automatic save does not mark a caption reviewed. Empty or unfinished reasoning without a usable caption still reports an error. The server unloads when a batch ends or is cancelled, freeing its VRAM before training.
