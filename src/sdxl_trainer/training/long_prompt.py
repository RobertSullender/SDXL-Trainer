"""SDXL native-window or long-caption conditioning with unchanged CLIP context.

Each tokenizer encodes the complete caption independently and splits its content
IDs into 75-token windows, each wrapped in BOS/EOS and padded to 77. The two
streams use the same number of windows; missing windows are encoded as an empty
caption. Their penultimate hidden states are concatenated on the feature axis,
then windows on the sequence axis. The CLIP-G pooled embedding comes from the
first window, matching common SDXL long-prompt practice. This is chunked
conditioning, not unlimited-context CLIP: context does not cross window bounds.
The optional native policy uses only the first 75 content tokens from each
tokenizer, matching ordinary SDXL conditioning without altering caption files.

Sources: the Diffusers SDXL pipeline uses both penultimate hidden states and the
second encoder's pooled projection, while Compel documents multiple native CLIP
windows and a first-window pooled embedding:
https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
https://github.com/damian0815/compel/blob/main/doc/compel.md
"""
from __future__ import annotations

import torch

WINDOW = 77
CONTENT_TOKENS = WINDOW - 2
CACHE_VERSION = "sdxl-chunk75-first-pooled-v1"


def _content_ids(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False, truncation=False, padding=False, verbose=False)["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if not isinstance(ids, list) or any(type(token) is not int for token in ids):
        raise ValueError("CLIP tokenizer must return a single list of token IDs.")
    return ids


def _window_ids(tokenizer, content: list[int]) -> list[int]:
    if len(content) > CONTENT_TOKENS:
        raise ValueError("A CLIP window cannot contain more than 75 content tokens.")
    bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
    if any(type(value) is not int for value in (bos, eos, pad)):
        raise ValueError("Both CLIP tokenizers must define BOS, EOS and padding token IDs.")
    return [bos, *content, eos] + [pad] * (CONTENT_TOKENS - len(content))


def encode_long_prompt(text: str, tokenizers, encoders, device, *, caption_mode="chunked"):
    """Return selected token conditioning and CLIP-G's first-window pooled vector.

    Callers control ``torch.no_grad`` and autocast. Cross-device ``.to`` keeps
    the autograd path for trainable encoders, including CPU/GPU offloading.
    Short captions produce the same 77-token IDs and hidden states as the
    ordinary Diffusers SDXL pipeline. No attention mask is supplied, matching
    the established worker and Diffusers CLIP conditioning path.
    ``chunked`` preserves every content token; ``native`` encodes one window
    per tokenizer, truncating only training conditioning after 75 content IDs.
    """
    if not isinstance(text, str) or len(tokenizers) != 2 or len(encoders) != 2:
        raise ValueError("A caption and exactly two SDXL tokenizers and encoders are required.")
    if caption_mode not in {"chunked", "native"}:
        raise ValueError("Caption mode must be chunked or native.")
    streams = [_content_ids(tokenizer, text) for tokenizer in tokenizers]
    if caption_mode == "native":
        streams = [ids[:CONTENT_TOKENS] for ids in streams]
    chunks = max(1, *((len(ids) + CONTENT_TOKENS - 1) // CONTENT_TOKENS for ids in streams))
    encoded_streams = []
    pooled = None
    for index, (tokenizer, encoder, ids) in enumerate(zip(tokenizers, encoders, streams)):
        encoder_device = next(encoder.parameters()).device
        windows = []
        for chunk in range(chunks):
            content = ids[chunk * CONTENT_TOKENS:(chunk + 1) * CONTENT_TOKENS]
            input_ids = torch.tensor([_window_ids(tokenizer, content)], dtype=torch.long, device=encoder_device)
            result = encoder(input_ids, output_hidden_states=True, return_dict=True)
            windows.append(result.hidden_states[-2].to(device))
            if index == 1 and chunk == 0:
                pooled = getattr(result, "text_embeds", None)
                if pooled is not None:
                    pooled = pooled.to(device)
        encoded_streams.append(torch.cat(windows, dim=1))
    if pooled is None:
        raise ValueError("CLIP-G did not produce projected pooled text embeddings.")
    return torch.cat(encoded_streams, dim=-1), pooled


def pad_prompt_batch(results, tokenizers, encoders, device, *, empty_prompt=None):
    """Pad shorter captions with encoded empty windows, not zero embeddings.

    Results may be cached frozen-text embeddings or live autograd tensors. The
    extra empty window is encoded once for this batch. Return an attention mask
    that hides only whole windows appended for batching, so a short caption's
    UNet conditioning does not change merely because a long caption is nearby.
    Normal within-window padding remains unmasked, matching SDXL's native path.
    """
    if not results:
        raise ValueError("A prompt batch must contain at least one caption.")
    longest = max(prompt.shape[1] for prompt, _ in results)
    if any(prompt.ndim != 3 or prompt.shape[0] != 1 or prompt.shape[1] % WINDOW for prompt, _ in results):
        raise ValueError("Invalid cached SDXL prompt conditioning shape.")
    empty = None
    if any(prompt.shape[1] != longest for prompt, _ in results):
        if empty_prompt is None:
            empty, _ = encode_long_prompt("", tokenizers, encoders, device)
        else:
            empty = empty_prompt
            if empty.ndim != 3 or tuple(empty.shape[:2]) != (1, WINDOW) or empty.shape[-1] != results[0][0].shape[-1]:
                raise ValueError("Invalid cached empty-window conditioning shape.")
    prompts = []
    masks = []
    for prompt, _ in results:
        missing = (longest - prompt.shape[1]) // WINDOW
        prompts.append(torch.cat([prompt, empty.repeat(1, missing, 1)], dim=1) if missing else prompt)
        if empty is not None:
            masks.append(torch.cat([torch.ones((1, prompt.shape[1]), dtype=torch.long, device=prompt.device),
                                    torch.zeros((1, longest - prompt.shape[1]), dtype=torch.long, device=prompt.device)], dim=1))
    mask = torch.cat(masks, dim=0) if masks else None
    return torch.cat(prompts, dim=0), torch.cat([pooled for _, pooled in results], dim=0), mask
