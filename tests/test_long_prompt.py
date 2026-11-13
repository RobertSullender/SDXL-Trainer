"""Actual CLIP models validate chunking, native parity, gradients and padding."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from sdxl_trainer.training.long_prompt import encode_long_prompt, pad_prompt_batch


@pytest.fixture
def clips(tmp_path):
    torch.manual_seed(3)
    (tmp_path/'vocab.json').write_text(json.dumps({'<|startoftext|>': 0, '<|endoftext|>': 1, 'a</w>': 2, 'b</w>': 3}))
    (tmp_path/'merges.txt').write_text('#version: 0.2\n')
    first = CLIPTokenizer(str(tmp_path/'vocab.json'), str(tmp_path/'merges.txt'), model_max_length=77)
    second = CLIPTokenizer(str(tmp_path/'vocab.json'), str(tmp_path/'merges.txt'), model_max_length=77)
    first_encoder = CLIPTextModel(CLIPTextConfig(vocab_size=4, hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=77, bos_token_id=0, eos_token_id=1, pad_token_id=1))
    second_encoder = CLIPTextModelWithProjection(CLIPTextConfig(vocab_size=4, hidden_size=12, intermediate_size=24, num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=77, projection_dim=12, bos_token_id=0, eos_token_id=1, pad_token_id=1))
    first_encoder.eval(); second_encoder.eval()
    return (first, second), (first_encoder, second_encoder)


@pytest.mark.parametrize('caption_mode', ['chunked', 'native'])
def test_short_caption_exactly_matches_native_sdxl_conditioning(clips, caption_mode):
    tokenizers, encoders = clips
    text = 'a b a'
    actual, pooled = encode_long_prompt(text, tokenizers, encoders, 'cpu', caption_mode=caption_mode)
    direct = []
    with torch.no_grad():
        for tokenizer, encoder in zip(tokenizers, encoders):
            ids = tokenizer(text, padding='max_length', max_length=77, truncation=False, return_tensors='pt').input_ids
            result = encoder(ids, output_hidden_states=True, return_dict=True)
            direct.append(result.hidden_states[-2])
            if getattr(result, 'text_embeds', None) is not None:
                expected_pooled = result.text_embeds
    assert actual.shape == (1, 77, 20)
    assert torch.equal(actual.detach(), torch.cat(direct, dim=-1))
    assert torch.equal(pooled.detach(), expected_pooled)


def test_native_long_caption_matches_direct_truncation_without_tail_gradients(clips):
    tokenizers, encoders = clips
    text = ' '.join(['a'] * 75 + ['b'] * 90)
    observed = [[], []]
    handles = [encoder.register_forward_pre_hook(lambda _module, args, index=index: observed[index].append(args[0].detach().clone()))
               for index, encoder in enumerate(encoders)]
    try:
        prompt, pooled = encode_long_prompt(text, tokenizers, encoders, 'cpu', caption_mode='native')
    finally:
        for handle in handles:
            handle.remove()
    direct = []
    with torch.no_grad():
        for index, (tokenizer, encoder) in enumerate(zip(tokenizers, encoders)):
            ids = tokenizer(text, padding='max_length', max_length=77, truncation=True, return_tensors='pt').input_ids
            assert len(observed[index]) == 1
            assert torch.equal(observed[index][0], ids)
            result = encoder(ids, output_hidden_states=True, return_dict=True)
            direct.append(result.hidden_states[-2])
            if getattr(result, 'text_embeds', None) is not None:
                expected_pooled = result.text_embeds
    assert prompt.shape == (1, 77, 20)
    assert torch.equal(prompt.detach(), torch.cat(direct, dim=-1))
    assert torch.equal(pooled.detach(), expected_pooled)
    prompt.square().sum().backward()
    for encoder in encoders:
        grad = encoder.get_input_embeddings().weight.grad
        assert grad is not None
        assert torch.count_nonzero(grad[2]).item() > 0
        assert torch.count_nonzero(grad[3]).item() == 0


def test_unknown_caption_policy_is_rejected(clips):
    tokenizers, encoders = clips
    with pytest.raises(ValueError, match='Caption mode'):
        encode_long_prompt('a', tokenizers, encoders, 'cpu', caption_mode='unknown')


def test_long_caption_keeps_tail_and_trains_both_clip_encoders(clips):
    tokenizers, encoders = clips
    # The unique 'b' appears only after the native first 75 content tokens.
    first = ' '.join(['a'] * 75)
    prompt, pooled = encode_long_prompt(first + ' b', tokenizers, encoders, 'cpu')
    reference, pooled_first = encode_long_prompt(first, tokenizers, encoders, 'cpu')
    assert prompt.shape == (1, 154, 20)
    assert torch.equal(prompt[:, :77], reference)
    assert torch.equal(pooled, pooled_first)  # pooled CLIP-G is first-window only
    prompt[:, 77:, :].square().sum().backward()
    for encoder in encoders:
        embedding = encoder.get_input_embeddings().weight
        assert embedding.grad is not None and torch.count_nonzero(embedding.grad[3]).item() > 0


def test_variable_length_batch_uses_encoded_empty_window_and_preserves_gradients(clips):
    tokenizers, encoders = clips
    short = encode_long_prompt('a', tokenizers, encoders, 'cpu')
    long = encode_long_prompt(' '.join(['a'] * 150 + ['b']), tokenizers, encoders, 'cpu')
    prompt, pooled, mask = pad_prompt_batch([short, long], tokenizers, encoders, 'cpu')
    empty, _ = encode_long_prompt('', tokenizers, encoders, 'cpu')
    assert prompt.shape == (2, 231, 20)
    assert pooled.shape == (2, 12)
    assert mask.shape == (2, 231)
    assert torch.equal(mask[0], torch.tensor([1] * 77 + [0] * 154))
    assert torch.all(mask[1] == 1)
    assert torch.equal(prompt[0:1, 77:154], empty)
    assert torch.equal(prompt[0:1, 154:231], empty)
    assert torch.equal(prompt[1:2], long[0])
    prompt[0, 77:].square().sum().backward()
    assert all(encoder.get_input_embeddings().weight.grad is not None for encoder in encoders)


def test_chunked_counting_does_not_emit_untruncated_length_warning(clips, caplog):
    from sdxl_trainer.tokens import TokenCounter
    tokenizers, encoders = clips
    text = ' '.join(['a'] * 151)
    with caplog.at_level('WARNING'):
        counts = TokenCounter(*tokenizers).count(text)
        prompt, _ = encode_long_prompt(text, tokenizers, encoders, 'cpu')
    assert counts.chunks == 3
    assert prompt.shape[1] == 231
    assert not any('sequence length is longer than' in record.message.lower() for record in caplog.records)


@pytest.mark.parametrize(('content_count','windows'), [(256,4),(512,7),(1024,14)])
def test_very_long_captions_preserve_all_clip_windows_and_tail_gradients(clips, content_count, windows):
    tokenizers, encoders = clips
    observed = [[], []]
    handles = [encoder.register_forward_pre_hook(lambda _module, args, index=index: observed[index].append(args[0].detach().clone()))
               for index, encoder in enumerate(encoders)]
    try:
        text = ' '.join(['a'] * (content_count - 1) + ['b'])
        prompt, _ = encode_long_prompt(text, tokenizers, encoders, 'cpu')
        assert prompt.shape == (1, 77 * windows, 20)
        assert all(len(calls) == windows for calls in observed)
        # The final content token occurs after every earlier native CLIP window.
        final_position = (content_count - 1) % 75 + 1
        assert all(int(calls[-1][0, final_position]) == 3 for calls in observed)
        prompt[:, (windows - 1) * 77:, :].square().sum().backward()
        for encoder in encoders:
            grad = encoder.get_input_embeddings().weight.grad
            assert grad is not None and torch.count_nonzero(grad[3]).item() > 0
    finally:
        for handle in handles:
            handle.remove()
