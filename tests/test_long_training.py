"""Long captions exercise the actual tiny SDXL worker, not just tokenization."""
import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from test_training import tiny_sdxl_model, legacy_training_config


@pytest.mark.parametrize('train_text', [False, True])
def test_tiny_sdxl_trains_mixed_short_long_captions_and_frozen_cache(tiny_sdxl_model, tmp_path, monkeypatch, train_text):
    if any(importlib.util.find_spec(name) is None for name in ('torch','safetensors')):
        pytest.skip('Local ML runtime is not installed')
    import torch
    from safetensors.torch import load_file
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import TrainingConfig, worker

    monkeypatch.setattr(dataset, 'BUCKETS', ((64, 64),))
    original_preflight = worker.preflight
    def tiny_preflight(*args, **kwargs):
        result = original_preflight(*args, **kwargs)
        result.errors = [e for e in result.errors if 'free disk space' not in e and 'GiB VRAM is free' not in e and 'CPU floor' not in e]
        return result
    monkeypatch.setattr(worker, 'preflight', tiny_preflight)
    monkeypatch.setattr(worker, '_STOP', False)
    paths=[]
    for index, color in enumerate(('purple', 'green')):
        path=tmp_path / f'{index}.png'
        Image.new('RGB', (64,64), color).save(path)
        paths.append(path)
    project=dataset.DatasetProject.create(tmp_path/'project')
    project.import_images(paths)
    project.save_caption(1, 'a')
    project.save_caption(2, ' '.join(['a']*151))
    counter=TokenCounter.from_model(tiny_sdxl_model)
    assert counter.count(project.caption_text(2)).chunks >= 3
    finalized=project.finalize_in_project(counter)
    config=legacy_training_config(model_path=str(tiny_sdxl_model), dataset_path=str(finalized),
                          output_dir=str(tmp_path/'run'), mode='full', train_text_encoder=train_text,
                          train_vae=False, device='cpu', precision='fp32', optimizer='sgd',
                          batch_size=2, gradient_accumulation_steps=1,
                          max_steps=2 if not train_text else 1, save_every=10,
                          warmup_steps=0, gradient_checkpointing=True, cache_frozen_text=True)
    worker.run(config)
    assert (Path(config.output_dir)/'diffusers/model_index.json').is_file()
    if train_text:
        from transformers import CLIPTextModel, CLIPTextModelWithProjection
        for name, cls in (('text_encoder', CLIPTextModel), ('text_encoder_2', CLIPTextModelWithProjection)):
            state=load_file(str(Path(config.output_dir)/'checkpoints/step-00000001'/f'{name}.safetensors'))
            baseline=cls.from_pretrained(str(tiny_sdxl_model/name), local_files_only=True).state_dict()
            assert state.keys() == baseline.keys()
            assert any(not torch.equal(value, baseline[key]) for key, value in state.items()), name
    else:
        cache=list((Path(config.output_dir)/'working/text').glob('*.safetensors'))
        assert len(cache)==2
        lengths=sorted(load_file(str(p))['prompt'].shape[1] for p in cache)
        assert lengths == [77,231]


def test_masked_mixed_length_batch_matches_short_caption_alone(tiny_sdxl_model):
    import torch
    from diffusers import StableDiffusionXLPipeline
    from sdxl_trainer.training.long_prompt import encode_long_prompt, pad_prompt_batch
    pipe = StableDiffusionXLPipeline.from_pretrained(str(tiny_sdxl_model), local_files_only=True, use_safetensors=True).to('cpu')
    pipe.unet.eval(); pipe.text_encoder.eval(); pipe.text_encoder_2.eval()
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    encoders = [pipe.text_encoder, pipe.text_encoder_2]
    with torch.no_grad():
        short = encode_long_prompt('a', tokenizers, encoders, 'cpu')
        long = encode_long_prompt(' '.join(['a']*151), tokenizers, encoders, 'cpu')
        prompt, pooled, mask = pad_prompt_batch([short, long], tokenizers, encoders, 'cpu')
        _, _, no_mask = pad_prompt_batch([short, short], tokenizers, encoders, 'cpu')
        assert no_mask is None
        latent = torch.randn((1,4,8,8), generator=torch.Generator().manual_seed(23))
        time_ids = torch.tensor([[64,64,0,0,64,64]], dtype=pooled.dtype)
        solo = pipe.unet(latent, 1, short[0], added_cond_kwargs={'text_embeds':short[1], 'time_ids':time_ids}).sample
        batched = pipe.unet(latent.repeat(2,1,1,1), 1, prompt, encoder_attention_mask=mask,
                            added_cond_kwargs={'text_embeds':pooled, 'time_ids':time_ids.repeat(2,1)}).sample
    torch.testing.assert_close(batched[:1], solo, rtol=1e-4, atol=3e-5)


def test_1024_content_token_caption_finalizes_and_reaches_unet_without_truncation(tiny_sdxl_model, tmp_path, monkeypatch):
    import torch
    from diffusers import UNet2DConditionModel
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import TrainingConfig, worker

    monkeypatch.setattr(dataset, 'BUCKETS', ((64, 64),))
    original_preflight = worker.preflight
    def tiny_preflight(*args, **kwargs):
        result = original_preflight(*args, **kwargs)
        result.errors = [e for e in result.errors if 'free disk space' not in e and 'GiB VRAM is free' not in e and 'CPU floor' not in e]
        return result
    monkeypatch.setattr(worker, 'preflight', tiny_preflight)
    monkeypatch.setattr(worker, '_STOP', False)
    image = tmp_path / 'original.png'
    Image.new('RGB', (64,64), 'purple').save(image)
    project = dataset.DatasetProject.create(tmp_path/'project')
    project.import_images([image])
    caption = ' '.join(['a'] * 1023 + ['b'])
    project.save_caption(1, caption)
    counter = TokenCounter.from_model(tiny_sdxl_model)
    counts = counter.count(caption)
    assert counts.clip_l == counts.clip_g == 1026 and counts.chunks == 14
    final = project.finalize_in_project(counter)
    sealed = dataset.verify_finalized(final)
    assert (final/'Img-1.txt').read_text() == caption
    assert sealed['entries'][0]['tokens']['clip_l'] == 1026
    seen = []
    original_forward = UNet2DConditionModel.forward
    def observe_unet(self, *args, **kwargs):
        context = args[2]
        seen.append((context.shape[1], bool(torch.count_nonzero(context[:, -77:]).item())))
        return original_forward(self, *args, **kwargs)
    monkeypatch.setattr(UNet2DConditionModel, 'forward', observe_unet)
    config = legacy_training_config(model_path=str(tiny_sdxl_model), dataset_path=str(final), output_dir=str(tmp_path/'run'),
                            mode='full', train_text_encoder=False, train_vae=False, device='cpu', precision='fp32',
                            optimizer='sgd', batch_size=1, gradient_accumulation_steps=1, max_steps=1,
                            save_every=10, warmup_steps=0, gradient_checkpointing=False)
    worker.run(config)
    assert seen == [(1078, True)]
    assert (Path(config.output_dir)/'diffusers/model_index.json').is_file()
