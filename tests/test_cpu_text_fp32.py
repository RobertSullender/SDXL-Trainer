"""Explicit fp32 CPU text precision and mixed-precision resume contracts."""
from pathlib import Path
import pytest
from PIL import Image
from test_training import tiny_sdxl_model, legacy_training_config


def test_fp32_cpu_text_option_validation_and_legacy_identity():
    from sdxl_trainer.training import TrainingConfig
    from sdxl_trainer.training.worker import _config_identity
    assert any('requires both text encoders' in e for e in TrainingConfig(cpu_text_precision='fp32').validate())
    assert not any('FP32 CPU text precision' in e for e in TrainingConfig(
        cpu_text_precision='fp32', train_text_encoder=True, device='cuda', component_offload='text_encoders_cpu').validate())
    assert not any('FP32 CPU text precision' in e for e in TrainingConfig(
        cpu_text_precision='fp32', train_text_encoder=True, device='cpu').validate())
    assert _config_identity(TrainingConfig())['cpu_text_precision'] == 'match'
    assert _config_identity(TrainingConfig(cpu_text_precision='fp32'))['cpu_text_precision'] == 'fp32'


def test_fp32_cpu_text_uses_native_backend_only_for_other_reduced_cpu_components(monkeypatch):
    import torch
    from sdxl_trainer.training import TrainingConfig, worker
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(worker, '_run', lambda *args: torch.backends.mkldnn.enabled)
    values=dict(model_precision='bf16', precision='bf16', train_text_encoder=True,
                device='cuda', component_offload='text_encoders_cpu', monitor_gpu=False, profile_phases=False)
    with torch.backends.mkldnn.flags(enabled=True):
        assert worker.run(TrainingConfig(**values, cpu_text_precision='fp32'))
        assert not worker.run(TrainingConfig(**values, cpu_text_precision='match'))
        values.update(component_offload='text_and_vae_cpu', train_vae=True, vae_precision='bf16')
        assert not worker.run(TrainingConfig(**values, cpu_text_precision='fp32'))


@pytest.mark.parametrize('mode', ['full', 'lora'])
def test_cpu_fp32_text_bf16_unet_multicaption_training_and_resume(tiny_sdxl_model, tmp_path, monkeypatch, mode):
    import torch
    from diffusers import StableDiffusionXLPipeline
    from safetensors.torch import load_file
    from transformers import CLIPTextModel, CLIPTextModelWithProjection
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import TrainingConfig, worker

    monkeypatch.setattr(dataset, 'BUCKETS', ((64, 64),))
    preflight = worker.preflight
    def tiny(*args, **kwargs):
        result = preflight(*args, **kwargs)
        result.errors = [e for e in result.errors if 'free disk space' not in e and 'CPU floor' not in e]
        return result
    monkeypatch.setattr(worker, 'preflight', tiny)
    images=[]
    for index, color in enumerate(('purple','green')):
        image=tmp_path/f'{index}.png'; Image.new('RGB',(64,64),color).save(image); images.append(image)
    project=dataset.DatasetProject.create(tmp_path/'project'); project.import_images(images)
    project.save_caption(1,'a'); project.save_caption(2,' '.join(['a']*151))
    finalized=project.finalize_in_project(TokenCounter.from_model(tiny_sdxl_model))

    originals = {}
    for name, cls in (('text_encoder',CLIPTextModel), ('text_encoder_2',CLIPTextModelWithProjection)):
        originals[name] = cls.from_pretrained(str(tiny_sdxl_model/name),local_files_only=True).state_dict()
    loaded=[]
    pipeline_load=StableDiffusionXLPipeline.from_pretrained.__func__
    def capture_pipeline(cls,*args,**kwargs):
        if 'text_encoder' in kwargs:
            for name in ('text_encoder','text_encoder_2'):
                state=kwargs[name].state_dict()
                assert all(torch.equal(state[k],v) for k,v in originals[name].items())
                assert any(not torch.equal(v, v.bfloat16().float()) for v in state.values() if v.is_floating_point())
            loaded.append(True)
        return pipeline_load(cls,*args,**kwargs)
    monkeypatch.setattr(StableDiffusionXLPipeline,'from_pretrained',classmethod(capture_pipeline))
    observed=[]
    for cls in (CLIPTextModel, CLIPTextModelWithProjection):
        forward=cls.forward
        def capture_forward(self,*args,_forward=forward,**kwargs):
            result=_forward(self,*args,**kwargs)
            if self.training:
                assert next(self.parameters()).dtype == torch.float32
                assert not torch.is_autocast_enabled('cpu')
                assert result.hidden_states[-2].dtype == torch.float32
                observed.append(type(self).__name__)
            return result
        monkeypatch.setattr(cls,'forward',capture_forward)

    values=dict(model_path=str(tiny_sdxl_model),dataset_path=str(finalized),mode=mode,
        model_precision='bf16',precision='bf16',cpu_text_precision='fp32',vae_precision='fp32',
        train_text_encoder=True,device='cpu',optimizer='adamw_disk',optimizer_update_device='cpu',
        optimizer_gradient_storage='disk',batch_size=2,gradient_accumulation_steps=1,max_steps=2,
        save_every=1,warmup_steps=0,rank=2,lora_alpha=2,learning_rate=.001,text_encoder_lr=.001,
        monitor_gpu=False,profile_phases=False)
    resumed=legacy_training_config(**values,output_dir=str(tmp_path/'resumed'))
    monkeypatch.setattr(worker,'_STOP',True);worker.run(resumed)
    checkpoint=Path(resumed.output_dir)/'checkpoints/step-00000001'
    resumed.resume_from=str(checkpoint)
    monkeypatch.setattr(worker,'_STOP',False);worker.run(resumed)
    straight=legacy_training_config(**values,output_dir=str(tmp_path/'straight'));worker.run(straight)
    assert len(loaded)==3
    assert 'CLIPTextModel' in observed and 'CLIPTextModelWithProjection' in observed
    for name in ('unet','text_encoder','text_encoder_2'):
        actual=load_file(str(Path(resumed.output_dir)/'checkpoints/step-00000002'/f'{name}.safetensors'))
        expected=load_file(str(Path(straight.output_dir)/'checkpoints/step-00000002'/f'{name}.safetensors'))
        assert actual.keys()==expected.keys()
        assert all(torch.equal(value,expected[key]) for key,value in actual.items())
        dtype = torch.bfloat16 if name == 'unet' and mode == 'full' else torch.float32
        assert all(value.dtype==dtype for value in actual.values() if value.is_floating_point())
        if mode=='lora':
            assert any('lora_B' in key and torch.count_nonzero(value) for key,value in actual.items())
        else:
            if name.startswith('text_encoder'):
                assert any(not torch.equal(value,originals[name][key]) for key,value in actual.items())
    output=Path(straight.output_dir)/'diffusers'
    for name,dtype in (('unet',torch.bfloat16),('text_encoder',torch.float32),('text_encoder_2',torch.float32)):
        weights={}
        for file in (output/name).glob('*.safetensors'):
            weights.update(load_file(str(file)))
        assert weights and all(v.dtype==dtype for v in weights.values() if v.is_floating_point())


@pytest.mark.parametrize('precision', ['bf16','fp16'])
@pytest.mark.parametrize('mode', ['full','lora'])
def test_cuda_mixed_precision_unet_and_fp32_cpu_text(tiny_sdxl_model, tmp_path, monkeypatch, precision, mode):
    import torch
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    if precision=='bf16' and not torch.cuda.is_bf16_supported(): pytest.skip('CUDA bf16 unavailable')
    from diffusers import UNet2DConditionModel
    from safetensors.torch import load_file
    from transformers import CLIPTextModel, CLIPTextModelWithProjection
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import TrainingConfig, worker
    monkeypatch.setattr(dataset,'BUCKETS',((64,64),)); monkeypatch.setattr(worker,'_STOP',False)
    preflight=worker.preflight
    def tiny(*args,**kwargs):
        result=preflight(*args,**kwargs)
        result.errors=[e for e in result.errors if 'free disk space' not in e and 'GiB VRAM is free' not in e]
        return result
    monkeypatch.setattr(worker,'preflight',tiny)
    files=[]
    for index,color in enumerate(('purple','green')):
        image=tmp_path/f'{index}.png';Image.new('RGB',(64,64),color).save(image);files.append(image)
    project=dataset.DatasetProject.create(tmp_path/'project');project.import_images(files)
    project.save_caption(1,'a');project.save_caption(2,' '.join(['a']*151))
    finalized=project.finalize_in_project(TokenCounter.from_model(tiny_sdxl_model))
    initial={}
    for name,cls in (('text_encoder',CLIPTextModel),('text_encoder_2',CLIPTextModelWithProjection)):
        initial[name]=cls.from_pretrained(str(tiny_sdxl_model/name),local_files_only=True).state_dict()
    forward=UNet2DConditionModel.forward; observed=[]
    def capture(self,sample,timestep,encoder_hidden_states,*args,**kwargs):
        assert torch.is_autocast_enabled('cuda')
        added=kwargs['added_cond_kwargs']
        assert sample.dtype==encoder_hidden_states.dtype==added['text_embeds'].dtype==added['time_ids'].dtype==torch.float32
        result=forward(self,sample,timestep,encoder_hidden_states,*args,**kwargs)
        assert result[0].dtype==({'bf16':torch.bfloat16,'fp16':torch.float16}[precision])
        observed.append(True)
        return result
    monkeypatch.setattr(UNet2DConditionModel,'forward',capture)
    run=tmp_path/'run'
    config=legacy_training_config(model_path=str(tiny_sdxl_model),dataset_path=str(finalized),output_dir=str(run),mode=mode,
        model_precision=precision,precision=precision,cpu_text_precision='fp32',vae_precision='fp32',
        train_text_encoder=True,device='cuda',component_offload='text_encoders_cpu',optimizer='adamw_disk',
        optimizer_update_device='cpu',optimizer_gradient_storage='disk',batch_size=2,gradient_accumulation_steps=1,
        max_steps=1,save_every=1,warmup_steps=0,rank=2,lora_alpha=2,learning_rate=.001,text_encoder_lr=.001,
        loss_scale_initial=128,monitor_gpu=False,profile_phases=False)
    worker.run(config)
    assert observed
    for name in ('text_encoder','text_encoder_2'):
        trained=load_file(str(run/'checkpoints/step-00000001'/f'{name}.safetensors'))
        if mode=='full':
            assert any(not torch.equal(value,initial[name][key]) for key,value in trained.items())
        else:
            assert any('lora_B' in key and torch.count_nonzero(value) for key,value in trained.items())
