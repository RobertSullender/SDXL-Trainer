"""Actual reduced model weights use fp32 masters; fp16 overflow never updates."""
import importlib.util
import json
from pathlib import Path

import pytest
from test_training import tiny_sdxl_model, legacy_training_config

pytestmark = pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='ML runtime unavailable')


def test_dynamic_loss_scaler_backoff_growth_and_resume():
    from sdxl_trainer.training.loss_scaler import DynamicLossScaler
    scaler = DynamicLossScaler(128, growth_interval=2)
    scaler.overflow()
    assert scaler.scale == 64 and scaler.good_steps == 0
    scaler.success()
    restored = DynamicLossScaler(128, growth_interval=2)
    restored.load_state_dict(scaler.state_dict())
    restored.success()
    assert restored.scale == 128 and restored.good_steps == 0
    restored.scale = 1.0
    restored.overflow()
    assert restored.scale == .5
    copy = DynamicLossScaler(128, growth_interval=2)
    copy.load_state_dict(restored.state_dict())
    assert copy.scale == .5
    with pytest.raises(ValueError):
        restored.load_state_dict({'scale': float('nan'), 'growth_interval': 2, 'good_steps': 0})


@pytest.mark.parametrize('dtype_name', ['float16', 'bfloat16'])
def test_reduced_weights_keep_fp32_adam_master_and_resume(tmp_path, dtype_name):
    import torch
    from sdxl_trainer.training.disk_optimizer import create_disk_adamw
    dtype = getattr(torch, dtype_name)
    p = torch.nn.Parameter(torch.tensor([1., -2., 0.1], dtype=dtype))
    ref = torch.nn.Parameter(p.detach().float().clone())
    optimizer = create_disk_adamw([{'named_params':[('x', p)], 'lr':1e-4}], tmp_path/'work', update_device='cpu', weight_decay=.01)
    ordinary = torch.optim.AdamW([{'params':[ref], 'lr':1e-4}], weight_decay=.01, foreach=False)
    assert optimizer.has_master_weights and optimizer.update_backend == 'cpu'
    for i in range(6):
        gradient = torch.tensor([.02, -.04, .15]) * (1 + i / 10)
        scale = 128.
        (p.float() * gradient * scale).sum().backward()
        # Parameters receive gradients at their own dtype before hooks spill.
        ref.grad = (gradient * scale).to(dtype).float() / scale
        optimizer.step(loss_scale=scale); optimizer.zero_grad()
        ordinary.step(); ordinary.zero_grad()
        with optimizer.mapped('master_weights.bin', optimizer.records[0]) as master:
            assert torch.allclose(master, ref.detach(), atol=2e-7, rtol=2e-6)
        assert torch.equal(p.detach(), ref.detach().to(dtype))
    checkpoint=tmp_path/'checkpoint'
    optimizer.save_state(checkpoint)
    assert (checkpoint/'master_weights.bin').is_file()
    q=torch.nn.Parameter(p.detach().clone())
    resumed=create_disk_adamw([{'named_params':[('x', q)], 'lr':1e-4}], tmp_path/'resumed', update_device='cpu', weight_decay=.01)
    resumed.load_state(checkpoint)
    for param,opt in ((p,optimizer),(q,resumed)):
        (param.float().sum()*128).backward()
        opt.step(loss_scale=128);opt.zero_grad()
    assert torch.equal(p,q)
    with optimizer.mapped('master_weights.bin',optimizer.records[0]) as a, resumed.mapped('master_weights.bin',resumed.records[0]) as b:
        assert torch.equal(a,b)


def test_overflow_does_not_mutate_masters_or_moments(tmp_path):
    import torch
    from sdxl_trainer.training.disk_optimizer import create_disk_adamw, NonFiniteGradients
    p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float16))
    optimizer=create_disk_adamw([{'named_params':[('p',p)],'lr':1e-3}],tmp_path,update_device='cpu')
    (p.float()*1e10).sum().backward()
    before=p.detach().clone()
    with pytest.raises(NonFiniteGradients): optimizer.step(loss_scale=4096)
    assert torch.equal(before,p) and optimizer.records[0]['step']==0
    with optimizer.mapped('master_weights.bin',optimizer.records[0]) as master:
        assert master.item()==1.
    optimizer.zero_grad()
    (p.float()*128).sum().backward(); optimizer.step(loss_scale=128)
    assert optimizer.records[0]['step']==1


def test_mixed_dtype_disk_staging_preserves_bits_hooks_and_tied_parameters(tmp_path):
    import torch
    from sdxl_trainer.training.disk_staging import DiskModuleStager
    class Mixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a=torch.nn.Parameter(torch.tensor([1.,-2.,3.],dtype=torch.bfloat16))
            self.b=torch.nn.Parameter(torch.tensor([.3,-.7],dtype=torch.float32))
            self.c=torch.nn.Parameter(torch.tensor([.125],dtype=torch.float16))
            self.tied=self.a
    model=Mixed(); before={n:p.detach().clone() for n,p in model.named_parameters()}
    ids={n:id(p) for n,p in model.named_parameters()}; seen=[]
    hook=model.a.register_hook(lambda g:seen.append(True) or g)
    stage=DiskModuleStager(model,tmp_path,batch_bytes=8)
    stage.park();stage.restore('cpu')
    assert model.tied is model.a
    for name,p in model.named_parameters():
        assert id(p)==ids[name] and torch.equal(p,before[name]) and p.dtype==before[name].dtype
    model.a.float().sum().backward();assert seen
    hook.remove();stage.close()


@pytest.fixture(scope='module')
def tiny_variant_only(tiny_sdxl_model, tmp_path_factory):
    import torch
    from diffusers import StableDiffusionXLPipeline
    destination=tmp_path_factory.mktemp('variant-only')/'model'
    pipeline=StableDiffusionXLPipeline.from_pretrained(str(tiny_sdxl_model),local_files_only=True,use_safetensors=True)
    pipeline.to(dtype=torch.float16)
    pipeline.save_pretrained(str(destination),safe_serialization=True,variant='fp16')
    for name in ('unet','vae','text_encoder','text_encoder_2'):
        assert list((destination/name).glob('*.fp16.safetensors'))
        assert not [p for p in (destination/name).glob('*.safetensors') if '.fp16.' not in p.name]
    return destination


@pytest.mark.parametrize('optimizer_backend', ['cpu', 'cuda'])
@pytest.mark.parametrize('component_offload', ['text_cpu_staged_unet', 'stage_unet_gpu_text'])
@pytest.mark.parametrize('weight_dtype,mode,vae_dtype', [
    ('fp16','full','fp32'), ('bf16','full','fp32'), ('fp16','full','fp16'), ('bf16','full','bf16'),
    ('fp16','lora','fp16'), ('bf16','lora','bf16'),
])
def test_tiny_cuda_variant_only_training_all_components(tmp_path, monkeypatch, tiny_variant_only, weight_dtype, mode, vae_dtype, component_offload, optimizer_backend):
    import torch
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    if component_offload == 'text_cpu_staged_unet' and optimizer_backend == 'cuda': pytest.skip('Mixed CPU text requires CPU optimizer updates')
    if weight_dtype == 'bf16' and not torch.cuda.is_bf16_supported(): pytest.skip('bf16 CUDA unavailable')
    from PIL import Image
    from safetensors import safe_open
    from safetensors.torch import load_file
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import TrainingConfig, worker
    monkeypatch.setattr(dataset,'BUCKETS',((64,64),))
    original_preflight=worker.preflight
    def tiny_preflight(*args,**kwargs):
        report=original_preflight(*args,**kwargs)
        report.errors=[e for e in report.errors if 'free disk space' not in e and 'GiB VRAM is free' not in e]
        return report
    monkeypatch.setattr(worker,'preflight',tiny_preflight)
    monkeypatch.setattr(worker,'_STOP',False)
    model_type={'fp16':torch.float16,'bf16':torch.bfloat16}[weight_dtype]
    vae_type={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[vae_dtype]
    base_vae=AutoencoderKL.from_pretrained(str(tiny_variant_only/'vae'),variant='fp16',torch_dtype=vae_type,local_files_only=True,use_safetensors=True)
    baseline_pipeline=StableDiffusionXLPipeline.from_pretrained(str(tiny_variant_only),variant='fp16',torch_dtype=model_type,vae=base_vae,local_files_only=True,use_safetensors=True)
    baseline={name:{key:value.detach().clone() for key,value in getattr(baseline_pipeline,name).state_dict().items()} for name in ('unet','text_encoder','text_encoder_2','vae')}
    del baseline_pipeline,base_vae
    image=tmp_path/'source.png';Image.new('RGB',(64,64),'purple').save(image)
    project=dataset.DatasetProject.create(tmp_path/'project');project.import_images([image]);project.save_caption(1,'a')
    finalized=project.finalize_in_project(TokenCounter.from_model(tiny_variant_only))
    run=tmp_path/'run'
    config=legacy_training_config(model_path=str(tiny_variant_only),dataset_path=str(finalized),output_dir=str(run),mode=mode,
      model_variant='fp16',model_precision=weight_dtype,precision=weight_dtype,vae_precision=vae_dtype,
      train_text_encoder=True,train_vae=True,device='cuda',component_offload=component_offload,
      optimizer='adamw_disk',optimizer_update_device=optimizer_backend,optimizer_gradient_storage='disk',
      learning_rate=.001,text_encoder_lr=.001,vae_lr=.001,loss_scale_initial=128,
      batch_size=1,gradient_accumulation_steps=1,max_steps=1,save_every=1,rank=2,lora_alpha=2,warmup_steps=0)
    events=[]; original_emit=worker.emit
    def capture(event_type,**values):
        events.append((event_type,values))
        original_emit(event_type,**values)
    monkeypatch.setattr(worker,'emit',capture)
    worker.run(config)
    placements=[values.get('message','') for event_type,values in events if event_type=='status' and 'Component placement:' in values.get('message','')]
    expected_location='cpu, cpu' if component_offload=='text_cpu_staged_unet' else 'cuda:0, cuda:0'
    assert placements and f'text encoders {expected_location}' in placements[0]
    checkpoint=run/'checkpoints/step-00000001'
    info=json.loads((checkpoint/'optimizer/optimizer.json').read_text())
    assert info['master_weights'] == (mode == 'full')
    assert info['update_backend'] == optimizer_backend
    assert (checkpoint/'optimizer/master_weights.bin').is_file() == (mode == 'full')
    assert (run/'diffusers/model_index.json').is_file()
    state=load_file(str(checkpoint/'unet.safetensors'))
    floating=[x for x in state.values() if x.is_floating_point()]
    assert floating and all(x.dtype==(getattr(torch,{'fp16':'float16','bf16':'bfloat16'}[weight_dtype]) if mode=='full' else torch.float32) for x in floating)
    for component in ('unet','text_encoder','text_encoder_2','vae'):
        state=load_file(str(checkpoint/(component+'.safetensors')))
        assert state,component
        if mode=='lora':
            assert any('lora_B' in k and torch.count_nonzero(v) for k,v in state.items()),component
        else:
            assert any(key in baseline[component] and not torch.equal(value, baseline[component][key])
                       for key,value in state.items()),component
        # Verify final Diffusers weights really retain the chosen component
        # storage dtype; do not rely on the checkpoint alone.
        saved=list((run/'diffusers'/component).glob('*.safetensors'))
        assert saved,component
        with safe_open(str(saved[0]),framework='pt',device='cpu') as output:
            floating=[output.get_tensor(key) for key in output.keys() if output.get_tensor(key).is_floating_point()]
            expected_dtype=vae_type if component=='vae' else model_type
            assert floating and all(value.dtype==expected_dtype for value in floating),component


def test_scaled_overflow_retries_same_batch_and_restores_checkpoint_state(tmp_path, monkeypatch, tiny_sdxl_model):
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import TrainingConfig, worker, disk_optimizer
    monkeypatch.setattr(dataset,'BUCKETS',((64,64),))
    original_preflight=worker.preflight
    def tiny_preflight(*args,**kwargs):
        report=original_preflight(*args,**kwargs)
        report.errors=[e for e in report.errors if 'free disk space' not in e and 'system RAM is available' not in e]
        return report
    monkeypatch.setattr(worker,'preflight',tiny_preflight)
    paths=[]
    for index,color in enumerate(('purple','green')):
        path=tmp_path/f'original-{index}.png';Image.new('RGB',(64,64),color).save(path);paths.append(path)
    project=dataset.DatasetProject.create(tmp_path/'project');project.import_images(paths)
    for i in (1,2):project.save_caption(i,'a')
    finalized=project.finalize_in_project(TokenCounter.from_model(tiny_sdxl_model))
    base=dict(model_path=str(tiny_sdxl_model),dataset_path=str(finalized),mode='full',
              model_precision='fp32',precision='fp32',vae_precision='fp16',train_text_encoder=True,
              train_vae=True,device='cpu',optimizer='adamw_disk',optimizer_update_device='cpu',
              loss_scale_growth_interval=10,batch_size=1,gradient_accumulation_steps=1,
              max_steps=2,save_every=1,warmup_steps=0,learning_rate=.001,vae_lr=.001)
    original_factory=disk_optimizer.create_disk_adamw; calls={'overflow':0}
    def intercept(*args,**kwargs):
        optimizer=original_factory(*args,**kwargs); original_step=optimizer.step
        def first_overflow(*args,**kwargs):
            if not calls['overflow']:
                calls['overflow']+=1
                raise disk_optimizer.NonFiniteGradients('Injected overflow')
            return original_step(*args,**kwargs)
        optimizer.step=first_overflow
        return optimizer
    monkeypatch.setattr(disk_optimizer,'create_disk_adamw',intercept)
    monkeypatch.setattr(worker,'_STOP',False)
    events=[]; original_emit=worker.emit
    def emit(event_type,**values):
        events.append((event_type,values)); original_emit(event_type,**values)
        if event_type=='progress' and values.get('step')==1: worker._STOP=True
    monkeypatch.setattr(worker,'emit',emit)
    run=tmp_path/'retry';config=legacy_training_config(**base,output_dir=str(run),loss_scale_initial=128)
    worker.run(config)
    ckpt=run/'checkpoints/step-00000001'
    state=torch.load(ckpt/'rng.pt',map_location='cpu',weights_only=True)
    assert calls['overflow']==1 and state['loss_scaler']['scale']==64 and state['loss_scaler']['good_steps']==1
    assert any(t=='status' and 'retrying the same images' in v.get('message','') for t,v in events)
    # A fresh run at the reduced scale must match the retried first step exactly.
    monkeypatch.setattr(disk_optimizer,'create_disk_adamw',original_factory)
    reference=tmp_path/'reference';ref=legacy_training_config(**base,output_dir=str(reference),loss_scale_initial=64)
    monkeypatch.setattr(worker,'_STOP',False)
    worker.run(ref)
    for component in ('unet','text_encoder','text_encoder_2','vae'):
        a=load_file(str(ckpt/f'{component}.safetensors'))
        b=load_file(str(reference/'checkpoints/step-00000001'/f'{component}.safetensors'))
        assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a),component
    monkeypatch.setattr(worker,'_STOP',False)
    config.resume_from=str(ckpt)
    worker.run(config)
    resumed=torch.load(run/'checkpoints/step-00000002/rng.pt',map_location='cpu',weights_only=True)
    assert resumed['loss_scaler']['scale']==64 and resumed['loss_scaler']['good_steps']==2
    # Cancellation during overflow must not loop forever or publish step zero.
    calls['overflow']=0
    monkeypatch.setattr(disk_optimizer,'create_disk_adamw',intercept)
    monkeypatch.setattr(worker,'_STOP',True)
    events.clear()
    cancelled=tmp_path/'cancelled';worker.run(legacy_training_config(**base,output_dir=str(cancelled),loss_scale_initial=128))
    assert not (cancelled/'checkpoints').exists()
    assert any(t=='complete' and v.get('step')==0 and 'no checkpoint was created' in v.get('message','') for t,v in events)


def test_resume_rejects_unsafe_or_truncated_masters_and_bad_scaler_before_writing(tmp_path):
    import torch
    from sdxl_trainer.training import TrainingConfig
    from sdxl_trainer.training.worker import _config_identity,validate_resume_identity
    checkpoint=tmp_path/'checkpoints/step-00000001';(checkpoint/'optimizer').mkdir(parents=True)
    config=legacy_training_config(model_path=str(tmp_path/'model'),dataset_path=str(tmp_path/'dataset'),
                          output_dir=str(tmp_path/'new-run'),resume_from=str(checkpoint),mode='full',
                          model_precision='fp16',precision='fp16')
    info={'version':1,'step':1,'epoch':0,'batch_cursor':0,'config':_config_identity(config),
          'dataset_fingerprint':'data','model_fingerprint':'model','original_max_steps':config.max_steps}
    (checkpoint/'checkpoint.json').write_text(json.dumps(info))
    for name in ('unet.safetensors','optimizer/first_moment.bin','optimizer/second_moment.bin'):
        (checkpoint/name).write_bytes(b'old!')
    torch.save({'loss_scaler':{'scale':128.,'good_steps':0,'growth_interval':2000}},checkpoint/'rng.pt')
    (checkpoint/'optimizer/optimizer.json').write_text(json.dumps({'records':[{'offset':0,'numel':1}],
                                                                  'groups':[],'master_weights':True,'update_backend':'cpu'}))
    target=tmp_path/'do-not-touch';target.write_bytes(b'1234')
    moment=checkpoint/'optimizer/first_moment.bin'
    moment.unlink();moment.symlink_to(target)
    with pytest.raises(ValueError,match='regular state file'):validate_resume_identity(config,'data','model')
    moment.unlink();moment.write_bytes(b'12')
    with pytest.raises(ValueError,match='optimizer state is truncated'):validate_resume_identity(config,'data','model')
    moment.write_bytes(b'old!')
    master=checkpoint/'optimizer/master_weights.bin';master.symlink_to(target)
    with pytest.raises(ValueError,match='regular fp32 master'):validate_resume_identity(config,'data','model')
    master.unlink();master.write_bytes(b'12')
    with pytest.raises(ValueError,match='truncated fp32 master'):validate_resume_identity(config,'data','model')
    master.write_bytes(b'1234')
    torch.save({'loss_scaler':{'scale':float('nan'),'good_steps':0,'growth_interval':2000}},checkpoint/'rng.pt')
    with pytest.raises(ValueError,match='loss scaler'):validate_resume_identity(config,'data','model')
    assert not Path(config.output_dir).exists() and target.read_bytes()==b'1234'


@pytest.mark.parametrize("train_text", [True, False])
def test_bf16_cpu_text_forward_and_backward_use_native_backend(tmp_path, monkeypatch, tiny_variant_only, train_text):
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from diffusers import StableDiffusionXLPipeline
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import TrainingConfig, worker, long_prompt
    monkeypatch.setattr(dataset,'BUCKETS',((64,64),))
    original_preflight=worker.preflight
    def tiny_preflight(*args,**kwargs):
        report=original_preflight(*args,**kwargs)
        report.errors=[e for e in report.errors if 'free disk space' not in e and 'system RAM is available' not in e]
        return report
    monkeypatch.setattr(worker,'preflight',tiny_preflight)
    monkeypatch.setattr(worker,'_STOP',False)
    seen=[];original_encode=long_prompt.encode_long_prompt
    def check_backend(*args,**kwargs):
        seen.append(torch.backends.mkldnn.enabled)
        return original_encode(*args,**kwargs)
    monkeypatch.setattr(long_prompt,'encode_long_prompt',check_backend)
    original=StableDiffusionXLPipeline.from_pretrained(str(tiny_variant_only),variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True,use_safetensors=True)
    baseline={name:{k:v.clone() for k,v in getattr(original,name).state_dict().items()} for name in ('text_encoder','text_encoder_2')};del original
    image=tmp_path/'source.png';Image.new('RGB',(64,64),'purple').save(image)
    project=dataset.DatasetProject.create(tmp_path/'project');project.import_images([image]);project.save_caption(1,'a')
    finalized=project.finalize_in_project(TokenCounter.from_model(tiny_variant_only))
    run=tmp_path/'run'
    config=legacy_training_config(model_path=str(tiny_variant_only),dataset_path=str(finalized),output_dir=str(run),
      mode='full',model_variant='fp16',model_precision='bf16',precision='bf16',vae_precision='fp32',
      train_text_encoder=train_text,train_vae=False,device='cpu',optimizer='adamw_disk',optimizer_update_device='cpu',
      learning_rate=.001,text_encoder_lr=.001,batch_size=1,gradient_accumulation_steps=1,max_steps=1,
      save_every=1,warmup_steps=0)
    worker.run(config)
    assert seen
    if train_text:
        assert not any(seen)
    for component in baseline:
        if train_text:
            state=load_file(str(run/'checkpoints/step-00000001'/f'{component}.safetensors'))
            assert any(k in baseline[component] and not torch.equal(v,baseline[component][k]) for k,v in state.items()),component
        else:
            assert not (run/'checkpoints/step-00000001'/f'{component}.safetensors').exists()
        # Frozen models must retain the explicitly selected bf16 storage dtype
        # when the complete Diffusers output is saved.
        saved=next((run/'diffusers'/component).glob('*.safetensors'))
        from safetensors import safe_open
        with safe_open(str(saved),framework='pt',device='cpu') as weights:
            assert all(weights.get_tensor(k).dtype==torch.bfloat16 for k in weights.keys() if weights.get_tensor(k).is_floating_point())


def test_explicit_variant_preflight_requires_every_component_variant(tmp_path, tiny_variant_only):
    from sdxl_trainer.training import TrainingConfig, preflight
    import shutil
    model=tmp_path/'variant-model';shutil.copytree(tiny_variant_only,model)
    config=TrainingConfig(model_path=str(model),dataset_path=str(tmp_path/'missing-dataset'),
                          output_dir=str(tmp_path/'run'),model_variant='fp16')
    assert not [e for e in preflight(config).errors if 'variant is missing' in e]
    encoder=model/'text_encoder'
    for path in encoder.glob('*.fp16.safetensors'): path.unlink()
    (encoder/'model.fp16garbage.safetensors').write_bytes(b'bad')
    errors=preflight(config).errors
    assert any('fp16 variant is missing' in e and 'text_encoder/' in e for e in errors)
    # A shard is not enough: Diffusers needs the corresponding valid index and
    # all referenced shards. Test current and legacy names, with no fallback.
    one='model.fp16-00001-of-00002.safetensors';two='model.fp16-00002-of-00002.safetensors'
    (encoder/one).write_bytes(b'1')
    (encoder/'model.safetensors.index.fp16.json').write_text(json.dumps({'weight_map':{'a':one,'b':two}}))
    assert any('text_encoder/' in e and 'variant is missing' in e for e in preflight(config).errors)
    (encoder/two).write_bytes(b'2')
    assert not [e for e in preflight(config).errors if 'variant is missing' in e]
    (encoder/'model.safetensors.index.fp16.json').unlink()
    legacy='model-00001-of-00001.fp16.safetensors';(encoder/legacy).write_bytes(b'3')
    (encoder/'model.safetensors.fp16.index.json').write_text(json.dumps({'weight_map':{'a':legacy}}))
    assert not [e for e in preflight(config).errors if 'variant is missing' in e]


@pytest.mark.parametrize("vae_dtype", ["fp32", "bf16"])
@pytest.mark.parametrize("cpu_text_precision", ["match", "fp32"])
def test_bf16_named_variant_only_loads_and_trains(tmp_path, monkeypatch, tiny_sdxl_model, vae_dtype, cpu_text_precision):
    import torch
    from PIL import Image
    from diffusers import StableDiffusionXLPipeline
    from safetensors.torch import load_file
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import TrainingConfig,worker
    variant=tmp_path/'bf16-model'
    pipeline=StableDiffusionXLPipeline.from_pretrained(str(tiny_sdxl_model),local_files_only=True,use_safetensors=True)
    pipeline.to(dtype=torch.bfloat16);pipeline.save_pretrained(str(variant),variant='bf16',safe_serialization=True);del pipeline
    for name in ('unet','vae','text_encoder','text_encoder_2'):
        assert list((variant/name).glob('*.bf16.safetensors'))
        assert not [p for p in (variant/name).glob('*.safetensors') if '.bf16.' not in p.name]
    monkeypatch.setattr(dataset,'BUCKETS',((64,64),))
    original=worker.preflight
    def tiny_preflight(*args,**kwargs):
        report=original(*args,**kwargs)
        report.errors=[e for e in report.errors if 'free disk space' not in e and 'system RAM is available' not in e]
        return report
    monkeypatch.setattr(worker,'preflight',tiny_preflight);monkeypatch.setattr(worker,'_STOP',False)
    image=tmp_path/'one.png';Image.new('RGB',(64,64),'blue').save(image)
    project=dataset.DatasetProject.create(tmp_path/'project');project.import_images([image]);project.save_caption(1,'a')
    finalized=project.finalize_in_project(TokenCounter.from_model(variant));run=tmp_path/'run'
    config=legacy_training_config(model_path=str(variant),dataset_path=str(finalized),output_dir=str(run),
      mode='full',model_variant='bf16',model_precision='bf16',precision='bf16',vae_precision=vae_dtype,cpu_text_precision=cpu_text_precision,
      train_text_encoder=True,train_vae=True,device='cpu',optimizer='adamw_disk',optimizer_update_device='cpu',
      learning_rate=.001,text_encoder_lr=.001,vae_lr=.001,batch_size=1,gradient_accumulation_steps=1,
      max_steps=1,save_every=1,warmup_steps=0)
    worker.run(config)
    state=load_file(str(run/'checkpoints/step-00000001/unet.safetensors'))
    assert state and all(v.dtype==torch.bfloat16 for v in state.values() if v.is_floating_point())
    expected=torch.float32 if cpu_text_precision=='fp32' else torch.bfloat16
    for name in ('text_encoder','text_encoder_2'):
        text=load_file(str(run/'checkpoints/step-00000001'/f'{name}.safetensors'))
        assert all(v.dtype==expected for v in text.values() if v.is_floating_point())
    assert (run/'diffusers/model_index.json').is_file()


def test_mixed_precision_scaled_accumulation_and_global_clipping_match_fp32_reference(tmp_path):
    import torch
    from sdxl_trainer.training.disk_optimizer import create_disk_adamw
    parameters=[torch.nn.Parameter(torch.tensor([.5,-.75],dtype=dtype)) for dtype in (torch.float16,torch.bfloat16,torch.float32)]
    reference=[torch.nn.Parameter(p.detach().float().clone()) for p in parameters]
    names=[(str(i),p) for i,p in enumerate(parameters)]
    disk=create_disk_adamw([{'named_params':names,'lr':.002}],tmp_path,update_device='cpu',weight_decay=.03)
    ordinary=torch.optim.AdamW(reference,lr=.002,weight_decay=.03,foreach=False)
    scale=128.
    for step in range(3):
        expected=[torch.zeros_like(p) for p in reference]
        for batch in range(2):
            grad=torch.tensor([.4+step*.1,-.8+batch*.1])
            sum((p.float()*grad*scale).sum() for p in parameters).backward()
            for index,p in enumerate(parameters):expected[index]+=(grad*scale).to(p.dtype).float()/scale
        for p,g in zip(reference,expected):p.grad=g
        torch.nn.utils.clip_grad_norm_(reference,.2)
        ordinary.step();ordinary.zero_grad()
        disk.step(loss_scale=scale,max_grad_norm=.2);disk.zero_grad()
        for record,ref,p in zip(disk.records,reference,parameters):
            with disk.mapped('master_weights.bin',record) as master:
                assert torch.allclose(master,ref.detach(),atol=2e-7,rtol=2e-6)
            assert torch.equal(p.detach(),ref.detach().to(p.dtype))


@pytest.mark.parametrize('weight_dtype',['fp16','bf16'])
def test_cuda_frozen_text_is_saved_in_selected_weight_dtype(tmp_path,monkeypatch,tiny_variant_only,weight_dtype):
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    if weight_dtype=='bf16' and not torch.cuda.is_bf16_supported():pytest.skip('bf16 CUDA unavailable')
    from PIL import Image
    from safetensors import safe_open
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import TrainingConfig,worker
    monkeypatch.setattr(dataset,'BUCKETS',((64,64),));monkeypatch.setattr(worker,'_STOP',False)
    preflight=worker.preflight
    def tiny(*args,**kwargs):
        result=preflight(*args,**kwargs)
        result.errors=[e for e in result.errors if 'free disk space' not in e and 'GiB VRAM is free' not in e]
        return result
    monkeypatch.setattr(worker,'preflight',tiny)
    image=tmp_path/'image.png';Image.new('RGB',(64,64),'red').save(image)
    project=dataset.DatasetProject.create(tmp_path/'project');project.import_images([image]);project.save_caption(1,'a')
    finalized=project.finalize_in_project(TokenCounter.from_model(tiny_variant_only));run=tmp_path/'run'
    config=legacy_training_config(model_path=str(tiny_variant_only),dataset_path=str(finalized),output_dir=str(run),
      mode='full',model_variant='fp16',model_precision=weight_dtype,precision=weight_dtype,vae_precision='fp32',
      train_text_encoder=False,train_vae=False,device='cuda',optimizer='adamw_disk',optimizer_update_device='cpu',
      loss_scale_initial=128,batch_size=1,gradient_accumulation_steps=1,max_steps=1,save_every=1,warmup_steps=0)
    worker.run(config)
    expected={'fp16':torch.float16,'bf16':torch.bfloat16}[weight_dtype]
    for component in ('text_encoder','text_encoder_2'):
        saved=next((run/'diffusers'/component).glob('*.safetensors'))
        with safe_open(str(saved),framework='pt',device='cpu') as weights:
            assert all(weights.get_tensor(k).dtype==expected for k in weights.keys() if weights.get_tensor(k).is_floating_point())


def test_gpu_text_staging_preflight_counts_text_vram_and_preserves_resume_identity(tmp_path):
    from sdxl_trainer.training import TrainingConfig,preflight
    from sdxl_trainer.training.worker import _config_identity
    base=dict(model_path=str(tmp_path/'missing-model'),dataset_path=str(tmp_path/'missing-data'),
              output_dir=str(tmp_path/'run'),train_text_encoder=True,train_vae=True,
              model_precision='bf16',precision='bf16')
    cpu=TrainingConfig(**base,component_offload='text_cpu_staged_unet')
    gpu=TrainingConfig(**base,component_offload='stage_unet_gpu_text')
    assert preflight(gpu).estimated_minimum_gib-preflight(cpu).estimated_minimum_gib == pytest.approx(1.5)
    assert preflight(gpu).estimated_disk_gib == preflight(cpu).estimated_disk_gib
    assert _config_identity(gpu) != _config_identity(cpu)


@pytest.mark.parametrize('weight_dtype',['fp16','bf16'])
def test_cuda_streamed_fp32_masters_clip_accumulate_overflow_and_resume(tmp_path, weight_dtype):
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    if weight_dtype=='bf16' and not torch.cuda.is_bf16_supported():pytest.skip('bf16 CUDA unavailable')
    from sdxl_trainer.training.disk_optimizer import create_disk_adamw,NonFiniteGradients
    dtype={'fp16':torch.float16,'bf16':torch.bfloat16}[weight_dtype]
    base=torch.linspace(-.9,.9,17,device='cuda',dtype=torch.float32)
    other=torch.linspace(.2,.5,9,device='cuda',dtype=torch.float32)
    def params():
        return [torch.nn.Parameter(base.to(dtype).clone()),torch.nn.Parameter(other.clone())]
    def groups(p):return [{'named_params':[('low',p[0]),('vae_fp32',p[1])],'lr':1e-5}]
    cpu_params,gpu_params=params(),params()
    cpu=create_disk_adamw(groups(cpu_params),tmp_path/'cpu',update_device='cpu',weight_decay=.03,gradient_storage='disk')
    gpu=create_disk_adamw(groups(gpu_params),tmp_path/'gpu',update_device='cuda',weight_decay=.03,gradient_storage='disk')
    assert cpu.has_master_weights and gpu.has_master_weights and gpu.update_backend=='cuda'
    assert len(gpu._scratch)==5 and sum(t.numel()*t.element_size() for t in gpu._scratch)<=80*1024**2
    gpu._chunk_elements=3  # Exercise multiple chunks and an incomplete tail.
    def backward(p,shift):
        for batch in range(2):
            g=torch.linspace(.1+shift,.4+shift,p[0].numel(),device='cuda')
            h=torch.linspace(-.3,.2+batch*.01,p[1].numel(),device='cuda')
            ((p[0].float()*g).sum()+(p[1].float()*h).sum()).mul(128.).backward()
    initial=gpu_params[0].detach().clone()
    for step in range(3):
        for p,opt in ((cpu_params,cpu),(gpu_params,gpu)):
            backward(p,step*.01);opt.step(loss_scale=128,max_grad_norm=.4);opt.zero_grad()
        for i in range(2):
            for name in ('master_weights.bin','first_moment.bin','second_moment.bin'):
                with cpu.mapped(name,cpu.records[i]) as a,gpu.mapped(name,gpu.records[i]) as b:
                    torch.testing.assert_close(a,b,rtol=5e-5,atol=3e-7)
            with gpu.mapped('master_weights.bin',gpu.records[i]) as master:
                assert torch.equal(gpu_params[i].detach().cpu().reshape(-1),master.to(gpu_params[i].dtype))
    # Small updates accumulate in fp32 masters even if BF16/FP16 live weights
    # round back to their original values for individual steps.
    with gpu.mapped('master_weights.bin',gpu.records[0]) as master:
        assert not torch.equal(master,initial.float().cpu())
    ckpt=tmp_path/'ckpt';gpu.save_state(ckpt)
    metadata=json.loads((ckpt/'optimizer.json').read_text())
    assert metadata['update_backend']=='cuda' and metadata['master_weights']
    replay=[torch.nn.Parameter(p.detach().clone()) for p in gpu_params]
    resumed=create_disk_adamw(groups(replay),tmp_path/'resumed',update_device='auto',weight_decay=.03,gradient_storage='disk')
    assert resumed.update_backend=='cpu'  # Reduced-precision auto remains CPU.
    resumed.load_state(ckpt)
    assert resumed.update_backend=='cuda';resumed._chunk_elements=3
    for p,opt in ((gpu_params,gpu),(replay,resumed)):
        backward(p,.12);opt.step(loss_scale=128,max_grad_norm=.4);opt.zero_grad()
    for i in range(2):
        assert torch.equal(gpu_params[i],replay[i])
        for name in ('master_weights.bin','first_moment.bin','second_moment.bin'):
            with gpu.mapped(name,gpu.records[i]) as a,resumed.mapped(name,resumed.records[i]) as b:
                assert torch.equal(a,b)
    # Overflow detection precedes any CUDA update, including the master file.
    old=[p.detach().clone() for p in gpu_params];old_steps=[r['step'] for r in gpu.records]
    (gpu_params[0].float()*float('inf')).sum().backward()
    with pytest.raises(NonFiniteGradients):gpu.step(loss_scale=128)
    assert all(torch.equal(a,b) for a,b in zip(old,gpu_params)) and old_steps==[r['step'] for r in gpu.records]
    gpu.zero_grad()


def test_cuda_reduced_master_choice_is_explicit_and_rejects_mixed_placement():
    from sdxl_trainer.training import TrainingConfig
    common=dict(model_precision='bf16',precision='bf16',mode='full',train_text_encoder=True,
                train_vae=True,optimizer='adamw_disk',optimizer_update_device='cuda')
    gpu=TrainingConfig(**common,component_offload='stage_unet_gpu_text')
    assert not any('CUDA' in e for e in gpu.validate())
    cpu_text=TrainingConfig(**common,component_offload='text_cpu_staged_unet')
    assert any('not CUDA' in e for e in cpu_text.validate())
    cpu_only=TrainingConfig(**common,device='cpu')
    assert any('require the CUDA' in e for e in cpu_only.validate())


@pytest.mark.parametrize('device,backend',[('cpu','cpu'),('cuda','cuda')])
def test_reduced_master_overflow_does_not_publish_invalid_checkpoint(tmp_path,device,backend):
    import torch
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from sdxl_trainer.training.disk_optimizer import create_disk_adamw
    p=torch.nn.Parameter(torch.tensor([65504.],device=device,dtype=torch.float16))
    opt=create_disk_adamw([{'named_params':[('weight',p)],'lr':1.}],tmp_path/'work',update_device=backend,weight_decay=0.)
    old=tmp_path/'old';opt.save_state(old)
    before={name:(old/name).read_bytes() for name in ('optimizer.json','master_weights.bin','first_moment.bin','second_moment.bin')}
    (-p.float()*128).sum().backward()
    with pytest.raises(FloatingPointError,match='exceeded the selected model weight precision'):
        opt.step(loss_scale=128)
    assert p.item()==65504.
    with pytest.raises(RuntimeError,match='optimizer boundary'):opt.save_state(tmp_path/'partial')
    assert before=={name:(old/name).read_bytes() for name in before}
    assert not (tmp_path/'partial').exists()


@pytest.mark.parametrize('device,backend',[('cpu','cpu'),('cuda','cuda')])
def test_reduced_master_detects_nonfinite_moment_even_when_weight_stays_finite(tmp_path,device,backend):
    import torch
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    if device=='cuda' and not torch.cuda.is_bf16_supported():pytest.skip('bf16 CUDA unavailable')
    from sdxl_trainer.training.disk_optimizer import create_disk_adamw
    p=torch.nn.Parameter(torch.tensor([0.],device=device,dtype=torch.bfloat16))
    opt=create_disk_adamw([{'named_params':[('weight',p)],'lr':.001}],tmp_path/'work',update_device=backend,weight_decay=0.)
    (p.float()*1e30).sum().backward()
    with pytest.raises(FloatingPointError,match='non-finite'):
        opt.step(max_grad_norm=0)
    assert p.item()==0.
    with pytest.raises(RuntimeError,match='optimizer boundary'):opt.save_state(tmp_path/'partial')
