"""CPU contract tests for the optional 8-bit optimizer integration."""
import json
import os
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('safetensors')
from sdxl_trainer.training.paged_adamw import PagedAdamWCheckpoint, prepare_paged_gradients, check_paged_finite
from sdxl_trainer.training.config import TrainingConfig, preflight
from test_training import tiny_sdxl_model, legacy_training_config


class FakePaged(torch.optim.Optimizer):
    def __init__(self, parameters):
        super().__init__([{'params':parameters,'lr':.02}], {'lr':.02,'betas':(.9,.999),'eps':1e-8,'weight_decay':.01})
        self.initialized=False
    def check_overrides(self): pass
    def init_state(self, group, p, group_index, parameter_index):
        a=torch.zeros_like(p,dtype=torch.uint8);b=torch.zeros_like(p,dtype=torch.uint8)
        a.is_paged=True;b.is_paged=True
        self.state[p].update(step=0,state1=a,state2=b,qmap1=torch.tensor([.1,.2]),absmax1=torch.tensor([0.]))
    @torch.no_grad()
    def step(self):
        for gi,group in enumerate(self.param_groups):
            for pi,p in enumerate(group['params']):
                if p.grad is None:continue
                if not self.state[p]:self.init_state(group,p,gi,pi)
                state=self.state[p];state['step']+=1
                state['state1'].add_(1);state['state2'].add_(2);state['absmax1'].add_(.1)
                p.sub_(p.grad * group['lr'])


RUNTIME={'bitsandbytes':'0.49.2','torch':'test','torch_cuda':'13.2','cuda_override':'130','device_capability':[12,0]}


def pair(tmp_path,name,dtype):
    params=[torch.nn.Parameter(torch.tensor([1.,-2.],dtype=dtype)),torch.nn.Parameter(torch.tensor([.5],dtype=torch.float32))]
    opt=FakePaged(params)
    groups=[{'named_params':[(f'p{i}',p) for i,p in enumerate(params)],'lr':.02}]
    return params,opt,PagedAdamWCheckpoint(opt,groups,RUNTIME)


def advance(params,opt):
    for p in params:p.grad=torch.ones_like(p)
    opt.step();opt.zero_grad()


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16,torch.float16])
def test_paged_state_checkpoint_restores_paging_and_exact_continuation(tmp_path,dtype):
    params,opt,checkpoint=pair(tmp_path,'a',dtype)
    advance(params,opt);advance(params,opt)
    path=tmp_path/'checkpoint';checkpoint.save_state(path)
    other,restored,replay=pair(tmp_path,'b',dtype)
    for target,source in zip(other,params):target.data.copy_(source.data)
    replay.load_state(path)
    for source,target in zip(params,other):
        left,right=opt.state[source],restored.state[target]
        assert left['step']==right['step']
        assert right['state1'].is_paged and right['state2'].is_paged
        for key in ('state1','state2','qmap1','absmax1'):assert torch.equal(left[key],right[key])
    advance(params,opt);advance(other,restored)
    assert all(torch.equal(a,b) for a,b in zip(params,other))
    assert all(torch.equal(opt.state[a]['state1'],restored.state[b]['state1']) for a,b in zip(params,other))


def test_paged_checkpoint_rejects_runtime_layout_and_symlinks(tmp_path):
    params,opt,checkpoint=pair(tmp_path,'a',torch.float32);advance(params,opt)
    path=tmp_path/'checkpoint';checkpoint.save_state(path)
    saved=(path/'paged.json').read_bytes()
    other,restored,replay=pair(tmp_path,'b',torch.float32)
    replay.runtime={**RUNTIME,'bitsandbytes':'other'}
    with pytest.raises(ValueError,match='runtime differs'):replay.load_state(path)
    replay.runtime=RUNTIME
    data=json.loads(saved);data['records'][0]['name']='wrong';(path/'paged.json').write_text(json.dumps(data))
    with pytest.raises(ValueError,match='layout differs'):replay.load_state(path)
    data=json.loads(saved);data['groups'][0]['eps']=.5;(path/'paged.json').write_text(json.dumps(data))
    with pytest.raises(ValueError,match='hyperparameter values differ'):replay.load_state(path)
    assert not restored.state
    data=json.loads(saved);data['groups'][0]['lr']=-1;(path/'paged.json').write_text(json.dumps(data))
    with pytest.raises(ValueError,match='learning rate is invalid'):replay.load_state(path)
    (path/'paged.json').write_bytes(saved)
    state=path/'state.safetensors';backup=tmp_path/'state';state.rename(backup);state.symlink_to(backup)
    with pytest.raises(ValueError,match='regular state files'):replay.load_state(path)


def test_paged_resume_runtime_or_header_failure_preserves_run_config(tmp_path,monkeypatch):
    import sdxl_trainer.training.paged_adamw as module
    from sdxl_trainer.training.worker import _config_identity,validate_resume_identity
    run=tmp_path/'run'; checkpoint=run/'checkpoints/step-00000001'; checkpoint.mkdir(parents=True)
    config=TrainingConfig(model_path=str(tmp_path/'model'),dataset_path=str(tmp_path/'dataset'),output_dir=str(run),
                          resume_from=str(checkpoint),optimizer='paged_adamw8bit')
    config.save(run/'training-config.json');before=(run/'training-config.json').read_bytes()
    info={'version':1,'step':1,'epoch':0,'batch_cursor':0,'config':_config_identity(config),
          'dataset_fingerprint':'dataset','model_fingerprint':'model','original_max_steps':config.max_steps}
    (checkpoint/'checkpoint.json').write_text(json.dumps(info))
    for name in ('unet.safetensors','rng.pt'):(checkpoint/name).write_bytes(b'unchanged')
    params,opt,snapshot=pair(tmp_path,'a',torch.float32);advance(params,opt);snapshot.save_state(checkpoint/'optimizer')
    monkeypatch.setattr(module,'load_paged_runtime',lambda _: (None,{**RUNTIME,'cuda_override':'different'}))
    with pytest.raises(ValueError,match='runtime differs'):
        validate_resume_identity(config,'dataset','model')
    assert (run/'training-config.json').read_bytes()==before
    monkeypatch.setattr(module,'load_paged_runtime',lambda _: (None,RUNTIME))
    (checkpoint/'optimizer/state.safetensors').write_bytes(b'truncated')
    with pytest.raises(Exception):validate_resume_identity(config,'dataset','model')
    assert (run/'training-config.json').read_bytes()==before


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32])
def test_paged_clipping_uses_safe_global_norm_and_unscales_once(dtype):
    p=torch.nn.Parameter(torch.ones(10_000,dtype=dtype))
    p.grad=torch.full_like(p,1000.)
    norm=prepare_paged_gradients([p],loss_scale=2.,max_grad_norm=10.,chunk_elements=17)
    assert norm==pytest.approx(50_000.,rel=1e-6)
    assert torch.linalg.vector_norm(p.grad.double()).item()==pytest.approx(10.,rel=.01)


def test_paged_nonfinite_gradient_fails_without_mutating_gradients_or_weights():
    from sdxl_trainer.training.disk_optimizer import NonFiniteGradients
    p=torch.nn.Parameter(torch.ones(2,dtype=torch.float16));p.grad=torch.tensor([float('inf'),8.],dtype=p.dtype)
    before=p.detach().clone();grad=p.grad.clone()
    with pytest.raises(NonFiniteGradients):prepare_paged_gradients([p],128.,1.)
    assert torch.equal(p.detach(),before) and torch.equal(p.grad,grad)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1.])
def test_paged_nonfinite_or_negative_quantization_scale_is_rejected(tmp_path,bad):
    params,opt,checkpoint=pair(tmp_path,'a',torch.float32);advance(params,opt)
    opt.state[params[0]]['absmax1'][0]=bad
    with pytest.raises(FloatingPointError,match='invalid optimizer state'):
        check_paged_finite(opt,params)
    with pytest.raises(FloatingPointError,match='invalid optimizer state'):
        checkpoint.save_state(tmp_path/'bad')
    assert not (tmp_path/'bad/paged.json').exists()


def test_paged_optimizer_preflight_reports_tradeoffs_and_placement(tmp_path):
    good=TrainingConfig(optimizer='paged_adamw8bit',mode='full',model_precision='bf16',precision='bf16',
                        model_path=str(tmp_path/'model'),dataset_path=str(tmp_path/'dataset'),output_dir=str(tmp_path/'run'))
    assert not any('Disk AdamW' in e for e in good.validate())
    assert any('quantizes optimizer moments' in warning and 'without fp32 masters' in warning for warning in preflight(good).warnings)
    for changes in ({'device':'cpu'}, {'train_text_encoder':True,'component_offload':'text_encoders_cpu'},
                    {'train_vae':True,'component_offload':'stage_unet_gpu_text'}):
        bad=TrainingConfig.from_dict({**good.to_dict(),**changes})
        assert any('Paged 8-bit AdamW' in error for error in bad.validate())
    bad=TrainingConfig.from_dict({**good.to_dict(),'paged_cuda_override':'13.0'})
    assert any('override' in error for error in bad.validate())


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16,torch.float16])
def test_native_cuda_paged_checkpoint_restores_managed_state(tmp_path,dtype):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    if dtype==torch.bfloat16 and not torch.cuda.is_bf16_supported():pytest.skip('CUDA bf16 unavailable')
    from sdxl_trainer.training.paged_adamw import load_paged_runtime,probe_paged_runtime
    override=os.environ.get('SDXL_TEST_BNB_CUDA_OVERRIDE','')
    bnb,runtime=load_paged_runtime(override)
    probe_paged_runtime(bnb,{torch.float32:'fp32',torch.bfloat16:'bf16',torch.float16:'fp16'}[dtype])
    def make():
        parameter=torch.nn.Parameter(torch.ones(100_352,dtype=dtype,device='cuda'))
        groups=[{'named_params':[('unet.weight',parameter)],'lr':.1}]
        optimizer=bnb.optim.PagedAdamW8bit([{'params':[parameter],'lr':.1}])
        return parameter,optimizer,PagedAdamWCheckpoint(optimizer,groups,runtime)
    parameter,optimizer,checkpoint=make()
    parameter.grad=torch.ones_like(parameter);optimizer.step();optimizer.zero_grad();torch.cuda.synchronize()
    assert optimizer.state[parameter]['state1'].is_paged
    checkpoint.save_state(tmp_path/'checkpoint')
    resumed,restored,replay=make();resumed.data.copy_(parameter.data);replay.load_state(tmp_path/'checkpoint')
    assert restored.state[resumed]['state1'].is_paged and restored.state[resumed]['state2'].is_paged
    for a,b in ((parameter,optimizer),(resumed,restored)):
        a.grad=torch.ones_like(a);b.step();b.zero_grad()
    torch.cuda.synchronize()
    assert torch.equal(parameter,resumed)
    for key in optimizer.state[parameter]:
        a,b=optimizer.state[parameter][key],restored.state[resumed][key]
        assert torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b


@pytest.mark.parametrize('mode',['full','lora'])
def test_native_cuda_paged_tiny_sdxl_two_images_and_resume(tmp_path,monkeypatch,tiny_sdxl_model,mode):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    if not torch.cuda.is_bf16_supported():pytest.skip('CUDA bf16 unavailable')
    from PIL import Image
    from safetensors.torch import load_file
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import worker
    monkeypatch.setattr(dataset,'BUCKETS',((64,64),))
    original=worker.preflight
    def tiny(*args,**kwargs):
        result=original(*args,**kwargs)
        result.errors=[e for e in result.errors if 'free disk space' not in e and 'GiB VRAM is free' not in e]
        return result
    monkeypatch.setattr(worker,'preflight',tiny)
    files=[]
    for i,color in enumerate(('blue','purple')):
        p=tmp_path/f'{i}.png';Image.new('RGB',(64,64),color).save(p);files.append(p)
    project=dataset.DatasetProject.create(tmp_path/'project');project.import_images(files)
    project.save_caption(1,'a');project.save_caption(2,' '.join(['a']*151))
    finalized=project.finalize_in_project(TokenCounter.from_model(tiny_sdxl_model))
    values=dict(model_path=str(tiny_sdxl_model),dataset_path=str(finalized),mode=mode,
        model_precision='bf16',precision='bf16',vae_precision='fp32',train_text_encoder=False,
        device='cuda',optimizer='paged_adamw8bit',paged_cuda_override=os.environ.get('SDXL_TEST_BNB_CUDA_OVERRIDE',''),
        batch_size=1,gradient_accumulation_steps=1,max_steps=2,save_every=1,warmup_steps=0,
        rank=2,lora_alpha=2,learning_rate=.01,monitor_gpu=False,profile_phases=False)
    config=legacy_training_config(**values,output_dir=str(tmp_path/'resumed'))
    monkeypatch.setattr(worker,'_STOP',True);worker.run(config)
    config.resume_from=str(Path(config.output_dir)/'checkpoints/step-00000001')
    monkeypatch.setattr(worker,'_STOP',False);worker.run(config)
    straight=legacy_training_config(**values,output_dir=str(tmp_path/'straight'));worker.run(straight)
    a=load_file(str(Path(config.output_dir)/'checkpoints/step-00000002/unet.safetensors'))
    b=load_file(str(Path(straight.output_dir)/'checkpoints/step-00000002/unet.safetensors'))
    assert a.keys()==b.keys() and all(torch.equal(value,b[key]) for key,value in a.items())
