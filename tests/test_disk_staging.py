import torch
import pytest
from sdxl_trainer.training.disk_staging import DiskModuleStager


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_snapshot_reuse_avoids_rewriting_and_detects_mutation(tmp_path, dtype, monkeypatch):
    model = torch.nn.Linear(5, 3).to(dtype=dtype)
    model.register_buffer("counter", torch.tensor([1]))
    stager = DiskModuleStager(model, tmp_path, batch_bytes=16)
    evictions = []
    original_evict = stager._evict
    def record_evict(start, end):
        evictions.append((start, end))
        original_evict(start, end)
    monkeypatch.setattr(stager, "_evict", record_evict)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    stager.park()
    assert evictions  # The first snapshot is written.
    stager.restore("cpu")
    for count in (2, 3):
        # Buffers change independently and must always be moved afresh.
        model.counter.fill_(count)
        evictions.clear()
        stager.park()
        assert not evictions  # The unchanged parameter snapshot is reused.
        stager.restore("cpu")
        assert model.counter.item() == count
        assert torch.equal(model.weight, initial["weight"])
    with torch.no_grad():
        model.weight.add_(1)  # Version counter detects ordinary mutation.
    updated = model.weight.detach().clone()
    evictions.clear()
    stager.park()
    assert evictions
    stager.restore("cpu")
    assert torch.equal(model.weight, updated)
    # Raw .data writes can bypass PyTorch's version counter: invalidate first.
    stager.invalidate()
    model.weight.data.add_(2)
    raw_updated = model.weight.detach().clone()
    evictions.clear()
    stager.park()
    assert evictions
    stager.restore("cpu")
    assert torch.equal(model.weight, raw_updated)
    stager.close()


def test_snapshot_reuse_preserves_accumulated_updates_and_resume(tmp_path):
    torch.manual_seed(17)
    optimized = torch.nn.Linear(4, 3)
    baseline = torch.nn.Linear(4, 3)
    baseline.load_state_dict(optimized.state_dict())
    stager = DiskModuleStager(optimized, tmp_path, batch_bytes=16)
    batches = [torch.randn(2, 4) for _ in range(4)]
    def step(model, tensors, stage=None):
        accumulated = [torch.zeros_like(p) for p in model.parameters()]
        for tensor in tensors:
            if stage:
                stage.park(); stage.restore("cpu")
            model(tensor).square().mean().backward()
            for value, parameter in zip(accumulated, model.parameters()):
                value.add_(parameter.grad)
                parameter.grad = None
            if stage:
                stage.park(); stage.restore("cpu")
        if stage:
            stage.invalidate()
        with torch.no_grad():
            for parameter, gradient in zip(model.parameters(), accumulated):
                parameter.sub_(gradient * .01)
        if stage:
            stage.invalidate()
    step(optimized, batches[:2], stager)
    step(baseline, batches[:2])
    assert all(torch.equal(left, right) for left, right in zip(optimized.parameters(), baseline.parameters()))
    checkpoint = {key: value.clone() for key, value in optimized.state_dict().items()}
    # Checkpoint load after a valid staging snapshot must discard the old image.
    stager.park(); stager.restore("cpu")
    stager.invalidate()
    optimized.load_state_dict(checkpoint)
    step(optimized, batches[2:], stager)
    baseline.load_state_dict(checkpoint)
    step(baseline, batches[2:])
    assert all(torch.equal(left, right) for left, right in zip(optimized.parameters(), baseline.parameters()))
    stager.close()


def test_staging_preserves_fp32_values_hooks_and_repeated_updates(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(17, 3), torch.nn.Linear(3, 2))
    baseline = {k: v.detach().clone() for k, v in model.state_dict().items()}
    identities = {n: p for n, p in model.named_parameters()}
    counts = {n: 0 for n in identities}
    for n, p in identities.items():
        p.register_post_accumulate_grad_hook(lambda _, name=n: counts.__setitem__(name, counts[name] + 1))
    stager = DiskModuleStager(model, tmp_path, batch_bytes=13)
    stager.park()
    assert all(p.device.type == 'cpu' for p in model.parameters())
    assert all(torch.equal(baseline[n], v) for n, v in model.state_dict().items())
    stager.restore('cpu')
    assert all(p is dict(model.named_parameters())[n] for n, p in identities.items())
    assert all(torch.equal(baseline[n], v) for n, v in model.state_dict().items())
    for _ in range(2):
        model(torch.randn(2, 17)).sum().backward()
        with torch.no_grad():
            for p in model.parameters():
                p.sub_(p.grad * .01)
                p.grad = None
        stager.park()
        stager.restore('cpu')
    assert all(count == 2 for count in counts.values())
    assert all(not torch.equal(baseline[n], v) for n, v in model.state_dict().items())
    stager.close()


def test_staging_failure_invalidates_future_transfers(tmp_path, monkeypatch):
    model = torch.nn.Linear(17, 3)
    stager = DiskModuleStager(model, tmp_path, batch_bytes=1)
    monkeypatch.setattr(stager, '_evict', lambda *args: (_ for _ in ()).throw(OSError('simulated write failure')))
    with pytest.raises(OSError, match='simulated write failure'):
        stager.park()
    with pytest.raises(RuntimeError, match='invalidated'):
        stager.restore('cpu')
    fd = stager.file.fileno()
    stager.close()
    with pytest.raises(OSError):
        import os
        os.fstat(fd)


def test_restore_clone_failure_releases_mmap_with_retained_traceback(tmp_path, monkeypatch):
    stager = DiskModuleStager(torch.nn.Linear(2, 2), tmp_path)
    stager.park()

    def fail_clone(tensor, *args, **kwargs):
        # Like an exception from the native clone operation, do not keep the
        # exported tensor in this test helper's Python traceback frame.
        del tensor
        raise MemoryError('simulated clone failure')

    monkeypatch.setattr(torch.Tensor, 'clone', fail_clone)
    with pytest.raises(MemoryError, match='simulated clone failure') as error:
        stager.restore('cpu')
    assert error.value.__traceback__ is not None
    assert stager.poisoned
    stager.close()
    assert stager.closed


def test_reused_snapshot_failure_poisoned_and_discarded(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 2)
    stager = DiskModuleStager(model, tmp_path)
    stager.park(); stager.restore('cpu')
    monkeypatch.setattr(stager, '_move_buffers', lambda *args: (_ for _ in ()).throw(OSError('buffer transfer failed')))
    with pytest.raises(OSError, match='buffer transfer failed'):
        stager.park()
    assert not stager._snapshot_valid
    with pytest.raises(RuntimeError, match='invalidated'):
        stager.restore('cpu')
    stager.close()


def test_staging_rejects_live_gradients_before_changing_parameters(tmp_path):
    model = torch.nn.Linear(2, 2)
    model(torch.ones(1, 2)).sum().backward()
    before = [p.detach().clone() for p in model.parameters()]
    stager = DiskModuleStager(model, tmp_path)
    with pytest.raises(RuntimeError, match='gradients spilled'):
        stager.park()
    assert all(torch.equal(a, b) for a,b in zip(before, model.parameters()))
    assert not stager.poisoned
    stager.close()


def test_staging_refuses_shared_parameter_storage(tmp_path):
    model = torch.nn.Module()
    shared = torch.randn(2, 2)
    model.register_parameter("first", torch.nn.Parameter(shared))
    model.register_parameter("second", torch.nn.Parameter(shared))
    with pytest.raises(ValueError, match="sharing storage"):
        DiskModuleStager(model, tmp_path)


def test_staging_preallocation_failure_preserves_weights(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 2)
    before = [p.detach().clone() for p in model.parameters()]
    def fail(*args):
        raise OSError("simulated full disk")
    monkeypatch.setattr("os.posix_fallocate", fail)
    with pytest.raises(RuntimeError, match="reserve disk space"):
        DiskModuleStager(model, tmp_path)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_unlinked_staging_file_closes_without_clobbering_existing_files(tmp_path):
    import os
    existing = tmp_path / 'keep.txt'
    existing.write_text('keep')
    stager = DiskModuleStager(torch.nn.Linear(2, 2), tmp_path)
    fd = stager.file.fileno()
    assert os.fstat(fd).st_nlink == 0
    assert list(tmp_path.iterdir()) == [existing]
    stager.park()
    stager.close()
    with pytest.raises(OSError):
        os.fstat(fd)
    assert existing.read_text() == 'keep'


def test_staging_rejects_noncontiguous_parameters(tmp_path):
    model = torch.nn.Module()
    model.register_parameter('weight', torch.nn.Parameter(torch.randn(2, 3).T))
    with pytest.raises(ValueError, match='contiguous fp32'):
        DiskModuleStager(model, tmp_path)


def test_tied_parameter_and_shared_integer_buffer_remain_shared(tmp_path):
    model = torch.nn.Module()
    parameter = torch.nn.Parameter(torch.randn(2, 2))
    model.register_parameter('a', parameter)
    model.register_parameter('b', parameter)
    model.child = torch.nn.Module()
    shared = torch.tensor([1, 2], dtype=torch.int64)
    model.register_buffer('indices', shared)
    model.child.register_buffer('indices', shared)
    original = parameter.detach().clone()
    stager = DiskModuleStager(model, tmp_path, batch_bytes=8)
    stager.park()
    stager.restore('cpu')
    assert model.a is model.b is parameter
    assert model.indices is model.child.indices
    assert model.indices.dtype == torch.int64
    assert torch.equal(original, parameter)
    stager.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_disk_staging_round_trips_gpu_bitwise_and_keeps_hooks(tmp_path):
    torch.manual_seed(123)
    model = torch.nn.Sequential(torch.nn.Linear(17, 3), torch.nn.Linear(3, 2)).cuda()
    identities = dict(model.named_parameters())
    baseline = {n: p.detach().cpu().clone() for n, p in identities.items()}
    hook_counts = {n: 0 for n in identities}
    for name, parameter in identities.items():
        parameter.register_post_accumulate_grad_hook(lambda _, name=name: hook_counts.__setitem__(name, hook_counts[name] + 1))
    stager = DiskModuleStager(model, tmp_path, batch_bytes=13)
    stager.park()
    assert all(p.device.type == 'cpu' for p in model.parameters())
    assert all(torch.equal(baseline[n], p) for n, p in model.named_parameters())
    stager.restore('cuda')
    assert all(p is dict(model.named_parameters())[n] for n, p in identities.items())
    assert all(torch.equal(baseline[n], p.detach().cpu()) for n, p in model.named_parameters())
    for _ in range(2):
        model(torch.randn(2, 17, device='cuda')).sum().backward()
        with torch.no_grad():
            for p in model.parameters():
                p.sub_(p.grad * .01)
                p.grad = None
        stager.park()
        stager.restore('cuda')
    assert all(count == 2 for count in hook_counts.values())
    stager.close()
