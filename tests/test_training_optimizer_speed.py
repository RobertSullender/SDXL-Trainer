"""Disk AdamW must stay numerically faithful while avoiding per-tensor syncs."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("Optional ML runtime not installed", allow_module_level=True)
import numpy as np
import torch

from sdxl_trainer.training.disk_optimizer import create_disk_adamw
from test_training import legacy_training_config


def make_parameters(device="cpu"):
    return [torch.nn.Parameter(torch.tensor([[0.5, -0.4], [1.3, -0.1]], device=device)),
            torch.nn.Parameter(torch.tensor([0.1, -1.0], device=device))]


def make_optimizer(parameters, root):
    return create_disk_adamw([
        {"named_params": [("unet.weight", parameters[0])], "lr": 0.01},
        {"named_params": [("unet.bias", parameters[1])], "lr": 0.003},
    ], root, betas=(0.8, 0.95), eps=1e-7, weight_decay=0.04)


def make_reference(parameters):
    return torch.optim.AdamW([{"params": [parameters[0]], "lr": 0.01},
                              {"params": [parameters[1]], "lr": 0.003}],
                             betas=(0.8, 0.95), eps=1e-7,
                             weight_decay=0.04, foreach=False)


def accumulate(parameters, values):
    for scale in values:
        loss = ((parameters[0] * scale).sin().square().sum()
                + (parameters[1] * (1.2 - scale)).square().sum()) / len(values)
        loss.backward()


def update_pair(reference, expected, actual, disk, batches, clip=0.7):
    accumulate(reference, batches)
    accumulate(actual, batches)
    norm = torch.nn.utils.clip_grad_norm_(reference, clip, error_if_nonfinite=True).item()
    expected.step()
    actual_norm = disk.step(max_grad_norm=clip)
    expected.zero_grad(set_to_none=True)
    disk.zero_grad(set_to_none=True)
    assert actual_norm == pytest.approx(norm, abs=1e-6, rel=1e-6)
    for a, b in zip(reference, actual):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)


def test_accumulation_clipping_groups_and_resume_match_adamw(tmp_path):
    reference = make_parameters()
    actual = make_parameters()
    expected = make_reference(reference)
    disk = make_optimizer(actual, tmp_path / "working")
    for batches in ([0.1, 0.8, 1.3, 0.4], [0.2, 0.5], [1.2, 0.9, 0.3]):
        update_pair(reference, expected, actual, disk, batches)
    checkpoint = tmp_path / "checkpoint"
    disk.save_state(checkpoint)
    resumed = [torch.nn.Parameter(p.detach().clone()) for p in actual]
    resumed_optimizer = make_optimizer(resumed, tmp_path / "resumed")
    resumed_optimizer.load_state(checkpoint)
    for batches in ([0.3, 0.6], [1.1, 0.2, 0.8, 0.9]):
        update_pair(reference, expected, resumed, resumed_optimizer, batches)
        accumulate(actual, batches)
        disk.step(max_grad_norm=0.7)
        disk.zero_grad(set_to_none=True)
        for left, right in zip(actual, resumed):
            assert torch.equal(left, right), "Resumed training must be bitwise identical"


def test_working_maps_do_not_flush_every_tensor(tmp_path, monkeypatch):
    flushes = []
    original_flush = np.memmap.flush
    def record_flush(mapping):
        flushes.append(Path(mapping.filename).name)
        return original_flush(mapping)
    monkeypatch.setattr(np.memmap, "flush", record_flush)
    parameters = make_parameters()
    optimizer = make_optimizer(parameters, tmp_path / "working")
    for _ in range(3):
        accumulate(parameters, [0.2, 0.4, 0.6, 0.8])
        optimizer.step(max_grad_norm=0.5)
        optimizer.zero_grad(set_to_none=True)
    assert flushes == []
    optimizer.save_state(tmp_path / "checkpoint")
    assert flushes == ["first_moment.bin", "second_moment.bin"]


@pytest.mark.parametrize("eager", [False, True])
def test_evicted_accumulated_gradients_preserve_values_and_update(tmp_path, eager):
    """Eviction between microbatches must not reset or change AdamW inputs."""
    reference = make_parameters()
    actual = make_parameters()
    def disk(params, path, *, eagerly=False):
        return create_disk_adamw([{"named_params": [("unet.weight", params[0])], "lr": 0.01},
                                  {"named_params": [("unet.bias", params[1])], "lr": 0.003}], path,
                                 betas=(0.8, 0.95), eps=1e-7, weight_decay=0.04, gradient_storage="disk", eager_gradient_eviction=eagerly)
    expected = disk(reference, tmp_path / "reference")
    optimizer = disk(actual, tmp_path / "evicted", eagerly=eager)
    optimizer._writeback_batch_bytes = 20  # Reverse-order records share a page.
    batches = []
    original_evict = optimizer._evict_pending_gradients
    def record_batch():
        batches.append(list(optimizer._pending_gradient_ranges))
        return original_evict()
    optimizer._evict_pending_gradients = record_batch
    for _ in range(3):
        for value in (0.2, 0.7, 1.2):
            accumulate(reference, [value])
            accumulate(actual, [value])
            active = set(optimizer.active)
            gradients = np.array(optimizer._maps["gradients.bin"])
            optimizer.evict_accumulated_gradients()
            assert optimizer.active == active
            np.testing.assert_array_equal(optimizer._maps["gradients.bin"], gradients)
        assert optimizer.step(max_grad_norm=0.7) == pytest.approx(expected.step(max_grad_norm=0.7), abs=1e-12)
        optimizer.zero_grad(set_to_none=True)
        expected.zero_grad(set_to_none=True)
        for a, b in zip(actual, reference):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    if eager:
        assert any(len(batch) == 2 and batch[0][0] > batch[1][0] for batch in batches), "Reverse-order hooks must be coalesced safely."


def test_gradient_eviction_failure_keeps_accumulation_active(tmp_path, monkeypatch):
    parameters = make_parameters()
    optimizer = create_disk_adamw([{"named_params": [(f"p{i}", p) for i, p in enumerate(parameters)], "lr": 0.01}],
                                  tmp_path / "working", gradient_storage="disk")
    accumulate(parameters, [0.2])
    active = set(optimizer.active)
    def fail(*args):
        raise OSError("simulated advice failure")
    monkeypatch.setattr(os, "posix_fadvise", fail)
    with pytest.raises(OSError, match="simulated advice failure"):
        optimizer.evict_accumulated_gradients()
    assert optimizer.active == active
    with pytest.raises(RuntimeError, match="optimizer boundary"):
        optimizer.save_state(tmp_path / "partial")


def test_eager_gradient_eviction_failure_prevents_checkpoint(tmp_path, monkeypatch):
    parameters = make_parameters()
    optimizer = create_disk_adamw([{"named_params": [(f"p{i}", p) for i, p in enumerate(parameters)], "lr": 0.01}],
                                  tmp_path / "working", gradient_storage="disk", eager_gradient_eviction=True)
    optimizer._writeback_batch_bytes = 1
    def fail(*args):
        raise OSError("simulated advice failure")
    monkeypatch.setattr(os, "posix_fadvise", fail)
    with pytest.raises(OSError, match="simulated advice failure"):
        accumulate(parameters, [0.2])
    assert optimizer.active
    assert optimizer._pending_gradient_ranges
    with pytest.raises(RuntimeError, match="optimizer boundary"):
        optimizer.save_state(tmp_path / "partial")


@pytest.mark.parametrize("storage", ["ram", "disk"])
def test_completed_ranges_are_batched_flushed_and_evicted(tmp_path, monkeypatch, storage):
    _mock_available_ram(monkeypatch, 64 * 1024 * 1024)
    parameters = make_parameters()
    optimizer = create_disk_adamw([{"named_params": [(f"p{i}", p) for i, p in enumerate(parameters)], "lr": 0.01}],
                                  tmp_path / "working", update_device="cpu", gradient_storage=storage)
    optimizer._writeback_batch_bytes = 8
    actual = optimizer._evict_moment_range
    ranges = []
    def record_range(start, end):
        if end > start:
            ranges.append((start, end))
        return actual(start, end)
    monkeypatch.setattr(optimizer, "_evict_moment_range", record_range)
    advised = []
    original_advise = os.posix_fadvise
    def record_advice(fd, offset, length, advice):
        advised.append(Path(os.readlink(f"/proc/self/fd/{fd}")).name)
        return original_advise(fd, offset, length, advice)
    monkeypatch.setattr(os, "posix_fadvise", record_advice)
    reference = make_parameters()
    expected = torch.optim.AdamW([{"params": reference, "lr": 0.01}], weight_decay=0.01, foreach=False)
    # Both records overlap the same filesystem page; their distinct gradients
    # must survive aligned flushes without corrupting neighboring moments.
    update_pair(reference, expected, parameters, optimizer, [0.2, 0.4], clip=0.5)
    assert ranges == [(0, 16), (16, 24)]
    assert advised.count("first_moment.bin") == 2
    assert advised.count("second_moment.bin") == 2
    assert advised.count("gradients.bin") == (2 if storage == "disk" else 0)
    optimizer.save_state(tmp_path / "checkpoint")
    resumed = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    resumed_optimizer = create_disk_adamw([{"named_params": [(f"p{i}", p) for i, p in enumerate(resumed)], "lr": 0.01}],
                                         tmp_path / "resumed", update_device="cpu", gradient_storage=storage)
    resumed_optimizer.load_state(tmp_path / "checkpoint")
    for _ in range(3):
        accumulate(parameters, [.3, .8]); accumulate(resumed, [.3, .8])
        optimizer.step(max_grad_norm=.5); resumed_optimizer.step(max_grad_norm=.5)
        optimizer.zero_grad(set_to_none=True); resumed_optimizer.zero_grad(set_to_none=True)
        for left, right in zip(parameters, resumed):
            assert torch.equal(left, right)


def test_moment_writeback_failure_does_not_publish_partial_checkpoint(tmp_path, monkeypatch):
    _mock_available_ram(monkeypatch, 64 * 1024 * 1024)
    parameters = make_parameters()
    optimizer = create_disk_adamw([{"named_params": [(f"p{i}", p) for i, p in enumerate(parameters)], "lr": .01}],
                                  tmp_path / "working", update_device="cpu", gradient_storage="ram")
    accumulate(parameters, [.2]); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    optimizer.save_state(tmp_path / "previous-checkpoint")
    before = {p.name: p.read_bytes() for p in (tmp_path / "previous-checkpoint").iterdir()}
    original_map = optimizer._maps["first_moment.bin"]
    attempted_eviction = []
    class BadMapping:
        def flush(self, *args): raise OSError("simulated sync failure")
        def madvise(self, *args): attempted_eviction.append(True)
    class MapProxy:
        _mmap = BadMapping()
        def __getitem__(self, item): return original_map[item]
    optimizer._maps["first_moment.bin"] = MapProxy()
    accumulate(parameters, [.4])
    with pytest.raises(OSError, match="simulated sync failure"):
        optimizer.step()
    assert not attempted_eviction
    with pytest.raises(RuntimeError, match="optimizer boundary"):
        optimizer.save_state(tmp_path / "partial")
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "previous-checkpoint").iterdir()}


def test_checkpoint_requires_step_boundary_and_rejects_truncated_state(tmp_path):
    parameters = make_parameters()
    optimizer = make_optimizer(parameters, tmp_path / "working")
    accumulate(parameters, [0.2])
    with pytest.raises(RuntimeError, match="optimizer boundary"):
        optimizer.save_state(tmp_path / "unfinished")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    checkpoint = tmp_path / "checkpoint"
    optimizer.save_state(checkpoint)
    first_before = (tmp_path / "working/first_moment.bin").read_bytes()
    steps_before = [r["step"] for r in optimizer.records]
    (checkpoint / "second_moment.bin").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="Truncated"):
        optimizer.load_state(checkpoint)
    assert (tmp_path / "working/first_moment.bin").read_bytes() == first_before
    assert [r["step"] for r in optimizer.records] == steps_before


def test_bounded_optimizer_copy_handles_short_io_and_streaming_writeback(tmp_path, monkeypatch):
    import sdxl_trainer.training.disk_optimizer as module
    payload = os.urandom(1_200_123)
    source = tmp_path / 'source'; source.write_bytes(payload)
    dest = tmp_path / 'snapshot'
    read, write, sync = os.read, os.write, os.fsync
    syncs = []
    def short_read(fd, count): return read(fd, min(count, 4093))
    def short_write(fd, data): return write(fd, data[:3071])
    def counted_sync(fd): syncs.append(fd); return sync(fd)
    monkeypatch.setattr(module.os, 'read', short_read)
    monkeypatch.setattr(module.os, 'write', short_write)
    monkeypatch.setattr(module.os, 'fsync', counted_sync)
    module._copy_state_file_bounded(source, dest, len(payload), create=True,
                                    chunk_bytes=32_768, writeback_bytes=256_000)
    assert dest.read_bytes() == payload
    assert len(syncs) >= 4
    # Resume overwrites the existing working file without truncating its mmap.
    working = tmp_path / 'working'; working.write_bytes(b'\0' * len(payload))
    mapped = np.memmap(working, dtype='uint8', mode='r+')
    module._copy_state_file_bounded(dest, working, len(payload), create=False,
                                    chunk_bytes=32_768, writeback_bytes=256_000)
    assert bytes(mapped) == payload


def test_bounded_optimizer_copy_rejects_symlinks_and_preserves_existing(tmp_path):
    from sdxl_trainer.training.disk_optimizer import _copy_state_file_bounded
    source=tmp_path/'source'; source.write_bytes(b'original data')
    existing=tmp_path/'existing'; existing.write_bytes(b'preserve this')
    with pytest.raises(FileExistsError):
        _copy_state_file_bounded(source, existing, source.stat().st_size, create=True)
    assert existing.read_bytes()==b'preserve this'
    source_link=tmp_path/'source-link';source_link.symlink_to(source)
    with pytest.raises(OSError):
        _copy_state_file_bounded(source_link, tmp_path/'new', source.stat().st_size, create=True)
    dest_link=tmp_path/'dest-link';dest_link.symlink_to(existing)
    with pytest.raises(OSError):
        _copy_state_file_bounded(source, dest_link, source.stat().st_size, create=False)
    assert existing.read_bytes()==b'preserve this'


@pytest.mark.parametrize('failure', ['short-read','sync'])
def test_bounded_optimizer_copy_failure_removes_incomplete_file(tmp_path, monkeypatch, failure):
    import sdxl_trainer.training.disk_optimizer as module
    source=tmp_path/'source';source.write_bytes(os.urandom(2048))
    dest=tmp_path/'partial'
    if failure=='short-read':
        read=os.read; calls=[]
        def interrupted(fd, count):
            calls.append(1)
            return read(fd, min(count,128)) if len(calls)==1 else b''
        monkeypatch.setattr(module.os,'read',interrupted)
    else:
        def failed(fd): raise OSError('injected sync failure')
        monkeypatch.setattr(module.os,'fsync',failed)
    with pytest.raises((ValueError,OSError)):
        module._copy_state_file_bounded(source,dest,2048,create=True,chunk_bytes=128,writeback_bytes=128)
    assert not dest.exists()
    assert source.stat().st_size==2048


def test_optimizer_copy_failure_never_publishes_metadata_or_changes_prior_checkpoint(tmp_path,monkeypatch):
    import sdxl_trainer.training.disk_optimizer as module
    params=make_parameters(); optimizer=make_optimizer(params,tmp_path/'work')
    accumulate(params,[.2]);optimizer.step();optimizer.zero_grad()
    old=tmp_path/'old';optimizer.save_state(old)
    before={file.name:file.read_bytes() for file in old.iterdir()}
    copy=module._copy_state_file_bounded
    def fail_second(source,destination,*args,**kwargs):
        if Path(source).name=='second_moment.bin':raise OSError('injected checkpoint copy failure')
        return copy(source,destination,*args,**kwargs)
    monkeypatch.setattr(module,'_copy_state_file_bounded',fail_second)
    with pytest.raises(OSError,match='injected checkpoint'):
        optimizer.save_state(tmp_path/'partial')
    assert not (tmp_path/'partial/optimizer.json').exists()
    assert before=={file.name:file.read_bytes() for file in old.iterdir()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_parameters_match_fp32_adamw_after_accumulation(tmp_path):
    reference = make_parameters("cuda")
    actual = make_parameters("cuda")
    expected = make_reference(reference)
    disk = make_optimizer(actual, tmp_path / "working")
    for _ in range(4):
        update_pair(reference, expected, actual, disk, [0.2, 0.7, 1.2, 0.4])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_mixed_device_optimizer_matches_adamw_with_accumulation(tmp_path):
    """CPU text and CUDA UNet contribute to one clipped, full-fp32 update."""
    reference = [torch.nn.Parameter(torch.tensor([0.5, -0.4], device="cuda")),
                 torch.nn.Parameter(torch.tensor([1.3, -0.1], device="cpu"))]
    actual = [torch.nn.Parameter(p.detach().clone()) for p in reference]
    expected = torch.optim.AdamW(reference, lr=0.01, betas=(0.8, 0.95), eps=1e-7, weight_decay=0.04, foreach=False)
    optimizer = create_disk_adamw([{"named_params": [("unet", actual[0]), ("clip", actual[1])], "lr": 0.01}],
                                  tmp_path / "working-mixed", betas=(0.8, 0.95), eps=1e-7, weight_decay=0.04)
    assert optimizer.update_backend == "cpu"
    for _ in range(3):
        for scale in (0.2, 0.7):
            for params in (reference, actual):
                ((params[0] * scale).sin().square().sum() + (params[1].to("cuda") * (1.2 - scale)).square().sum()).div(2).backward()
        norm = torch.nn.utils.clip_grad_norm_(reference, 0.7, error_if_nonfinite=True).item()
        expected.step()
        assert optimizer.step(max_grad_norm=0.7) == pytest.approx(norm, abs=1e-6, rel=1e-6)
        expected.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        for a, b in zip(reference, actual):
            torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)

# Reuse the existing tiny SDXL fixture; these tests never read or modify the
# user's real models, projects, checkpoints, or captions.
from importlib.util import module_from_spec, spec_from_file_location
_fixture_spec = spec_from_file_location("sdxl_existing_test_fixture", Path(__file__).with_name("test_training.py"))
_fixture_module = module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
tiny_sdxl_model = _fixture_module.tiny_sdxl_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("mode", ["lora", "full"])
@pytest.mark.parametrize("component_offload", ["none", "text_encoders_cpu", "text_and_vae_cpu", "text_cpu_staged_unet", "stage_unet_gpu_text"])
def test_cuda_tiny_all_components_training(tmp_path, monkeypatch, tiny_sdxl_model, mode, component_offload):
    """All components update, including across a CPU/GPU autograd boundary."""
    from PIL import Image
    from diffusers import StableDiffusionXLPipeline
    from safetensors.torch import load_file
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import TrainingConfig, worker

    monkeypatch.setattr(dataset, "BUCKETS", ((64, 64),))
    original_preflight = worker.preflight
    def tiny_preflight(*args, **kwargs):
        report = original_preflight(*args, **kwargs)
        # Production estimates intentionally describe real SDXL, not this tiny
        # synthetic model; no device allocation or dependency errors are ignored.
        report.errors = [e for e in report.errors if "free disk space" not in e and "GiB VRAM is free" not in e]
        return report
    monkeypatch.setattr(worker, "preflight", tiny_preflight)
    monkeypatch.setattr(worker, "_STOP", False)
    original = StableDiffusionXLPipeline.from_pretrained(str(tiny_sdxl_model), local_files_only=True, use_safetensors=True)
    baseline = {name: {k: v.detach().clone() for k, v in getattr(original, name).state_dict().items()}
                for name in ("unet", "text_encoder", "text_encoder_2", "vae")}
    del original
    image = tmp_path / "original.png"
    Image.new("RGB", (64, 64), "purple").save(image)
    project = dataset.DatasetProject.create(tmp_path / "project")
    project.import_images([image])
    project.save_caption(1, "a")
    finalized = project.finalize_in_project(TokenCounter.from_model(tiny_sdxl_model))
    run = tmp_path / "run"
    config = legacy_training_config(model_path=str(tiny_sdxl_model), dataset_path=str(finalized),
                            output_dir=str(run), mode=mode, train_text_encoder=True,
                            train_vae=True, device="cuda", precision="bf16", batch_size=1,
                            gradient_accumulation_steps=1, max_steps=1, save_every=1,
                            rank=2, lora_alpha=2, gradient_checkpointing=True, warmup_steps=0,
                            # Exercise staged post-backward cleanup with optional
                            # SNR tensors present as well as the default path.
                            snr_gamma=5.0 if component_offload == "stage_unet_gpu_text" else 0.0,
                            component_offload=component_offload)
    torch.cuda.reset_peak_memory_stats()
    worker.run(config)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    assert peak > 0
    for name in ("unet", "text_encoder", "text_encoder_2", "vae"):
        state = load_file(str(run / "checkpoints/step-00000001" / f"{name}.safetensors"))
        if mode == "lora":
            assert any("lora_B" in key and torch.count_nonzero(value) for key, value in state.items()), name
        else:
            assert any(not torch.equal(value, baseline[name][key]) for key, value in state.items()), name
    assert (run / "diffusers/model_index.json").is_file()
    import json
    optimizer_info = json.loads((run / "checkpoints/step-00000001/optimizer/optimizer.json").read_text())
    if component_offload in {"text_encoders_cpu", "text_and_vae_cpu", "text_cpu_staged_unet"}:
        assert optimizer_info["update_backend"] == "cpu"
    elif component_offload == "stage_unet_gpu_text":
        assert optimizer_info["update_backend"] == "cuda"
    print(f"tiny SDXL {mode}: all four components updated on CUDA; peak allocated {peak/2**20:.1f} MiB")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("mode", ["lora", "full"])
@pytest.mark.parametrize("component_offload", ["text_encoders_cpu", "text_cpu_staged_unet", "stage_unet_gpu_text"])
def test_cpu_text_cuda_training_resumes_with_two_images(tmp_path, monkeypatch, tiny_sdxl_model, mode, component_offload):
    """Checkpoint preserves CPU and GPU RNG and optimizer state across placement."""
    from PIL import Image
    from safetensors.torch import load_file
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import TrainingConfig, worker
    monkeypatch.setattr(dataset, "BUCKETS", ((64, 64),))
    original_preflight = worker.preflight
    def tiny_preflight(*args, **kwargs):
        report = original_preflight(*args, **kwargs)
        report.errors = [e for e in report.errors if "free disk space" not in e and "GiB VRAM is free" not in e]
        return report
    monkeypatch.setattr(worker, "preflight", tiny_preflight)
    images = []
    for index, color in enumerate(("purple", "green")):
        path = tmp_path / f"original-{index}.png"
        Image.new("RGB", (64, 64), color).save(path)
        images.append(path)
    project = dataset.DatasetProject.create(tmp_path / "project")
    project.import_images(images)
    for index in (1, 2):
        project.save_caption(index, "a")
    finalized = project.finalize_in_project(TokenCounter.from_model(tiny_sdxl_model))
    config = legacy_training_config(model_path=str(tiny_sdxl_model), dataset_path=str(finalized),
                            output_dir=str(tmp_path / "resumed"), mode=mode, train_text_encoder=True,
                            train_vae=True, device="cuda", precision="bf16", batch_size=1,
                            gradient_accumulation_steps=2, max_steps=2, save_every=1,
                            rank=2, lora_alpha=2, gradient_checkpointing=True, warmup_steps=0,
                            component_offload=component_offload)
    monkeypatch.setattr(worker, "_STOP", True)
    worker.run(config)
    config.resume_from = str(Path(config.output_dir) / "checkpoints/step-00000001")
    monkeypatch.setattr(worker, "_STOP", False)
    worker.run(config)
    uninterrupted = TrainingConfig.from_dict({**config.to_dict(), "output_dir": str(tmp_path / "uninterrupted"), "resume_from": ""})
    worker.run(uninterrupted)
    # Independently seeded GPU executions may differ by small floating-point
    # amounts in nondeterministic kernels; RNG and sampler state must still be
    # identical at the checkpoint, and continued weights must remain close.
    resumed_rng = torch.load(Path(config.output_dir) / "checkpoints/step-00000001/rng.pt", map_location="cpu", weights_only=True)
    expected_rng = torch.load(Path(uninterrupted.output_dir) / "checkpoints/step-00000001/rng.pt", map_location="cpu", weights_only=True)
    assert torch.equal(resumed_rng["torch"], expected_rng["torch"])
    assert all(torch.equal(a, b) for a, b in zip(resumed_rng["cuda"], expected_rng["cuda"]))
    assert resumed_rng["python"] == expected_rng["python"]
    assert torch.equal(resumed_rng["numpy_keys"], expected_rng["numpy_keys"])
    assert all(resumed_rng[k] == expected_rng[k] for k in ("numpy_position", "numpy_gauss", "numpy_cached"))
    import json
    resumed_info = json.loads((Path(config.output_dir) / "checkpoints/step-00000001/checkpoint.json").read_text())
    expected_info = json.loads((Path(uninterrupted.output_dir) / "checkpoints/step-00000001/checkpoint.json").read_text())
    assert all(resumed_info[k] == expected_info[k] for k in ("step", "epoch", "batch_cursor", "config"))
    left = Path(config.output_dir) / "checkpoints/step-00000001/optimizer"
    right = Path(uninterrupted.output_dir) / "checkpoints/step-00000001/optimizer"
    assert json.loads((left / "optimizer.json").read_text()) == json.loads((right / "optimizer.json").read_text())
    for name in ("first_moment.bin", "second_moment.bin"):
        np.testing.assert_allclose(np.memmap(left / name, dtype="float32", mode="r"),
                                   np.memmap(right / name, dtype="float32", mode="r"), rtol=1e-3, atol=1e-6)
    for component in ("unet", "text_encoder", "text_encoder_2", "vae"):
        actual = load_file(str(Path(config.output_dir) / "checkpoints/step-00000002" / f"{component}.safetensors"))
        expected = load_file(str(Path(uninterrupted.output_dir) / "checkpoints/step-00000002" / f"{component}.safetensors"))
        first = load_file(str(Path(config.output_dir) / "checkpoints/step-00000001" / f"{component}.safetensors"))
        assert any(not torch.equal(actual[name], first[name]) for name in actual), f"{component} stopped updating after the first step"
        assert actual.keys() == expected.keys()
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-3, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_staging_write_failure_closes_temporary_store_and_saves_no_checkpoint(tmp_path, monkeypatch, tiny_sdxl_model):
    from PIL import Image
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import TrainingConfig, worker
    from sdxl_trainer.training.disk_staging import DiskModuleStager
    monkeypatch.setattr(dataset, "BUCKETS", ((64, 64),))
    original_preflight = worker.preflight
    def tiny_preflight(*args, **kwargs):
        report = original_preflight(*args, **kwargs)
        report.errors = [e for e in report.errors if "free disk space" not in e and "GiB VRAM is free" not in e]
        return report
    monkeypatch.setattr(worker, "preflight", tiny_preflight)
    monkeypatch.setattr(worker, "_STOP", False)
    image = tmp_path / "original.png"
    Image.new("RGB", (64, 64), "purple").save(image)
    project = dataset.DatasetProject.create(tmp_path / "project")
    project.import_images([image]); project.save_caption(1, "a")
    finalized = project.finalize_in_project(TokenCounter.from_model(tiny_sdxl_model))
    stores = []
    original_init = DiskModuleStager.__init__
    def capture_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.batch_bytes = 1
        stores.append(self)
    monkeypatch.setattr(DiskModuleStager, "__init__", capture_init)
    def fail_write(*args):
        raise OSError("simulated staged write failure")
    monkeypatch.setattr(DiskModuleStager, "_evict", fail_write)
    run = tmp_path / "run"
    config = legacy_training_config(model_path=str(tiny_sdxl_model), dataset_path=str(finalized), output_dir=str(run),
                            mode="full", train_text_encoder=True, train_vae=True, device="cuda", precision="bf16",
                            batch_size=1, gradient_accumulation_steps=1, max_steps=1, save_every=1,
                            gradient_checkpointing=True, warmup_steps=0, component_offload="text_cpu_staged_unet")
    with pytest.raises(OSError, match="simulated staged write failure"):
        worker.run(config)
    assert len(stores) == 1 and stores[0].closed and stores[0].file.closed
    assert not (run / "checkpoints").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_frozen_vae_cache_preserves_training_and_uses_posterior_cache(tmp_path, monkeypatch, tiny_sdxl_model):
    """Fresh posterior samples must yield the same fp32 weights with caching on/off."""
    from PIL import Image
    from safetensors.torch import load_file
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import TrainingConfig, worker

    monkeypatch.setattr(dataset, "BUCKETS", ((64, 64),))
    original_preflight = worker.preflight
    def tiny_preflight(*args, **kwargs):
        report = original_preflight(*args, **kwargs)
        report.errors = [e for e in report.errors if "free disk space" not in e and "GiB VRAM is free" not in e]
        return report
    monkeypatch.setattr(worker, "preflight", tiny_preflight)
    monkeypatch.setattr(worker, "_STOP", False)
    image = tmp_path / "original.png"
    Image.new("RGB", (64, 64), "purple").save(image)
    project = dataset.DatasetProject.create(tmp_path / "project")
    project.import_images([image]); project.save_caption(1, "a")
    finalized = project.finalize_in_project(TokenCounter.from_model(tiny_sdxl_model))
    for cache in (True, False):
        run = tmp_path / ("cached" if cache else "uncached")
        config = legacy_training_config(model_path=str(tiny_sdxl_model), dataset_path=str(finalized),
                                output_dir=str(run), mode="full", train_text_encoder=False,
                                train_vae=False, device="cuda", precision="bf16", batch_size=1,
                                gradient_accumulation_steps=1, max_steps=2, save_every=2,
                                gradient_checkpointing=True, warmup_steps=0, cache_frozen_vae=cache)
        worker.run(config)
    cache_files = list((tmp_path / "cached/working/vae").glob("*.safetensors"))
    assert len(cache_files) == 1
    cached = load_file(str(tmp_path / "cached/checkpoints/step-00000002/unet.safetensors"))
    uncached = load_file(str(tmp_path / "uncached/checkpoints/step-00000002/unet.safetensors"))
    assert cached.keys() == uncached.keys()
    for key in cached:
        torch.testing.assert_close(cached[key], uncached[key], rtol=0, atol=0, msg=lambda m: f"{key}: {m}")


def _mock_available_ram(monkeypatch, available_kib):
    read_text = Path.read_text
    def replacement(path, *args, **kwargs):
        if path == Path('/proc/meminfo'):
            return f'MemAvailable: {available_kib} kB\n'
        return read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', replacement)


def test_ram_gradients_are_bitwise_identical_to_disk_and_resume(tmp_path, monkeypatch):
    _mock_available_ram(monkeypatch, 64 * 1024 * 1024)
    left, right = make_parameters(), make_parameters()
    def build(params, name, storage):
        return create_disk_adamw([{'named_params': [('unet.weight', params[0])], 'lr': .01},
                                  {'named_params': [('unet.bias', params[1])], 'lr': .003}],
                                 tmp_path / name, betas=(.8, .95), eps=1e-7, weight_decay=.04,
                                 update_device='cpu', gradient_storage=storage)
    ram, disk = build(left, 'ram', 'ram'), build(right, 'disk', 'disk')
    assert ram.gradient_storage == 'ram' and disk.gradient_storage == 'disk'
    assert not (tmp_path / 'ram/gradients.bin').exists()
    for batches in ([.2, .7, .4, 1.2], [.3, .9], [1.2, .1, .6]):
        for params, opt in ((left, ram), (right, disk)):
            accumulate(params, batches); opt.step(max_grad_norm=.7); opt.zero_grad(set_to_none=True)
        for a, b in zip(left, right): assert torch.equal(a, b)
    ram.save_state(tmp_path / 'checkpoint')
    resumed_params = [torch.nn.Parameter(p.detach().clone()) for p in left]
    resumed = build(resumed_params, 'resumed-disk', 'disk')
    resumed.load_state(tmp_path / 'checkpoint')
    for params, opt in ((left, ram), (resumed_params, resumed)):
        accumulate(params, [.2, .8, .9, .3]); opt.step(max_grad_norm=.7); opt.zero_grad(set_to_none=True)
    for a, b in zip(left, resumed_params): assert torch.equal(a, b)


def test_ram_storage_respects_available_memory_reserve(tmp_path, monkeypatch):
    _mock_available_ram(monkeypatch, 1024 * 1024)
    params = make_parameters()
    groups = [{'named_params': [(f'p{i}', p) for i, p in enumerate(params)], 'lr': .01}]
    auto = create_disk_adamw(groups, tmp_path / 'auto', update_device='cpu', gradient_storage='auto')
    assert auto.gradient_storage == 'disk'
    with pytest.raises(RuntimeError, match='12 GiB reserve'):
        create_disk_adamw(groups, tmp_path / 'requested', update_device='cpu', gradient_storage='ram')


def test_reduced_master_ram_reserve_covers_checkpoint_headroom(tmp_path, monkeypatch):
    from sdxl_trainer.training.disk_optimizer import _ram_gradient_minimum_available
    gib = 1024**3
    sdxl_gradients = 2_567_463_684 * 4
    assert _ram_gradient_minimum_available(sdxl_gradients, True) > 30 * gib
    assert _ram_gradient_minimum_available(sdxl_gradients, False) == 12 * gib + sdxl_gradients
    # 4 KiB of gradients fits the legacy 12 GiB reserve, but there is not
    # room for the additional 4 KiB reduced-model master allowance.
    _mock_available_ram(monkeypatch, 12 * 1024**2 + 6)
    def make(name, dtype, storage):
        p = torch.nn.Parameter(torch.ones(1024, dtype=dtype))
        return create_disk_adamw([{'named_params': [('p', p)], 'lr': .01}], tmp_path / name,
                                 update_device='cpu', gradient_storage=storage)
    assert make('fp32', torch.float32, 'ram').gradient_storage == 'ram'
    assert make('reduced-auto', torch.bfloat16, 'auto').gradient_storage == 'disk'
    with pytest.raises(RuntimeError, match='checkpoint/export headroom'):
        make('reduced-forced', torch.bfloat16, 'ram')


def test_ram_buffer_release_reacquires_and_preserves_accumulation_and_retry(tmp_path, monkeypatch):
    from sdxl_trainer.training.disk_optimizer import NonFiniteGradients
    _mock_available_ram(monkeypatch, 64 * 1024 * 1024)
    params = [torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float16)) for _ in range(2)]
    def make(p, root, storage):
        return create_disk_adamw([{'named_params': [('p', p)], 'lr': .001}], tmp_path / root,
                                 update_device='cpu', gradient_storage=storage)
    ram = make(params[0], 'ram-release', 'ram')
    disk = make(params[1], 'disk-reference', 'disk')
    ram.release_gradient_buffer()
    assert ram._grad_buffer is None
    for p, optimizer in zip(params, (ram, disk)):
        for factor in (.5, .75):
            (p.float() * factor * 128).sum().backward()
        if optimizer is ram:
            with pytest.raises(RuntimeError, match='gradients or mapped views'):
                ram.release_gradient_buffer()
        optimizer.step(loss_scale=128)
        optimizer.zero_grad()
    assert torch.equal(*params)
    ram.release_gradient_buffer()
    assert ram._grad_buffer is None
    # A retry after an overflow still starts from an unmodified master.
    for p, optimizer in zip(params, (ram, disk)):
        (p.float() * float('inf')).sum().backward()
        with pytest.raises(NonFiniteGradients):
            optimizer.step(loss_scale=128)
        optimizer.zero_grad()
        (p.float() * 128).sum().backward()
        optimizer.step(loss_scale=128)
        optimizer.zero_grad()
    assert torch.equal(*params)
    with ram.mapped('gradients.bin', ram.records[0]) as view:
        with pytest.raises(RuntimeError, match='mapped views'):
            ram.release_gradient_buffer()
    with pytest.raises(RuntimeError, match='mapped views'):
        ram.release_gradient_buffer()
    del view
    ram.release_gradient_buffer()


def test_released_ram_gradients_recheck_headroom_before_next_backward(tmp_path, monkeypatch):
    _mock_available_ram(monkeypatch, 64 * 1024 * 1024)
    p = torch.nn.Parameter(torch.ones(2))
    optimizer = create_disk_adamw([{'named_params': [('p', p)], 'lr': .01}], tmp_path / 'work',
                                  update_device='cpu', gradient_storage='ram')
    optimizer.release_gradient_buffer()
    _mock_available_ram(monkeypatch, 1024 * 1024)
    with pytest.raises(RuntimeError, match='resume the last completed checkpoint'):
        p.sum().backward()
    assert optimizer._grad_buffer is None and not optimizer.active


def test_auto_gradient_storage_uses_conservative_memory_and_swap_headroom(tmp_path, monkeypatch):
    import sdxl_trainer.training.disk_optimizer as module
    real_read = Path.read_text
    def configure(available_gib, swap_free_gib):
        def read(path, *args, **kwargs):
            if path == Path('/proc/meminfo'):
                return (f'MemAvailable: {available_gib * 1024**2} kB\n'
                        f'SwapTotal: {8 * 1024**2} kB\n'
                        f'SwapFree: {swap_free_gib * 1024**2} kB\n')
            return real_read(path, *args, **kwargs)
        monkeypatch.setattr(module.Path, 'read_text', read)
    def create(name):
        p = torch.nn.Parameter(torch.ones(4))
        return create_disk_adamw([{'named_params': [('p', p)], 'lr': .01}], tmp_path / name,
                                 update_device='cpu', gradient_storage='auto')
    configure(12, 4)
    assert create('insufficient-reserve').gradient_storage == 'disk'
    configure(25, 4)
    assert create('typical-32gb-host').gradient_storage == 'ram'
    configure(64, 1)
    assert create('low-swap').gradient_storage == 'disk'
    configure(64, 4)
    assert create('ample-headroom').gradient_storage == 'ram'


def test_backend_mismatch_rejected_before_loading_optimizer_state(tmp_path):
    import json
    params = make_parameters()
    groups = [{'named_params': [(f'p{i}', p) for i, p in enumerate(params)], 'lr': .01}]
    optimizer = create_disk_adamw(groups, tmp_path / 'working', update_device='cpu', gradient_storage='disk')
    accumulate(params, [.2]); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    checkpoint = tmp_path / 'checkpoint'; optimizer.save_state(checkpoint)
    metadata_file = checkpoint / 'optimizer.json'
    metadata = json.loads(metadata_file.read_text()); metadata['update_backend'] = 'cuda'
    metadata_file.write_text(json.dumps(metadata))
    before = [(tmp_path / 'working' / name).read_bytes() for name in ('first_moment.bin','second_moment.bin')]
    steps = [r['step'] for r in optimizer.records]
    with pytest.raises(ValueError, match='choose cuda or auto'):
        optimizer.load_state(checkpoint)
    assert before == [(tmp_path / 'working' / name).read_bytes() for name in ('first_moment.bin','second_moment.bin')]
    assert [r['step'] for r in optimizer.records] == steps


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available')
def test_cuda_backend_chunked_updates_resume_and_legacy_cpu(tmp_path):
    import json
    cpu_math, gpu_math = make_parameters('cuda'), make_parameters('cuda')
    groups = lambda params: [{'named_params': [('w', params[0])], 'lr': .01},
                             {'named_params': [('b', params[1])], 'lr': .003}]
    cpu = create_disk_adamw(groups(cpu_math), tmp_path / 'cpu', update_device='cpu', gradient_storage='ram')
    gpu = create_disk_adamw(groups(gpu_math), tmp_path / 'gpu', update_device='cuda', gradient_storage='ram')
    assert gpu.update_backend == 'cuda'
    assert sum(t.numel() * t.element_size() for t in gpu._scratch) <= 64 * 1024 * 1024
    # Force multiple chunks, including one smaller final chunk, while retaining
    # the same bounded, reusable scratch buffers.
    gpu._chunk_elements = 3
    for batch in ([.1,.8,.4,.3], [.2,.7], [1.1,.3,.5]):
        for params,opt in ((cpu_math,cpu),(gpu_math,gpu)):
            accumulate(params,batch); opt.step(max_grad_norm=.7); opt.zero_grad(set_to_none=True)
        for a,b in zip(cpu_math,gpu_math): torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-7)
    gpu.save_state(tmp_path / 'checkpoint')
    meta = json.loads((tmp_path/'checkpoint/optimizer.json').read_text())
    assert meta['update_backend'] == 'cuda'
    resumed_params = [torch.nn.Parameter(p.detach().clone()) for p in gpu_math]
    resumed = create_disk_adamw(groups(resumed_params), tmp_path/'resumed', update_device='auto', gradient_storage='disk')
    resumed.load_state(tmp_path / 'checkpoint')
    assert resumed.update_backend == 'cuda'
    resumed._chunk_elements = 3
    for params,opt in ((gpu_math,gpu),(resumed_params,resumed)):
        accumulate(params,[.3,.9]); opt.step(max_grad_norm=.7); opt.zero_grad(set_to_none=True)
    for a,b in zip(gpu_math,resumed_params): assert torch.equal(a,b)
    cpu.save_state(tmp_path / 'legacy')
    legacy_meta = tmp_path/'legacy/optimizer.json'
    value=json.loads(legacy_meta.read_text()); value.pop('update_backend'); legacy_meta.write_text(json.dumps(value))
    legacy_params = [torch.nn.Parameter(p.detach().clone()) for p in cpu_math]
    legacy = create_disk_adamw(groups(legacy_params), tmp_path/'legacy-work', update_device='auto', gradient_storage='ram')
    legacy.load_state(tmp_path/'legacy')
    assert legacy.update_backend == 'cpu'
    for params,opt in ((cpu_math,cpu),(legacy_params,legacy)):
        accumulate(params,[.3,.9]); opt.step(max_grad_norm=.7); opt.zero_grad(set_to_none=True)
    for a,b in zip(cpu_math,legacy_params): assert torch.equal(a,b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available')
def test_noncontiguous_gpu_parameters_never_use_streaming_backend(tmp_path):
    parameter = torch.nn.Parameter(torch.arange(16,device='cuda',dtype=torch.float32).reshape(4,4).T)
    assert not parameter.is_contiguous()
    groups = [{'named_params': [('p',parameter)], 'lr':.01}]
    automatic = create_disk_adamw(groups,tmp_path/'auto-noncontiguous',update_device='auto',gradient_storage='disk')
    assert automatic.update_backend == 'cpu'
    with pytest.raises(ValueError,match='contiguous'):
        create_disk_adamw(groups,tmp_path/'forced-noncontiguous',update_device='cuda',gradient_storage='disk')
