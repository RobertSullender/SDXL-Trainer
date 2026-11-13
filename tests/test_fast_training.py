"""Fast preparation retains long captions and exact resumable sampling."""
from pathlib import Path
import json

from PIL import Image
import pytest

from test_training import tiny_sdxl_model
from sdxl_trainer.training.config import LEGACY_TRAINING_FIELDS, TrainingConfig, fast_training_settings


def test_fast_settings_preserve_complete_captions_and_mode_appropriate_optimizers():
    for mode, optimizer in (("full", "paged_adamw8bit"), ("lora", "adamw")):
        values = fast_training_settings(mode)
        assert values["optimizer"] == optimizer
        assert values["caption_mode"] == "chunked"
        assert values["model_precision"] == values["precision"] == "bf16"
        assert not values["train_text_encoder"] and not values["train_vae"]
        assert not values["cpu_offload_saved_tensors"]
        assert values["precompute_frozen_cache"] and values["adamw_fused"]
        assert not ({"learning_rate", "max_steps", "seed", "batch_size", "model_path"} & values.keys())


def test_old_recipes_keep_original_numerical_options():
    values = TrainingConfig().to_dict()
    for name in LEGACY_TRAINING_FIELDS:
        values.pop(name)
    loaded = TrainingConfig.from_dict(values)
    for name, expected in LEGACY_TRAINING_FIELDS.items():
        assert getattr(loaded, name) == expected
    assert loaded.caption_mode == "chunked"


def _dataset(tiny_sdxl_model, tmp_path, monkeypatch, mixed=False):
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import worker
    sizes = ((64, 64), (64, 128)) if mixed else ((64, 64), (64, 64))
    monkeypatch.setattr(dataset, "BUCKETS", tuple(dict.fromkeys(sizes)))
    old = worker.preflight
    def small(*args, **kwargs):
        result = old(*args, **kwargs)
        result.errors = [e for e in result.errors if "CPU floor" not in e and "free disk space" not in e]
        return result
    monkeypatch.setattr(worker, "preflight", small)
    paths = []
    for i, (size, color) in enumerate(zip(sizes, ("purple", "green"))):
        path = tmp_path / f"source-{i}.png"
        Image.new("RGB", size, color).save(path)
        paths.append(path)
    project = dataset.DatasetProject.create(tmp_path / "draft")
    project.import_images(paths)
    project.save_caption(1, "a")
    project.save_caption(2, " ".join(["a"] * 151))
    counter = TokenCounter.from_model(tiny_sdxl_model)
    if mixed:
        return project.finalize_buckets_in_project(counter)[0].parent
    return project.finalize_in_project(counter)


def _config(model, dataset, output, **extra):
    return TrainingConfig(model_path=str(model), dataset_path=str(dataset), output_dir=str(output),
        mode="full", optimizer="sgd", model_precision="fp32", precision="fp32", device="cpu",
        train_text_encoder=False, train_vae=False, batch_size=1, gradient_accumulation_steps=1,
        max_steps=2, save_every=1, warmup_steps=0, learning_rate=.001,
        caption_mode="chunked", frozen_vae_cache_mode="sample", precompute_frozen_cache=True,
        frozen_text_cache_ram_mib=4, vae_cache_ram_mib=4,
        monitor_gpu=False, profile_phases=False, **extra)


@pytest.mark.parametrize("mixed", [False, True], ids=["single-snapshot", "all-buckets"])
def test_prepared_caches_cover_two_images_keep_all_windows_and_resume_exactly(tiny_sdxl_model, tmp_path, monkeypatch, mixed):
    import torch
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from safetensors.torch import load_file
    from sdxl_trainer.training import worker
    finalized = _dataset(tiny_sdxl_model, tmp_path, monkeypatch, mixed=mixed)
    ordering, sequence_lengths, spatial_shapes = [], [], []
    old_encode = AutoencoderKL.encode
    old_step = torch.optim.SGD.step
    old_unet = UNet2DConditionModel.forward
    def encode(self, *args, **kwargs):
        ordering.append("encode")
        return old_encode(self, *args, **kwargs)
    def step(self, *args, **kwargs):
        ordering.append("update")
        return old_step(self, *args, **kwargs)
    def unet(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
        sequence_lengths.append(encoder_hidden_states.shape[1])
        spatial_shapes.append(tuple(sample.shape[-2:]))
        return old_unet(self, sample, timestep, encoder_hidden_states, *args, **kwargs)
    monkeypatch.setattr(AutoencoderKL, "encode", encode)
    monkeypatch.setattr(torch.optim.SGD, "step", step)
    monkeypatch.setattr(UNet2DConditionModel, "forward", unet)
    old_emit = worker.emit
    def stop_after_first(kind, **values):
        old_emit(kind, **values)
        if kind == "progress" and values["step"] == 1:
            worker.request_stop()
    monkeypatch.setattr(worker, "emit", stop_after_first)
    monkeypatch.setattr(worker, "_STOP", False)
    config = _config(tiny_sdxl_model, finalized, tmp_path / "resumed")
    if mixed:
        config.batch_size = 2  # Different sizes still run in separate microbatches.
    worker.run(config)
    assert ordering == ["encode", "encode", "update"]
    checkpoint = Path(config.output_dir) / "checkpoints/step-00000001"
    assert checkpoint.is_dir()
    monkeypatch.setattr(worker, "emit", old_emit)
    monkeypatch.setattr(worker, "_STOP", False)
    config.resume_from = str(checkpoint)
    worker.run(config)
    # A resumed preparation reads persistent caches, without invoking the VAE.
    assert ordering == ["encode", "encode", "update", "update"]
    assert sorted(sequence_lengths) == [77, 231]
    assert len(set(spatial_shapes)) == (2 if mixed else 1)
    cache = Path(config.output_dir) / "working/text"
    assert sorted(load_file(str(p))["prompt"].shape[1] for p in cache.glob("*.safetensors")) == [77, 231]
    assert len(list((Path(config.output_dir) / "working/latents").glob("*.safetensors"))) == 2
    straight = _config(tiny_sdxl_model, finalized, tmp_path / "straight")
    straight.batch_size = config.batch_size
    worker.run(straight)
    baseline = load_file(str(Path(straight.output_dir) / "checkpoints/step-00000002/unet.safetensors"))
    for name, tensor in load_file(str(Path(config.output_dir) / "checkpoints/step-00000002/unet.safetensors")).items():
        torch.testing.assert_close(tensor, baseline[name], rtol=0, atol=0)
    assert (Path(config.output_dir) / "diffusers/model_index.json").is_file()


def test_cancel_during_preparation_keeps_zero_step_semantics(tiny_sdxl_model, tmp_path, monkeypatch, capsys):
    from sdxl_trainer.training import worker
    finalized = _dataset(tiny_sdxl_model, tmp_path, monkeypatch)
    monkeypatch.setattr(worker, "_STOP", True)
    config = _config(tiny_sdxl_model, finalized, tmp_path / "cancelled")
    worker.run(config)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert any(e["type"] == "complete" and e.get("stopped") and e["step"] == 0 for e in events)
    assert not (Path(config.output_dir) / "checkpoints").exists()


def test_tf32_policy_is_explicit_and_restored(monkeypatch):
    import torch
    from sdxl_trainer.training import worker
    original = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    monkeypatch.setattr(worker, "_run", lambda *args: (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
    assert worker.run(TrainingConfig(allow_tf32=True, precision="bf16", monitor_gpu=False, profile_phases=False)) == (True, True)
    assert worker.run(TrainingConfig(allow_tf32=True, precision="fp32", model_precision="fp32", monitor_gpu=False, profile_phases=False)) == (False, False)
    assert worker.run(TrainingConfig(allow_tf32=False, tf32_policy="legacy", monitor_gpu=False, profile_phases=False)) == original
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == original
