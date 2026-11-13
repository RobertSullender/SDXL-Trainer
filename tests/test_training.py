"""CPU-safe checks for training controls, deterministic batching and checkpoint mapping."""
from __future__ import annotations
from dataclasses import fields
import importlib.util
import json
from pathlib import Path
import pytest

from sdxl_trainer.training import TrainingConfig, preflight
from sdxl_trainer.training.export import unet_key, vae_key, text_encoder_2_key, _component_files
from sdxl_trainer.training.worker import bucket_batches, lr_factor, _config_identity


def legacy_training_config(**overrides):
    """Select the original numerical path explicitly for its regression tests."""
    return TrainingConfig(**{
        "model_precision": "fp32", "optimizer": "adamw_disk",
        "paged_cuda_override": "", "precompute_frozen_cache": False,
        "frozen_vae_cache_mode": "posterior", "frozen_text_cache_ram_mib": 0,
        "vae_cache_ram_mib": 128, "cpu_offload_saved_tensors": True,
        "adamw_fused": False, "allow_tf32": False, "caption_mode": "chunked",
        **overrides,
    })


def test_configuration_is_strict_and_documented(tmp_path):
    config = TrainingConfig()
    assert all(f.metadata.get("help") and f.metadata.get("label") for f in fields(config))
    destination = tmp_path / "config.json"
    config.save(destination)
    assert TrainingConfig.load(destination) == config
    with pytest.raises(ValueError, match="Unknown"):
        TrainingConfig.from_dict({"secret_network_upload": True})
    with pytest.raises(ValueError, match="Invalid type"):
        TrainingConfig.from_dict({"batch_size": True})
    with pytest.raises(ValueError, match="Invalid type"):
        TrainingConfig.from_dict({"learning_rate": float("nan")})
    assert TrainingConfig(batch_size=0).validate()
    assert any("reconstruction" in error for error in TrainingConfig(train_vae=True, vae_reconstruction_weight=0).validate())
    assert any("requires the CUDA" in error for error in TrainingConfig(device="cpu", component_offload="text_encoders_cpu").validate())
    assert any("not CUDA" in error for error in TrainingConfig(train_text_encoder=True, component_offload="text_encoders_cpu", optimizer="adamw_disk", optimizer_update_device="cuda").validate())
    assert any("requires Disk AdamW" in error for error in TrainingConfig(train_vae=True, component_offload="text_cpu_staged_unet", optimizer="sgd").validate())
    assert any("not RAM" in error for error in TrainingConfig(train_vae=True, component_offload="text_cpu_staged_unet", optimizer_gradient_storage="ram").validate())
    assert any("requires Disk AdamW" in error for error in TrainingConfig(train_vae=True, component_offload="stage_unet_gpu_text", optimizer="sgd").validate())
    assert any("not RAM" in error for error in TrainingConfig(train_vae=True, component_offload="stage_unet_gpu_text", optimizer_gradient_storage="ram").validate())
    assert not any("not CUDA" in error for error in TrainingConfig(train_text_encoder=True, train_vae=True, component_offload="stage_unet_gpu_text", optimizer_update_device="cuda").validate())


def test_preflight_does_not_require_ml_imports(tmp_path):
    config = TrainingConfig(model_path=str(tmp_path / "missing-model"), dataset_path=str(tmp_path / "missing-dataset"), output_dir=str(tmp_path / "run"))
    result = preflight(config)
    assert not result.ok
    assert any("unet" in error for error in result.errors)
    assert result.estimated_disk_gib > 0
    assert any("not a fit guarantee" in warning for warning in result.warnings)


@pytest.mark.parametrize("variant", ["fp16", "bf16"])
def test_preflight_warns_about_reduced_source_even_with_fp32_weights(tmp_path, variant):
    config = TrainingConfig(model_path=str(tmp_path / "model"), dataset_path=str(tmp_path / "dataset"),
                            output_dir=str(tmp_path / "run"), model_variant=variant,
                            model_precision="fp32", vae_precision="fp32", precision="fp32")
    warnings = preflight(config).warnings
    assert any(variant in warning and "source weights" in warning and "cannot recover" in warning
               and "VAE" in warning for warning in warnings)
    config.model_variant = "default"
    assert not any("source weights" in warning for warning in preflight(config).warnings)


def test_variant_preflight_helps_with_app_generated_models(tmp_path):
    model = tmp_path / "model"
    (model / "unet").mkdir(parents=True)
    config = TrainingConfig(model_path=str(model), dataset_path=str(tmp_path / "dataset"),
                            output_dir=str(tmp_path / "run"), model_variant="fp16")
    assert any("default file variant" in error and "unet/" in error for error in preflight(config).errors)


def test_output_cannot_overlap_input(tmp_path):
    config = TrainingConfig(model_path=str(tmp_path / "model"), dataset_path=str(tmp_path / "dataset"), output_dir=str(tmp_path / "dataset" / "run"))
    assert any("separate from the dataset" in error for error in preflight(config).errors)


def test_bucket_batches_are_repeatable_and_never_mix_shapes():
    entries = [{"width": 1024, "height": 1024}] * 7 + [{"width": 832, "height": 1216}] * 3
    batches = bucket_batches(entries, 3, 81, 0)
    assert batches == bucket_batches(entries, 3, 81, 0)
    assert sorted(index for batch in batches for index in batch) == list(range(10))
    for batch in batches:
        assert 1 <= len(batch) <= 3
        assert len({(entries[i]["width"], entries[i]["height"]) for i in batch}) == 1
    assert batches != bucket_batches(entries, 3, 81, 1)


def test_learning_rate_schedule_and_resume_identity():
    assert lr_factor(0, 100, 10, "constant") == pytest.approx(0.1)
    assert lr_factor(9, 100, 10, "constant") == 1
    assert lr_factor(50, 100, 0, "linear") == 0.5
    assert lr_factor(100, 100, 0, "cosine") == 0
    assert _config_identity(TrainingConfig(resume_from="somewhere")) == _config_identity(TrainingConfig())
    assert _config_identity(TrainingConfig(dataset_path="/old/project/finalized/Dataset-1")) == _config_identity(TrainingConfig(dataset_path="/new/project/finalized/Dataset-1"))
    assert _config_identity(TrainingConfig(rank=32)) != _config_identity(TrainingConfig())
    assert _config_identity(TrainingConfig(component_offload="text_encoders_cpu")) != _config_identity(TrainingConfig())


def test_resume_rejects_changed_optimizer_backend_before_writing(tmp_path):
    from sdxl_trainer.training.worker import validate_resume_identity
    checkpoint = tmp_path / "checkpoints/step-00000001"
    (checkpoint / "optimizer").mkdir(parents=True)
    config = legacy_training_config(model_path=str(tmp_path / "model"), dataset_path=str(tmp_path / "dataset"),
                            output_dir=str(tmp_path / "new-run"), resume_from=str(checkpoint),
                            optimizer_update_device="cuda")
    info = {"version": 1, "step": 1, "epoch": 0, "batch_cursor": 0,
            "config": _config_identity(config), "dataset_fingerprint": "dataset",
            "model_fingerprint": "model", "original_max_steps": config.max_steps}
    (checkpoint / "checkpoint.json").write_text(json.dumps(info))
    for name in ("unet.safetensors", "rng.pt", "optimizer/first_moment.bin", "optimizer/second_moment.bin"):
        (checkpoint / name).write_bytes(b"preserve")
    # Missing update_backend is the legacy CPU optimizer checkpoint format.
    (checkpoint / "optimizer/optimizer.json").write_text(json.dumps({"records": [{"offset": 0, "numel": 2}], "groups": []}))
    with pytest.raises(ValueError, match="cpu optimizer update backend"):
        validate_resume_identity(config, "dataset", "model")
    assert not Path(config.output_dir).exists()
    assert (checkpoint / "optimizer/first_moment.bin").read_bytes() == b"preserve"
    config.optimizer_update_device = "auto"
    assert validate_resume_identity(config, "dataset", "model")["step"] == 1
    # Older checkpoints did not save placement; their default is the original
    # no-component-offload behavior, and a changed placement must be rejected.
    info["config"].pop("component_offload")
    (checkpoint / "checkpoint.json").write_text(json.dumps(info))
    assert validate_resume_identity(config, "dataset", "model")["step"] == 1
    config.component_offload = "text_encoders_cpu"
    with pytest.raises(ValueError, match="differ"):
        validate_resume_identity(config, "dataset", "model")


@pytest.mark.parametrize("source,expected", [
    ("time_embedding.linear_1.weight", "time_embed.0.weight"),
    ("down_blocks.1.resnets.0.norm1.weight", "input_blocks.4.0.in_layers.0.weight"),
    ("down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.weight", "input_blocks.8.1.transformer_blocks.0.attn1.to_q.weight"),
    ("down_blocks.1.downsamplers.0.conv.weight", "input_blocks.6.0.op.weight"),
    ("up_blocks.0.upsamplers.0.conv.weight", "output_blocks.2.2.conv.weight"),
    ("mid_block.resnets.1.time_emb_proj.bias", "middle_block.2.emb_layers.1.bias"),
])
def test_unet_checkpoint_mapping(source, expected):
    assert unet_key(source) == "model.diffusion_model." + expected


def test_unknown_unet_weights_fail_instead_of_silent_mislabel():
    with pytest.raises(ValueError, match="Unsupported"):
        unet_key("secret_unknown.weight")


def test_vae_and_dual_clip_mapping():
    assert vae_key("decoder.up_blocks.0.resnets.2.conv_shortcut.weight") == "first_stage_model.decoder.up.3.block.2.nin_shortcut.weight"
    assert vae_key("encoder.mid_block.attentions.0.to_out.0.weight") == "first_stage_model.encoder.mid.attn_1.proj_out.weight"
    assert text_encoder_2_key("text_model.encoder.layers.7.mlp.fc1.weight") == "conditioner.embedders.1.model.transformer.resblocks.7.mlp.c_fc.weight"
    assert text_encoder_2_key("text_projection.weight") == "conditioner.embedders.1.model.text_projection"


def test_export_shard_index_cannot_escape_folder(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="Unsafe"):
        _component_files(tmp_path)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Optional ML runtime not installed")
def test_disk_adamw_matches_torch_with_accumulation_and_resume(tmp_path):
    import torch
    from sdxl_trainer.training.disk_optimizer import create_disk_adamw
    torch.manual_seed(9)
    reference = torch.nn.Linear(3, 2)
    actual = torch.nn.Linear(3, 2)
    actual.load_state_dict(reference.state_dict())
    expected_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.03, foreach=False)
    actual_optimizer = create_disk_adamw([{"named_params": list(actual.named_parameters()), "lr": 0.01}], tmp_path / "optimizer", weight_decay=0.03)
    for _ in range(3):
        for _ in range(2):
            x = torch.randn(5, 3)
            reference(x).square().mean().div(2).backward()
            actual(x).square().mean().div(2).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        expected_optimizer.step()
        actual_optimizer.step(max_grad_norm=1.0)
        expected_optimizer.zero_grad(set_to_none=True)
        actual_optimizer.zero_grad(set_to_none=True)
    for left, right in zip(reference.parameters(), actual.parameters()):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)
        assert right.grad is None
    actual_optimizer.save_state(tmp_path / "checkpoint")
    actual_optimizer.load_state(tmp_path / "checkpoint")


@pytest.fixture(scope="module")
def tiny_sdxl_model(tmp_path_factory):
    """No downloaded weights: construct a tiny SDXL-shaped pipeline locally."""
    if any(importlib.util.find_spec(name) is None for name in ("torch", "diffusers", "transformers", "peft", "safetensors")):
        pytest.skip("Optional ML runtime not installed")
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    torch.set_num_threads(2)
    torch.manual_seed(3)
    root = tmp_path_factory.mktemp("tiny-sdxl")
    tokenizer_dir = root / "tokenizer-assets"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "vocab.json").write_text(json.dumps({"<|startoftext|>": 0, "<|endoftext|>": 1, "a</w>": 2}))
    (tokenizer_dir / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(str(tokenizer_dir / "vocab.json"), str(tokenizer_dir / "merges.txt"), model_max_length=77)
    text_one = CLIPTextModel(CLIPTextConfig(vocab_size=3, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=77, bos_token_id=0, eos_token_id=1, pad_token_id=1))
    text_two = CLIPTextModelWithProjection(CLIPTextConfig(vocab_size=3, hidden_size=48, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=77, projection_dim=48, bos_token_id=0, eos_token_id=1, pad_token_id=1))
    unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1, block_out_channels=(16, 32, 32), down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"), cross_attention_dim=80, attention_head_dim=4, norm_num_groups=4, addition_embed_type="text_time", addition_time_embed_dim=8, projection_class_embeddings_input_dim=96)
    vae = AutoencoderKL(in_channels=3, out_channels=3, block_out_channels=(8, 8, 16, 16), down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4, latent_channels=4, layers_per_block=1, norm_num_groups=4, sample_size=64)
    pipeline = StableDiffusionXLPipeline(unet=unet, vae=vae, text_encoder=text_one, text_encoder_2=text_two, tokenizer=tokenizer, tokenizer_2=tokenizer, scheduler=DDPMScheduler(num_train_timesteps=10), add_watermarker=False)
    pipeline.save_pretrained(str(root / "model"), safe_serialization=True)
    return root / "model"


def test_tiny_sharded_diffusers_training_output_roundtrip(tiny_sdxl_model, tmp_path, monkeypatch, capsys):
    """The worker's configured shard size preserves all four trained components."""
    import torch
    from PIL import Image
    from diffusers import StableDiffusionXLPipeline
    from safetensors.torch import load_file
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import worker
    monkeypatch.setattr(dataset, "BUCKETS", ((64, 64),))
    preflight = worker.preflight
    def synthetic_preflight(*args, **kwargs):
        report = preflight(*args, **kwargs)
        report.errors = [e for e in report.errors if "free disk space" not in e and "system RAM is available" not in e]
        return report
    monkeypatch.setattr(worker, "preflight", synthetic_preflight)
    monkeypatch.setattr(worker, "_STOP", False)
    source = tmp_path / "original.png"
    Image.new("RGB", (64, 64), "purple").save(source)
    project = dataset.DatasetProject.create(tmp_path / "draft")
    project.import_images([source]); project.save_caption(1, "a")
    finalized = project.finalize_in_project(TokenCounter.from_model(tiny_sdxl_model))
    run = tmp_path / "run"
    config = legacy_training_config(model_path=str(tiny_sdxl_model), dataset_path=str(finalized), output_dir=str(run),
        mode="full", train_text_encoder=True, train_vae=True, device="cpu", precision="fp32", batch_size=1,
        gradient_accumulation_steps=1, max_steps=1, save_every=1, warmup_steps=0, monitor_gpu=False,
        final_export_shard_mib=1)
    worker.run(config)
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    assert any('component weights' in event.get('message', '') for event in messages)
    assert any('optimizer state' in event.get('message', '') for event in messages)
    output = run / "diffusers"
    assert len(_component_files(output / "unet")) > 1
    reloaded = StableDiffusionXLPipeline.from_pretrained(str(output), local_files_only=True, use_safetensors=True)
    for name in ("unet", "text_encoder", "text_encoder_2", "vae"):
        checkpoint = load_file(str(run / "checkpoints/step-00000001" / f"{name}.safetensors"))
        stored = {}
        for shard in _component_files(output / name):
            stored.update(load_file(str(shard)))
        restored = getattr(reloaded, name).state_dict()
        assert checkpoint.keys() == stored.keys() == restored.keys()
        for key, value in checkpoint.items():
            assert value.dtype == stored[key].dtype == restored[key].dtype
            assert torch.equal(value, stored[key])
            assert torch.equal(value, restored[key])
    # The synthetic UNet has width 80 rather than real SDXL's 2048. Override
    # only that exporter's architecture guard to exercise real shard conversion.
    from sdxl_trainer.training.export import export_checkpoint
    config_path = output / "unet" / "config.json"
    original_config = config_path.read_text()
    synthetic_config = json.loads(original_config)
    synthetic_config["cross_attention_dim"] = 2048
    config_path.write_text(json.dumps(synthetic_config))
    try:
        converted = load_file(str(export_checkpoint(output, tmp_path / "converted.safetensors")))
    finally:
        config_path.write_text(original_config)
    assert torch.equal(converted["model.diffusion_model.input_blocks.0.0.weight"],
                       reloaded.unet.state_dict()["conv_in.weight"])
    assert converted["model.diffusion_model.input_blocks.0.0.weight"].dtype == reloaded.unet.conv_in.weight.dtype
    assert any(key.startswith("first_stage_model.") for key in converted)
    assert any(key.startswith("conditioner.embedders.0.") for key in converted)
    assert any(key.startswith("conditioner.embedders.1.") for key in converted)
    # A sharded pipeline may intentionally mix dtypes across components.
    reloaded.unet.to(dtype=torch.bfloat16)
    reloaded.text_encoder.to(dtype=torch.float16)
    reloaded.text_encoder_2.to(dtype=torch.bfloat16)
    mixed = tmp_path / "mixed-dtypes"
    reloaded.save_pretrained(str(mixed), safe_serialization=True, max_shard_size=1024 * 1024)
    for name in ("unet", "text_encoder", "text_encoder_2", "vae"):
        stored = {}
        for shard in _component_files(mixed / name):
            stored.update(load_file(str(shard)))
        expected = getattr(reloaded, name).state_dict()
        assert stored.keys() == expected.keys()
        for key, value in expected.items():
            assert stored[key].dtype == value.dtype
            assert torch.equal(stored[key], value)
    assert worker._config_identity(config) == worker._config_identity(
        TrainingConfig.from_dict({**config.to_dict(), "final_export_shard_mib": 8192}))


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_tiny_joint_training_exact_resume_and_diffusers_output(tiny_sdxl_model, tmp_path, monkeypatch, mode):
    """Exercises all components, backprop, disk optimizer, safe stop and exact resume.

    Tiny 64px test bucket and waived SDXL disk-space estimate apply only to this
    synthetic model. Production still verifies the canonical SDXL bucket table.
    """
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from diffusers import StableDiffusionXLPipeline
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import worker
    monkeypatch.setattr(dataset, "BUCKETS", ((64, 64),))
    original_preflight = worker.preflight

    def synthetic_preflight(*args, **kwargs):
        report = original_preflight(*args, **kwargs)
        report.errors = [e for e in report.errors if "free disk space" not in e]
        return report

    monkeypatch.setattr(worker, "preflight", synthetic_preflight)
    original = tmp_path / "original.png"
    Image.new("RGB", (64, 64), "purple").save(original)
    draft = dataset.DatasetProject.create(tmp_path / "draft")
    draft.import_images([original])
    draft.save_caption(1, "a")
    finalized = draft.finalize_in_project(TokenCounter.from_model(tiny_sdxl_model))
    run_root = draft.root / "runs" / "resumed"
    config = legacy_training_config(model_path=str(tiny_sdxl_model), dataset_path=str(finalized), output_dir=str(run_root), mode=mode, train_text_encoder=True, train_vae=True, device="cpu", precision="fp32", batch_size=1, gradient_accumulation_steps=2, max_steps=2, save_every=1, rank=2, lora_alpha=2, gradient_checkpointing=True, warmup_steps=0)
    monkeypatch.setattr(worker, "_STOP", True)
    worker.run(config)
    assert (run_root / "checkpoints" / "step-00000001" / "checkpoint.json").is_file()
    # Simulate legacy metadata, then move the enclosing project externally,
    # including its snapshot and checkpoints, and resume at its new location.
    metadata = run_root / "checkpoints" / "step-00000001" / "checkpoint.json"
    legacy = json.loads(metadata.read_text())
    legacy["config"]["dataset_path"] = str(finalized)
    metadata.write_text(json.dumps(legacy))
    moved = draft.root.rename(tmp_path / "Moved project")
    run_root = moved / "runs" / "resumed"
    config.dataset_path = str(moved / "finalized" / "Dataset-1")
    config.output_dir = str(run_root)
    monkeypatch.setattr(worker, "_STOP", False)
    config.resume_from = str(run_root / "checkpoints" / "step-00000001")
    # Invalid resume identity must not overwrite saved controls or truncate live moments.
    protected = [run_root / "training-config.json", run_root / "working" / "optimizer" / "first_moment.bin", run_root / "working" / "optimizer" / "second_moment.bin"]
    before = {path: path.read_bytes() for path in protected}
    for change, message in (({"rank": 3}, "Resume controls"), ({"max_steps": 3}, "original total step")):
        invalid = TrainingConfig.from_dict({**config.to_dict(), **change})
        with pytest.raises(ValueError, match=message):
            worker.run(invalid)
        assert all(path.read_bytes() == content for path, content in before.items())
    metadata_file = Path(config.resume_from) / "checkpoint.json"
    original_metadata = metadata_file.read_bytes()
    for fingerprint in ("model_fingerprint", "dataset_fingerprint"):
        altered = json.loads(original_metadata)
        altered[fingerprint] = "invalid-fingerprint"
        metadata_file.write_text(json.dumps(altered))
        try:
            with pytest.raises(ValueError, match="Resume controls"):
                worker.run(config)
            assert all(path.read_bytes() == content for path, content in before.items())
        finally:
            metadata_file.write_bytes(original_metadata)
    # Even a self-consistent edited snapshot has a different sealed manifest;
    # merely changing the dataset path must not bypass resume integrity.
    snapshot = Path(config.dataset_path)
    manifest_file = snapshot / "manifest.json"
    caption_file = snapshot / "Img-1.txt"
    original_manifest, original_caption = manifest_file.read_bytes(), caption_file.read_bytes()
    changed_manifest = json.loads(original_manifest)
    caption_file.write_text("a a")
    changed_manifest["entries"][0]["caption_sha256"] = dataset.sha256(caption_file)
    changed_manifest["entries"][0]["tokens"] = TokenCounter.from_model(tiny_sdxl_model).count("a a").to_dict()
    manifest_file.write_text(json.dumps(changed_manifest))
    try:
        dataset.verify_finalized(snapshot)
        with pytest.raises(ValueError, match="Resume controls"):
            worker.run(config)
        assert all(path.read_bytes() == content for path, content in before.items())
    finally:
        manifest_file.write_bytes(original_manifest)
        caption_file.write_bytes(original_caption)
    worker.run(config)
    uninterrupted = TrainingConfig.from_dict({**config.to_dict(), "output_dir": str(tmp_path / "uninterrupted"), "resume_from": ""})
    worker.run(uninterrupted)
    if mode == "lora":
        assert (run_root / "adapters/lora_comfyui.safetensors").is_file()
        assert (run_root / "adapters/lora_diffusers.safetensors").is_file()
    for component in ("unet", "text_encoder", "text_encoder_2", "vae"):
        resumed = load_file(str(run_root / "checkpoints" / "step-00000002" / f"{component}.safetensors"))
        straight = load_file(str(tmp_path / "uninterrupted" / "checkpoints" / "step-00000002" / f"{component}.safetensors"))
        assert resumed.keys() == straight.keys()
        for name in resumed:
            torch.testing.assert_close(resumed[name], straight[name], atol=0, rtol=0)
    merged = StableDiffusionXLPipeline.from_pretrained(str(run_root / "diffusers"), local_files_only=True, use_safetensors=True)
    assert not any("lora" in name for name in merged.unet.state_dict())
    assert not any("lora" in name for name in merged.vae.state_dict())
    assert merged.text_encoder is not None and merged.text_encoder_2 is not None


@pytest.mark.parametrize("train_text_encoder", [False, True])
def test_frozen_vae_cache_matches_uncached_training_bitwise(tiny_sdxl_model, tmp_path, monkeypatch, train_text_encoder):
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import worker
    monkeypatch.setattr(dataset, "BUCKETS", ((64, 64),))
    original_preflight = worker.preflight
    def synthetic_preflight(*args, **kwargs):
        report = original_preflight(*args, **kwargs)
        report.errors = [e for e in report.errors if "free disk space" not in e]
        return report
    monkeypatch.setattr(worker, "preflight", synthetic_preflight)
    monkeypatch.setattr(worker, "_STOP", False)
    original = tmp_path / "image.png"
    Image.new("RGB", (64, 64), "purple").save(original)
    project = dataset.DatasetProject.create(tmp_path / "project")
    project.import_images([original]); project.save_caption(1, "a")
    finalized = project.finalize_in_project(TokenCounter.from_model(tiny_sdxl_model))
    finalized_image = finalized / dataset.verify_finalized(finalized)["entries"][0]["image"]
    open_image = Image.open
    reads = []
    def count_open(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path) == finalized_image:
            reads.append(path)
        return open_image(path, *args, **kwargs)
    monkeypatch.setattr(Image, "open", count_open)
    read_counts = {}
    for use_cache in (False, True):
        reads.clear()
        config = legacy_training_config(model_path=str(tiny_sdxl_model), dataset_path=str(finalized),
            output_dir=str(tmp_path / ("cached" if use_cache else "uncached")), mode="full",
            train_text_encoder=train_text_encoder, train_vae=False, device="cpu", precision="fp32",
            batch_size=1, gradient_accumulation_steps=2, max_steps=2, save_every=1,
            gradient_checkpointing=True, warmup_steps=0, cache_frozen_vae=use_cache,
            profile_phases=True, monitor_gpu=False)
        worker.run(config)
        read_counts[use_cache] = len(reads)
    # Four image presentations; cached training decodes only the first one.
    # Finalized-dataset validation reads the image in both runs equally.
    assert read_counts[False] - read_counts[True] == 3
    for component in (["unet", "text_encoder", "text_encoder_2"] if train_text_encoder else ["unet"]):
        path = f"checkpoints/step-00000002/{component}.safetensors"
        cached = load_file(str(tmp_path / "cached" / path))
        original = load_file(str(tmp_path / "uncached" / path))
        for name in cached:
            torch.testing.assert_close(cached[name], original[name], atol=0, rtol=0)
    assert list((tmp_path / "cached/working/vae").glob("*.safetensors"))
    assert not (tmp_path / "uncached/working/vae").exists()


def test_resume_enables_cache_without_changing_weights_or_existing_checkpoint(tiny_sdxl_model, tmp_path, monkeypatch):
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import worker
    monkeypatch.setattr(dataset, "BUCKETS", ((64, 64),))
    original_preflight = worker.preflight
    def synthetic_preflight(*args, **kwargs):
        report = original_preflight(*args, **kwargs)
        report.errors = [e for e in report.errors if "free disk space" not in e]
        return report
    monkeypatch.setattr(worker, "preflight", synthetic_preflight)
    image = tmp_path / "image.png"
    Image.new("RGB", (64, 64), "orange").save(image)
    project = dataset.DatasetProject.create(tmp_path / "project")
    project.import_images([image]); project.save_caption(1, "a")
    finalized = project.finalize_in_project(TokenCounter.from_model(tiny_sdxl_model))
    values = dict(model_path=str(tiny_sdxl_model), dataset_path=str(finalized), mode="full",
        train_text_encoder=False, train_vae=False, device="cpu", precision="fp32", batch_size=1,
        gradient_accumulation_steps=2, max_steps=2, save_every=1, warmup_steps=0, monitor_gpu=False)
    config = legacy_training_config(**values, output_dir=str(tmp_path / "resumed"), cache_frozen_vae=False)
    monkeypatch.setattr(worker, "_STOP", True)
    worker.run(config)
    checkpoint = tmp_path / "resumed/checkpoints/step-00000001"
    # Pre-override checkpoints had no CPU text precision field.
    legacy_info = json.loads((checkpoint / "checkpoint.json").read_text())
    legacy_info["config"].pop("cpu_text_precision")
    legacy_info["config"].pop("paged_cuda_override")
    (checkpoint / "checkpoint.json").write_text(json.dumps(legacy_info, indent=2))
    old = (checkpoint / "checkpoint.json").read_bytes()
    # A checkpoint written by the earlier whole-component Safetensors writer
    # remains a valid resume source after introducing the streaming writer.
    from safetensors.torch import save_file
    legacy_unet = checkpoint / "unet.safetensors"
    legacy_state = {name: value.clone() for name, value in load_file(str(legacy_unet)).items()}
    save_file(legacy_state, str(legacy_unet))
    config.cache_frozen_vae = True
    config.profile_phases = False
    config.resume_from = str(checkpoint)
    monkeypatch.setattr(worker, "_STOP", False)
    worker.run(config)
    straight = legacy_training_config(**values, output_dir=str(tmp_path / "straight"), cache_frozen_vae=False)
    worker.run(straight)
    path = "checkpoints/step-00000002/unet.safetensors"
    resumed = load_file(str(tmp_path / "resumed" / path))
    original = load_file(str(tmp_path / "straight" / path))
    for name in resumed:
        torch.testing.assert_close(resumed[name], original[name], atol=0, rtol=0)
    assert (checkpoint / "checkpoint.json").read_bytes() == old


def test_interrupted_stream_does_not_publish_checkpoint(tiny_sdxl_model, tmp_path, monkeypatch):
    from PIL import Image
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import checkpoint_stream, worker
    monkeypatch.setattr(dataset, "BUCKETS", ((64, 64),))
    original_preflight = worker.preflight
    def synthetic_preflight(*args, **kwargs):
        report = original_preflight(*args, **kwargs)
        report.errors = [e for e in report.errors if "free disk space" not in e]
        return report
    monkeypatch.setattr(worker, "preflight", synthetic_preflight)
    monkeypatch.setattr(worker, "_STOP", False)
    source = tmp_path / "original.png"
    Image.new("RGB", (64, 64), "purple").save(source)
    project = dataset.DatasetProject.create(tmp_path / "project")
    project.import_images([source]); project.save_caption(1, "a")
    finalized = project.finalize_in_project(TokenCounter.from_model(tiny_sdxl_model))
    real_save = checkpoint_stream.save_safetensors_stream
    def fail_second(tensors, path, *args, **kwargs):
        if Path(path).name == "text_encoder.safetensors":
            raise OSError("injected checkpoint interruption")
        return real_save(tensors, path, *args, **kwargs)
    monkeypatch.setattr(checkpoint_stream, "save_safetensors_stream", fail_second)
    run = tmp_path / "run"
    config = legacy_training_config(model_path=str(tiny_sdxl_model), dataset_path=str(finalized), output_dir=str(run),
        mode="full", train_text_encoder=True, device="cpu", precision="fp32", batch_size=1,
        gradient_accumulation_steps=1, max_steps=1, save_every=1, warmup_steps=0, monitor_gpu=False)
    with pytest.raises(OSError, match="injected checkpoint interruption"):
        worker.run(config)
    assert not (run / "checkpoints/step-00000001").exists()
    assert not (run / "checkpoints/step-00000001.tmp/checkpoint.json").exists()


def test_nonfinite_vae_latents_stop_before_unet_or_checkpoint(tiny_sdxl_model, tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    from PIL import Image
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from sdxl_trainer import dataset
    from sdxl_trainer.tokens import TokenCounter
    from sdxl_trainer.training import worker
    monkeypatch.setattr(dataset, "BUCKETS", ((64,64),))
    preflight=worker.preflight
    def synthetic(*args,**kwargs):
        result=preflight(*args,**kwargs)
        result.errors=[e for e in result.errors if 'free disk space' not in e]
        return result
    monkeypatch.setattr(worker,'preflight',synthetic);monkeypatch.setattr(worker,'_STOP',False)
    source=tmp_path/'source.png';Image.new('RGB',(64,64),'purple').save(source)
    project=dataset.DatasetProject.create(tmp_path/'project');project.import_images([source]);project.save_caption(1,'a')
    finalized=project.finalize_in_project(TokenCounter.from_model(tiny_sdxl_model))
    monkeypatch.setattr(AutoencoderKL,'encode',lambda self,pixels: SimpleNamespace(
        latent_dist=SimpleNamespace(sample=lambda: torch.full((pixels.shape[0],4,8,8),float('nan')))))
    def no_unet(*args,**kwargs): pytest.fail('UNet must not run on non-finite VAE latents')
    monkeypatch.setattr(UNet2DConditionModel,'forward',no_unet)
    run=tmp_path/'run'
    config=legacy_training_config(model_path=str(tiny_sdxl_model),dataset_path=str(finalized),output_dir=str(run),
        mode='full',device='cpu',precision='fp32',batch_size=1,gradient_accumulation_steps=1,max_steps=1,
        save_every=1,warmup_steps=0,monitor_gpu=False,cache_frozen_vae=False)
    with pytest.raises(FloatingPointError,match='Non-finite VAE latents'):
        worker.run(config)
    assert not (run/'checkpoints/step-00000001').exists()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None or importlib.util.find_spec("safetensors") is None, reason="Optional ML runtime not installed")
@pytest.mark.parametrize("clip_l_prefix", ["", "text_model."])
def test_single_file_export_preserves_qkv_projection_and_vae_shapes(tmp_path, monkeypatch, clip_l_prefix):
    import torch
    from safetensors.torch import load_file, save_file
    from sdxl_trainer.training.export import export_checkpoint
    model = tmp_path / "model"
    for name in ("unet", "vae", "text_encoder", "text_encoder_2"):
        (model / name).mkdir(parents=True)
    (model / "unet" / "config.json").write_text(json.dumps({"in_channels": 4, "block_out_channels": [32, 64, 128], "cross_attention_dim": 2048}))
    save_file({"conv_in.weight": torch.ones(2, 4, 3, 3)}, str(model / "unet" / "diffusion_pytorch_model.safetensors"))
    save_file({"encoder.mid_block.attentions.0.to_q.weight": torch.ones(2, 2)}, str(model / "vae" / "diffusion_pytorch_model.safetensors"))
    save_file({clip_l_prefix + "embeddings.token_embedding.weight": torch.ones(3, 2)}, str(model / "text_encoder" / "model.safetensors"))
    encoder = {f"text_model.encoder.layers.0.self_attn.{name}_proj.weight": torch.full((2, 2), float(index)) for index, name in enumerate("qkv")}
    encoder["text_projection.weight"] = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    save_file(encoder, str(model / "text_encoder_2" / "model.safetensors"))
    output = export_checkpoint(model, tmp_path / "merged.safetensors")
    tensors = load_file(str(output))
    torch.testing.assert_close(tensors["conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight"], torch.ones(3, 2))
    assert "conditioner.embedders.0.transformer.embeddings.token_embedding.weight" not in tensors
    assert tensors["first_stage_model.encoder.mid.attn_1.q.weight"].shape == (2, 2, 1, 1)
    qkv = tensors["conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight"]
    assert qkv.shape == (6, 2)
    torch.testing.assert_close(qkv, torch.cat([torch.full((2, 2), float(i)) for i in range(3)]))
    torch.testing.assert_close(tensors["conditioner.embedders.1.model.text_projection"], encoder["text_projection.weight"].T.contiguous())
    with pytest.raises(ValueError, match="already exists"):
        export_checkpoint(model, output)
    clip_file = model / "text_encoder" / "model.safetensors"
    save_file({"text_model.embeddings.token_embedding.weight": torch.ones(3, 2),
               "embeddings.token_embedding.weight": torch.zeros(3, 2)}, str(clip_file))
    with pytest.raises(ValueError, match="Duplicate mapped tensor"):
        export_checkpoint(model, tmp_path / "duplicate.safetensors")
    assert not (tmp_path / "duplicate.safetensors").exists()
    save_file({clip_l_prefix + "embeddings.token_embedding.weight": torch.ones(3, 2)}, str(clip_file))
    import safetensors.torch
    original_save = safetensors.torch.save_file
    competing = tmp_path / "racing.safetensors"

    def racing_save(*args, **kwargs):
        original_save(*args, **kwargs)
        competing.write_bytes(b"another process published this file")

    monkeypatch.setattr(safetensors.torch, "save_file", racing_save)
    with pytest.raises(FileExistsError):
        export_checkpoint(model, competing)
    assert competing.read_bytes() == b"another process published this file"
    assert not list(tmp_path.glob("racing.safetensors.*.tmp"))


@pytest.mark.parametrize("filename", ["lora_diffusers.safetensors", "lora_comfyui.safetensors"])
def test_portable_standard_lora_preserves_nondefault_alpha(tiny_sdxl_model, tmp_path, filename):
    """Both portable formats preserve alpha/rank for UNet and both CLIPs."""
    import torch
    from diffusers import StableDiffusionXLPipeline
    from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model
    from peft.tuners.tuners_utils import BaseTunerLayer
    from sdxl_trainer.training.worker import save_portable_lora
    torch.manual_seed(41)
    original = StableDiffusionXLPipeline.from_pretrained(str(tiny_sdxl_model), local_files_only=True, use_safetensors=True)
    components = {"unet": original.unet, "text_encoder": original.text_encoder, "text_encoder_2": original.text_encoder_2}
    states = {}
    for name, component in components.items():
        targets = ["q_proj", "k_proj", "v_proj", "out_proj"] if name.startswith("text_encoder") else ["to_q", "to_k", "to_v", "to_out.0"]
        inject_adapter_in_model(LoraConfig(r=2, lora_alpha=6, target_modules=targets, lora_dropout=0.2), component)
        # Both factors nonzero, making an alpha-loss regression readily observable.
        with torch.no_grad():
            for parameter_name, parameter in component.named_parameters():
                if "lora_" in parameter_name:
                    parameter.uniform_(-0.2, 0.2)
        component.eval()
        states[name] = get_peft_model_state_dict(component)
    unchanged = {name: {key: value.clone() for key, value in state.items()} for name, state in states.items()}
    folder = tmp_path / "portable"
    save_portable_lora(folder, states, alpha=6, rank=2)
    assert (folder / "pytorch_lora_weights.safetensors").read_bytes() == (folder / "lora_diffusers.safetensors").read_bytes()
    for name, state in states.items():
        for key, value in state.items():
            torch.testing.assert_close(value, unchanged[name][key], atol=0, rtol=0)
    reloaded = StableDiffusionXLPipeline.from_pretrained(str(tiny_sdxl_model), local_files_only=True, use_safetensors=True)
    reloaded.load_lora_weights(str(folder), weight_name=filename, local_files_only=True)
    reloaded.fuse_lora()
    for name, component in components.items():
        for module_name, module in list(component.named_modules()):
            if isinstance(module, BaseTunerLayer):
                module.merge(safe_merge=True)
                parent_name, _, child = module_name.rpartition(".")
                parent = component.get_submodule(parent_name) if parent_name else component
                setattr(parent, child, module.get_base_layer())
        # Unload fused tuner wrappers on the consumer side before comparing keys.
    reloaded.unload_lora_weights()
    for name, component in components.items():
        actual = getattr(reloaded, name).state_dict()
        expected = component.state_dict()
        assert actual.keys() == expected.keys()
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-6)


def test_resume_rejects_foreign_nonempty_output_without_writing(tmp_path):
    from sdxl_trainer.training.worker import run
    output = tmp_path / "precious"
    (output / "working" / "optimizer").mkdir(parents=True)
    config_file = output / "training-config.json"
    moment = output / "working" / "optimizer" / "first_moment.bin"
    config_file.write_bytes(b"existing configuration must survive")
    moment.write_bytes(b"existing moment state must survive")
    foreign = tmp_path / "other-run" / "checkpoints" / "step-00000001"
    foreign.mkdir(parents=True)
    (foreign / "checkpoint.json").write_text("{}")
    config = TrainingConfig(output_dir=str(output), resume_from=str(foreign))
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="same run"):
        run(config)
    after = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert after == before


@pytest.mark.parametrize("relative", ["working", "working/optimizer/first_moment.bin", "checkpoints", "diffusers.tmp", "training-config.json"])
def test_run_rejects_symlinked_managed_paths(tmp_path, relative):
    from sdxl_trainer.training.config import output_safety_errors
    output = tmp_path / "run"
    output.mkdir()
    target = tmp_path / "untouched"
    target.mkdir()
    (target / "sentinel").write_bytes(b"original")
    linked = output / relative
    linked.parent.mkdir(parents=True, exist_ok=True)
    linked.symlink_to(target, target_is_directory=True)
    errors = output_safety_errors(TrainingConfig(output_dir=str(output)))
    assert any("symbolic link" in error for error in errors)
    assert (target / "sentinel").read_bytes() == b"original"


def test_resume_rejects_newer_checkpoint_before_mutating_run(tmp_path):
    from sdxl_trainer.training.config import output_safety_errors
    output = tmp_path / "run"
    for step in (1, 2):
        checkpoint = output / "checkpoints" / f"step-{step:08d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "checkpoint.json").write_text("{}")
    TrainingConfig().save(output / "training-config.json")
    config = TrainingConfig(output_dir=str(output), resume_from=str(output / "checkpoints" / "step-00000001"))
    assert any("newer checkpoint" in error for error in output_safety_errors(config))
