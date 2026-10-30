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


def test_preflight_does_not_require_ml_imports(tmp_path):
    config = TrainingConfig(model_path=str(tmp_path / "missing-model"), dataset_path=str(tmp_path / "missing-dataset"), output_dir=str(tmp_path / "run"))
    result = preflight(config)
    assert not result.ok
    assert any("unet" in error for error in result.errors)
    assert result.estimated_disk_gib > 0
    assert any("not a fit guarantee" in warning for warning in result.warnings)


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
    assert _config_identity(TrainingConfig(rank=32)) != _config_identity(TrainingConfig())


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
    draft.finalize(tmp_path / "dataset", TokenCounter.from_model(tiny_sdxl_model))
    config = TrainingConfig(model_path=str(tiny_sdxl_model), dataset_path=str(tmp_path / "dataset"), output_dir=str(tmp_path / "resumed"), mode=mode, train_text_encoder=True, train_vae=True, device="cpu", precision="fp32", batch_size=1, gradient_accumulation_steps=2, max_steps=2, save_every=1, rank=2, lora_alpha=2, gradient_checkpointing=True, warmup_steps=0)
    monkeypatch.setattr(worker, "_STOP", True)
    worker.run(config)
    assert (tmp_path / "resumed" / "checkpoints" / "step-00000001" / "checkpoint.json").is_file()
    monkeypatch.setattr(worker, "_STOP", False)
    config.resume_from = str(tmp_path / "resumed" / "checkpoints" / "step-00000001")
    # Invalid resume identity must not overwrite saved controls or truncate live moments.
    protected = [tmp_path / "resumed" / "training-config.json", tmp_path / "resumed" / "working" / "optimizer" / "first_moment.bin", tmp_path / "resumed" / "working" / "optimizer" / "second_moment.bin"]
    before = {path: path.read_bytes() for path in protected}
    for change, message in (({"rank": 3}, "Resume controls"), ({"max_steps": 3}, "original total step")):
        invalid = TrainingConfig.from_dict({**config.to_dict(), **change})
        with pytest.raises(ValueError, match=message):
            worker.run(invalid)
        assert all(path.read_bytes() == content for path, content in before.items())
    metadata_file = Path(config.resume_from) / "checkpoint.json"
    original_metadata = metadata_file.read_bytes()
    altered = json.loads(original_metadata)
    altered["model_fingerprint"] = "invalid-model-fingerprint"
    metadata_file.write_text(json.dumps(altered))
    try:
        with pytest.raises(ValueError, match="Resume controls"):
            worker.run(config)
        assert all(path.read_bytes() == content for path, content in before.items())
    finally:
        metadata_file.write_bytes(original_metadata)
    worker.run(config)
    uninterrupted = TrainingConfig.from_dict({**config.to_dict(), "output_dir": str(tmp_path / "uninterrupted"), "resume_from": ""})
    worker.run(uninterrupted)
    for component in ("unet", "text_encoder", "text_encoder_2", "vae"):
        resumed = load_file(str(tmp_path / "resumed" / "checkpoints" / "step-00000002" / f"{component}.safetensors"))
        straight = load_file(str(tmp_path / "uninterrupted" / "checkpoints" / "step-00000002" / f"{component}.safetensors"))
        assert resumed.keys() == straight.keys()
        for name in resumed:
            torch.testing.assert_close(resumed[name], straight[name], atol=0, rtol=0)
    merged = StableDiffusionXLPipeline.from_pretrained(str(tmp_path / "resumed" / "diffusers"), local_files_only=True, use_safetensors=True)
    assert not any("lora" in name for name in merged.unet.state_dict())
    assert not any("lora" in name for name in merged.vae.state_dict())
    assert merged.text_encoder is not None and merged.text_encoder_2 is not None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None or importlib.util.find_spec("safetensors") is None, reason="Optional ML runtime not installed")
def test_single_file_export_preserves_qkv_projection_and_vae_shapes(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file, save_file
    from sdxl_trainer.training.export import export_checkpoint
    model = tmp_path / "model"
    for name in ("unet", "vae", "text_encoder", "text_encoder_2"):
        (model / name).mkdir(parents=True)
    (model / "unet" / "config.json").write_text(json.dumps({"in_channels": 4, "block_out_channels": [32, 64, 128], "cross_attention_dim": 2048}))
    save_file({"conv_in.weight": torch.ones(2, 4, 3, 3)}, str(model / "unet" / "diffusion_pytorch_model.safetensors"))
    save_file({"encoder.mid_block.attentions.0.to_q.weight": torch.ones(2, 2)}, str(model / "vae" / "diffusion_pytorch_model.safetensors"))
    save_file({"text_model.embeddings.token_embedding.weight": torch.ones(3, 2)}, str(model / "text_encoder" / "model.safetensors"))
    encoder = {f"text_model.encoder.layers.0.self_attn.{name}_proj.weight": torch.full((2, 2), float(index)) for index, name in enumerate("qkv")}
    encoder["text_projection.weight"] = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    save_file(encoder, str(model / "text_encoder_2" / "model.safetensors"))
    output = export_checkpoint(model, tmp_path / "merged.safetensors")
    tensors = load_file(str(output))
    assert tensors["first_stage_model.encoder.mid.attn_1.q.weight"].shape == (2, 2, 1, 1)
    qkv = tensors["conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight"]
    assert qkv.shape == (6, 2)
    torch.testing.assert_close(qkv, torch.cat([torch.full((2, 2), float(i)) for i in range(3)]))
    torch.testing.assert_close(tensors["conditioner.embedders.1.model.text_projection"], encoder["text_projection.weight"].T.contiguous())
    with pytest.raises(ValueError, match="already exists"):
        export_checkpoint(model, output)
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


def test_portable_standard_lora_preserves_nondefault_alpha(tiny_sdxl_model, tmp_path):
    """A standard Diffusers reload must retain alpha/rank for UNet and both CLIPs."""
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
    save_portable_lora(tmp_path / "portable", states, alpha=6, rank=2)
    for name, state in states.items():
        for key, value in state.items():
            torch.testing.assert_close(value, unchanged[name][key], atol=0, rtol=0)
    reloaded = StableDiffusionXLPipeline.from_pretrained(str(tiny_sdxl_model), local_files_only=True, use_safetensors=True)
    reloaded.load_lora_weights(str(tmp_path / "portable"), weight_name="pytorch_lora_weights.safetensors", local_files_only=True)
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
