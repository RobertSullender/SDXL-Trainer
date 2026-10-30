"""Isolated, network-disabled-by-policy SDXL worker. JSON events are emitted to stdout."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback

# Set before importing Hugging Face, Torch or tokenizers.
for _name, _value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ[_name] = _value

from .config import TrainingConfig, output_safety_errors, preflight

_STOP = False


def emit(event_type, **values):
    print(json.dumps({"type": event_type, **values}), flush=True)


def request_stop(signum=None, frame=None):
    global _STOP
    _STOP = True
    emit("status", message="Stop requested. Finishing the optimizer step, then saving a resumable checkpoint.")


def lr_factor(step, total, warmup, schedule):
    warmup = min(warmup, total)
    if step < warmup:
        return (step + 1) / max(1, warmup)
    fraction = (step - warmup) / max(1, total - warmup)
    if schedule == "linear":
        return max(0.0, 1 - fraction)
    if schedule == "cosine":
        return (1 + math.cos(math.pi * min(1.0, fraction))) / 2
    return 1.0


def bucket_batches(entries, batch_size, seed, epoch):
    """Deterministic batches with exactly one tensor shape per batch, no dropped images."""
    buckets = {}
    for i, entry in enumerate(entries):
        buckets.setdefault((entry["width"], entry["height"]), []).append(i)
    rng = random.Random(seed + epoch)
    batches = []
    for key in sorted(buckets):
        values = buckets[key][:]
        rng.shuffle(values)
        batches.extend(values[i : i + batch_size] for i in range(0, len(values), batch_size))
    rng.shuffle(batches)
    return batches


def _config_identity(config):
    ignored = {"resume_from", "output_dir", "max_steps", "save_every", "checkpoints_to_keep"}
    return {k: v for k, v in config.to_dict().items() if k not in ignored}


def validate_resume_identity(config, dataset_fingerprint, model_fingerprint):
    """Validate checkpoint identity before creating or truncating any run files."""
    errors = output_safety_errors(config)
    if errors:
        raise ValueError("\n".join(errors))
    if not config.resume_from:
        return None
    checkpoint = Path(config.resume_from).expanduser().resolve(strict=True)
    info = json.loads((checkpoint / "checkpoint.json").read_text())
    if info.get("version") != 1:
        raise ValueError("Unsupported checkpoint version.")
    if info.get("config") != _config_identity(config) or info.get("dataset_fingerprint") != dataset_fingerprint or info.get("model_fingerprint") != model_fingerprint:
        raise ValueError("Resume controls, base model or dataset differ from the checkpoint. Use the saved configuration and unchanged inputs.")
    if info.get("original_max_steps") != config.max_steps:
        raise ValueError("Exact resume requires the original total step count because it defines the learning-rate schedule.")
    for field in ("step", "epoch", "batch_cursor"):
        if type(info.get(field)) is not int or info[field] < 0:
            raise ValueError(f"Invalid checkpoint sampler state: {field}")
    if not 0 < info["step"] <= config.max_steps:
        raise ValueError("Checkpoint step is outside the configured run.")
    if checkpoint.name != f"step-{info['step']:08d}":
        raise ValueError("Checkpoint directory name does not match its saved optimizer step.")
    names = ["unet.safetensors", "rng.pt"]
    if config.train_text_encoder:
        names += ["text_encoder.safetensors", "text_encoder_2.safetensors"]
    if config.train_vae:
        names += ["vae.safetensors"]
    names += ["optimizer/optimizer.json", "optimizer/first_moment.bin", "optimizer/second_moment.bin"] if config.optimizer == "adamw_disk" else ["optimizer.pt"]
    for name in names:
        asset = checkpoint / name
        if asset.is_symlink() or not asset.is_file():
            raise ValueError(f"Checkpoint is missing a regular state file: {name}")
    output = Path(config.output_dir).expanduser()
    if output.is_dir() and any(output.iterdir()):
        previous = TrainingConfig.load(output / "training-config.json")
        if _config_identity(previous) != _config_identity(config) or previous.max_steps != config.max_steps:
            raise ValueError("Existing run configuration does not match the resume checkpoint.")
    return info


def portable_lora_state(state, alpha, rank):
    """Fold alpha/rank into B for portable Diffusers adapters with unit scaling.

    Standalone Diffusers weight files do not reliably carry PEFT alpha metadata.
    Folding scaling into B preserves the effective delta when a consumer infers
    alpha=rank. Raw PEFT state and resumable checkpoints remain unchanged.
    """
    scale = alpha / rank
    return {name: tensor * scale if ".lora_B." in name else tensor for name, tensor in state.items()}


def save_portable_lora(destination, adapter_states, *, alpha, rank):
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    portable = {name: convert_state_dict_to_diffusers(portable_lora_state(state, alpha, rank)) for name, state in adapter_states.items() if name != "vae"}
    StableDiffusionXLPipeline.save_lora_weights(str(destination), unet_lora_layers=portable["unet"], text_encoder_lora_layers=portable.get("text_encoder"), text_encoder_2_lora_layers=portable.get("text_encoder_2"), safe_serialization=True)


def _weights(component):
    return {name: value.detach().cpu().contiguous() for name, value in component.state_dict().items()}


def run(config):
    report = preflight(config, check_runtime=True)
    for warning in report.warnings:
        emit("status", message=warning)
    if report.errors:
        raise ValueError("\n".join(report.errors))
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler, StableDiffusionXLPipeline
    from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model, set_peft_model_state_dict
    from safetensors.torch import load_file, save_file
    from ..dataset import verify_finalized, sha256
    from ..tokens import TokenCounter, tokenizer_fingerprint
    from .disk_optimizer import create_disk_adamw

    model_path = Path(config.model_path).resolve()
    dataset_path = Path(config.dataset_path).resolve()
    output = Path(config.output_dir).resolve()
    manifest = verify_finalized(dataset_path)
    counter = TokenCounter.from_model(model_path)
    fingerprints = {"tokenizer_sha256": tokenizer_fingerprint(model_path / "tokenizer"), "tokenizer_2_sha256": tokenizer_fingerprint(model_path / "tokenizer_2")}
    recorded = manifest.get("tokenizers", {})
    for name, fingerprint in fingerprints.items():
        if recorded.get(name) != fingerprint:
            raise ValueError("Dataset tokenizers differ from the selected base model. Re-finalize with this model's tokenizers.")
    captions = []
    for entry in manifest["entries"]:
        text = (dataset_path / entry["caption"]).read_text(encoding="utf-8")
        if not counter.count(text).valid:
            raise ValueError(f"{entry['caption']} exceeds a CLIP context. Captions are never silently truncated.")
        captions.append(text)
    if not captions:
        raise ValueError("Dataset contains no images.")
    dataset_fingerprint = sha256(dataset_path / "manifest.json")
    base_digest = hashlib.sha256()
    for asset in sorted(model_path.rglob("*")):
        if asset.is_file() and asset.suffix in {".json", ".safetensors", ".txt"}:
            base_digest.update(str(asset.relative_to(model_path)).encode())
            base_digest.update(sha256(asset).encode())
    model_fingerprint = base_digest.hexdigest()
    resume_info = validate_resume_identity(config, dataset_fingerprint, model_fingerprint)
    output.mkdir(parents=True, exist_ok=True)
    config.save(output / "training-config.json")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    # TF32 is not enabled: preserve the selected precision rather than silently lowering it.
    dtype = torch.float32
    emit("status", message="Loading the local Diffusers model. No network access is requested.")
    pipeline = StableDiffusionXLPipeline.from_pretrained(str(model_path), local_files_only=True, use_safetensors=True, torch_dtype=dtype)
    if pipeline.unet.config.in_channels != 4 or len(pipeline.unet.config.block_out_channels) != 3:
        raise ValueError("Only SDXL base architecture (three blocks, four latent channels) is supported; refiner and inpainting models require a different pipeline.")
    pipeline.set_progress_bar_config(disable=True)
    unet, vae = pipeline.unet, pipeline.vae
    encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]
    scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    components = {"unet": unet}
    if config.train_text_encoder:
        components.update({"text_encoder": encoders[0], "text_encoder_2": encoders[1]})
    if config.train_vae:
        components["vae"] = vae
    for model in [unet, vae, *encoders]:
        model.requires_grad_(False)
        model.eval()
    for name, model in components.items():
        if config.mode == "lora":
            targets = ["q_proj", "k_proj", "v_proj", "out_proj"] if name.startswith("text_encoder") else ["to_q", "to_k", "to_v", "to_out.0"]
            adapter_config = LoraConfig(r=config.rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=targets, init_lora_weights=True, bias="none")
            inject_adapter_in_model(adapter_config, model)
            for p in model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        else:
            model.float().requires_grad_(True)
        model.train()
        if config.gradient_checkpointing:
            if hasattr(model, "enable_gradient_checkpointing"):
                model.enable_gradient_checkpointing()
            elif hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if config.mode == "lora" and config.gradient_checkpointing and name.startswith("text_encoder") and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.to(config.device)
    # Frozen encoders stay on CPU and need not compete with training for device VRAM.
    # VAE is always fp32; encode on CPU when frozen, avoiding SDXL fp16 VAE overflows.
    vae.float()
    if not config.train_vae:
        vae.to("cpu")
    if not config.train_text_encoder:
        for model in encoders:
            model.float().to("cpu")
    anchor_vae = None
    if config.train_vae:
        from diffusers import AutoencoderKL
        anchor_vae = AutoencoderKL.from_pretrained(str(model_path / "vae"), local_files_only=True, use_safetensors=True, torch_dtype=torch.float32).eval().requires_grad_(False).to("cpu")
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    # No tiling during VAE training: blending tiles changes the training objective.
    groups = []
    for name, model in components.items():
        lr = config.text_encoder_lr if name.startswith("text_encoder") else config.vae_lr if name == "vae" else config.learning_rate
        groups.append({"named_params": [(f"{name}.{n}", p) for n, p in model.named_parameters() if p.requires_grad], "lr": lr})
    all_parameters = [p for g in groups for _, p in g["named_params"]]
    if not all_parameters:
        raise ValueError("No trainable parameters were selected.")
    plain_groups = [{"params": [p for _, p in g["named_params"]], "lr": g["lr"]} for g in groups]
    if config.optimizer == "adamw_disk":
        optimizer = create_disk_adamw(groups, output / "working" / "optimizer", betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon, weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(plain_groups, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon, weight_decay=config.weight_decay, foreach=False)
    else:
        optimizer = torch.optim.SGD(plain_groups, weight_decay=config.weight_decay)
    base_lrs = [g["lr"] for g in groups]
    step, epoch, batch_cursor = 0, 0, 0
    training_started = time.monotonic()
    text_cache = output / "working" / "text"
    text_cache.mkdir(parents=True, exist_ok=True)

    def state_for(name, model):
        state = get_peft_model_state_dict(model) if config.mode == "lora" else model.state_dict()
        return {key: value.detach().cpu().contiguous() for key, value in state.items()}

    def save_checkpoint():
        destination = output / "checkpoints" / f"step-{step:08d}"
        if destination.exists():
            raise ValueError(f"Checkpoint already exists and was not overwritten: {destination}. Resume the latest checkpoint or choose a new empty output folder.")
        temporary = destination.with_name(destination.name + ".tmp")
        if temporary.exists():
            raise ValueError(f"Incomplete checkpoint folder already exists: {temporary}. Inspect or remove it before retrying; no files were overwritten.")
        temporary.mkdir(parents=True, exist_ok=False)
        for name, model in components.items():
            save_file(state_for(name, model), str(temporary / f"{name}.safetensors"))
        rng = {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy_keys": torch.from_numpy(np.random.get_state()[1].astype(np.int64)), "numpy_position": np.random.get_state()[2], "numpy_gauss": np.random.get_state()[3], "numpy_cached": np.random.get_state()[4]}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng, temporary / "rng.pt")
        if config.optimizer == "adamw_disk":
            optimizer.save_state(temporary / "optimizer")
        else:
            torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        info = {"version": 1, "step": step, "epoch": epoch, "batch_cursor": batch_cursor, "config": _config_identity(config), "dataset_fingerprint": dataset_fingerprint, "model_fingerprint": model_fingerprint, "original_max_steps": config.max_steps}
        (temporary / "checkpoint.json").write_text(json.dumps(info, indent=2))
        temporary.rename(destination)
        emit("checkpoint", path=str(destination), step=step)
        existing = sorted(p for p in destination.parent.glob("step-*") if p.is_dir() and (p / "checkpoint.json").is_file())
        for old in existing[:-config.checkpoints_to_keep]:
            if old != destination:
                shutil.rmtree(old)
        return destination

    if config.resume_from:
        checkpoint = Path(config.resume_from).resolve()
        info = resume_info
        for name, model in components.items():
            state = load_file(str(checkpoint / f"{name}.safetensors"), device="cpu")
            if config.mode == "lora":
                set_peft_model_state_dict(model, state)
            else:
                model.load_state_dict(state, strict=True)
        if config.optimizer == "adamw_disk":
            optimizer.load_state(checkpoint / "optimizer")
        else:
            optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location=config.device, weights_only=True))
        rng = torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=True)
        torch.set_rng_state(rng["torch"])
        random.setstate(rng["python"])
        np.random.set_state(("MT19937", rng["numpy_keys"].numpy().astype(np.uint32), rng["numpy_position"], rng["numpy_gauss"], rng["numpy_cached"]))
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        step, epoch, batch_cursor = info["step"], info["epoch"], info["batch_cursor"]
        emit("status", message=f"Resumed optimizer step {step}, epoch {epoch}, batch {batch_cursor}.")

    def encode_captions(indices):
        results = []
        for index in indices:
            text = captions[index]
            cache_key = hashlib.sha256(text.encode()).hexdigest()
            cache_file = text_cache / (cache_key + ".safetensors")
            if not config.train_text_encoder and config.cache_frozen_text and cache_file.is_file():
                result = load_file(str(cache_file))
                results.append((result["prompt"].to(config.device), result["pooled"].to(config.device)))
                continue
            embeddings = []
            pooled = None
            context = nullcontext() if config.train_text_encoder else torch.no_grad()
            with context:
                for tokenizer, encoder in zip(tokenizers, encoders):
                    ids = tokenizer(text, padding="max_length", max_length=77, truncation=False, return_tensors="pt").input_ids
                    encoded = encoder(ids.to(next(encoder.parameters()).device), output_hidden_states=True, return_dict=True)
                    embeddings.append(encoded.hidden_states[-2].to(config.device))
                    if getattr(encoded, "text_embeds", None) is not None:
                        pooled = encoded.text_embeds.to(config.device)
                if pooled is None:
                    raise ValueError("CLIP-G did not produce projected pooled text embeddings.")
                prompt = torch.cat(embeddings, dim=-1)
            if not config.train_text_encoder and config.cache_frozen_text:
                save_file({"prompt": prompt.detach().cpu().contiguous(), "pooled": pooled.detach().cpu().contiguous()}, str(cache_file))
            results.append((prompt, pooled))
        return torch.cat([x[0] for x in results]), torch.cat([x[1] for x in results])

    batches = bucket_batches(manifest["entries"], config.batch_size, config.seed, epoch)
    optimizer.zero_grad(set_to_none=True)
    emit("status", message=f"Training {sum(p.numel() for p in all_parameters):,} fp32 parameters; {config.optimizer}.")
    while step < config.max_steps:
        accumulated_loss = accumulated_vae = 0.0
        for micro_step in range(config.gradient_accumulation_steps):
            if batch_cursor >= len(batches):
                epoch += 1
                batch_cursor = 0
                batches = bucket_batches(manifest["entries"], config.batch_size, config.seed, epoch)
            indices = batches[batch_cursor]
            batch_cursor += 1
            entries = [manifest["entries"][i] for i in indices]
            arrays = []
            for entry in entries:
                with Image.open(dataset_path / entry["image"]) as image:
                    arrays.append(np.array(image.convert("RGB"), dtype=np.float32) / 127.5 - 1)
            pixels_cpu = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
            saved = torch.autograd.graph.save_on_cpu(pin_memory=False) if config.cpu_offload_saved_tensors and config.device == "cuda" else nullcontext()
            # Text encoders and UNet run under autocast; VAE remains explicit fp32.
            with saved:
                autocast = torch.autocast(device_type=config.device, dtype=torch.bfloat16, cache_enabled=False) if config.precision == "bf16" else nullcontext()
                with autocast:
                    prompt, pooled = encode_captions(indices)
                vae_device = next(vae.parameters()).device
                with torch.no_grad():
                    # Diffusion never drives the VAE toward a degenerate latent space.
                    posterior = vae.encode(pixels_cpu.to(vae_device)).latent_dist
                    latents = posterior.sample().detach().to(config.device) * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                if config.noise_offset:
                    noise = noise + config.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (len(indices),), device=config.device).long()
                noisy = scheduler.add_noise(latents, noise, timesteps)
                time_ids = []
                for entry in entries:
                    width, height = entry["original_size"]
                    top, left = entry["crop_top_left"]
                    time_ids.append([height, width, top, left, entry["height"], entry["width"]])
                added = {"text_embeds": pooled, "time_ids": torch.tensor(time_ids, device=config.device, dtype=prompt.dtype)}
                with (torch.autocast(device_type=config.device, dtype=torch.bfloat16, cache_enabled=False) if config.precision == "bf16" else nullcontext()):
                    prediction = unet(noisy.to(prompt.dtype), timesteps, prompt, added_cond_kwargs=added, return_dict=False)[0]
                if scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif scheduler.config.prediction_type == "v_prediction":
                    target = scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported training prediction type: {scheduler.config.prediction_type}")
                loss_per_image = F.mse_loss(prediction.float(), target.float(), reduction="none").mean(dim=(1, 2, 3))
                if config.snr_gamma > 0:
                    alpha = scheduler.alphas_cumprod.to(config.device)[timesteps]
                    snr = alpha / (1 - alpha)
                    denominator = snr + (1 if scheduler.config.prediction_type == "v_prediction" else 0)
                    loss_per_image = loss_per_image * (torch.minimum(snr, torch.full_like(snr, config.snr_gamma)) / denominator)
                loss = loss_per_image.mean()
                accumulated_loss += loss.detach().item() / config.gradient_accumulation_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite diffusion loss. No checkpoint was overwritten.")
                (loss / config.gradient_accumulation_steps).backward()
                # Separate VAE graph avoids retaining the full diffusion graph and its activations.
                if config.train_vae:
                    pixels = pixels_cpu.to(vae_device)
                    with torch.no_grad():
                        reference = anchor_vae.encode(pixels_cpu).latent_dist.mean.to(vae_device)
                    adapted_posterior = vae.encode(pixels).latent_dist
                    reconstruction = vae.decode(adapted_posterior.sample()).sample
                    reconstruction_loss = F.l1_loss(reconstruction.float(), pixels.float())
                    kl = adapted_posterior.kl().mean()
                    anchor = F.mse_loss(adapted_posterior.mean.float(), reference.float())
                    vae_loss = config.vae_reconstruction_weight * reconstruction_loss + config.vae_kl_weight * kl + config.vae_anchor_weight * anchor
                    if not torch.isfinite(vae_loss):
                        raise FloatingPointError("Non-finite VAE loss. No checkpoint was overwritten.")
                    accumulated_vae += vae_loss.detach().item() / config.gradient_accumulation_steps
                    (vae_loss / config.gradient_accumulation_steps).backward()
            del pixels_cpu, arrays, prediction, loss, prompt, pooled, latents, noisy, noise
        factor = lr_factor(step, config.max_steps, config.warmup_steps, config.lr_scheduler)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * factor
        if config.optimizer == "adamw_disk":
            norm = optimizer.step(max_grad_norm=config.max_grad_norm)
        else:
            norm = torch.nn.utils.clip_grad_norm_(all_parameters, config.max_grad_norm if config.max_grad_norm else float("inf"), error_if_nonfinite=True).item()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        emit("progress", step=step, total=config.max_steps, loss=accumulated_loss, vae_loss=accumulated_vae if config.train_vae else None, learning_rate=optimizer.param_groups[0]["lr"], grad_norm=norm, epoch=epoch, elapsed_seconds=round(time.monotonic() - training_started, 2))
        if step % config.save_every == 0 or _STOP or step == config.max_steps:
            save_checkpoint()
        if _STOP:
            emit("complete", stopped=True, step=step, message="Stopped safely; resume from the saved checkpoint.")
            return
    # Save reusable adapters before merging, then create a complete Diffusers model.
    if config.mode == "lora":
        adapter_dir = output / "adapters"
        adapter_dir.mkdir(exist_ok=True)
        adapter_states = {name: state_for(name, model) for name, model in components.items()}
        for name, state in adapter_states.items():
            save_file(state, str(adapter_dir / f"{name}.safetensors"))
            (adapter_dir / f"{name}.json").write_text(json.dumps({"rank": config.rank, "lora_alpha": config.lora_alpha, "lora_dropout": config.lora_dropout, "base_model": str(model_path)}, indent=2))
        save_portable_lora(adapter_dir, adapter_states, alpha=config.lora_alpha, rank=config.rank)
        # Replace tuner layers with their merged base layers, including VAE adapters.
        from peft.tuners.tuners_utils import BaseTunerLayer
        for model in components.values():
            model.eval()
            for name, module in list(model.named_modules()):
                if isinstance(module, BaseTunerLayer):
                    module.merge(safe_merge=True)
                    parent_name, _, child = name.rpartition(".")
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child, module.get_base_layer())
            for attr in ("peft_config", "_hf_peft_config_loaded"):
                if hasattr(model, attr):
                    if attr == "_hf_peft_config_loaded":
                        setattr(model, attr, False)
                    else:
                        delattr(model, attr)
    emit("status", message="Saving complete Diffusers output with UNet, both text encoders and VAE.")
    destination = output / "diffusers"
    temporary = output / "diffusers.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    # Keep fp32 base and adapter master weights in the merged output; autocast
    # only affects forward computation, never persisted model precision.
    pipeline.save_pretrained(str(temporary), safe_serialization=True, max_shard_size="2GB")
    if destination.exists():
        raise ValueError("A completed Diffusers output already exists; it was not overwritten.")
    temporary.rename(destination)
    emit("complete", stopped=False, step=step, path=str(destination), message="Training complete. Diffusers output is ready for optional single-file export.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        config = TrainingConfig.load(args.config)
        if args.preflight:
            from dataclasses import asdict
            result = preflight(config, check_runtime=True)
            emit("preflight", **asdict(result), ok=result.ok)
            return 0 if result.ok else 1
        run(config)
        return 0
    except Exception as exc:
        emit("error", message=str(exc), traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
