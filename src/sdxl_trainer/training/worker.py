"""Isolated, network-disabled-by-policy SDXL worker. JSON events are emitted to stdout."""
from __future__ import annotations
import argparse
from contextlib import ExitStack, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback

# Set before importing Hugging Face, Torch or tokenizers.
for _name, _value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ[_name] = _value

from .config import CPU_TEXT_OFFLOAD, STAGED_UNET_OFFLOAD, LEGACY_TRAINING_FIELDS, TrainingConfig, output_safety_errors, preflight

_STOP = False
_EMIT_LOCK = threading.RLock()  # SIGINT may emit while the main thread is already printing.


def emit(event_type, **values):
    with _EMIT_LOCK:
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


def sample_vae_latents(posterior, scaling_factor, device, vae_precision):
    """Sample exactly once and fail at the first non-finite latent boundary."""
    import torch
    latents = posterior.sample().detach().to(device) * scaling_factor
    if not torch.isfinite(latents).all().item():
        message = (f"Non-finite VAE latents after sampling and scaling (VAE precision: {vae_precision}). "
                   "Check the base model weights, image data and VAE scaling configuration.")
        if vae_precision == "fp16":
            message += (" FP16 VAE computation can overflow; try fp32 or bf16 VAE precision, "
                        "or a base model containing an fp16-safe VAE.")
        raise FloatingPointError(message + " No checkpoint was overwritten.")
    return latents


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
    # A project may be moved externally after checkpointing; the sealed dataset's
    # manifest fingerprint, not its filesystem location, identifies its data.
    ignored = {"resume_from", "output_dir", "dataset_path", "max_steps", "save_every", "checkpoints_to_keep",
               "cache_frozen_vae", "vae_cache_ram_mib", "frozen_text_cache_ram_mib", "profile_phases", "monitor_gpu", "optimizer_update_device", "optimizer_gradient_storage", "final_export_shard_mib"}
    if config.frozen_vae_cache_mode == "sample" and not config.train_vae:
        ignored.remove("cache_frozen_vae")  # fixed samples versus fresh draws changes training
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
    saved_config = info.get("config")
    if isinstance(saved_config, dict):
        # Older version-1 checkpoints recorded the path in their identity.
        # Compare a copy so their metadata is not modified on disk.
        ignored = {"dataset_path", "cache_frozen_vae", "vae_cache_ram_mib", "frozen_text_cache_ram_mib", "profile_phases", "monitor_gpu", "optimizer_update_device", "optimizer_gradient_storage", "final_export_shard_mib"}
        if config.frozen_vae_cache_mode == "sample" and not config.train_vae:
            ignored.remove("cache_frozen_vae")
        saved_config = {key: value for key, value in saved_config.items() if key not in ignored}
        # Version-1 checkpoints created before component placement was exposed
        # used the original all-selected-components-on-compute-device behavior.
        saved_config.setdefault("component_offload", "none")
        saved_config.setdefault("model_variant", "default")
        saved_config.setdefault("model_precision", "fp32")
        saved_config.setdefault("cpu_text_precision", "match")
        saved_config.setdefault("paged_cuda_override", "")
        saved_config.setdefault("vae_precision", "fp32")
        saved_config.setdefault("loss_scale_initial", 4096.0)
        saved_config.setdefault("loss_scale_growth_interval", 2000)
        for name, value in LEGACY_TRAINING_FIELDS.items():
            if name not in ignored:
                saved_config.setdefault(name, value)
    if saved_config != _config_identity(config) or info.get("dataset_fingerprint") != dataset_fingerprint or info.get("model_fingerprint") != model_fingerprint:
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
    if config.optimizer == "adamw_disk":
        names += ["optimizer/optimizer.json", "optimizer/first_moment.bin", "optimizer/second_moment.bin"]
    elif config.optimizer == "paged_adamw8bit":
        names += ["optimizer/paged.json", "optimizer/state.safetensors"]
    else:
        names += ["optimizer.pt"]
    if config.optimizer in {"adamw_disk", "paged_adamw8bit"} and (checkpoint / "optimizer").is_symlink():
        raise ValueError("Checkpoint optimizer state must be a real directory.")
    for name in names:
        asset = checkpoint / name
        if asset.is_symlink() or not asset.is_file():
            raise ValueError(f"Checkpoint is missing a regular state file: {name}")
    if config.optimizer == "paged_adamw8bit":
        from .paged_adamw import load_paged_runtime, validate_paged_snapshot
        _, runtime_info = load_paged_runtime(config.paged_cuda_override)
        validate_paged_snapshot(checkpoint / "optimizer", runtime_info)
    if config.optimizer == "adamw_disk":
        optimizer_info = json.loads((checkpoint / "optimizer/optimizer.json").read_text())
        saved_backend = optimizer_info.get("update_backend", "cpu")
        if saved_backend not in {"cpu", "cuda"}:
            raise ValueError("Checkpoint has an unsupported optimizer update backend.")
        if config.optimizer_update_device not in {"auto", saved_backend}:
            raise ValueError(f"This checkpoint uses the {saved_backend} optimizer update backend. Select auto or {saved_backend} to resume exactly.")
        records = optimizer_info.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("Checkpoint optimizer parameter metadata is missing.")
        expected_bytes = 0
        for record in records:
            if (not isinstance(record, dict) or type(record.get("offset")) is not int or type(record.get("numel")) is not int
                or record["offset"] != expected_bytes or record["numel"] < 0):
                raise ValueError("Checkpoint optimizer parameter layout is invalid.")
            expected_bytes += 4 * record["numel"]
        if expected_bytes <= 0:
            raise ValueError("Checkpoint optimizer parameter layout is empty.")
        for name in ("first_moment.bin", "second_moment.bin"):
            if (checkpoint / "optimizer" / name).stat().st_size != expected_bytes:
                raise ValueError(f"Checkpoint optimizer state is truncated: {name}.")
        expected_masters = config.mode == "full" and (config.model_precision != "fp32" or (config.train_vae and config.vae_precision != "fp32"))
        if optimizer_info.get("master_weights", False) != expected_masters:
            raise ValueError("This checkpoint does not contain the expected fp32 master weights.")
        if expected_masters:
            master = checkpoint / "optimizer/master_weights.bin"
            if master.is_symlink() or not master.is_file():
                raise ValueError("This checkpoint is missing regular fp32 master weights.")
            if master.stat().st_size != expected_bytes:
                raise ValueError("This checkpoint has truncated fp32 master weights.")
    if config.precision == "fp16" or (config.train_vae and config.vae_precision == "fp16"):
        import torch
        from .loss_scaler import DynamicLossScaler
        saved_rng = torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=True)
        scaler = DynamicLossScaler(config.loss_scale_initial, config.loss_scale_growth_interval)
        if not isinstance(saved_rng, dict):
            raise ValueError("This checkpoint has invalid RNG and loss-scaler state.")
        scaler.load_state_dict(saved_rng.get("loss_scaler"))
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


def _comfyui_lora_state(adapter_states, *, alpha, rank):
    """Map raw PEFT factors to the SDXL names accepted by ComfyUI's loader."""
    import torch

    result = {}
    prefixes = {"unet": "lora_unet_", "text_encoder": "lora_te1_", "text_encoder_2": "lora_te2_"}
    for component, prefix in prefixes.items():
        state = adapter_states.get(component, {})
        for key, down in state.items():
            module, _, factor = key.rpartition(".lora_")
            if factor == "B.weight" and module + ".lora_A.weight" in state:
                continue
            if not module or factor != "A.weight":
                raise ValueError(f"Unsupported or unpaired {component} LoRA tensor: {key}")
            up = state.get(module + ".lora_B.weight")
            if (up is None or down.ndim != 2 or up.ndim != 2
                    or down.shape[0] != rank or up.shape[1] != rank):
                raise ValueError(f"Invalid {component} LoRA factor shapes or rank: {module}")
            name = prefix + module.replace(".", "_")
            if name + ".alpha" in result:
                raise ValueError(f"Duplicate ComfyUI LoRA module name: {name}")
            result[name + ".lora_down.weight"] = down
            result[name + ".lora_up.weight"] = up
            # The ComfyUI loader applies alpha / rank to these raw factors.
            result[name + ".alpha"] = torch.tensor(float(alpha), dtype=torch.float32)
    return result


def save_portable_lora(destination, adapter_states, *, alpha, rank):
    """Save separate Diffusers and ComfyUI adapters from the same trained factors."""
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from safetensors.torch import save_file

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lora-", dir=destination) as temporary:
        stage = Path(temporary)
        comfy = _comfyui_lora_state(adapter_states, alpha=alpha, rank=rank)
        save_file(comfy, str(stage / "lora_comfyui.safetensors"), metadata={"format": "pt"})
        portable = {name: convert_state_dict_to_diffusers(portable_lora_state(state, alpha, rank))
                    for name, state in adapter_states.items() if name != "vae"}
        StableDiffusionXLPipeline.save_lora_weights(
            str(stage), unet_lora_layers=portable["unet"],
            text_encoder_lora_layers=portable.get("text_encoder"),
            text_encoder_2_lora_layers=portable.get("text_encoder_2"), safe_serialization=True,
        )
        # Keep Diffusers folder discovery working, sharing storage when possible.
        canonical = stage / "pytorch_lora_weights.safetensors"
        named = stage / "lora_diffusers.safetensors"
        try:
            os.link(canonical, named)
        except OSError:
            shutil.copyfile(canonical, named)
        for path in stage.iterdir():
            os.replace(path, destination / path.name)


def _weights(component):
    return {name: value.detach().cpu().contiguous() for name, value in component.state_dict().items()}


def run(config):
    """Ensure the optional monitor is stopped on completion, cancellation, or failure."""
    import torch
    from .telemetry import StepTelemetry
    profiling = getattr(config, "profile_phases", True)
    telemetry = StepTelemetry(config.device, gpu_enabled=profiling, synchronize_gpu=profiling,
                              monitor_gpu=getattr(config, "monitor_gpu", True),
                              on_sample=lambda sample: emit("gpu", **sample))
    telemetry.start()
    backend_settings = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        # Explicit policy, including strict FP32. Restore process state for tests
        # and callers embedding the worker rather than using its isolated CLI.
        if config.tf32_policy == "explicit":
            tf32 = bool(config.allow_tf32 and config.precision != "fp32" and config.device == "cuda")
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
        with ExitStack() as resources:
            # Some x86 oneDNN backends select reduced-precision forward
            # primitives whose backward is unsupported on this CPU. Cover the
            # entire graph lifetime, not merely the encoder's forward. The
            # native CPU path is slower but preserves the selected dtype.
            cpu_reduced_text = config.train_text_encoder and config.model_precision != "fp32" and config.cpu_text_precision != "fp32" and (
                config.device == "cpu" or config.component_offload in CPU_TEXT_OFFLOAD)
            cpu_reduced_vae = config.train_vae and config.vae_precision != "fp32" and (
                config.device == "cpu" or config.component_offload == "text_and_vae_cpu")
            cpu_reduced_unet = config.device == "cpu" and config.model_precision != "fp32"
            if cpu_reduced_text or cpu_reduced_vae or cpu_reduced_unet:
                resources.enter_context(torch.backends.mkldnn.flags(enabled=False))
            return _run(config, telemetry, resources)
    finally:
        telemetry.close()
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = backend_settings


def _run(config, telemetry, resources):
    report = preflight(config, check_runtime=True)
    for warning in report.warnings:
        emit("status", message=warning)
    if report.errors:
        raise ValueError("\n".join(report.errors))
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline
    from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model, set_peft_model_state_dict
    from safetensors.torch import load_file, save_file
    from ..dataset import load_training_dataset, sha256
    from ..tokens import TokenCounter, tokenizer_fingerprint
    from .disk_optimizer import NonFiniteGradients, create_disk_adamw
    from .loss_scaler import DynamicLossScaler
    from .long_prompt import CACHE_VERSION, encode_long_prompt, pad_prompt_batch
    paged_runtime = None
    if config.optimizer == "paged_adamw8bit":
        from .paged_adamw import load_paged_runtime
        paged_runtime = load_paged_runtime(config.paged_cuda_override)

    model_path = Path(config.model_path).resolve()
    dataset_path = Path(config.dataset_path).expanduser().absolute()
    output = Path(config.output_dir).resolve()
    manifest, dataset_fingerprint = load_training_dataset(dataset_path)
    counter = TokenCounter.from_model(model_path)
    fingerprints = {"tokenizer_sha256": tokenizer_fingerprint(model_path / "tokenizer"), "tokenizer_2_sha256": tokenizer_fingerprint(model_path / "tokenizer_2")}
    recorded = manifest.get("tokenizers", {})
    for name, fingerprint in fingerprints.items():
        if recorded.get(name) != fingerprint:
            raise ValueError("Dataset tokenizers differ from the selected base model. Re-finalize with this model's tokenizers.")
    captions = []
    for entry in manifest["entries"]:
        text = (dataset_path / entry["caption"]).read_text(encoding="utf-8")
        actual = counter.count(text)
        recorded_tokens = entry.get("tokens", {})
        if (actual.clip_l, actual.clip_g) != (recorded_tokens.get("clip_l"), recorded_tokens.get("clip_g")):
            raise ValueError(f"{entry['caption']} token counts do not match the finalized manifest. Re-finalize this dataset.")
        captions.append(text)
    if not captions:
        raise ValueError("Dataset contains no images.")
    base_digest = hashlib.sha256()
    for asset in sorted(model_path.rglob("*")):
        if asset.is_file() and asset.suffix in {".json", ".safetensors", ".txt"}:
            base_digest.update(str(asset.relative_to(model_path)).encode())
            base_digest.update(sha256(asset).encode())
    model_fingerprint = base_digest.hexdigest()
    resume_info = validate_resume_identity(config, dataset_fingerprint, model_fingerprint)
    output.mkdir(parents=True, exist_ok=True)
    if not (config.resume_from and config.optimizer == "paged_adamw8bit"):
        config.save(output / "training-config.json")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    emit("status", message=f"Training caption context: {config.caption_mode}; TF32 policy {config.tf32_policy} (matmul {torch.backends.cuda.matmul.allow_tf32}, cuDNN {torch.backends.cudnn.allow_tf32}).")
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = dtype_map[config.model_precision]
    vae_dtype = dtype_map[config.vae_precision]
    variant = None if config.model_variant == "default" else config.model_variant
    emit("status", message="Loading the local Diffusers model. No network access is requested.")
    # Load the VAE independently when its requested precision differs; loading
    # it in fp16 and then upcasting would permanently round its source weights.
    load_kwargs = {"local_files_only": True, "use_safetensors": True, "torch_dtype": dtype}
    if variant:
        load_kwargs["variant"] = variant
    if vae_dtype != dtype:
        vae_kwargs = {"local_files_only": True, "use_safetensors": True, "torch_dtype": vae_dtype}
        if variant:
            vae_kwargs["variant"] = variant
        load_kwargs["vae"] = AutoencoderKL.from_pretrained(str(model_path / "vae"), **vae_kwargs)
    if config.cpu_text_precision == "fp32" and dtype != torch.float32:
        # Load directly at the selected precision: casting a previously
        # reduced encoder back to fp32 cannot restore source weight bits.
        # Selected variant files may themselves already be rounded.
        from transformers import CLIPTextModel, CLIPTextModelWithProjection
        text_kwargs = {"local_files_only": True, "use_safetensors": True, "torch_dtype": torch.float32}
        if variant:
            text_kwargs["variant"] = variant
        load_kwargs["text_encoder"] = CLIPTextModel.from_pretrained(str(model_path / "text_encoder"), **text_kwargs)
        load_kwargs["text_encoder_2"] = CLIPTextModelWithProjection.from_pretrained(str(model_path / "text_encoder_2"), **text_kwargs)
    pipeline = StableDiffusionXLPipeline.from_pretrained(str(model_path), **load_kwargs)
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

    # Prepare frozen conditioning before trainable weights occupy the GPU.
    # Cache tensors live on disk and in bounded CPU RAM, never permanently on
    # CUDA. Trainable text/VAE paths deliberately bypass these caches.
    from .fast_cache import FrozenSampleCache, FrozenTextCache
    from .vae_cache import make_frozen_vae_cache
    vae_cache = (make_frozen_vae_cache(config, output, model_fingerprint, vae)
                 if config.frozen_vae_cache_mode == "posterior" else None)
    sample_cache = (FrozenSampleCache(output / "working" / "latents", model_fingerprint,
                    seed=config.seed, vae_precision=config.vae_precision,
                    latent_channels=int(vae.config.latent_channels),
                    scale_factor=2 ** (len(vae.config.block_out_channels) - 1),
                    scaling_factor=float(vae.config.scaling_factor), output_dtype=dtype,
                    memory_limit_bytes=config.vae_cache_ram_mib * 1024 * 1024)
                    if not config.train_vae and config.cache_frozen_vae and config.frozen_vae_cache_mode == "sample" else None)
    frozen_text_cache = (FrozenTextCache(output / "working" / "text", model_fingerprint,
                        caption_mode=config.caption_mode, precision=config.model_precision,
                        encoding_policy=f"{CACHE_VERSION}/{'prepare-' + config.device if config.precompute_frozen_cache else 'lazy-cpu'}/{config.precision}/tf32-{config.allow_tf32}",
                        memory_limit_bytes=config.frozen_text_cache_ram_mib * 1024 * 1024,
                        prompt_width=sum(int(encoder.config.hidden_size) for encoder in encoders),
                        pooled_width=int(encoders[1].config.projection_dim))
                        if not config.train_text_encoder and config.cache_frozen_text else None)
    empty_conditioning = None

    def cache_pixels(entry):
        with Image.open(dataset_path / entry["image"]) as image:
            array = np.array(image.convert("RGB"), dtype=np.float32) / 127.5 - 1
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()

    def frozen_encode(text, device):
        with torch.no_grad(), (torch.autocast(device_type=config.device, dtype=dtype_map[config.precision], cache_enabled=False)
                              if config.precision != "fp32" and config.precompute_frozen_cache else nullcontext()):
            prompt, pooled = encode_long_prompt(text, tokenizers, encoders, device, caption_mode=config.caption_mode)
            return prompt.to(dtype=dtype), pooled.to(dtype=dtype)

    if config.precompute_frozen_cache and any(cache is not None for cache in (sample_cache, vae_cache, frozen_text_cache)):
        preparation_started = time.monotonic()
        timings = {}
        devices = [torch.device(config.device).index or 0] if config.device.startswith("cuda") else []
        # Encoding must not advance training RNG, including rebuilding disposable
        # caches on resume. Fixed samples use their own content-derived RNG.
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            if sample_cache is not None or vae_cache is not None:
                started = time.monotonic()
                emit("status", message=f"Preparing frozen VAE cache on {config.device}: {len(captions)} images.")
                vae.to(config.device)
                for index, entry in enumerate(manifest["entries"]):
                    if _STOP:
                        emit("complete", stopped=True, step=0, message="Stopped during cache preparation; completed caches are retained.")
                        return
                    if sample_cache is not None:
                        sample_cache.get(entry, lambda entry=entry: vae.encode(cache_pixels(entry).to(device=config.device, dtype=vae_dtype)).latent_dist)
                    else:
                        vae_cache.posterior(vae, lambda _: cache_pixels(entry).to(device=config.device, dtype=vae_dtype), [entry])
                    if (index + 1) % 50 == 0 or index + 1 == len(captions):
                        emit("status", message=f"Prepared VAE cache {index + 1}/{len(captions)}.")
                vae.to("cpu")
                if config.device == "cuda":
                    torch.cuda.empty_cache()
                timings["vae_seconds"] = round(time.monotonic() - started, 3)
            if frozen_text_cache is not None:
                started = time.monotonic()
                emit("status", message=f"Preparing full {config.caption_mode} frozen text cache on {config.device}: {len(captions)} captions.")
                for encoder in encoders:
                    encoder.to(config.device)
                for index, text in enumerate(captions):
                    if _STOP:
                        emit("complete", stopped=True, step=0, message="Stopped during cache preparation; completed caches are retained.")
                        return
                    frozen_text_cache.get(text, lambda text=text: frozen_encode(text, config.device))
                    if (index + 1) % 50 == 0 or index + 1 == len(captions):
                        emit("status", message=f"Prepared text cache {index + 1}/{len(captions)}.")
                if config.batch_size > 1:
                    empty_conditioning = frozen_encode("", config.device)[0].detach().cpu()
                for encoder in encoders:
                    encoder.to("cpu")
                if config.device == "cuda":
                    torch.cuda.empty_cache()
                timings["text_seconds"] = round(time.monotonic() - started, 3)
        emit("preparation", total_seconds=round(time.monotonic() - preparation_started, 3), **timings)
    for name, model in components.items():
        if config.mode == "lora":
            targets = ["q_proj", "k_proj", "v_proj", "out_proj"] if name.startswith("text_encoder") else ["to_q", "to_k", "to_v", "to_out.0"]
            adapter_config = LoraConfig(r=config.rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=targets, init_lora_weights=True, bias="none")
            inject_adapter_in_model(adapter_config, model)
            for p in model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        else:
            model.requires_grad_(True)
        model.train()
        if config.gradient_checkpointing:
            if hasattr(model, "enable_gradient_checkpointing"):
                model.enable_gradient_checkpointing()
            elif hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if config.mode == "lora" and config.gradient_checkpointing and name.startswith("text_encoder") and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        target_device = config.device
        if name.startswith("text_encoder") and config.component_offload in CPU_TEXT_OFFLOAD:
            target_device = "cpu"
        elif name == "vae" and config.component_offload == "text_and_vae_cpu":
            target_device = "cpu"
        # Establish placement before the optimizer hooks and parameter groups
        # are created. Parameters are never migrated mid-graph or mid-update.
        model.to(target_device)
    # Frozen encoders stay on CPU and need not compete with training for device VRAM.
    # VAE dtype was selected while loading; do not cast the entire module here,
    # which would also reduce LoRA adapters that intentionally remain fp32.
    if not config.train_vae:
        vae.to("cpu")
    if not config.train_text_encoder:
        for model in encoders:
            if config.model_precision == "fp32":
                model.float()
            model.to("cpu")
    anchor_vae = None
    if config.train_vae:
        anchor_kwargs = {"local_files_only": True, "use_safetensors": True, "torch_dtype": torch.float32}
        if variant:
            anchor_kwargs["variant"] = variant
        anchor_vae = AutoencoderKL.from_pretrained(str(model_path / "vae"), **anchor_kwargs).eval().requires_grad_(False).to("cpu")
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    # No tiling during VAE training: blending tiles changes the training objective.
    if config.component_offload != "none":
        emit("status", message=f"Component placement: UNet {next(unet.parameters()).device}; text encoders {next(encoders[0].parameters()).device}, {next(encoders[1].parameters()).device}; VAE {next(vae.parameters()).device}. Text weights: {'fp32' if config.cpu_text_precision == 'fp32' else config.model_precision}; VAE: {config.vae_precision}.")
    stage_unet = config.component_offload in STAGED_UNET_OFFLOAD and config.train_vae
    if stage_unet:
        emit("status", message="UNet weights are temporarily disk-backed on CPU during VAE work and return to the GPU before UNet work and each optimizer update. Transfers preserve their selected precision and may be very slow.")
    groups = []
    for name, model in components.items():
        lr = config.text_encoder_lr if name.startswith("text_encoder") else config.vae_lr if name == "vae" else config.learning_rate
        groups.append({"named_params": [(f"{name}.{n}", p) for n, p in model.named_parameters() if p.requires_grad], "lr": lr})
    all_parameters = [p for g in groups for _, p in g["named_params"]]
    if not all_parameters:
        raise ValueError("No trainable parameters were selected.")
    plain_groups = [{"params": [p for _, p in g["named_params"]], "lr": g["lr"]} for g in groups]
    paged_checkpoint = None
    if config.optimizer == "adamw_disk":
        # Staging needs the host RAM for UNet weights. Auto deliberately uses
        # releasable disk gradient pages rather than anonymous RAM gradients.
        gradient_storage = "disk" if stage_unet and config.optimizer_gradient_storage == "auto" else config.optimizer_gradient_storage
        optimizer = create_disk_adamw(groups, output / "working" / "optimizer", betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon, weight_decay=config.weight_decay, update_device=config.optimizer_update_device, gradient_storage=gradient_storage, eager_gradient_eviction=stage_unet)
    elif config.optimizer == "paged_adamw8bit":
        from .paged_adamw import PagedAdamWCheckpoint
        bnb, runtime_info = paged_runtime
        optimizer = bnb.optim.PagedAdamW8bit(plain_groups, betas=(config.adam_beta1, config.adam_beta2),
                                            eps=config.adam_epsilon, weight_decay=config.weight_decay)
        paged_checkpoint = PagedAdamWCheckpoint(optimizer, groups, runtime_info)
    elif config.optimizer == "adamw":
        fused = bool(config.adamw_fused and all(p.device.type == "cuda" for p in all_parameters)
                     and len({p.device for p in all_parameters}) == 1)
        optimizer = torch.optim.AdamW(plain_groups, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon, weight_decay=config.weight_decay,
                                      **({"fused": True} if fused else {"foreach": False}))
        emit("status", message=f"AdamW GPU fusion {'enabled' if fused else 'disabled'}.")
    else:
        optimizer = torch.optim.SGD(plain_groups, weight_decay=config.weight_decay)
    unet_stager = None
    if stage_unet:
        from .disk_staging import DiskModuleStager
        unet_stager = DiskModuleStager(unet, output / "working")
        resources.callback(unet_stager.close)
    base_lrs = [g["lr"] for g in groups]
    scaler = DynamicLossScaler(config.loss_scale_initial, config.loss_scale_growth_interval) if config.precision == "fp16" or (config.train_vae and config.vae_precision == "fp16") else None
    step, epoch, batch_cursor = 0, 0, 0
    training_started = time.monotonic()
    if config.vae_precision != "fp32" and not config.train_vae and config.cache_frozen_vae and config.frozen_vae_cache_mode == "posterior":
        emit("status", message="Frozen VAE caching is bypassed for reduced-precision VAE computation to retain its selected sampling semantics.")

    def state_for(name, model):
        state = get_peft_model_state_dict(model) if config.mode == "lora" else model.state_dict()
        return {key: value.detach().cpu().contiguous() for key, value in state.items()}

    def save_checkpoint():
        checkpoint_started = time.monotonic()
        # The complete-caption working set can occupy several GiB. Release
        # disposable RAM copies before model/optimizer serialization; persistent
        # validated entries remain available for the next step or resume.
        for cache in (frozen_text_cache, sample_cache, vae_cache):
            if cache is not None:
                cache.clear_memory()
        destination = output / "checkpoints" / f"step-{step:08d}"
        if destination.exists():
            raise ValueError(f"Checkpoint already exists and was not overwritten: {destination}. Resume the latest checkpoint or choose a new empty output folder.")
        temporary = destination.with_name(destination.name + ".tmp")
        if temporary.exists():
            raise ValueError(f"Incomplete checkpoint folder already exists: {temporary}. Inspect or remove it before retrying; no files were overwritten.")
        if config.optimizer == "adamw_disk":
            # A completed update no longer needs its accumulation buffer. Free
            # it before constructing checkpoint component copies, then lazily
            # reacquire it on the next backward if training continues.
            optimizer.release_gradient_buffer()
        temporary.mkdir(parents=True, exist_ok=False)
        from .checkpoint_stream import save_safetensors_stream
        emit("status", message=f"Saving checkpoint step {step}: component weights.")
        for name, model in components.items():
            state = get_peft_model_state_dict(model) if config.mode == "lora" else model.state_dict()
            save_safetensors_stream(state, temporary / f"{name}.safetensors")
            del state
        rng = {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy_keys": torch.from_numpy(np.random.get_state()[1].astype(np.int64)), "numpy_position": np.random.get_state()[2], "numpy_gauss": np.random.get_state()[3], "numpy_cached": np.random.get_state()[4]}
        if scaler is not None:
            rng["loss_scaler"] = scaler.state_dict()
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng, temporary / "rng.pt")
        emit("status", message=f"Saving checkpoint step {step}: optimizer state.")
        if config.optimizer == "adamw_disk":
            optimizer.save_state(temporary / "optimizer")
        elif config.optimizer == "paged_adamw8bit":
            paged_checkpoint.save_state(temporary / "optimizer")
        else:
            torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        info = {"version": 1, "step": step, "epoch": epoch, "batch_cursor": batch_cursor, "config": _config_identity(config), "dataset_fingerprint": dataset_fingerprint, "model_fingerprint": model_fingerprint, "original_max_steps": config.max_steps}
        (temporary / "checkpoint.json").write_text(json.dumps(info, indent=2))
        temporary.rename(destination)
        emit("checkpoint", path=str(destination), step=step,
             checkpoint_seconds=round(time.monotonic() - checkpoint_started, 3))
        existing = sorted(p for p in destination.parent.glob("step-*") if p.is_dir() and (p / "checkpoint.json").is_file())
        for old in existing[:-config.checkpoints_to_keep]:
            if old != destination:
                shutil.rmtree(old)
        return destination

    if config.resume_from:
        checkpoint = Path(config.resume_from).resolve()
        info = resume_info
        if unet_stager:
            unet_stager.invalidate()
        for name, model in components.items():
            state = load_file(str(checkpoint / f"{name}.safetensors"), device="cpu")
            if config.mode == "lora":
                set_peft_model_state_dict(model, state)
            else:
                model.load_state_dict(state, strict=True)
        if config.optimizer == "adamw_disk":
            optimizer.load_state(checkpoint / "optimizer")
        elif config.optimizer == "paged_adamw8bit":
            paged_checkpoint.load_state(checkpoint / "optimizer")
        else:
            optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location=config.device, weights_only=True))
        rng = torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=True)
        torch.set_rng_state(rng["torch"])
        random.setstate(rng["python"])
        np.random.set_state(("MT19937", rng["numpy_keys"].numpy().astype(np.uint32), rng["numpy_position"], rng["numpy_gauss"], rng["numpy_cached"]))
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if scaler is not None:
            scaler.load_state_dict(rng.get("loss_scaler"))
        step, epoch, batch_cursor = info["step"], info["epoch"], info["batch_cursor"]
        if config.optimizer == "paged_adamw8bit":
            config.save(output / "training-config.json")
        emit("status", message=f"Resumed optimizer step {step}, epoch {epoch}, batch {batch_cursor}.")

    def encode_captions(indices):
        nonlocal empty_conditioning
        results = []
        context = nullcontext() if config.train_text_encoder else torch.no_grad()
        with context:
            for index in indices:
                text = captions[index]
                if frozen_text_cache is not None:
                    prompt, pooled = frozen_text_cache.get(text, lambda: frozen_encode(text, config.device))
                    results.append((prompt.to(config.device), pooled.to(config.device)))
                    continue
                prompt, pooled = encode_long_prompt(text, tokenizers, encoders, config.device, caption_mode=config.caption_mode)
                results.append((prompt, pooled))
            empty = None
            if frozen_text_cache is not None and len({prompt.shape[1] for prompt, _ in results}) > 1:
                if empty_conditioning is None:
                    empty_conditioning = frozen_encode("", config.device)[0].detach().cpu()
                empty = empty_conditioning.to(config.device)
            return pad_prompt_batch(results, tokenizers, encoders, config.device, empty_prompt=empty)

    batches = bucket_batches(manifest["entries"], config.batch_size, config.seed, epoch)
    optimizer.zero_grad(set_to_none=True)
    backend = f" · updates on {optimizer.update_backend} · gradients in {optimizer.gradient_storage}" if config.optimizer == "adamw_disk" else ""
    text_precision = "fp32" if config.cpu_text_precision == "fp32" else config.model_precision
    emit("status", message=f"Training {sum(p.numel() for p in all_parameters):,} parameters; UNet weights {config.model_precision}, text weights {text_precision}, VAE {config.vae_precision}; {config.optimizer}{backend}.")
    while step < config.max_steps:
        # A failed fp16 attempt is retried on exactly the same images/noise and
        # RNG streams. No optimizer state or scheduler step changes on overflow.
        if scaler is not None:
            retry_position = (epoch, batch_cursor)
            retry_torch = torch.get_rng_state()
            retry_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            retry_python = random.getstate()
            retry_numpy = np.random.get_state()
        loss_scale = scaler.scale if scaler is not None else 1.0
        accumulated_loss = accumulated_vae = 0.0
        telemetry.begin_step()
        image_count = 0
        def phase(name, gpu=False):
            return telemetry.phase(name, gpu=gpu) if getattr(config, "profile_phases", True) else nullcontext()
        for micro_step in range(config.gradient_accumulation_steps):
            if batch_cursor >= len(batches):
                epoch += 1
                batch_cursor = 0
                batches = bucket_batches(manifest["entries"], config.batch_size, config.seed, epoch)
            indices = batches[batch_cursor]
            image_count += len(indices)
            batch_cursor += 1
            entries = [manifest["entries"][i] for i in indices]
            def load_images(selected):
                with phase("load_images"):
                    arrays = []
                    for entry in selected:
                        with Image.open(dataset_path / entry["image"]) as image:
                            arrays.append(np.array(image.convert("RGB"), dtype=np.float32) / 127.5 - 1)
                    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
            # Frozen posterior hits need no decoded pixels. Misses are decoded
            # lazily in entry order, without drawing or caching latent noise.
            pixels_cpu = None if vae_cache is not None or sample_cache is not None else load_images(entries)
            saved = torch.autograd.graph.save_on_cpu(pin_memory=False) if config.cpu_offload_saved_tensors and config.device == "cuda" else nullcontext()
            # Text encoders and UNet use the chosen autocast; VAE computation
            # is separate and uses its explicitly selected precision.
            with saved:
                autocast = torch.autocast(device_type=config.device, dtype=dtype_map[config.precision], cache_enabled=False) if config.precision != "fp32" else nullcontext()
                with phase("text_encoding", gpu=config.train_text_encoder and next(encoders[0].parameters()).device.type == "cuda"), autocast:
                    # CPU autocast is independent of CUDA autocast; disable it
                    # explicitly so the opt-in text forward is truly fp32 in
                    # both all-CPU and CPU-text/GPU-UNet configurations.
                    text_context = torch.autocast(device_type="cpu", enabled=False) if config.cpu_text_precision == "fp32" else nullcontext()
                    with text_context:
                        prompt, pooled, prompt_mask = encode_captions(indices)
                vae_device = next(vae.parameters()).device
                if stage_unet:
                    with phase("offload_unet"):
                        unet_stager.park()
                        torch.cuda.empty_cache()
                with phase("vae_encoding", gpu=vae_device.type == "cuda"), torch.no_grad():
                    # Diffusion never drives the VAE toward a degenerate latent space.
                    if sample_cache is not None:
                        latents = torch.cat([sample_cache.get(entry, lambda entry=entry: vae.encode(cache_pixels(entry).to(device=vae_device, dtype=vae_dtype)).latent_dist)
                                             for entry in entries], dim=0).to(config.device)
                    else:
                        posterior = (vae_cache.posterior(vae, lambda i: load_images([entries[i]]), entries) if vae_cache is not None
                                     else vae.encode(pixels_cpu.to(device=vae_device, dtype=vae_dtype)).latent_dist)
                        latents = sample_vae_latents(posterior, vae.config.scaling_factor, config.device, config.vae_precision)
                if stage_unet:
                    del posterior
                    with phase("restore_unet"):
                        torch.cuda.empty_cache()
                        unet_stager.restore(config.device)
                with phase("prepare_diffusion", gpu=True):
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
                with phase("unet_forward", gpu=True), (torch.autocast(device_type=config.device, dtype=dtype_map[config.precision], cache_enabled=False) if config.precision != "fp32" else nullcontext()):
                    prediction = unet(noisy.to(prompt.dtype), timesteps, prompt, encoder_attention_mask=prompt_mask, added_cond_kwargs=added, return_dict=False)[0]
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
                with phase("backward", gpu=True):
                    (loss * loss_scale / config.gradient_accumulation_steps).backward()
                if stage_unet:
                    # The diffusion backward is complete and its scalar loss
                    # already recorded. Drop the GPU graph/output references
                    # before VAE training; the next microbatch creates a fresh
                    # graph, while accumulated gradients live in Disk AdamW.
                    del prediction, loss, loss_per_image, prompt, pooled, prompt_mask
                    del latents, noisy, noise, timesteps, target, added, time_ids
                    if config.snr_gamma > 0:
                        del alpha, snr, denominator
                # Separate VAE graph avoids retaining the full diffusion graph and its activations.
                if config.train_vae:
                    if stage_unet:
                        with phase("flush_gradients"):
                            optimizer.evict_accumulated_gradients()
                        with phase("offload_unet"):
                            unet_stager.park()
                            torch.cuda.empty_cache()
                    with phase("vae_training", gpu=vae_device.type == "cuda"):
                        pixels = pixels_cpu.to(device=vae_device, dtype=vae_dtype)
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
                        (vae_loss * loss_scale / config.gradient_accumulation_steps).backward()
                    if stage_unet:
                        # The completed VAE graph is no longer used; drop its
                        # output references before restoring the large UNet.
                        del pixels, reference, adapted_posterior, reconstruction, reconstruction_loss, kl, anchor, vae_loss
                        with phase("flush_gradients"):
                            optimizer.evict_accumulated_gradients()
                        with phase("restore_unet"):
                            torch.cuda.empty_cache()
                            unet_stager.restore(config.device)
            if stage_unet:
                del pixels_cpu
            else:
                del pixels_cpu, prediction, loss, prompt, pooled, prompt_mask, latents, noisy, noise
        factor = lr_factor(step, config.max_steps, config.warmup_steps, config.lr_scheduler)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * factor
        try:
            with phase("optimizer", gpu=True):
                # The optimizer may mutate parameters through raw .data/views,
                # so invalidate before it begins, including on partial failure.
                if unet_stager:
                    unet_stager.invalidate()
                if config.optimizer == "adamw_disk":
                    norm = optimizer.step(max_grad_norm=config.max_grad_norm, loss_scale=loss_scale)
                elif config.optimizer == "paged_adamw8bit":
                    from .paged_adamw import prepare_paged_gradients, check_paged_finite
                    norm = prepare_paged_gradients(all_parameters, loss_scale, config.max_grad_norm)
                    optimizer.step()
                    check_paged_finite(optimizer, all_parameters)
                else:
                    if loss_scale != 1:
                        for parameter in all_parameters:
                            if parameter.grad is not None:
                                parameter.grad.div_(loss_scale)
                    norm = torch.nn.utils.clip_grad_norm_(all_parameters, config.max_grad_norm if config.max_grad_norm else float("inf"), error_if_nonfinite=False).item()
                    if not math.isfinite(norm):
                        raise NonFiniteGradients("Non-finite gradients; no optimizer update was applied.")
                    optimizer.step()
                if unet_stager:
                    unet_stager.invalidate()
                optimizer.zero_grad(set_to_none=True)
        except NonFiniteGradients:
            optimizer.zero_grad(set_to_none=True)
            if scaler is None:
                raise
            scaler.overflow()
            epoch, batch_cursor = retry_position
            batches = bucket_batches(manifest["entries"], config.batch_size, config.seed, epoch)
            torch.set_rng_state(retry_torch)
            if retry_cuda is not None:
                torch.cuda.set_rng_state_all(retry_cuda)
            random.setstate(retry_python)
            np.random.set_state(retry_numpy)
            if _STOP:
                checkpoint = output / "checkpoints" / f"step-{step:08d}"
                if step and not checkpoint.exists():
                    save_checkpoint()
                emit("complete", stopped=True, step=step,
                     message=("Stopped after an fp16 overflow; the last completed step is saved." if step else
                              "Stopped after an fp16 overflow before the first optimizer update; no checkpoint was created."))
                return
            emit("status", message=f"FP16 gradient overflow; retrying the same images with loss scale {scaler.scale:g}.")
            continue
        if scaler is not None:
            scaler.success()
        step += 1
        metrics = telemetry.finish_step(image_count, step)
        emit("progress", step=step, total=config.max_steps, loss=accumulated_loss, vae_loss=accumulated_vae if config.train_vae else None, learning_rate=optimizer.param_groups[0]["lr"], grad_norm=norm, epoch=epoch, elapsed_seconds=round(time.monotonic() - training_started, 2))
        if getattr(config, "profile_phases", True):
            emit("telemetry", **metrics)
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
        emit("status", message="Saved lora_comfyui.safetensors and lora_diffusers.safetensors in adapters/.")
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
    # Save the selected model/VAE storage precision; fp32 optimizer masters
    # are resume state and are not silently substituted into the final model.
    pipeline.save_pretrained(str(temporary), safe_serialization=True, max_shard_size=config.final_export_shard_mib * 1024 * 1024)
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
