"""Serializable training controls. Importing this module never imports the ML runtime."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
import importlib.util
import json
import math
import re
import shutil

CPU_TEXT_OFFLOAD = frozenset({"text_encoders_cpu", "text_and_vae_cpu", "text_cpu_staged_unet"})
STAGED_UNET_OFFLOAD = frozenset({"text_cpu_staged_unet", "stage_unet_gpu_text"})
DEFAULT_UNET_LEARNING_RATES = {"lora": 1e-4, "full": 1e-6}
_CHECKPOINT_STEP = re.compile(r"step-([0-9]+)")

# Recipes/checkpoints written before the fast profile must retain their original
# numerical behavior even though new application defaults are faster.
LEGACY_TRAINING_FIELDS = {
    "model_precision": "fp32", "optimizer": "adamw_disk", "paged_cuda_override": "",
    "cpu_offload_saved_tensors": True, "vae_cache_ram_mib": 128,
    "caption_mode": "chunked", "frozen_vae_cache_mode": "posterior",
    "precompute_frozen_cache": False, "frozen_text_cache_ram_mib": 0,
    "allow_tf32": False, "tf32_policy": "legacy", "adamw_fused": False,
}


def fast_training_settings(mode="lora"):
    """Performance selections only: preserve paths, rates, duration and prompts."""
    return {
        "train_text_encoder": False, "train_vae": False,
        "optimizer": "paged_adamw8bit" if mode == "full" else "adamw",
        "paged_cuda_override": "130", "adamw_fused": True,
        "model_precision": "bf16", "precision": "bf16", "vae_precision": "fp32",
        "cpu_text_precision": "match", "device": "cuda", "component_offload": "none",
        "gradient_checkpointing": True, "cpu_offload_saved_tensors": False,
        "cache_frozen_text": True, "cache_frozen_vae": True,
        "frozen_vae_cache_mode": "sample", "precompute_frozen_cache": True,
        "frozen_text_cache_ram_mib": 4096, "vae_cache_ram_mib": 1024,
        "caption_mode": "chunked", "allow_tf32": True, "tf32_policy": "explicit",
    }


def knob(default, label, help, group="Training", **extra):
    return field(default=default, metadata={"label": label, "help": help, "group": group, **extra})


def _variant_weights_complete(component: Path, part: str, variant: str) -> bool:
    """Recognize actual Diffusers/Transformers variant files, including shards.

    Requiring an index for shards prevents an unrelated or partial file from
    making an explicitly selected variant appear available before a loader
    silently falls back to canonical weights.
    """
    prefix = "diffusion_pytorch_model" if part in {"unet", "vae"} else "model"
    if (component / f"{prefix}.{variant}.safetensors").is_file():
        return True
    # Both current and legacy Diffusers variant index conventions exist.
    indexes = (component / f"{prefix}.safetensors.index.{variant}.json",
               component / f"{prefix}.safetensors.{variant}.index.json")
    escaped_prefix, escaped_variant = re.escape(prefix), re.escape(variant)
    modern = re.compile(rf"{escaped_prefix}\.{escaped_variant}-(\d{{5}})-of-(\d{{5}})\.safetensors\Z")
    legacy = re.compile(rf"{escaped_prefix}-(\d{{5}})-of-(\d{{5}})\.{escaped_variant}\.safetensors\Z")
    for index in indexes:
        try:
            mapping = json.loads(index.read_text()).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                continue
            names = set(mapping.values())
            for pattern in (modern, legacy):
                matches = [pattern.fullmatch(name) if isinstance(name, str) else None for name in names]
                if not all(matches) or not all((component / name).is_file() for name in names):
                    continue
                numbers = [(int(match[1]), int(match[2])) for match in matches]
                total = numbers[0][1]
                if total and len(numbers) == total and all(count == total for _, count in numbers) and {number for number, _ in numbers} == set(range(1, total + 1)):
                    return True
        except (OSError, ValueError, TypeError):
            continue
    return False


@dataclass
class TrainingConfig:
    model_path: str = knob("", "Base Diffusers model", "Local SDXL base pipeline folder. All model loads are local-only.", "Files", path="directory")
    dataset_path: str = knob("", "Finalized dataset", "Select a finalized bucket, or select its export revision folder to train all of its aspect ratios in one run.", "Files", path="directory")
    output_dir: str = knob("", "Run output", "New, empty run folder. Existing runs require an explicit resume checkpoint.", "Files", path="directory")
    mode: str = knob("lora", "Training method", "Switch between saved LoRA and full fine-tune settings, including each method's run output and resume checkpoint. The model, file variant, and dataset are shared.", choices=["lora", "full"])
    train_text_encoder: bool = knob(False, "Train both text encoders", "Update CLIP-L and CLIP-G. Both tokenizers remain frozen.")
    train_vae: bool = knob(False, "Train VAE", "Experimental: jointly adapt the VAE with reconstruction, KL and latent anchoring losses. Review reconstructions before use.")
    batch_size: int = knob(1, "Micro batch", "Images in one bucket per forward pass. Last bucket batch can be smaller.", min=1, max=64)
    gradient_accumulation_steps: int = knob(4, "Gradient accumulation", "Average this many micro batches before one optimizer update.", min=1, max=1024)
    max_steps: int = knob(1000, "Optimizer steps", "Total optimizer updates, including any resumed steps.", min=1, max=10000000)
    learning_rate: float = knob(DEFAULT_UNET_LEARNING_RATES["lora"], "UNet learning rate", "UNet optimizer learning rate for the selected method.", "Optimization", min=1e-9, max=1.0, decimals=9)
    text_encoder_lr: float = knob(0.000005, "Text encoder learning rate", "Separate learning rate shared by CLIP-L and CLIP-G.", "Optimization", min=1e-9, max=1.0, decimals=9)
    vae_lr: float = knob(0.000001, "VAE learning rate", "VAE updates can change the latent space; keep this conservative.", "Optimization", min=1e-9, max=1.0, decimals=9)
    optimizer: str = knob("adamw", "Optimizer", "Fast LoRA uses fused GPU AdamW; fast full training uses paged 8-bit AdamW. Disk AdamW retains fp32 moments and masters. Paged 8-bit updates live weights directly without fp32 masters and requires all trainable components on one GPU. These numerical methods can produce different results.", "Optimization", choices=["adamw_disk", "paged_adamw8bit", "adamw", "sgd"])
    paged_cuda_override: str = knob("130", "8-bit CUDA binary override", "The local pinned fast runtime uses 130. Empty uses the PyTorch CUDA version. An override must pass the local native GPU probe and applies only to the isolated worker.", "Optimization")
    adamw_fused: bool = knob(True, "Fused GPU AdamW", "Use fused CUDA kernels for ordinary AdamW when all trainable weights are on one GPU. CPU or mixed placement uses ordinary AdamW. Ignored by Disk AdamW, paged AdamW and SGD; arithmetic can differ slightly.", "Optimization")
    optimizer_update_device: str = knob("auto", "Disk AdamW update device", "Auto selects CPU for fp32 masters of reduced-precision weights and otherwise may use the GPU. Select CUDA explicitly to stream fp32 master/moment updates on one GPU; this needs extra VRAM. Existing checkpoints retain their original update backend.", "Optimization", choices=["auto", "cuda", "cpu"])
    optimizer_gradient_storage: str = knob("auto", "Disk AdamW gradient storage", "Auto uses fp32 system RAM for accumulated gradients only when sufficient headroom is available, otherwise disk. Reduced-weight training with fp32 masters requires extra checkpoint/export headroom; RAM may be rejected even when explicitly selected. When the UNet is staged for VAE training, auto uses disk; RAM is unavailable. Optimizer moments remain disk-backed.", "Optimization", choices=["auto", "ram", "disk"])
    adam_beta1: float = knob(0.9, "Adam beta 1", "First moment decay.", "Optimization", min=0.0, max=0.99999)
    adam_beta2: float = knob(0.999, "Adam beta 2", "Second moment decay.", "Optimization", min=0.0, max=0.999999)
    adam_epsilon: float = knob(1e-8, "Adam epsilon", "Numerical stability constant.", "Optimization", min=1e-12, max=1.0, decimals=12)
    weight_decay: float = knob(0.01, "Weight decay", "Decoupled weight decay for AdamW; regular weight decay for SGD.", "Optimization", min=0.0, max=1.0)
    lr_scheduler: str = knob("constant", "Learning rate schedule", "Warmup then constant, linear decay, or cosine decay.", "Optimization", choices=["constant", "linear", "cosine"])
    warmup_steps: int = knob(100, "Warmup steps", "Ramp from zero to configured learning rates. Capped at total steps.", "Optimization", min=0, max=10000000)
    max_grad_norm: float = knob(1.0, "Gradient norm limit", "Global L2 gradient clipping; zero disables clipping.", "Optimization", min=0.0, max=10000.0)
    rank: int = knob(16, "LoRA rank", "Rank of attention adapters in each selected component.", "LoRA", min=1, max=1024)
    lora_alpha: int = knob(16, "LoRA alpha", "Adapter scaling numerator (effective scaling alpha / rank).", "LoRA", min=1, max=4096)
    lora_dropout: float = knob(0.0, "LoRA dropout", "Dropout probability on adapter inputs.", "LoRA", min=0.0, max=0.99)
    model_variant: str = knob("default", "Diffusers file variant", "Select the filenames in the chosen base model: default for ordinary files and app-generated outputs, or fp16/bf16 for matching variants. This choice does not change automatically when switching models and is independent of training weight and forward precision. No download occurs during training.", "Files", choices=["default", "fp16", "bf16"])
    model_precision: str = knob("bf16", "UNet and text weight precision", "Weight storage on the model. Fast defaults use bf16. Disk AdamW supplies fp32 masters for reduced full training; paged AdamW updates live weights directly. LoRA adapters stay fp32. Reduced weights change numerical precision.", "Memory", choices=["fp32", "bf16", "fp16"])
    cpu_text_precision: str = knob("match", "Trainable CPU text precision", "Match the selected model precision, or explicitly use fp32 for both trainable CPU text encoders. FP32 may use more RAM and may be faster; reduced-precision source files cannot regain lost precision.", "Memory", choices=["match", "fp32"])
    vae_precision: str = knob("fp32", "VAE weight and compute precision", "SDXL's original VAE can be numerically unstable in fp16. The recommended default is fp32; lower precision is an explicit experiment and can produce non-finite values.", "Memory", choices=["fp32", "bf16", "fp16"])
    precision: str = knob("bf16", "Forward precision", "Autocast UNet and text operations to bf16 or fp16, or use fp32. FP16 uses dynamic gradient loss scaling. VAE precision is selected separately.", "Memory", choices=["bf16", "fp16", "fp32"])
    loss_scale_initial: float = knob(4096.0, "Initial fp16 loss scale", "Initial dynamic loss scale for fp16 training. On overflow the same batch is retried at half the scale; scaler state is saved in checkpoints.", "Memory", min=1.0, max=16777216.0)
    loss_scale_growth_interval: int = knob(2000, "Loss scale growth interval", "Successful optimizer updates between loss-scale increases in fp16 mode.", "Memory", min=1, max=10000000)
    device: str = knob("cuda", "Compute device", "CUDA for NVIDIA GPU. CPU is extremely slow but available as a memory fallback.", "Memory", choices=["cuda", "cpu"])
    component_offload: str = knob("none", "Component offload", "Keep text encoders on CPU or GPU; optionally park the UNet in a temporary disk-backed mapping while the GPU runs the VAE. Trainable CPU text follows its separate precision setting. Mixed-device Disk AdamW uses CPU updates. Transfers can be extremely slow; image resolution and the selected weight precision and VAE objective are preserved.", "Memory", choices=["none", "text_encoders_cpu", "text_and_vae_cpu", "text_cpu_staged_unet", "stage_unet_gpu_text"])
    gradient_checkpointing: bool = knob(True, "Gradient checkpointing", "Recompute activations in backward to reduce memory at a speed cost.", "Memory")
    cpu_offload_saved_tensors: bool = knob(False, "Offload saved activations", "Store backward tensors in system RAM. Reduces GPU memory; increases transfers and host RAM. Fast defaults keep activations on GPU.", "Memory")
    allow_tf32: bool = knob(True, "Allow TF32 acceleration", "Allow NVIDIA TF32 matmul and cuDNN acceleration when forward precision is bf16 or fp16. FP32 forward always disables TF32. This can change numerical results; the worker restores its previous backend settings when finished.", "Memory")
    tf32_policy: str = knob("explicit", "TF32 compatibility policy", "Explicit applies the TF32 acceleration checkbox and disables TF32 for FP32 forward. Legacy preserves the runtime's historical matmul/cuDNN defaults for old recipes and checkpoints, ignoring the checkbox; old GPU kernels need not be bitwise identical across library versions.", "Memory", choices=["explicit", "legacy"])
    caption_mode: str = knob("chunked", "Training caption context", "Chunked uses every caption token in multiple 77-position windows and remains the fast default. Optional native mode uses only the first 75 content tokens per CLIP. Caption files and captioning prompts remain complete in both modes.", "Memory", choices=["chunked", "native"])
    cache_frozen_text: bool = knob(True, "Cache frozen text embeddings", "Cache one caption's embeddings on disk when both text encoders are frozen.", "Memory")
    cache_frozen_vae: bool = knob(True, "Cache frozen VAE", "Cache one sampled latent or posterior parameters, according to the sampling choice below. Bypassed when training the VAE. Fresh-posterior caching requires fp32 VAE compute; sampled-latent caching also supports explicitly selected reduced VAE precision.", "Memory")
    frozen_vae_cache_mode: str = knob("sample", "Frozen VAE cache sampling", "Sample caches one deterministic sampled latent per image, like Artificial-It, for speed. Posterior retains fp32 means/log-variances and draws a fresh latent each visit. Training the VAE bypasses both. Fixed samples change training randomness.", "Memory", choices=["sample", "posterior"])
    precompute_frozen_cache: bool = knob(True, "Prepare frozen caches before training", "Encode frozen images and captions on the GPU before placing trainable components there. Subsequent steps use cached tensors. CPU mode prepares on CPU. Disable for the previous lazy CPU encoding path. Trainable components are never cached.", "Memory")
    frozen_text_cache_ram_mib: int = knob(4096, "Frozen text cache RAM limit (MiB)", "Bounded CPU RAM for validated text embeddings. The fast default reserves at most 4 GiB for complete-caption conditioning; memory is allocated only as used. Zero reads the disk cache on every visit.", "Memory", min=0, max=8192)
    vae_cache_ram_mib: int = knob(1024, "Frozen VAE cache RAM limit (MiB)", "Bounded CPU RAM for cached sampled latents or posterior parameters. Zero keeps only disk caching. The sampling mode above determines whether latents are reused or sampled afresh.", "Memory", min=0, max=8192)
    profile_phases: bool = knob(True, "Show phase timings", "Measure data, VAE, text, UNet, and optimizer time for each step. GPU timing synchronizes once at the step boundary.", "Monitoring")
    monitor_gpu: bool = knob(True, "Monitor NVIDIA GPU", "Sample GPU utilization and memory locally during training. Disable if monitoring is unavailable or unnecessary.", "Monitoring")
    snr_gamma: float = knob(0.0, "Min-SNR gamma", "Zero disables weighting. A typical experimental value is 5.", "Loss", min=0.0, max=1000.0)
    noise_offset: float = knob(0.0, "Noise offset", "Additional channel-wise Gaussian noise. Zero preserves the standard noise objective.", "Loss", min=0.0, max=1.0)
    vae_reconstruction_weight: float = knob(1.0, "VAE reconstruction weight", "Pixel L1 reconstruction objective, separate from detached diffusion latents.", "Loss", min=0.0, max=1000.0)
    vae_kl_weight: float = knob(0.000001, "VAE KL weight", "Posterior KL regularization averaged per image.", "Loss", min=0.0, max=1.0, decimals=9)
    vae_anchor_weight: float = knob(1.0, "VAE latent anchor weight", "Keep new encoder posterior means close to the frozen base VAE. Helps preserve the diffusion latent coordinate system.", "Loss", min=0.0, max=1000.0)
    seed: int = knob(42, "Seed", "Seed for sampler, noise, timesteps and initialization.", "Reproducibility", min=0, max=2147483647)
    save_every: int = knob(100, "Save interval", "Optimizer steps between resumable checkpoints.", "Files", min=1, max=10000000)
    checkpoints_to_keep: int = knob(2, "Checkpoints to keep", "Keep this many completed resumable checkpoints. Older checkpoints from this run are removed after a new one commits.", "Files", min=1, max=1000)
    final_export_shard_mib: int = knob(512, "Final Diffusers shard size (MiB)", "Target maximum per-file shard size for the final Diffusers output. Smaller shards can reduce temporary serialization memory but create more files. An individual tensor larger than the target is saved whole and can exceed it. Does not change model weights or resumable checkpoints.", "Files", min=1, max=16384)
    resume_from: str = knob("", "Resume checkpoint", "Select a run folder for its latest completed checkpoint, or select a specific checkpoint. The run settings load automatically and the output folder remains editable.", "Files", path="directory")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Training configuration must be a JSON object.")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown training controls: {', '.join(sorted(unknown))}")
        value = {**{key: val for key, val in LEGACY_TRAINING_FIELDS.items() if key not in value}, **value}
        defaults = cls()
        for name, val in value.items():
            expected = type(getattr(defaults, name))
            if expected is float:
                good = isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)
            else:
                good = isinstance(val, expected) and (expected is bool or not isinstance(val, bool))
            if not good:
                raise ValueError(f"Invalid type for {name}: expected {expected.__name__}.")
        return cls(**value)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def validate(self):
        errors = []
        for name, f in self.__dataclass_fields__.items():
            value = getattr(self, name)
            m = f.metadata
            if "choices" in m and value not in m["choices"]:
                errors.append(f"{m['label']}: choose one of {', '.join(m['choices'])}.")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value) or ("min" in m and value < m["min"]) or ("max" in m and value > m["max"]):
                    errors.append(f"{m['label']}: outside the supported range.")
        for name in ("model_path", "dataset_path", "output_dir"):
            if not getattr(self, name).strip():
                errors.append(f"{self.__dataclass_fields__[name].metadata['label']} is required.")
        if self.train_vae and self.vae_reconstruction_weight <= 0:
            errors.append("VAE training requires a positive reconstruction weight.")
        if self.train_vae and self.vae_anchor_weight <= 0:
            errors.append("VAE training requires a positive latent anchor weight.")
        if self.model_precision != "fp32" and self.precision != self.model_precision:
            errors.append("Reduced-precision UNet and text weights require the matching forward precision.")
        if self.cpu_text_precision == "fp32" and not (self.train_text_encoder and (self.device == "cpu" or self.component_offload in CPU_TEXT_OFFLOAD)):
            errors.append("FP32 CPU text precision requires both text encoders to be trainable and placed on CPU.")
        if self.mode == "full" and (self.model_precision != "fp32" or (self.train_vae and self.vae_precision != "fp32")) and self.optimizer not in {"adamw_disk", "paged_adamw8bit"}:
            errors.append("Full training with reduced-precision weights requires Disk AdamW and its fp32 master weights.")
        if self.paged_cuda_override and (not self.paged_cuda_override.isascii() or not self.paged_cuda_override.isdigit() or len(self.paged_cuda_override) not in (3,4)):
            errors.append("8-bit CUDA binary override must be empty or a version such as 130.")
        if self.optimizer == "paged_adamw8bit":
            if self.device != "cuda":
                errors.append("Paged 8-bit AdamW requires CUDA and all trainable weights on one GPU.")
            if (self.train_text_encoder and self.component_offload in CPU_TEXT_OFFLOAD) or (self.train_vae and (self.component_offload == "text_and_vae_cpu" or self.component_offload in STAGED_UNET_OFFLOAD)):
                errors.append("Paged 8-bit AdamW cannot use CPU-offloaded trainable components or staged UNet training; select no offload or Disk AdamW.")
        if self.optimizer == "adamw_disk" and self.optimizer_update_device == "cuda" and self.device != "cuda":
            errors.append("CUDA Disk AdamW updates require the CUDA compute device.")
        if self.device == "cpu" and self.precision == "fp16":
            errors.append("FP16 autocast training requires CUDA; choose bf16 or fp32 for CPU training.")
        if self.component_offload != "none" and self.device != "cuda":
            errors.append("Component offload requires the CUDA compute device; CPU mode already runs all components on CPU.")
        mixed = (self.train_text_encoder and self.component_offload in CPU_TEXT_OFFLOAD) or (self.train_vae and self.component_offload == "text_and_vae_cpu")
        if mixed and self.optimizer == "adamw_disk" and self.optimizer_update_device == "cuda":
            errors.append("Offloaded trainable components require Disk AdamW updates on auto or CPU, not CUDA.")
        if self.component_offload in STAGED_UNET_OFFLOAD and self.train_vae:
            if self.optimizer != "adamw_disk":
                errors.append("Staged UNet offload with VAE training requires Disk AdamW for bounded gradient memory.")
            if self.optimizer_gradient_storage == "ram":
                errors.append("Staged UNet offload with VAE training requires disk or auto gradient storage, not RAM.")
        return errors


@dataclass
class PreflightResult:
    errors: list[str]
    warnings: list[str]
    estimated_minimum_gib: float
    estimated_disk_gib: float

    @property
    def ok(self):
        return not self.errors


def checkpoint_run_folder(resume_from: str) -> Path | None:
    """Find the run folder from a standard checkpoint path."""
    checkpoint = Path(resume_from)
    if checkpoint.parent.name == "checkpoints" and _CHECKPOINT_STEP.fullmatch(checkpoint.name):
        return checkpoint.parent.parent
    return None


def has_checkpoint_metadata(checkpoint: Path) -> bool:
    metadata = checkpoint / "checkpoint.json"
    return not checkpoint.is_symlink() and not metadata.is_symlink() and metadata.is_file()


def run_checkpoints(run: Path) -> list[tuple[int, Path]]:
    """List completed checkpoints in descending optimizer step order."""
    checkpoints = run / "checkpoints"
    if checkpoints.is_symlink() or not checkpoints.is_dir():
        return []
    completed = []
    for candidate in checkpoints.iterdir():
        match = _CHECKPOINT_STEP.fullmatch(candidate.name)
        if match is not None and has_checkpoint_metadata(candidate):
            completed.append((int(match[1]), candidate))
    return sorted(completed, reverse=True)


def output_safety_errors(config: TrainingConfig):
    """Read-only protection for run folders and existing resume destinations."""
    errors = []
    output = Path(config.output_dir).expanduser()
    if output.is_symlink():
        errors.append("Run output must be a real directory, not a symbolic link.")
        return errors
    if output.exists() and not output.is_dir():
        return ["Run output must be a directory."]
    managed_directories = ("working", "checkpoints", "diffusers.tmp", "diffusers", "adapters")
    for name in (*managed_directories, "training-config.json"):
        child = output / name
        if child.is_symlink():
            errors.append(f"Run path must not be a symbolic link: {child}")
        elif child.exists():
            if name in managed_directories and not child.is_dir():
                errors.append(f"Managed run path must be a directory: {child}")
            elif name == "training-config.json" and not child.is_file():
                errors.append("training-config.json must be a regular file.")
            elif child.is_dir():
                for nested in child.rglob("*"):
                    if nested.is_symlink():
                        errors.append(f"Managed run files must not contain symbolic links: {nested}")
                        break
    nonempty = output.is_dir() and any(output.iterdir())
    if nonempty and not config.resume_from:
        errors.append("Run output is not empty. Choose a new folder or explicitly resume a checkpoint.")
    if config.resume_from:
        checkpoint = Path(config.resume_from).expanduser()
        if checkpoint.is_symlink() or (checkpoint / "checkpoint.json").is_symlink():
            errors.append("Resume checkpoint and its metadata must not be symbolic links.")
        if not (checkpoint / "checkpoint.json").is_file():
            errors.append("Resume folder is missing checkpoint.json.")
        if nonempty:
            if checkpoint.parent.name != "checkpoints" or checkpoint.parent.parent.resolve() != output.resolve():
                errors.append("A nonempty output can only resume a checkpoint owned by that same run. Choose the checkpoint's run folder or a new empty output folder.")
            if not (output / "training-config.json").is_file():
                errors.append("Existing resume output is missing its owning training-config.json.")
            if (output / "diffusers").exists():
                errors.append("This run already contains a completed Diffusers model. Resume into a new empty output folder.")
            match = _CHECKPOINT_STEP.fullmatch(checkpoint.name)
            if match:
                resumed_step = int(match[1])
                for other in (output / "checkpoints").glob("step-*"):
                    other_match = _CHECKPOINT_STEP.fullmatch(other.name)
                    if other_match and int(other_match[1]) > resumed_step:
                        errors.append("A newer checkpoint exists in this run. Resume the latest checkpoint or branch into a new empty output folder.")
                        break
    return errors


def preflight(config: TrainingConfig, check_runtime=False):
    errors = config.validate()
    warnings = []
    model = Path(config.model_path).expanduser()
    dataset = Path(config.dataset_path).expanduser()
    output = Path(config.output_dir).expanduser()
    for part in ("unet", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler"):
        if not (model / part).is_dir():
            errors.append(f"Base model is missing {part}/.")
    if not (model / "model_index.json").is_file():
        errors.append("Base model is missing model_index.json.")
    if config.model_variant != "default":
        for part in ("unet", "vae", "text_encoder", "text_encoder_2"):
            component = model / part
            if component.is_dir() and not _variant_weights_complete(component, part, config.model_variant):
                errors.append(f"Selected {config.model_variant} variant is missing safetensors weights in {part}/. Download that component's variant or choose the correct files. App-generated model outputs use the default file variant.")
    try:
        from ..dataset import load_training_dataset
        load_training_dataset(dataset)
    except Exception as exc:
        errors.append(f"Finalized dataset: {exc}")
    resolved_output = output.resolve()
    for source, label in ((model.resolve(), "base model"), (dataset.resolve(), "dataset")):
        if resolved_output == source or source in resolved_output.parents or resolved_output in source.parents:
            errors.append(f"Run output must be separate from the {label} folder.")
    errors.extend(output_safety_errors(config))
    # Conservative lower bounds, not a fit guarantee. SDXL base weights in fp32.
    trainable_gib = (10.0 + (3.0 if config.train_text_encoder else 0) + (0.32 if config.train_vae else 0)) if config.mode == "full" else 0.15 * max(1.0, config.rank / 16)
    gpu_text = config.train_text_encoder and config.component_offload not in CPU_TEXT_OFFLOAD
    gpu_vae = config.train_vae and config.component_offload != "text_and_vae_cpu"
    model_factor = 1.0 if config.model_precision == "fp32" else 0.5
    vae_factor = 1.0 if config.vae_precision == "fp32" else 0.5
    weights_gib = 10.0 * model_factor + (3.0 * model_factor if gpu_text else 0) + (0.32 * vae_factor if gpu_vae else 0)
    state_gib = 0 if config.optimizer == "adamw_disk" else trainable_gib * ((model_factor if config.mode == "full" else 1.0) if config.optimizer == "paged_adamw8bit" else (3 if config.optimizer == "adamw" else 1))
    minimum = weights_gib + state_gib + (1.5 if config.gradient_checkpointing else 5.0) * config.batch_size
    if config.mode == "lora":
        minimum = weights_gib + state_gib + (1.5 if config.gradient_checkpointing else 5.0) * config.batch_size
    # Allow the new checkpoint to commit before pruning old ones. Each saved
    # checkpoint contains selected weights plus both Adam moments (3x weights).
    master_weights = config.mode == "full" and (config.model_precision != "fp32" or (config.train_vae and config.vae_precision != "fp32"))
    checkpoint_multiplier = (3 + int(master_weights)) if config.optimizer.startswith("adamw") else 1
    working_multiplier = (3 + int(master_weights)) if config.optimizer == "adamw_disk" else 0
    disk = trainable_gib * (working_multiplier + checkpoint_multiplier * (config.checkpoints_to_keep + 1)) + 16
    if config.optimizer == "paged_adamw8bit":
        disk += trainable_gib * 0.5 * (config.checkpoints_to_keep + 1)  # approximate quantized moments
    if config.component_offload in STAGED_UNET_OFFLOAD and config.train_vae:
        disk += 10.0 * model_factor  # Anonymous disk-backed UNet staging, separate from optimizer state.
    ancestor = output
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if ancestor.exists() and shutil.disk_usage(ancestor).free / 2**30 < disk:
        errors.append(f"At least about {disk:.1f} GiB free disk space is needed for working state, retained checkpoints and atomic saves.")
    warnings.append(f"Estimated device memory floor: {minimum:.1f} GiB, excluding allocator overhead and some activations. This is not a fit guarantee.")
    if config.mode == "full" and config.learning_rate > 0.00001:
        warnings.append("UNet learning rate is high for a full fine-tune. Review it carefully; 1e-6 is a conservative starting point.")
    if config.mode == "full":
        warnings.append("Full fine-tuning is experimental on 16 GB. Disk AdamW and activation offload can be extremely slow.")
    if config.model_variant in ("fp16", "bf16"):
        warnings.append(f"The selected {config.model_variant} file variant may contain already-rounded source weights, including the VAE. Loading them as fp32 cannot recover their original precision; file variant and training precision are separate settings.")
    if config.optimizer == "paged_adamw8bit":
        warnings.append("Paged 8-bit AdamW quantizes optimizer moments and updates live weights without fp32 masters; bf16/fp16 gradient unscaling and small updates may round or underflow. It keeps gradients on GPU and paging is not a VRAM or RAM fit guarantee.")
    if config.cpu_text_precision == "fp32" and config.model_precision != "fp32":
        warnings.append("FP32 CPU text encoders can need about 1.5 GiB more system RAM than 16-bit text weights. Monitor RAM and swap; speed depends on the CPU and workload.")
    if config.model_precision != "fp32" or config.vae_precision != "fp32":
        warnings.append("Reduced-precision weights/compute change numerical behavior; fp32 optimizer masters do not restore precision lost in the model. Verify quality and stability.")
    if config.vae_precision == "fp16":
        warnings.append("The SDXL VAE can overflow in fp16. Select fp32 VAE precision if losses or latents are non-finite.")
    if config.train_vae:
        warnings.append("Joint VAE adaptation is experimental and has no quality guarantee. This uses a frozen-base latent anchor, L1 reconstruction and KL; diffusion gradients do not update the VAE.")
    if config.component_offload in CPU_TEXT_OFFLOAD:
        warnings.append("Offloaded trainable components use CPU computation, increasing system RAM use and training time. Monitor system RAM and swap; memory estimates are not a fit guarantee.")
    elif config.component_offload in STAGED_UNET_OFFLOAD and config.train_vae:
        warnings.append("Staged UNet weights use temporary disk space and can increase training time. Monitor VRAM, RAM, and swap; memory estimates are not a fit guarantee.")
    if config.device == "cpu":
        warnings.append("CPU training can be orders of magnitude slower. System RAM and swap still limit the run.")
    if config.precision == "bf16" and config.device == "cpu":
        warnings.append("CPU bf16 depends on the processor; fp32 is the safest compatibility setting.")
    if check_runtime:
        try:
            meminfo = dict((line.split(":", 1)[0], line.split(":", 1)[1].strip()) for line in Path("/proc/meminfo").read_text().splitlines())
            available_ram = float(meminfo["MemAvailable"].split()[0]) / 2**20
            if config.device == "cpu" and available_ram < minimum + 2.0:
                errors.append(f"Only {available_ram:.1f} GiB system RAM is available; the estimated CPU floor including overhead is {minimum + 2:.1f} GiB.")
            elif config.cpu_offload_saved_tensors and available_ram < 8.0:
                warnings.append(f"Only {available_ram:.1f} GiB system RAM is available for frozen components and offloaded activations. Close other applications before training.")
        except (OSError, KeyError, ValueError):
            pass
        for dependency in ("torch", "diffusers", "transformers", "peft", "safetensors", "numpy", "PIL"):
            if importlib.util.find_spec(dependency) is None:
                errors.append(f"Training runtime is missing {dependency}.")
        if not errors:
            import torch
            if config.device == "cuda":
                if not torch.cuda.is_available():
                    errors.append("CUDA is unavailable in the selected Python runtime.")
                else:
                    major, minor = torch.cuda.get_device_capability()
                    supported = torch.cuda.get_arch_list()
                    if major >= 12 and not any(x.startswith("sm_120") or x.startswith("compute_120") for x in supported):
                        warnings.append("Verify this PyTorch CUDA build supports Blackwell sm_120; CUDA 12.8+ is required.")
                    free = torch.cuda.mem_get_info()[0] / 2**30
                    if free < minimum:
                        message = f"Only {free:.1f} GiB VRAM is free; the estimated floor is {minimum:.1f} GiB. Reduce batch/components, use disk AdamW, or select CPU."
                        if config.model_precision != "fp32" or config.vae_precision != "fp32":
                            warnings.append(message + " Reduced-precision estimates are approximate; the run may still fail.")
                        else:
                            errors.append(message)
                    if (config.precision == "bf16" or config.model_precision == "bf16" or config.vae_precision == "bf16") and not torch.cuda.is_bf16_supported():
                        errors.append("Selected GPU does not support bf16. Choose fp32.")
                    if config.optimizer == "paged_adamw8bit" and not errors:
                        try:
                            from .paged_adamw import load_paged_runtime, probe_paged_runtime
                            bnb, _ = load_paged_runtime(config.paged_cuda_override)
                            probe_paged_runtime(bnb, config.model_precision if config.mode == "full" else "fp32")
                        except (RuntimeError, ValueError) as exc:
                            errors.append(str(exc))
    return PreflightResult(errors, warnings, round(minimum, 2), round(disk, 2))
