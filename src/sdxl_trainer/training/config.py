"""Serializable training controls. Importing this module never imports the ML runtime."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
import importlib.util
import json
import math
import re
import shutil


def knob(default, label, help, group="Training", **extra):
    return field(default=default, metadata={"label": label, "help": help, "group": group, **extra})


@dataclass
class TrainingConfig:
    model_path: str = knob("", "Base Diffusers model", "Local SDXL base pipeline folder. All model loads are local-only.", "Files", path="directory")
    dataset_path: str = knob("", "Finalized dataset", "Exported dataset folder containing a finalized manifest.json.", "Files", path="directory")
    output_dir: str = knob("", "Run output", "New, empty run folder. Existing runs require an explicit resume checkpoint.", "Files", path="directory")
    mode: str = knob("lora", "Training method", "LoRA adds trainable low-rank adapters. Full updates all selected weights.", choices=["lora", "full"])
    train_text_encoder: bool = knob(False, "Train both text encoders", "Update CLIP-L and CLIP-G. Both tokenizers remain frozen.")
    train_vae: bool = knob(False, "Train VAE", "Experimental: jointly adapt the VAE with reconstruction, KL and latent anchoring losses. Review reconstructions before use.")
    batch_size: int = knob(1, "Micro batch", "Images in one bucket per forward pass. Last bucket batch can be smaller.", min=1, max=64)
    gradient_accumulation_steps: int = knob(4, "Gradient accumulation", "Average this many micro batches before one optimizer update.", min=1, max=1024)
    max_steps: int = knob(1000, "Optimizer steps", "Total optimizer updates, including any resumed steps.", min=1, max=10000000)
    learning_rate: float = knob(0.0001, "UNet learning rate", "UNet optimizer learning rate; consider 1e-6 for a full fine-tune.", "Optimization", min=1e-9, max=1.0, decimals=9)
    text_encoder_lr: float = knob(0.000005, "Text encoder learning rate", "Separate learning rate shared by CLIP-L and CLIP-G.", "Optimization", min=1e-9, max=1.0, decimals=9)
    vae_lr: float = knob(0.000001, "VAE learning rate", "VAE updates can change the latent space; keep this conservative.", "Optimization", min=1e-9, max=1.0, decimals=9)
    optimizer: str = knob("adamw_disk", "Optimizer", "Disk AdamW spills gradients and fp32 moments to disk without quantization. Much slower, with far lower memory. Normal AdamW keeps state in device memory.", "Optimization", choices=["adamw_disk", "adamw", "sgd"])
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
    precision: str = knob("bf16", "Forward precision", "bf16 autocast with fp32 trainable weights and optimizer state, or entirely fp32. VAE always fp32. No fp16 loss scaling.", "Memory", choices=["bf16", "fp32"])
    device: str = knob("cuda", "Compute device", "CUDA for NVIDIA GPU. CPU is extremely slow but available as a memory fallback.", "Memory", choices=["cuda", "cpu"])
    gradient_checkpointing: bool = knob(True, "Gradient checkpointing", "Recompute activations in backward to reduce memory at a speed cost.", "Memory")
    cpu_offload_saved_tensors: bool = knob(True, "Offload saved activations", "Store backward tensors in system RAM. Reduces GPU memory; increases transfers and host RAM.", "Memory")
    cache_frozen_text: bool = knob(True, "Cache frozen text embeddings", "Cache one caption's embeddings on disk when both text encoders are frozen.", "Memory")
    snr_gamma: float = knob(0.0, "Min-SNR gamma", "Zero disables weighting. A typical experimental value is 5.", "Loss", min=0.0, max=1000.0)
    noise_offset: float = knob(0.0, "Noise offset", "Additional channel-wise Gaussian noise. Zero preserves the standard noise objective.", "Loss", min=0.0, max=1.0)
    vae_reconstruction_weight: float = knob(1.0, "VAE reconstruction weight", "Pixel L1 reconstruction objective, separate from detached diffusion latents.", "Loss", min=0.0, max=1000.0)
    vae_kl_weight: float = knob(0.000001, "VAE KL weight", "Posterior KL regularization averaged per image.", "Loss", min=0.0, max=1.0, decimals=9)
    vae_anchor_weight: float = knob(1.0, "VAE latent anchor weight", "Keep new encoder posterior means close to the frozen base VAE. Helps preserve the diffusion latent coordinate system.", "Loss", min=0.0, max=1000.0)
    seed: int = knob(42, "Seed", "Seed for sampler, noise, timesteps and initialization.", "Reproducibility", min=0, max=2147483647)
    save_every: int = knob(100, "Save interval", "Optimizer steps between resumable checkpoints.", "Files", min=1, max=10000000)
    checkpoints_to_keep: int = knob(2, "Checkpoints to keep", "Keep this many completed resumable checkpoints. Older checkpoints from this run are removed after a new one commits.", "Files", min=1, max=1000)
    resume_from: str = knob("", "Resume checkpoint", "Local checkpoint directory created by this app. Config and dataset identity must match.", "Files", path="directory")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Training configuration must be a JSON object.")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown training controls: {', '.join(sorted(unknown))}")
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
            match = re.fullmatch(r"step-(\d+)", checkpoint.name)
            if match:
                resumed_step = int(match[1])
                for other in (output / "checkpoints").glob("step-*"):
                    other_match = re.fullmatch(r"step-(\d+)", other.name)
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
    try:
        from ..dataset import verify_finalized
        verify_finalized(dataset)
    except Exception as exc:
        errors.append(f"Finalized dataset: {exc}")
    resolved_output = output.resolve()
    for source, label in ((model.resolve(), "base model"), (dataset.resolve(), "dataset")):
        if resolved_output == source or source in resolved_output.parents or resolved_output in source.parents:
            errors.append(f"Run output must be separate from the {label} folder.")
    errors.extend(output_safety_errors(config))
    # Conservative lower bounds, not a fit guarantee. SDXL base weights in fp32.
    trainable_gib = (10.0 + (3.0 if config.train_text_encoder else 0) + (0.32 if config.train_vae else 0)) if config.mode == "full" else 0.15 * max(1.0, config.rank / 16)
    weights_gib = 10.0 + (3.0 if config.train_text_encoder else 0) + (0.32 if config.train_vae else 0)
    state_gib = 0 if config.optimizer == "adamw_disk" else trainable_gib * (3 if config.optimizer == "adamw" else 1)
    minimum = weights_gib + state_gib + (1.5 if config.gradient_checkpointing else 5.0) * config.batch_size
    if config.mode == "lora":
        minimum = 10.0 + (3.0 if config.train_text_encoder else 0) + state_gib + (0.32 if config.train_vae else 0) + (1.5 if config.gradient_checkpointing else 5.0) * config.batch_size
    # Allow the new checkpoint to commit before pruning old ones. Each saved
    # checkpoint contains selected weights plus both Adam moments (3x weights).
    checkpoint_multiplier = 3 if config.optimizer.startswith("adamw") else 1
    working_multiplier = 3 if config.optimizer == "adamw_disk" else 0
    disk = trainable_gib * (working_multiplier + checkpoint_multiplier * (config.checkpoints_to_keep + 1)) + 16
    ancestor = output
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if ancestor.exists() and shutil.disk_usage(ancestor).free / 2**30 < disk:
        errors.append(f"At least about {disk:.1f} GiB free disk space is needed for working state, retained checkpoints and atomic saves.")
    warnings.append(f"Estimated device memory floor: {minimum:.1f} GiB, excluding allocator overhead and some activations. This is not a fit guarantee.")
    if config.mode == "full" and config.learning_rate > 0.00001:
        warnings.append("UNet learning rate is high for a full fine-tune. Review it carefully; 1e-6 is a conservative starting point.")
    if config.mode == "full":
        warnings.append("Full fine-tuning all fp32 weights is experimental on 16 GB. Disk AdamW and activation offload preserve precision but can be extremely slow.")
    if config.train_vae:
        warnings.append("Joint VAE adaptation is experimental and has no quality guarantee. This uses a frozen-base latent anchor, L1 reconstruction and KL; diffusion gradients do not update the VAE.")
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
                        errors.append(f"Only {free:.1f} GiB VRAM is free; the estimated floor is {minimum:.1f} GiB. Reduce batch/components, use disk AdamW, or select CPU.")
                    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
                        errors.append("Selected GPU does not support bf16. Choose fp32.")
    return PreflightResult(errors, warnings, round(minimum, 2), round(disk, 2))
