"""Frozen training tensors with validated disk storage and bounded CPU caching.

These caches are disposable run-local performance artifacts. They never modify
source images or captions, and callers must bypass them for trainable encoders.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile


_FORMAT = "sdxl-frozen-tensors-v1"
_PRECISIONS = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}


def _precision(value):
    name = str(value).removeprefix("torch.")
    aliases = {**{key: key for key in _PRECISIONS},
               **{dtype: key for key, dtype in _PRECISIONS.items()}}
    if name not in aliases:
        raise ValueError("Frozen caches support fp32, bf16, or fp16 tensors.")
    return aliases[name]


def _torch_dtype(precision):
    import torch
    return getattr(torch, _PRECISIONS[_precision(precision)])


def _key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _tensor_digest(tensors):
    """Hash exact CPU tensor bytes, including BF16 without NumPy conversion."""
    import torch
    result = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        result.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                                 separators=(",", ":")).encode())
        result.update(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


class _TensorCache:
    def __init__(self, directory, memory_limit_bytes, validate):
        if type(memory_limit_bytes) is not int or memory_limit_bytes < 0:
            raise ValueError("Frozen cache RAM limit must be a nonnegative integer.")
        self.directory = Path(directory)
        if self.directory.is_symlink() or self.directory.parent.is_symlink():
            raise ValueError("Frozen cache directories must not be symbolic links.")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.memory_limit_bytes = memory_limit_bytes
        self._validate = validate
        self._memory = OrderedDict()
        self._memory_bytes = 0

    @property
    def memory_bytes(self):
        return self._memory_bytes

    @property
    def memory_entries(self):
        return len(self._memory)

    def clear_memory(self):
        """Release CPU LRU tensors while retaining every validated disk entry."""
        self._memory.clear()
        self._memory_bytes = 0

    def path_for_key(self, key):
        return self.directory / f"{key}.safetensors"

    @staticmethod
    def _signature(path):
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Frozen cache is not a regular file: {path}")
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def _forget(self, key):
        old = self._memory.pop(key, None)
        if old is not None:
            self._memory_bytes -= old[2]

    @staticmethod
    def _copy(tensors):
        # CPU callers must not be able to mutate the protected cached copy.
        return {name: tensor.clone() for name, tensor in tensors.items()}

    def lookup(self, key, expected):
        path = self.path_for_key(key)
        signature = self._signature(path)
        if signature is None:
            self._forget(key)
            return None
        item = self._memory.get(key)
        if item is not None and item[0] == signature:
            self._memory.move_to_end(key)
            return self._copy(item[1])
        self._forget(key)
        try:
            from safetensors import safe_open
            with safe_open(str(path), framework="pt", device="cpu") as saved:
                metadata = saved.metadata() or {}
                if metadata.get("format") != _FORMAT or metadata.get("key") != key:
                    raise ValueError("Cache identity does not match.")
                tensors = {name: saved.get_tensor(name) for name in saved.keys()}
            self._validate(tensors, expected)
            if metadata.get("sha256") != _tensor_digest(tensors):
                raise ValueError("Cached tensor integrity check failed.")
            if signature != self._signature(path):
                raise ValueError("Cache changed while being read.")
        except Exception as exc:
            raise ValueError(f"Invalid frozen cache {path}. Remove this cache file and retry: {exc}") from exc
        size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        if size <= self.memory_limit_bytes:
            while self._memory and self._memory_bytes + size > self.memory_limit_bytes:
                _, old = self._memory.popitem(last=False)
                self._memory_bytes -= old[2]
            # safe_open tensors may alias the file: cache independent CPU copies.
            protected = self._copy(tensors)
            self._memory[key] = (signature, protected, size)
            self._memory_bytes += size
            return self._copy(protected)
        return self._copy(tensors)

    def publish(self, key, tensors, expected):
        from safetensors.torch import save_file
        self._validate(tensors, expected)
        descriptor, temporary = tempfile.mkstemp(prefix=".frozen-", suffix=".tmp", dir=self.directory)
        os.close(descriptor)
        try:
            save_file(tensors, temporary,
                      metadata={"format": _FORMAT, "key": key, "sha256": _tensor_digest(tensors)})
            try:
                os.link(temporary, self.path_for_key(key))
            except FileExistsError:
                # The winning publisher is checked below; never overwrite it.
                pass
        finally:
            Path(temporary).unlink(missing_ok=True)
        return self.lookup(key, expected)


class FrozenSampleCache:
    """One deterministic scaled latent per image, stored in model precision.

    ``posterior_factory`` is called without gradients on a cache miss and must
    return a VAE posterior with deterministic ``mean`` and ``logvar`` tensors.
    Sampling uses a dedicated per-image generator and FP32 arithmetic, without
    consuming global training RNG. Cached results are independent CPU tensors.
    """

    def __init__(self, directory, model_fingerprint, *, seed, vae_precision,
                 latent_channels, scale_factor, scaling_factor, output_dtype,
                 memory_limit_bytes=1024 * 1024 * 1024):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Frozen latent seed must be a nonnegative 63-bit integer.")
        if any(type(value) is not int or value <= 0 for value in (latent_channels, scale_factor)):
            raise ValueError("Latent channels and image scale factor must be positive integers.")
        if not isinstance(scaling_factor, (int, float)) or isinstance(scaling_factor, bool) or not math.isfinite(scaling_factor) or scaling_factor <= 0:
            raise ValueError("VAE scaling factor must be finite and positive.")
        self.model_fingerprint = str(model_fingerprint)
        self.seed = seed
        self.vae_precision = _precision(vae_precision)
        self.latent_channels = latent_channels
        self.scale_factor = scale_factor
        self.scaling_factor = float(scaling_factor)
        self.output_precision = _precision(output_dtype)
        self._cache = _TensorCache(directory, memory_limit_bytes, self._validate)

    @property
    def directory(self):
        return self._cache.directory

    @property
    def memory_bytes(self):
        return self._cache.memory_bytes

    @property
    def memory_entries(self):
        return self._cache.memory_entries

    def clear_memory(self):
        self._cache.clear_memory()

    def _identity(self, entry):
        image_hash = entry.get("image_sha256")
        if not isinstance(image_hash, str) or len(image_hash) != 64 or any(c not in "0123456789abcdef" for c in image_hash):
            raise ValueError("Finalized image must have a valid SHA256 content hash.")
        width, height = entry.get("width"), entry.get("height")
        if any(type(value) is not int or value <= 0 or value % self.scale_factor for value in (width, height)):
            raise ValueError("Finalized image dimensions must be positive multiples of the VAE scale factor.")
        shape = (1, self.latent_channels, height // self.scale_factor, width // self.scale_factor)
        key = _key({"version": 1, "mode": "fixed-fp32-sampled-latent", "image": image_hash,
                    "model": self.model_fingerprint, "seed": self.seed,
                    "width": width, "height": height, "channels": self.latent_channels,
                    "scale_factor": self.scale_factor, "scaling_factor": self.scaling_factor,
                    "vae_precision": self.vae_precision, "output_precision": self.output_precision})
        return key, shape

    def path_for(self, entry):
        return self._cache.path_for_key(self._identity(entry)[0])

    def _validate(self, tensors, expected_shape):
        import torch
        if set(tensors) != {"latent"}:
            raise ValueError("A frozen latent cache must contain exactly one latent tensor.")
        latent = tensors["latent"]
        if latent.device.type != "cpu" or tuple(latent.shape) != expected_shape or latent.dtype != _torch_dtype(self.output_precision):
            raise ValueError("Cached latent has an invalid shape, device, or dtype.")
        if not torch.isfinite(latent).all().item():
            raise ValueError("Cached latent contains non-finite values.")

    def lookup(self, entry):
        key, shape = self._identity(entry)
        result = self._cache.lookup(key, shape)
        return result["latent"] if result is not None else None

    def get(self, entry, posterior_factory):
        import torch
        key, shape = self._identity(entry)
        result = self._cache.lookup(key, shape)
        if result is not None:
            return result["latent"]
        with torch.no_grad():
            posterior = posterior_factory()
            mean, logvar = posterior.mean, posterior.logvar
            if not isinstance(mean, torch.Tensor) or not isinstance(logvar, torch.Tensor):
                raise ValueError("VAE posterior must provide mean and log-variance tensors.")
            if tuple(mean.shape) != shape or tuple(logvar.shape) != shape or mean.device != logvar.device:
                raise ValueError("VAE posterior shape/device does not match the finalized image bucket.")
            if not mean.is_floating_point() or not logvar.is_floating_point():
                raise ValueError("VAE posterior parameters must be floating-point tensors.")
            if not (torch.isfinite(mean).all() & torch.isfinite(logvar).all()).item():
                raise ValueError(f"Non-finite VAE posterior values (VAE precision: {self.vae_precision}).")
            # Generator creation and draws do not touch global CPU/CUDA RNG.
            generator = torch.Generator(device=mean.device)
            generator.manual_seed(int(hashlib.sha256((key + ":noise").encode()).hexdigest()[:16], 16) % (2**63 - 1))
            if getattr(posterior, "deterministic", False):
                sampled = mean.float()
            else:
                noise = torch.randn(shape, dtype=torch.float32, device=mean.device, generator=generator)
                sampled = mean.float() + torch.exp(0.5 * logvar.float()) * noise
            latent = (sampled * self.scaling_factor).to(device="cpu", dtype=_torch_dtype(self.output_precision)).contiguous()
        result = self._cache.publish(key, {"latent": latent}, shape)
        return result["latent"]


class FrozenTextCache:
    """Frozen SDXL conditioning keyed by caption and its complete encode policy."""

    def __init__(self, directory, model_fingerprint, *, caption_mode, precision,
                 encoding_policy, memory_limit_bytes=1024 * 1024 * 1024,
                 prompt_width=2048, pooled_width=1280):
        if caption_mode not in {"native", "chunked"}:
            raise ValueError("Caption mode must be native or chunked.")
        if not isinstance(encoding_policy, str) or not encoding_policy:
            raise ValueError("Frozen text encoding policy must be a nonempty string.")
        if any(type(value) is not int or value <= 0 for value in (prompt_width, pooled_width)):
            raise ValueError("SDXL text embedding widths must be positive integers.")
        self.model_fingerprint = str(model_fingerprint)
        self.caption_mode = caption_mode
        self.precision = _precision(precision)
        self.encoding_policy = encoding_policy
        self.prompt_width = prompt_width
        self.pooled_width = pooled_width
        self._cache = _TensorCache(directory, memory_limit_bytes, self._validate)

    @property
    def directory(self):
        return self._cache.directory

    @property
    def memory_bytes(self):
        return self._cache.memory_bytes

    @property
    def memory_entries(self):
        return self._cache.memory_entries

    def clear_memory(self):
        self._cache.clear_memory()

    def _identity(self, caption):
        if not isinstance(caption, str):
            raise ValueError("A frozen caption must be text.")
        return _key({"version": 1, "mode": "frozen-text", "model": self.model_fingerprint,
                     "caption": caption, "caption_mode": self.caption_mode,
                     "precision": self.precision, "encoding_policy": self.encoding_policy,
                     "prompt_width": self.prompt_width, "pooled_width": self.pooled_width})

    def path_for(self, caption):
        return self._cache.path_for_key(self._identity(caption))

    def _validate(self, tensors, _expected):
        import torch
        if set(tensors) != {"prompt", "pooled"}:
            raise ValueError("Frozen text cache must contain prompt and pooled tensors.")
        prompt, pooled = tensors["prompt"], tensors["pooled"]
        if (prompt.ndim != 3 or prompt.shape[0] != 1 or prompt.shape[2] != self.prompt_width
                or prompt.shape[1] <= 0 or prompt.shape[1] % 77
                or (self.caption_mode == "native" and prompt.shape[1] != 77)
                or tuple(pooled.shape) != (1, self.pooled_width)):
            raise ValueError("Frozen text embedding shape does not match the SDXL caption policy.")
        if any(tensor.device.type != "cpu" or tensor.dtype != _torch_dtype(self.precision) for tensor in tensors.values()):
            raise ValueError("Frozen text tensors have an invalid device or dtype.")
        if not all(torch.isfinite(tensor).all().item() for tensor in tensors.values()):
            raise ValueError("Frozen text embeddings contain non-finite values.")

    def lookup(self, caption):
        result = self._cache.lookup(self._identity(caption), None)
        return (result["prompt"], result["pooled"]) if result is not None else None

    def get(self, caption, encode_factory):
        import torch
        key = self._identity(caption)
        result = self._cache.lookup(key, None)
        if result is not None:
            return result["prompt"], result["pooled"]
        with torch.no_grad():
            encoded = encode_factory()
            if not isinstance(encoded, (tuple, list)) or len(encoded) != 2:
                raise ValueError("Frozen text encoder must return prompt and pooled tensors.")
            if not all(isinstance(tensor, torch.Tensor) for tensor in encoded):
                raise ValueError("Frozen text encoder must return prompt and pooled tensors.")
            tensors = {name: tensor.detach().to(device="cpu").contiguous()
                       for name, tensor in zip(("prompt", "pooled"), encoded)}
        result = self._cache.publish(key, tensors, None)
        return result["prompt"], result["pooled"]
