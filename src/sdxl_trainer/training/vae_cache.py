"""Cache frozen VAE posterior parameters, never sampled training latents."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from collections import OrderedDict


def make_frozen_vae_cache(config, output: Path, model_fingerprint: str, vae):
    """Never cache a trainable VAE, whose posterior changes after each update."""
    if config.train_vae or not getattr(config, "cache_frozen_vae", False) or getattr(config, "vae_precision", "fp32") != "fp32":
        return None
    return FrozenVaeCache(output / "working" / "vae", model_fingerprint,
                          int(vae.config.latent_channels),
                          2 ** (len(vae.config.block_out_channels) - 1),
                          memory_limit_bytes=getattr(config, "vae_cache_ram_mib", 128) * 1024 * 1024)


class FrozenVaeCache:
    """Store exact fp32 posteriors per image and base model under a run folder.

    Reconstructing the distribution preserves fresh random sampling for every
    training visit. Cache files are optional performance artifacts, never model
    or optimizer state. Existing invalid files fail closed rather than being
    silently trusted or overwritten.
    """

    def __init__(self, directory: str | Path, model_fingerprint: str, latent_channels: int, scale_factor: int,
                 memory_limit_bytes: int = 128 * 1024 * 1024):
        self.directory = Path(directory)
        self.model_fingerprint = model_fingerprint
        self.latent_channels = latent_channels
        self.scale_factor = scale_factor
        if type(memory_limit_bytes) is not int or memory_limit_bytes < 0:
            raise ValueError("Frozen VAE RAM cache limit must be a nonnegative integer.")
        self.memory_limit_bytes = memory_limit_bytes
        self._memory = OrderedDict()
        self._memory_bytes = 0
        if self.directory.is_symlink() or any(p.is_symlink() for p in (self.directory.parent,)):
            raise ValueError("Frozen VAE cache folders must not be symbolic links.")
        self.directory.mkdir(parents=True, exist_ok=True)

    def clear_memory(self):
        """Free cached CPU posterior copies without changing persistent entries."""
        self._memory.clear()
        self._memory_bytes = 0

    def _key(self, entry):
        image_hash = entry.get("image_sha256")
        if not isinstance(image_hash, str) or len(image_hash) != 64 or any(c not in "0123456789abcdef" for c in image_hash):
            raise ValueError("Finalized image has an invalid content hash.")
        return hashlib.sha256(json.dumps({"version": 1, "image": image_hash,
            "model": self.model_fingerprint, "width": entry["width"], "height": entry["height"],
            "channels": self.latent_channels, "scale": self.scale_factor}, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _digest(mean, logvar):
        h = hashlib.sha256()
        for tensor in (mean, logvar):
            h.update(str(tuple(tensor.shape)).encode())
            h.update(tensor.detach().contiguous().numpy().tobytes())
        return h.hexdigest()

    def _load(self, path, key, expected_shape):
        import torch
        from safetensors import safe_open
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Frozen VAE cache is not a regular file: {path}")
        try:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                metadata = f.metadata() or {}
                if set(f.keys()) != {"mean", "logvar"} or metadata.get("key") != key:
                    raise ValueError("Cache identity does not match.")
                mean, logvar = f.get_tensor("mean"), f.get_tensor("logvar")
            if any(t.dtype != torch.float32 or tuple(t.shape) != expected_shape or not torch.isfinite(t).all().item() for t in (mean, logvar)):
                raise ValueError("Cached posterior has an invalid shape, dtype, or value.")
            if metadata.get("sha256") != self._digest(mean, logvar):
                raise ValueError("Cached posterior integrity check failed.")
            return mean, logvar
        except Exception as exc:
            raise ValueError(f"Invalid frozen VAE cache {path}. Remove this cache file and retry: {exc}") from exc

    def _write(self, path, key, mean, logvar):
        from safetensors.torch import save_file
        fd, name = tempfile.mkstemp(prefix=".posterior-", suffix=".tmp", dir=self.directory)
        os.close(fd)
        try:
            save_file({"mean": mean.contiguous(), "logvar": logvar.contiguous()}, name,
                      metadata={"key": key, "sha256": self._digest(mean, logvar)})
            try:
                os.link(name, path)  # atomic publication, never overwrite an existing cache entry
            except FileExistsError:
                pass  # a concurrent publisher won; the caller validates the winner
        finally:
            Path(name).unlink(missing_ok=True)

    def _forget(self, key):
        item = self._memory.pop(key, None)
        if item is not None:
            self._memory_bytes -= item[2]

    @staticmethod
    def _signature(path):
        # Detect replacements, deletion, or modifications before using the RAM
        # copy. The validated tensor copy does not alias the mapped disk file.
        try:
            stat = path.lstat()
        except FileNotFoundError:
            return None
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Frozen VAE cache is not a regular file: {path}")
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def _lookup(self, path, key, expected_shape):
        signature = self._signature(path)
        if signature is None:
            self._forget(key)
            return None
        item = self._memory.get(key)
        if item is not None and item[0] == signature:
            self._memory.move_to_end(key)
            return item[1]
        self._forget(key)
        pair = self._load(path, key, expected_shape)
        # Record identity only after validation, and ensure the file was not
        # replaced or modified while being read.
        if signature != self._signature(path):
            raise ValueError(f"Frozen VAE cache changed while it was being read: {path}")
        size = sum(t.numel() * t.element_size() for t in pair)
        if size <= self.memory_limit_bytes:
            while self._memory and self._memory_bytes + size > self.memory_limit_bytes:
                _, old = self._memory.popitem(last=False)
                self._memory_bytes -= old[2]
            pair = tuple(t.clone() for t in pair)
            self._memory[key] = (signature, pair, size)
            self._memory_bytes += size
        return pair

    def posterior(self, vae, pixels, entries):
        import torch
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        if not entries or (not callable(pixels) and len(entries) != pixels.shape[0]):
            raise ValueError("VAE cache batch must match its finalized image entries.")
        pairs = []
        for i, entry in enumerate(entries):
            expected_shape = (1, self.latent_channels, entry["height"] // self.scale_factor,
                              entry["width"] // self.scale_factor)
            key = self._key(entry)
            path = self.directory / (key + ".safetensors")
            pair = self._lookup(path, key, expected_shape)
            if pair is None:
                image = pixels(i) if callable(pixels) else pixels[i:i + 1]
                if image.shape[0] != 1:
                    raise ValueError("VAE cache pixel loader must return exactly one image.")
                with torch.no_grad():
                    encoded = vae.encode(image).latent_dist
                mean = encoded.mean.detach().to(device="cpu", dtype=torch.float32).contiguous()
                logvar = encoded.logvar.detach().to(device="cpu", dtype=torch.float32).contiguous()
                if tuple(mean.shape) != expected_shape or tuple(logvar.shape) != expected_shape:
                    raise ValueError("VAE output shape does not match the finalized image bucket.")
                self._write(path, key, mean, logvar)
                pair = self._lookup(path, key, expected_shape)
            pairs.append(pair)
        mean = torch.cat([pair[0] for pair in pairs], dim=0)
        logvar = torch.cat([pair[1] for pair in pairs], dim=0)
        return DiagonalGaussianDistribution(torch.cat([mean, logvar], dim=1))
