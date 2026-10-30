"""Exact fp32 AdamW with disk-backed gradients and moments.

Gradient post-accumulation hooks spill each parameter immediately, keeping complete
UNet gradients out of device memory. Files are ordinary local memory maps; no
weight or optimizer quantization is used. This trades significant disk I/O for
memory. Filesystem encryption, if needed, is the user's OS responsibility.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import json
import math
import shutil


def create_disk_adamw(named_groups, root, *, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    import numpy as np
    import torch

    class DiskAdamW(torch.optim.Optimizer):
        def __init__(self):
            groups = []
            self.records = []
            self.active = set()
            self.hooks = []
            self.root = Path(root)
            self.root.mkdir(parents=True, exist_ok=True)
            offset = 0
            for group in named_groups:
                params = []
                for name, p in group["named_params"]:
                    if not p.requires_grad:
                        continue
                    if p.dtype != torch.float32:
                        raise ValueError("Disk AdamW requires fp32 trainable parameters.")
                    index = len(self.records)
                    self.records.append({"name": name, "p": p, "offset": offset, "numel": p.numel(), "step": 0, "group": len(groups)})
                    offset += p.numel() * 4
                    params.append(p)
                    self.hooks.append(p.register_post_accumulate_grad_hook(self._make_hook(index)))
                groups.append({"params": params, "lr": group["lr"]})
            self.total_bytes = offset
            for filename in ("gradients.bin", "first_moment.bin", "second_moment.bin"):
                with (self.root / filename).open("wb") as stream:
                    stream.truncate(offset)
            super().__init__(groups, {"lr": 1e-4, "betas": betas, "eps": eps, "weight_decay": weight_decay})

        @contextmanager
        def mapped(self, filename, record):
            mapped = np.memmap(self.root / filename, dtype="float32", mode="r+", offset=record["offset"], shape=(record["numel"],))
            tensor = torch.from_numpy(mapped)
            try:
                yield tensor
            finally:
                del tensor
                mapped.flush()
                # Discard clean pages from resident RAM when the platform supports it.
                try:
                    import mmap
                    mapped._mmap.madvise(mmap.MADV_DONTNEED)
                except (AttributeError, OSError, ValueError):
                    pass
                mapped._mmap.close()

        def _make_hook(self, index):
            def spill(parameter):
                if parameter.grad is None:
                    return
                record = self.records[index]
                gradient = parameter.grad.detach().float().cpu().reshape(-1)
                with self.mapped("gradients.bin", record) as disk:
                    if index in self.active:
                        disk.add_(gradient)
                    else:
                        disk.copy_(gradient)
                self.active.add(index)
                parameter.grad = None
            return spill

        @torch.no_grad()
        def gradient_norm(self):
            squared = 0.0
            for index in self.active:
                with self.mapped("gradients.bin", self.records[index]) as grad:
                    norm = torch.linalg.vector_norm(grad.double()).item()
                    squared += norm * norm
            return math.sqrt(squared)

        @torch.no_grad()
        def step(self, closure=None, max_grad_norm=0.0):
            if closure is not None:
                raise ValueError("Closures are not supported by disk AdamW.")
            norm = self.gradient_norm()
            if not math.isfinite(norm):
                raise FloatingPointError("Non-finite gradients; no optimizer update was applied.")
            scale = min(1.0, max_grad_norm / (norm + 1e-6)) if max_grad_norm > 0 else 1.0
            for index in sorted(self.active):
                record = self.records[index]
                p = record["p"]
                group = self.param_groups[record["group"]]
                beta1, beta2 = group["betas"]
                record["step"] += 1
                step = record["step"]
                with self.mapped("gradients.bin", record) as grad, self.mapped("first_moment.bin", record) as m, self.mapped("second_moment.bin", record) as v:
                    grad.mul_(scale)
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denominator = v.sqrt().div_(math.sqrt(1 - beta2**step)).add_(group["eps"])
                    update = m.div(denominator).mul_(group["lr"] / (1 - beta1**step))
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.sub_(update.reshape(p.shape).to(p.device))
            self.active.clear()
            return norm

        def zero_grad(self, set_to_none=True):
            self.active.clear()
            return super().zero_grad(set_to_none=set_to_none)

        def save_state(self, destination):
            if self.active:
                raise RuntimeError("Checkpoint only at an optimizer boundary.")
            destination = Path(destination)
            destination.mkdir(parents=True, exist_ok=True)
            metadata = {"records": [{k: v for k, v in r.items() if k != "p"} for r in self.records], "groups": [{k: v for k, v in g.items() if k != "params"} for g in self.param_groups]}
            (destination / "optimizer.json").write_text(json.dumps(metadata, indent=2))
            for name in ("first_moment.bin", "second_moment.bin"):
                shutil.copyfile(self.root / name, destination / name)

        def load_state(self, source):
            source = Path(source)
            metadata = json.loads((source / "optimizer.json").read_text())
            if len(metadata["records"]) != len(self.records):
                raise ValueError("Checkpoint optimizer parameter count differs.")
            for old, new in zip(metadata["records"], self.records):
                if any(old[k] != new[k] for k in ("name", "offset", "numel", "group")):
                    raise ValueError("Checkpoint optimizer parameter layout differs.")
                new["step"] = old["step"]
            for source_group, target in zip(metadata["groups"], self.param_groups):
                target.update(source_group)
            for name in ("first_moment.bin", "second_moment.bin"):
                if (source / name).stat().st_size != self.total_bytes:
                    raise ValueError("Truncated disk optimizer checkpoint.")
                shutil.copyfile(source / name, self.root / name)

    return DiskAdamW()
