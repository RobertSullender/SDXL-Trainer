"""Exact fp32 AdamW with disk-backed gradients and moments.

Gradient post-accumulation hooks spill each parameter immediately, keeping complete
UNet gradients out of device memory. Files are ordinary local memory maps; no
weight or optimizer quantization is used. Mappings stay open and use the OS page
cache rather than synchronously flushing every tensor. Completed fp32 moment
and disk-gradient ranges are flushed and evicted in bounded batches to reduce
swap pressure. Checkpoint copies explicitly flush and fsync the optimizer
moments. Filesystem encryption is the user's OS choice.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import errno
import json
import math
import mmap
import os
import stat
import threading
import weakref


class NonFiniteGradients(FloatingPointError):
    """A pre-update overflow which can safely be retried at a lower scale."""


def _ram_gradient_minimum_available(gradient_bytes, has_master_weights):
    # In addition to the gradient buffer and the original 12 GiB reserve,
    # reduced-precision full training needs room for checkpoint/export work
    # involving the fp32 master mapping. Its size equals the fp32 gradients.
    return 12 * 1024**3 + gradient_bytes * (2 if has_master_weights else 1)


def _ram_gradients_admissible(gradient_bytes, has_master_weights, automatic):
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
        available = next(int(line.split()[1]) * 1024 for line in lines if line.startswith("MemAvailable:"))
        swap_total = next((int(line.split()[1]) * 1024 for line in lines if line.startswith("SwapTotal:")), 0)
        swap_free = next((int(line.split()[1]) * 1024 for line in lines if line.startswith("SwapFree:")), 0)
    except (OSError, ValueError, StopIteration, IndexError):
        available = swap_total = swap_free = 0
    for limit_file, current_file in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                                     ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            limit = Path(limit_file).read_text().strip()
            if limit != "max" and int(limit) < 2**60:
                available = min(available, max(0, int(limit) - int(Path(current_file).read_text().strip())))
        except (OSError, ValueError):
            pass
    eviction_supported = hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED") and hasattr(mmap, "MADV_DONTNEED")
    swap_ok = not automatic or not swap_total or swap_free >= 2 * 1024**3
    return available >= _ram_gradient_minimum_available(gradient_bytes, has_master_weights) and swap_ok and eviction_supported


def _advise_drop(fd, start, length):
    if not length or not hasattr(os, 'posix_fadvise') or not hasattr(os, 'POSIX_FADV_DONTNEED'):
        return
    try:
        os.posix_fadvise(fd, start, length, os.POSIX_FADV_DONTNEED)
    except OSError as exc:
        if exc.errno not in {errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EINVAL}:
            raise


def _copy_state_file_bounded(source, destination, expected_bytes, *, create,
                             chunk_bytes=4 * 1024**2, writeback_bytes=128 * 1024**2):
    """Copy exact optimizer bytes with bounded buffers and advisory cache eviction.

    ``create`` writes a new checkpoint file exclusively; resume writes in place
    into the pre-existing working file so its mmap remains valid. A failed
    checkpoint copy removes its incomplete newly created file. The caller
    publishes checkpoint metadata only after all state files are complete.
    """
    if type(expected_bytes) is not int or expected_bytes < 0 or chunk_bytes <= 0 or writeback_bytes <= 0:
        raise ValueError('Invalid optimizer checkpoint copy size.')
    source = Path(source)
    destination = Path(destination)
    nofollow = getattr(os, 'O_NOFOLLOW', 0)
    src = os.open(source, os.O_RDONLY | nofollow)
    dst = None
    created = False
    try:
        src_stat = os.fstat(src)
        if not stat.S_ISREG(src_stat.st_mode) or src_stat.st_size != expected_bytes:
            raise ValueError(f'Invalid or truncated optimizer state file: {source}')
        flags = os.O_WRONLY | nofollow
        if create:
            flags |= os.O_CREAT | os.O_EXCL
        dst = os.open(destination, flags, 0o600)
        created = create
        dst_stat = os.fstat(dst)
        if not stat.S_ISREG(dst_stat.st_mode) or (not create and dst_stat.st_size != expected_bytes):
            raise ValueError(f'Invalid working optimizer state file: {destination}')
        if (src_stat.st_dev, src_stat.st_ino) == (dst_stat.st_dev, dst_stat.st_ino):
            raise ValueError('Optimizer checkpoint source and destination must be separate files.')
        copied = flushed = 0
        while copied < expected_bytes:
            data = os.read(src, min(chunk_bytes, expected_bytes - copied))
            if not data:
                raise ValueError(f'Optimizer state ended during copy: {source}')
            remaining = memoryview(data)
            while remaining:
                count = os.write(dst, remaining)
                if count <= 0:
                    raise OSError('Optimizer state copy made no write progress.')
                remaining = remaining[count:]
            copied += len(data)
            if copied - flushed >= writeback_bytes:
                os.fsync(dst)
                _advise_drop(src, flushed, copied - flushed)
                _advise_drop(dst, flushed, copied - flushed)
                flushed = copied
        os.fsync(dst)
        _advise_drop(src, flushed, copied - flushed)
        _advise_drop(dst, flushed, copied - flushed)
        if os.fstat(src).st_size != expected_bytes or os.fstat(dst).st_size != expected_bytes:
            raise ValueError('Optimizer state changed size during copy.')
    except BaseException:
        if dst is not None:
            os.close(dst)
            dst = None
        if created:
            destination.unlink(missing_ok=True)
        raise
    finally:
        if dst is not None:
            os.close(dst)
        os.close(src)


def create_disk_adamw(named_groups, root, *, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                      update_device="auto", gradient_storage="auto", eager_gradient_eviction=False):
    import numpy as np
    import torch

    class DiskAdamW(torch.optim.Optimizer):
        def __init__(self):
            if update_device not in {"auto", "cpu", "cuda"}:
                raise ValueError("Disk AdamW update device must be auto, cpu, or cuda.")
            if gradient_storage not in {"auto", "ram", "disk"}:
                raise ValueError("Gradient storage must be auto, ram, or disk.")
            self.requested_update_device = update_device
            self.update_backend = "cpu"
            self._scratch = None
            self._chunk_elements = 0
            self._gradient_lock = threading.RLock()
            self._gradient_views = 0
            self._gradient_view_refs = []
            self._pending_gradient_ranges = []
            self._pending_gradient_bytes = 0
            self._eager_gradient_eviction = eager_gradient_eviction
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
                    if p.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                        raise ValueError("Disk AdamW requires fp32, bf16 or fp16 trainable parameters.")
                    index = len(self.records)
                    self.records.append({"name": name, "p": p, "offset": offset, "numel": p.numel(), "step": 0, "group": len(groups)})
                    offset += p.numel() * 4
                    params.append(p)
                    self.hooks.append(p.register_post_accumulate_grad_hook(self._make_hook(index)))
                groups.append({"params": params, "lr": group["lr"]})
            self.total_bytes = offset
            # Half-precision parameter updates must not round away a small
            # learning rate on every step. Keep authoritative fp32 masters in
            # the disk mapping when any trainable parameter is reduced precision.
            self.has_master_weights = any(r["p"].dtype != torch.float32 for r in self.records)
            self.gradient_storage = "disk"
            self._grad_buffer = None
            self._automatic_gradient_storage = gradient_storage == "auto"
            if gradient_storage != "disk":
                # Mapped moment pages, frozen models, offloaded activations,
                # and other applications all compete with anonymous gradient
                # RAM. A single MemAvailable sample cannot guarantee no later
                # swap or OOM. Auto also requires swap headroom if swap exists.
                enough = _ram_gradients_admissible(offset, self.has_master_weights, self._automatic_gradient_storage)
                if not enough and gradient_storage == "ram":
                    extra = " plus additional fp32-master-sized checkpoint/export headroom" if self.has_master_weights else ""
                    raise RuntimeError(f"RAM gradient storage needs fp32 gradients, a 12 GiB reserve{extra}, and Linux cache eviction; choose disk or auto.")
                if enough:
                    try:
                        self._grad_buffer = torch.empty(offset // 4, dtype=torch.float32, device="cpu")
                        self.gradient_storage = "ram"
                    except (RuntimeError, MemoryError) as exc:
                        if gradient_storage == "ram":
                            raise RuntimeError("Could not allocate RAM gradient storage; choose disk or auto.") from exc
            names = ["first_moment.bin", "second_moment.bin"]
            if self.has_master_weights:
                names.append("master_weights.bin")
            if self.gradient_storage == "disk":
                names.insert(0, "gradients.bin")
            if self._eager_gradient_eviction and self.gradient_storage != "disk":
                raise ValueError("Eager gradient eviction requires disk-backed gradients.")
            for filename in names:
                with (self.root / filename).open("wb") as stream:
                    stream.truncate(offset)
            # One virtual mapping per file avoids opening, mmap'ing, syncing and
            # evicting a separate region for every parameter in every backward
            # pass. Only pages actually touched consume physical memory; the OS
            # can reclaim the file-backed cache when training needs more RAM.
            self._maps = {
                name: np.memmap(self.root / name, dtype="float32", mode="r+", shape=(offset // 4,))
                for name in names
            }
            super().__init__(groups, {"lr": 1e-4, "betas": betas, "eps": eps, "weight_decay": weight_decay})
            self._writeback_batch_bytes = 128 * 1024**2
            if self.has_master_weights:
                written = 0
                with torch.no_grad():
                    for record in self.records:
                        with self.mapped("master_weights.bin", record) as master:
                            master.copy_(record["p"].detach().cpu().float().reshape(-1))
                        end = record["offset"] + record["numel"] * 4
                        if end - written >= self._writeback_batch_bytes:
                            self._evict_file_range("master_weights.bin", written, end)
                            written = end
                    self._evict_file_range("master_weights.bin", written, self.total_bytes)
            if update_device == "cuda":
                self._set_backend("cuda")
            elif update_device == "auto" and not self.has_master_weights:
                try:
                    self._set_backend("cuda")
                except (RuntimeError, ValueError):
                    # Decide once, before any optimizer update. Never fall back
                    # halfway through a step or silently change a resume backend.
                    self._set_backend("cpu")

        def _set_backend(self, backend):
            if backend == "cpu":
                self._scratch = None
                self._chunk_elements = 0
                self.update_backend = "cpu"
                return
            if backend != "cuda":
                raise ValueError("Invalid optimizer checkpoint update backend.")
            if self.update_backend == "cuda" and self._scratch is not None:
                return
            parameters = [r["p"] for r in self.records]
            if not parameters or not torch.cuda.is_available():
                raise ValueError("CUDA optimizer updates require available CUDA parameters.")
            devices = {p.device for p in parameters}
            if len(devices) != 1 or next(iter(devices)).type != "cuda":
                raise ValueError("CUDA optimizer updates require all trainable weights on one CUDA device.")
            if any(not p.is_contiguous() for p in parameters):
                raise ValueError("CUDA optimizer updates require contiguous trainable weights; choose CPU updates instead.")
            device = next(iter(devices))
            # Four fp32 buffers preserve the existing fp32 path. Reduced
            # model weights additionally need one fp32 master buffer, so
            # updates never use rounded live weights as the source.
            elements = min(4 * 1024 * 1024, max(p.numel() for p in parameters))
            buffer_count = 5 if self.has_master_weights else 4
            bytes_needed = elements * 4 * buffer_count
            free, _ = torch.cuda.mem_get_info(device)
            if free < bytes_needed + 512 * 1024 * 1024:
                raise RuntimeError("Not enough free CUDA memory for optimizer scratch and reserve; choose CPU updates.")
            try:
                buffers = [torch.empty(elements, dtype=torch.float32, device=device) for _ in range(buffer_count)]
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError("Could not allocate CUDA optimizer scratch; choose CPU updates.") from exc
            self._scratch = buffers
            self._chunk_elements = elements
            self.update_backend = "cuda"

        @contextmanager
        def mapped(self, filename, record):
            start = record["offset"] // 4
            if filename == "gradients.bin" and self.gradient_storage == "ram":
                # Hold the lock until the view leaves scope so a checkpoint
                # cannot drop the allocation while a hook/update uses it.
                with self._gradient_lock:
                    if self._grad_buffer is None:
                        if not _ram_gradients_admissible(self.total_bytes, self.has_master_weights, self._automatic_gradient_storage):
                            raise RuntimeError("Insufficient RAM headroom to reacquire gradients after checkpointing; resume the last completed checkpoint with disk gradient storage.")
                        try:
                            self._grad_buffer = torch.empty(self.total_bytes // 4, dtype=torch.float32, device="cpu")
                        except (RuntimeError, MemoryError) as exc:
                            raise RuntimeError("Could not reallocate RAM gradients after checkpointing; resume using disk gradient storage.") from exc
                    self._gradient_views += 1
                    if len(self._gradient_view_refs) >= 512:
                        self._gradient_view_refs = [ref for ref in self._gradient_view_refs if ref() is not None]
                    view = self._grad_buffer[start : start + record["numel"]]
                    self._gradient_view_refs.append(weakref.ref(view))
                    try:
                        yield view
                    finally:
                        self._gradient_views -= 1
            else:
                yield torch.from_numpy(self._maps[filename][start : start + record["numel"]])

        def release_gradient_buffer(self):
            """Drop unused RAM gradients only at a completed update boundary."""
            with self._gradient_lock:
                self._gradient_view_refs = [ref for ref in self._gradient_view_refs if ref() is not None]
                if self.active or self._gradient_views or self._gradient_view_refs:
                    raise RuntimeError("Cannot release RAM gradients while gradients or mapped views are active.")
                if self.gradient_storage == "ram":
                    self._grad_buffer = None

        def _evict_file_range(self, name, start, end):
            if end <= start:
                return
            aligned = start // mmap.PAGESIZE * mmap.PAGESIZE
            length = end - aligned
            mapping = self._maps[name]._mmap
            mapping.flush(aligned, length)
            mapping.madvise(mmap.MADV_DONTNEED, aligned, length)
            with (self.root / name).open("rb") as stream:
                os.posix_fadvise(stream.fileno(), aligned, length, os.POSIX_FADV_DONTNEED)

        def _make_hook(self, index):
            def spill(parameter):
                if parameter.grad is None:
                    return
                record = self.records[index]
                # Transfer reduced-precision gradients before conversion, so a
                # temporary fp32 tensor does not consume extra scarce GPU VRAM.
                gradient = parameter.grad.detach().cpu().float().reshape(-1)
                # Several autograd hooks can write records sharing an OS page.
                # Serialize writes and flush/eviction so no pending hook can
                # mutate a page between writeback and the cache advice.
                with self._gradient_lock:
                    with self.mapped("gradients.bin", record) as disk:
                        if index in self.active:
                            disk.add_(gradient)
                        else:
                            disk.copy_(gradient)
                    self.active.add(index)
                    parameter.grad = None
                    if self._eager_gradient_eviction:
                        start = record["offset"]
                        end = start + record["numel"] * 4
                        self._pending_gradient_ranges.append((start, end))
                        self._pending_gradient_bytes += end - start
                        if self._pending_gradient_bytes >= self._writeback_batch_bytes:
                            self._evict_pending_gradients()
            return spill

        @torch.no_grad()
        def _evict_pending_gradients(self):
            """Flush/evict dirty records in merged page ranges, in any hook order."""
            if not self._pending_gradient_ranges:
                return
            # Adjacent records can share a page. Flush an entire overlapping
            # page once; lengths can extend to the non-page-aligned file end.
            merged = []
            for start, end in sorted(self._pending_gradient_ranges):
                start = (start // mmap.PAGESIZE) * mmap.PAGESIZE
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            mapping = self._maps["gradients.bin"]._mmap
            with (self.root / "gradients.bin").open("rb") as stream:
                for start, end in merged:
                    length = end - start
                    mapping.flush(start, length)
                    mapping.madvise(mmap.MADV_DONTNEED, start, length)
                    os.posix_fadvise(stream.fileno(), start, length, os.POSIX_FADV_DONTNEED)
            self._pending_gradient_ranges.clear()
            self._pending_gradient_bytes = 0

        @torch.no_grad()
        def gradient_norm(self):
            squared = 0.0
            for index in self.active:
                record = self.records[index]
                with self.mapped("gradients.bin", record) as grad:
                    norm = torch.linalg.vector_norm(grad.double()).item()
                    squared += norm * norm
                if self._eager_gradient_eviction:
                    # A full-model norm otherwise rereads and retains all
                    # ~13 GiB of gradient pages before the optimizer starts.
                    # These values were written at a backward boundary and
                    # remain unchanged by the read.
                    start = record["offset"]
                    end = start + record["numel"] * 4
                    self._pending_gradient_ranges.append((start, end))
                    self._pending_gradient_bytes += end - start
                    if self._pending_gradient_bytes >= self._writeback_batch_bytes:
                        self._evict_pending_gradients()
            if self._eager_gradient_eviction:
                self._evict_pending_gradients()
            return math.sqrt(squared)

        @torch.no_grad()
        def evict_accumulated_gradients(self):
            """Write active disk gradients and offer their cached pages for eviction.

            Used at a completed backward boundary before a large model is moved
            into host memory. Unlike an optimizer step this does not reset any
            gradient, parameter, per-parameter step or accumulated-active bit.
            Partial page overlaps are flushed before eviction, preserving other
            records. Kernel cache eviction is advisory, not a hard memory cap.
            """
            if self.gradient_storage != "disk":
                raise RuntimeError("Releasing accumulated gradient pages requires disk gradient storage.")
            if self._eager_gradient_eviction:
                with self._gradient_lock:
                    self._evict_pending_gradients()
                return
            if not self.active:
                return
            mapping = self._maps["gradients.bin"]._mmap
            with (self.root / "gradients.bin").open("rb") as stream:
                start = self.records[min(self.active)]["offset"]
                end = start
                def evict(first, last):
                    if last <= first:
                        return
                    aligned = (first // mmap.PAGESIZE) * mmap.PAGESIZE
                    length = last - aligned
                    mapping.flush(aligned, length)
                    mapping.madvise(mmap.MADV_DONTNEED, aligned, length)
                    os.posix_fadvise(stream.fileno(), aligned, length, os.POSIX_FADV_DONTNEED)
                for index in sorted(self.active):
                    record = self.records[index]
                    end = record["offset"] + record["numel"] * 4
                    if end - start >= self._writeback_batch_bytes:
                        evict(start, end)
                        start = end
                evict(start, end)

        @torch.no_grad()
        def _validate_master_update(self, weight, moment, variance, live_dtype):
            """Fail closed if a reduced-precision update cannot be represented.

            This check is only for the fp32-master path; the legacy fp32
            optimizer path is unchanged. A failure aborts the run before a
            checkpoint can be published; resume a prior complete checkpoint.
            """
            if not weight.numel():
                return
            lower, upper = torch.aminmax(weight)
            limit = torch.finfo(live_dtype).max
            valid = (torch.isfinite(lower) & torch.isfinite(upper)
                     & (lower >= -limit) & (upper <= limit)
                     & torch.isfinite(moment).all() & torch.isfinite(variance).all())
            if not valid.item():
                raise FloatingPointError(
                    "The fp32 optimizer master or moments became non-finite or exceeded the selected model weight precision. "
                    "Training stopped without publishing a checkpoint; resume a prior checkpoint with safer settings."
                )

        @torch.no_grad()
        def _cuda_update(self, parameter, grad, m, v, group, step, scale, master=None):
            beta1, beta2 = group["betas"]
            live_weight = parameter.view(-1)
            for start in range(0, parameter.numel(), self._chunk_elements):
                end = min(start + self._chunk_elements, parameter.numel())
                length = end - start
                gradient, moment, variance, denominator = (buffer[:length] for buffer in self._scratch[:4])
                if master is None:
                    weight = live_weight[start:end]
                else:
                    weight = self._scratch[4][:length]
                    weight.copy_(master[start:end])
                gradient.copy_(grad[start:end]).mul_(scale)
                moment.copy_(m[start:end])
                variance.copy_(v[start:end])
                moment.mul_(beta1).add_(gradient, alpha=1 - beta1)
                variance.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                torch.sqrt(variance, out=denominator)
                denominator.div_(math.sqrt(1 - beta2 ** step)).add_(group["eps"])
                torch.div(moment, denominator, out=gradient)
                gradient.mul_(group["lr"] / (1 - beta1 ** step))
                weight.mul_(1 - group["lr"] * group["weight_decay"])
                weight.sub_(gradient)
                if master is not None:
                    self._validate_master_update(weight, moment, variance, parameter.dtype)
                    # Copy once into selected live dtype, but preserve fp32
                    # master bits for the next optimizer update and resume.
                    live_weight[start:end].copy_(weight)
                    master[start:end].copy_(weight)
                # Device-to-host copies are blocking so checkpointing never sees
                # stale moments and these buffers can be reused safely.
                m[start:end].copy_(moment)
                v[start:end].copy_(variance)

        @torch.no_grad()
        def _evict_moment_range(self, start, end):
            """Write and evict a bounded completed range to reduce pressure.

            Without this, the two dirty 10 GiB moment maps can displace the
            pageable memory into swap on a 32 GiB machine. Disk gradients in the
            completed range are no longer needed after the weight update; write
            and advise their eviction too. Partial pages may overlap untouched
            records, so flushing precedes eviction and preserves those values.
            msync is performed per ~128 MiB range, not per parameter, and only
            after its update has completed. FP32 values are never reduced.
            """
            if end <= start:
                return
            aligned = (start // mmap.PAGESIZE) * mmap.PAGESIZE
            length = end - aligned
            names = ["first_moment.bin", "second_moment.bin"]
            if self.has_master_weights:
                names.append("master_weights.bin")
            if self.gradient_storage == "disk":
                names.append("gradients.bin")
            for name in names:
                mapping = self._maps[name]._mmap
                mapping.flush(aligned, length)
                mapping.madvise(mmap.MADV_DONTNEED, aligned, length)
                with (self.root / name).open("rb") as stream:
                    os.posix_fadvise(stream.fileno(), aligned, length, os.POSIX_FADV_DONTNEED)

        @torch.no_grad()
        def step(self, closure=None, max_grad_norm=0.0, loss_scale=1.0):
            if closure is not None:
                raise ValueError("Closures are not supported by disk AdamW.")
            if not math.isfinite(loss_scale) or loss_scale <= 0:
                raise ValueError("Loss scale must be finite and positive.")
            norm = self.gradient_norm() / loss_scale
            if not math.isfinite(norm):
                raise NonFiniteGradients("Non-finite gradients; no optimizer update was applied.")
            scale = (min(1.0, max_grad_norm / (norm + 1e-6)) if max_grad_norm > 0 else 1.0) / loss_scale
            written_to = 0
            for index in sorted(self.active):
                record = self.records[index]
                p = record["p"]
                group = self.param_groups[record["group"]]
                beta1, beta2 = group["betas"]
                record["step"] += 1
                step = record["step"]
                with self.mapped("gradients.bin", record) as grad, self.mapped("first_moment.bin", record) as m, self.mapped("second_moment.bin", record) as v:
                    if self.update_backend == "cuda":
                        if self.has_master_weights:
                            with self.mapped("master_weights.bin", record) as master:
                                self._cuda_update(p, grad, m, v, group, step, scale, master)
                        else:
                            self._cuda_update(p, grad, m, v, group, step, scale)
                    else:
                        grad.mul_(scale)
                        m.mul_(beta1).add_(grad, alpha=1 - beta1)
                        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                        denominator = v.sqrt().div_(math.sqrt(1 - beta2**step)).add_(group["eps"])
                        update = m.div(denominator).mul_(group["lr"] / (1 - beta1**step))
                        if self.has_master_weights:
                            with self.mapped("master_weights.bin", record) as master:
                                master.mul_(1 - group["lr"] * group["weight_decay"])
                                master.sub_(update.reshape(-1))
                                self._validate_master_update(master, m, v, p.dtype)
                                # Cast on the host before transfer. In
                                # particular, avoid an additional fp32 GPU
                                # allocation for reduced-precision weights.
                                p.copy_(master.reshape(p.shape).to(dtype=p.dtype))
                        else:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                            p.sub_(update.reshape(p.shape).to(p.device))
                end = record["offset"] + record["numel"] * 4
                if end - written_to >= self._writeback_batch_bytes:
                    self._evict_moment_range(written_to, end)
                    written_to = end
            if self.active:
                # Flush through the last updated record, rather than forcing
                # unrelated inactive model ranges to be written.
                record = self.records[max(self.active)]
                self._evict_moment_range(written_to, record["offset"] + record["numel"] * 4)
            self.active.clear()
            return norm

        def zero_grad(self, set_to_none=True):
            self.active.clear()
            return super().zero_grad(set_to_none=set_to_none)

        def save_state(self, destination):
            if self.active:
                raise RuntimeError("Checkpoint only at an optimizer boundary.")
            destination = Path(destination)
            if destination.is_symlink():
                raise ValueError("Optimizer checkpoint folder must not be a symbolic link.")
            destination.mkdir(parents=True, exist_ok=True)
            metadata = {"records": [{k: v for k, v in r.items() if k != "p"} for r in self.records], "groups": [{k: v for k, v in g.items() if k != "params"} for g in self.param_groups], "update_backend": self.update_backend, "master_weights": self.has_master_weights}
            names = ["first_moment.bin", "second_moment.bin"] + (["master_weights.bin"] if self.has_master_weights else [])
            for name in names:
                # A checkpoint must contain the completed state even if the OS
                # hasn't yet written dirty working pages to the source file.
                self._maps[name].flush()
                if hasattr(mmap, 'MADV_DONTNEED'):
                    self._maps[name]._mmap.madvise(mmap.MADV_DONTNEED)
                _copy_state_file_bounded(self.root / name, destination / name, self.total_bytes, create=True)
            # Publish metadata after both data files; an interrupted write must
            # not look like a complete optimizer snapshot.
            with (destination / "optimizer.json").open("x", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())

        def load_state(self, source):
            source = Path(source)
            if source.is_symlink() or (source / "optimizer.json").is_symlink():
                raise ValueError("Optimizer checkpoint must use a real directory and metadata file.")
            metadata = json.loads((source / "optimizer.json").read_text())
            backend = metadata.get("update_backend", "cpu")
            if metadata.get("master_weights", False) != self.has_master_weights:
                raise ValueError("Checkpoint fp32 master-weight layout differs from the selected model precision.")
            if backend not in {"cpu", "cuda"}:
                raise ValueError("Invalid optimizer checkpoint update backend.")
            if self.requested_update_device != "auto" and self.requested_update_device != backend:
                raise ValueError(f"Checkpoint optimizer updates use {backend}; choose {backend} or auto to resume exactly.")
            if len(metadata["records"]) != len(self.records):
                raise ValueError("Checkpoint optimizer parameter count differs.")
            for old, new in zip(metadata["records"], self.records):
                if any(old[k] != new[k] for k in ("name", "offset", "numel", "group")):
                    raise ValueError("Checkpoint optimizer parameter layout differs.")
            if len(metadata["groups"]) != len(self.param_groups):
                raise ValueError("Checkpoint optimizer parameter groups differ.")
            names = ["first_moment.bin", "second_moment.bin"] + (["master_weights.bin"] if self.has_master_weights else [])
            for name in names:
                path = source / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Optimizer checkpoint state must be a regular file.")
                if path.stat().st_size != self.total_bytes:
                    raise ValueError("Truncated disk optimizer checkpoint.")
            self._set_backend(backend)
            for old, new in zip(metadata["records"], self.records):
                new["step"] = old["step"]
            for source_group, target in zip(metadata["groups"], self.param_groups):
                target.update(source_group)
            for name in names:
                # Do not truncate a file with an active mmap: writing in place
                # preserves the mapping, and validates both lengths before any
                # optimizer state is changed.
                _copy_state_file_bounded(source / name, self.root / name, self.total_bytes, create=False)
                # Drop mapped pages left by construction before later updates.
                if hasattr(mmap, 'MADV_DONTNEED'):
                    self._maps[name]._mmap.madvise(mmap.MADV_DONTNEED)

    return DiskAdamW()
