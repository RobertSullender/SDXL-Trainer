"""Park fp32, fp16, and bf16 module parameters in an anonymous disk mmap.

Only move at completed graph boundaries. Errors poison this object: callers must
abort the run and may only resume a prior complete checkpoint. The temporary
file is unlinked immediately by TemporaryFile, including on a process kill.
"""
import mmap
import os
import tempfile
from pathlib import Path
import numpy as np
import torch


class DiskModuleStager:
    def __init__(self, module, workdir, batch_bytes=128 * 1024**2):
        self.module = module
        self.params = list(module.named_parameters())
        self.ids = {name: p for name, p in self.params}
        if not self.params or any(p.dtype not in (torch.float32, torch.float16, torch.bfloat16) or not p.is_contiguous() for _, p in self.params):
            raise ValueError('Disk staging requires contiguous fp32, fp16 or bf16 parameters.')
        # Distinct Parameter objects sharing storage could require aliasing
        # semantics that a sequential disk layout would not preserve.
        storage_ids = [p.untyped_storage().data_ptr() for _, p in self.params if p.numel()]
        if len(storage_ids) != len(set(storage_ids)):
            raise ValueError('Disk staging does not support distinct parameters sharing storage.')
        shared_buffers = {}
        for _, buffer in module.named_buffers(remove_duplicate=False):
            if not buffer.numel():
                continue
            storage = buffer.untyped_storage().data_ptr()
            if storage in storage_ids or (storage in shared_buffers and shared_buffers[storage] is not buffer):
                raise ValueError('Disk staging does not support buffers sharing storage with a different tensor.')
            shared_buffers[storage] = buffer
        # Keep byte offsets aligned for mixed half/fp32 LoRA layers. The raw
        # byte mapping also preserves bf16 bit patterns without a numpy bf16.
        self.ranges = []
        self.total = 0
        for _, parameter in self.params:
            size = parameter.element_size()
            self.total = (self.total + size - 1) // size * size
            begin = self.total
            self.total += parameter.numel() * size
            self.ranges.append((begin, self.total))
        self.batch_bytes = max(1, int(batch_bytes))
        self.file = tempfile.TemporaryFile(prefix='sdxl-unet-staging-', dir=Path(workdir))
        # Reserve actual blocks before touching an mmap. Sparse mmap writes can
        # otherwise terminate the process with SIGBUS on a full filesystem.
        try:
            os.posix_fallocate(self.file.fileno(), 0, self.total)
        except (AttributeError, OSError) as exc:
            self.file.close()
            raise RuntimeError('Unable to reserve disk space for temporary UNet staging.') from exc
        try:
            self.mapping = mmap.mmap(self.file.fileno(), self.total, access=mmap.ACCESS_WRITE)
            self.array = np.frombuffer(self.mapping, dtype=np.uint8)
        except BaseException:
            if hasattr(self, 'mapping'):
                self.mapping.close()
            self.file.close()
            raise
        self.parked = False
        self.poisoned = False
        self.closed = False
        self._batch_start = 0
        self._snapshot_valid = False
        self._restored_stamp = None

    def invalidate(self):
        """Invalidate the parked parameter snapshot before any weight mutation.

        Ordinary in-place PyTorch changes are also detected by version counters.
        Callers must explicitly invalidate before mutating through ``.data``,
        which can bypass those counters. The training worker owns all mutations.
        """
        self._snapshot_valid = False
        self._restored_stamp = None

    def _parameter_stamp(self):
        return tuple((p._version, p.untyped_storage().data_ptr(), p.device, p.dtype,
                      tuple(p.shape), tuple(p.stride())) for _, p in self.params)

    def _check(self, expect_parked):
        if self.closed:
            raise RuntimeError('The temporary staged-model store has already been closed.')
        if self.poisoned:
            raise RuntimeError('A prior staging error invalidated the run. Resume a completed checkpoint.')
        if self.parked != expect_parked:
            raise RuntimeError('Invalid staged-model device state.')
        current = dict(self.module.named_parameters())
        if current.keys() != self.ids.keys() or any(current[k] is not v for k, v in self.ids.items()):
            raise RuntimeError('Model parameter identity changed; optimizer hooks may be invalid.')
        if any(p.grad is not None for _, p in self.params):
            raise RuntimeError('Staged disk weights require gradients spilled before transfer.')

    def _evict(self, start, end):
        if end <= start:
            return
        start = start // mmap.PAGESIZE * mmap.PAGESIZE
        length = end - start
        self.mapping.flush(start, length)
        self.mapping.madvise(mmap.MADV_DONTNEED, start, length)
        os.posix_fadvise(self.file.fileno(), start, length, os.POSIX_FADV_DONTNEED)

    def _move_buffers(self, target):
        seen = {}
        for module in self.module.modules():
            for name, value in module._buffers.items():
                if value is not None:
                    if id(value) not in seen:
                        seen[id(value)] = value.to(target)
                    module._buffers[name] = seen[id(value)]

    @torch.no_grad()
    def park(self):
        self._check(False)
        reuse = self._snapshot_valid and self._restored_stamp == self._parameter_stamp()
        if not reuse:
            self.invalidate()
        begin = end = 0
        target = None
        try:
            for (_, parameter), (begin, end) in zip(self.params, self.ranges):
                target = torch.from_numpy(self.array[begin:end]).view(parameter.dtype).reshape(parameter.shape)
                if not reuse:
                    target.copy_(parameter.detach())
                parameter.data = target
                del target
                if not reuse and end - self._batch_start >= self.batch_bytes:
                    self._evict(self._batch_start, end)
                    self._batch_start = end
            if not reuse:
                self._evict(self._batch_start, end)
            self._move_buffers('cpu')
            self.parked = True
            self._snapshot_valid = True
            self._restored_stamp = None
        except BaseException:
            target = None  # A traceback must not retain an exported mmap view.
            self.invalidate()
            self.poisoned = True
            raise
        finally:
            self._batch_start = 0

    def close(self):
        """Release the unlinked staging file promptly, including on failure.

        A poisoned/parked model is discarded by the worker. Drop mapped tensor
        references without allocating another full UNet on the failing machine.
        Completed runs have restored every parameter to the original device.
        """
        if self.closed:
            return
        if self.parked or self.poisoned:
            for _, parameter in self.params:
                if parameter.device.type == 'cpu':
                    parameter.grad = None
                    parameter.data = torch.empty(0, dtype=parameter.dtype)
        self.array = None
        try:
            self.mapping.close()
        finally:
            self.file.close()
            self.closed = True

    @torch.no_grad()
    def restore(self, device):
        self._check(True)
        begin = end = 0
        data = None
        try:
            for (_, parameter), (begin, end) in zip(self.params, self.ranges):
                # clone when running the CPU prototype so restored parameters no
                # longer alias the temporary mapping; CUDA always copies.
                data = parameter.data.to(device)
                if data.device.type == 'cpu':
                    data = data.clone()
                parameter.data = data
                if end - self._batch_start >= self.batch_bytes:
                    self._evict(self._batch_start, end)
                    self._batch_start = end
            self._evict(self._batch_start, end)
            self._move_buffers(device)
            self.parked = False
            self._restored_stamp = self._parameter_stamp()
        except BaseException:
            # On CPU, .to(device) can return a view of the mmap. If cloning
            # fails, the traceback retains this frame while close() runs.
            data = None
            self.invalidate()
            self.poisoned = True
            raise
        finally:
            self._batch_start = 0

