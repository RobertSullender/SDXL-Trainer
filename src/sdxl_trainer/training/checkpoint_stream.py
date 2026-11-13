"""Bounded-memory writer for ordinary, resumable Safetensors component files."""
from __future__ import annotations

import json
import math
import errno
import os
from pathlib import Path
import struct
import sys


def _chunks(tensor, maximum_elements):
    """Yield bounded logical row-major slices, including noncontiguous tensors."""
    count = tensor.numel()
    if not count:
        return
    if count <= maximum_elements:
        yield tensor
        return
    if tensor.is_contiguous():
        flat = tensor.view(-1)
        for start in range(0, count, maximum_elements):
            yield flat[start:start + maximum_elements]
        return
    row_size = math.prod(tensor.shape[1:])
    if row_size <= maximum_elements:
        rows = max(1, maximum_elements // row_size)
        for start in range(0, tensor.shape[0], rows):
            yield tensor[start:start + rows]
    else:
        for row in tensor:
            yield from _chunks(row, maximum_elements)


def _writeback(output):
    output.flush()
    os.fsync(output.fileno())
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        # The file is only an unpublished checkpoint until the caller's
        # directory rename. Release completed clean page-cache pages when the
        # kernel supports it; this remains advisory rather than a RAM limit.
        try:
            os.posix_fadvise(output.fileno(), 0, output.tell(), os.POSIX_FADV_DONTNEED)
        except OSError as exc:
            # Some filesystems or platforms expose the API without supporting
            # this advice. Data was already synced; only the eviction hint may
            # be skipped. Other I/O errors remain fatal before publication.
            if exc.errno not in {errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EINVAL}:
                raise


def save_safetensors_stream(tensors, destination, chunk_bytes=32 * 1024**2,
                            writeback_bytes=256 * 1024**2):
    """Write standard Safetensors without materializing a whole component on CPU.

    The enclosing checkpoint directory is published only after all files are
    complete. Existing checkpoint files are never overwritten by this writer.
    Each tensor chunk is copied exactly, preserving dtype and logical order.
    """
    import torch
    if sys.byteorder != "little":
        raise RuntimeError("Streaming checkpoint serialization requires a little-endian host.")
    if type(chunk_bytes) is not int or chunk_bytes < 1 or type(writeback_bytes) is not int or writeback_bytes < 1:
        raise ValueError("Checkpoint chunk and writeback sizes must be positive integers.")
    dtypes = {torch.float64: "F64", torch.float32: "F32", torch.float16: "F16",
              torch.bfloat16: "BF16", torch.int64: "I64", torch.int32: "I32",
              torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL",
              torch.complex64: "C64"}
    for name, tag in (("uint16", "U16"), ("uint32", "U32"), ("uint64", "U64"),
                      ("float8_e4m3fn", "F8_E4M3"), ("float8_e4m3fnuz", "F8_E4M3FNUZ"),
                      ("float8_e5m2", "F8_E5M2"), ("float8_e5m2fnuz", "F8_E5M2FNUZ")):
        dtype = getattr(torch, name, None)
        if dtype is not None:
            dtypes[dtype] = tag
    header = {}
    offset = 0
    for name in sorted(tensors):
        value = tensors[name]
        if not isinstance(name, str) or name == "__metadata__" or not isinstance(value, torch.Tensor):
            raise ValueError("Checkpoint tensors must have ordinary string names and tensor values.")
        if value.layout != torch.strided or value.dtype not in dtypes or value.device.type == "meta":
            raise ValueError(f"Unsupported checkpoint tensor layout, dtype or device: {name}.")
        length = value.numel() * value.element_size()
        header[name] = {"dtype": dtypes[value.dtype], "shape": list(value.shape), "data_offsets": [offset, offset + length]}
        offset += length
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    destination = Path(destination)
    created = False
    try:
        with destination.open("xb") as output:
            created = True
            output.write(struct.pack("<Q", len(encoded)))
            output.write(encoded)
            written = 0
            with torch.no_grad():
                for name in sorted(tensors):
                    tensor = tensors[name].detach()
                    maximum = max(1, chunk_bytes // tensor.element_size())
                    for chunk in _chunks(tensor, maximum):
                        # Reinterpret bytes as uint8; this also supports BF16,
                        # whose dtype NumPy cannot represent directly.
                        cpu = torch.empty(chunk.numel(), dtype=chunk.dtype, device="cpu")
                        cpu.copy_(chunk.reshape(-1))
                        output.write(memoryview(cpu.view(torch.uint8).numpy()))
                        if output.tell() - written >= writeback_bytes:
                            _writeback(output)
                            written = output.tell()
            if output.tell() != written:
                _writeback(output)
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
