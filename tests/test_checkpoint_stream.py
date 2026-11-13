"""Resumable component snapshots are standard and bounded, including BF16."""
import errno
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import load_file, save_file
from sdxl_trainer.training import checkpoint_stream


def test_streamed_standard_format_preserves_dtype_shape_shared_and_strided(tmp_path):
    shared = torch.arange(120, dtype=torch.float32).reshape(10, 12)
    tensors = {
        "unet.shared": shared,
        "text_encoder.transpose": shared.t(),
        "text_encoder_2.sliced": shared[::2, 1::3],
        "text_encoder_2.expanded": torch.tensor([1.0, 2.0]).expand(9, 2),
        "vae.bfloat16": (shared / 3).to(torch.bfloat16),
        "vae.float16": (shared / 7).to(torch.float16),
        "scalar": torch.tensor(3, dtype=torch.int64),
        "empty": torch.empty((0, 4), dtype=torch.float32),
        "bool": torch.tensor([True, False]),
    }
    target = tmp_path / "stream.safetensors"
    checkpoint_stream.save_safetensors_stream(tensors, target, chunk_bytes=17)
    actual = load_file(str(target))
    assert actual.keys() == tensors.keys()
    for name, expected in tensors.items():
        assert actual[name].dtype == expected.dtype
        assert actual[name].shape == expected.shape
        assert torch.equal(actual[name], expected)
    # Each 8-byte element is alone in its chunk, including strided rows.
    single = {"transposed": torch.arange(100).reshape(10, 10).t()}
    checkpoint_stream.save_safetensors_stream(single, tmp_path / "strided.safetensors", chunk_bytes=3)
    assert torch.equal(load_file(str(tmp_path / "strided.safetensors"))["transposed"], single["transposed"])


def test_legacy_standard_safetensors_still_loads(tmp_path):
    target = tmp_path / "legacy.safetensors"
    original = {"layer": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)}
    save_file(original, str(target))
    assert torch.equal(load_file(str(target))["layer"], original["layer"])


def test_stream_failure_removes_partial_file_and_preserves_existing(tmp_path, monkeypatch):
    target = tmp_path / "partial.safetensors"
    original_chunks = checkpoint_stream._chunks
    def broken(tensor, limit):
        yield next(original_chunks(tensor, limit))
        raise OSError("injected write failure")
    monkeypatch.setattr(checkpoint_stream, "_chunks", broken)
    with pytest.raises(OSError, match="injected"):
        checkpoint_stream.save_safetensors_stream({"weight": torch.arange(10)}, target, chunk_bytes=8)
    assert not target.exists()
    target.write_bytes(b"existing checkpoint remains")
    with pytest.raises(FileExistsError):
        checkpoint_stream.save_safetensors_stream({"weight": torch.arange(10)}, target)
    assert target.read_bytes() == b"existing checkpoint remains"


def test_each_noncontiguous_chunk_is_bounded(tmp_path):
    tensor = torch.arange(2 * 7 * 9).reshape(2, 7, 9).transpose(0, 2)
    chunks = list(checkpoint_stream._chunks(tensor, 5))
    assert chunks and all(chunk.numel() <= 5 for chunk in chunks)
    actual = torch.cat([chunk.contiguous().reshape(-1) for chunk in chunks])
    assert torch.equal(actual, tensor.contiguous().reshape(-1))


def test_completed_checkpoint_ranges_are_written_back_before_publish(tmp_path, monkeypatch):
    calls = []
    writeback = checkpoint_stream._writeback
    def record(output):
        calls.append(output.tell())
        writeback(output)
    monkeypatch.setattr(checkpoint_stream, "_writeback", record)
    target = tmp_path / "output.safetensors"
    expected = {"weight": torch.arange(100)}
    checkpoint_stream.save_safetensors_stream(expected, target, chunk_bytes=32, writeback_bytes=64)
    assert len(calls) > 1 and calls == sorted(calls) and calls[-1] == target.stat().st_size
    assert torch.equal(load_file(str(target))["weight"], expected["weight"])


@pytest.mark.parametrize("unsupported", [errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL])
def test_unsupported_advisory_eviction_does_not_prevent_synced_checkpoint(tmp_path, monkeypatch, unsupported):
    if not hasattr(checkpoint_stream.os, "posix_fadvise"):
        pytest.skip("No POSIX advice interface")
    def no_advice(*args):
        raise OSError(unsupported, "advice unsupported")
    monkeypatch.setattr(checkpoint_stream.os, "posix_fadvise", no_advice)
    target = tmp_path / "model.safetensors"
    checkpoint_stream.save_safetensors_stream({"weight": torch.arange(5)}, target)
    assert torch.equal(load_file(str(target))["weight"], torch.arange(5))


@pytest.mark.parametrize("failed_operation", ["fsync", "posix_fadvise"])
def test_real_writeback_errors_remove_partial_checkpoint(tmp_path, monkeypatch, failed_operation):
    if not hasattr(checkpoint_stream.os, failed_operation):
        pytest.skip("Writeback interface unavailable")
    def failed(*args):
        raise OSError(errno.EIO, "injected writeback failure")
    monkeypatch.setattr(checkpoint_stream.os, failed_operation, failed)
    target = tmp_path / "model.safetensors"
    with pytest.raises(OSError, match="injected writeback failure"):
        checkpoint_stream.save_safetensors_stream({"weight": torch.arange(5)}, target)
    assert not target.exists()
