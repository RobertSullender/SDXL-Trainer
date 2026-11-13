"""Frozen training caches preserve identities, RNG, and bounded host memory."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from sdxl_trainer.training.fast_cache import FrozenSampleCache, FrozenTextCache


def entry(number=1):
    return {"image_sha256": f"{number:064x}", "width": 16, "height": 24}


def posterior(value=0.25, deterministic=False):
    return SimpleNamespace(mean=torch.full((1, 4, 3, 2), value, dtype=torch.float32),
                           logvar=torch.full((1, 4, 3, 2), -0.5),
                           deterministic=deterministic)


def samples(path, **kwargs):
    return FrozenSampleCache(path, "base-model", seed=kwargs.pop("seed", 42),
                             vae_precision="fp32", latent_channels=4, scale_factor=8,
                             scaling_factor=0.13025, output_dtype=kwargs.pop("output_dtype", "bf16"),
                             **kwargs)


def texts(path, **kwargs):
    return FrozenTextCache(path, "base-model", caption_mode=kwargs.pop("caption_mode", "native"),
                           precision=kwargs.pop("precision", "bf16"), encoding_policy=kwargs.pop("encoding_policy", "cuda-tf32"),
                           prompt_width=4, pooled_width=3, **kwargs)


def text_pair(windows=1, value=1.0, dtype=torch.bfloat16):
    return torch.full((1, 77 * windows, 4), value, dtype=dtype), torch.full((1, 3), value, dtype=dtype)


def test_fixed_sample_is_reproducible_after_cache_deletion_and_preserves_rng(tmp_path):
    cache = samples(tmp_path / "first")
    before = torch.get_rng_state().clone()
    first = cache.get(entry(), posterior)
    assert torch.equal(before, torch.get_rng_state())
    cache.path_for(entry()).unlink()
    assert cache.lookup(entry()) is None
    again = cache.get(entry(), posterior)
    rebuilt = samples(tmp_path / "second").get(entry(), posterior)
    assert torch.equal(first, again) and torch.equal(first, rebuilt)
    assert torch.equal(before, torch.get_rng_state())
    assert first.dtype == torch.bfloat16 and first.device.type == "cpu"
    assert not torch.equal(first, samples(tmp_path / "seed", seed=43).get(entry(), posterior))
    assert not torch.equal(first, cache.get(entry(2), posterior))


def test_sample_hit_skips_vae_and_results_do_not_alias_ram_cache(tmp_path):
    cache = samples(tmp_path)
    expected = cache.get(entry(), posterior)
    expected.zero_()
    def fail():
        pytest.fail("A cache hit must not encode the image again")
    actual = cache.get(entry(), fail)
    assert torch.count_nonzero(actual) > 0
    assert cache.memory_bytes == actual.numel() * actual.element_size()
    assert cache.memory_entries == 1


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_sample_scaling_and_storage_precision(tmp_path, precision):
    cache = samples(tmp_path, output_dtype=precision)
    latent = cache.get(entry(), lambda: posterior(2, deterministic=True))
    expected = torch.full_like(latent, 2 * 0.13025)
    assert torch.equal(latent, expected)
    assert torch.equal(latent, samples(tmp_path, output_dtype=precision).lookup(entry()))


def test_sample_lru_is_bounded_and_disk_only_is_supported(tmp_path):
    size = 1 * 4 * 3 * 2 * 2
    cache = samples(tmp_path / "bounded", memory_limit_bytes=size)
    first = cache.get(entry(), posterior)
    second = cache.get(entry(2), posterior)
    assert cache.memory_entries == 1 and cache.memory_bytes == size
    assert torch.equal(cache.lookup(entry()), first)
    assert cache.memory_entries == 1 and cache.memory_bytes == size
    disk = samples(tmp_path / "disk", memory_limit_bytes=0)
    assert torch.equal(disk.get(entry(), posterior), first)
    assert disk.memory_entries == 0 and disk.memory_bytes == 0


def test_clear_sample_ram_retains_disk_sample_and_skips_vae(tmp_path):
    cache = samples(tmp_path)
    expected = cache.get(entry(), posterior)
    path = cache.path_for(entry())
    original_bytes = path.read_bytes()
    assert cache.memory_bytes > 0 and cache.memory_entries == 1
    cache.clear_memory()
    assert cache.memory_bytes == cache.memory_entries == 0
    assert path.read_bytes() == original_bytes
    before = torch.get_rng_state().clone()
    actual = cache.get(entry(), lambda: pytest.fail("Clearing RAM must not invoke the VAE"))
    assert torch.equal(actual, expected) and torch.equal(before, torch.get_rng_state())
    assert 0 < cache.memory_bytes <= cache._cache.memory_limit_bytes
    cache.clear_memory()
    cache.clear_memory()
    assert cache.memory_bytes == cache.memory_entries == 0


def test_sample_rejects_bad_posterior_before_publishing(tmp_path):
    cache = samples(tmp_path)
    wrong = SimpleNamespace(mean=torch.zeros((1, 4, 2, 2)), logvar=torch.zeros((1, 4, 2, 2)))
    with pytest.raises(ValueError, match="posterior shape"):
        cache.get(entry(), lambda: wrong)
    bad = posterior()
    bad.mean[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite VAE posterior"):
        cache.get(entry(), lambda: bad)
    assert list(tmp_path.glob("*.safetensors")) == []


def test_sample_detects_corrupted_disk_entry_even_when_ram_cached(tmp_path):
    cache = samples(tmp_path)
    cache.get(entry(), posterior)
    cache.path_for(entry()).write_bytes(b"broken")
    with pytest.raises(ValueError, match="Invalid frozen cache"):
        cache.lookup(entry())
    with pytest.raises(ValueError, match="Invalid frozen cache"):
        cache.get(entry(), posterior)
    assert cache.path_for(entry()).read_bytes() == b"broken"


def test_sample_checks_saved_shape_and_integrity(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import save_file
    cache = samples(tmp_path)
    cache.get(entry(), posterior)
    path = cache.path_for(entry())
    with safe_open(str(path), framework="pt") as opened:
        metadata = opened.metadata()
        saved = opened.get_tensor("latent")
    save_file({"latent": saved + 1}, str(path), metadata=metadata)
    with pytest.raises(ValueError, match="integrity"):
        cache.lookup(entry())
    save_file({"latent": torch.zeros((1, 4, 1, 1), dtype=torch.bfloat16)}, str(path), metadata=metadata)
    with pytest.raises(ValueError, match="invalid shape"):
        cache.lookup(entry())


def test_text_cache_preserves_full_tensor_pair_and_skips_encoder(tmp_path):
    cache = texts(tmp_path)
    before = torch.get_rng_state().clone()
    prompt, pooled = cache.get("caption", text_pair)
    assert torch.equal(before, torch.get_rng_state())
    expected = prompt.clone()
    prompt.zero_()
    cached = cache.get("caption", lambda: pytest.fail("A cache hit must skip text encoders"))
    assert torch.equal(cached[0], expected)
    assert torch.equal(cached[1], pooled)
    assert all(tensor.device.type == "cpu" for tensor in cached)
    assert all(not tensor.requires_grad for tensor in cached)


def test_text_cache_keys_include_caption_mode_precision_and_encode_policy(tmp_path):
    native = texts(tmp_path)
    chunked = texts(tmp_path, caption_mode="chunked")
    fp32 = texts(tmp_path, precision="fp32")
    cpu = texts(tmp_path, encoding_policy="cpu-strict")
    keys = {cache.path_for("a") for cache in (native, chunked, fp32, cpu)}
    assert len(keys) == 4
    assert native.path_for("a") != native.path_for("b")
    assert chunked.lookup("a") is None
    chunked.get("a", lambda: text_pair(2))
    assert native.lookup("a") is None
    assert chunked.lookup("a")[0].shape[1] == 154


def test_text_cache_lru_budget_and_invalidation(tmp_path):
    size = (77 * 4 + 3) * 2
    cache = texts(tmp_path, memory_limit_bytes=size)
    cache.get("a", text_pair)
    cache.get("b", lambda: text_pair(value=2))
    assert cache.memory_bytes == size and cache.memory_entries == 1
    assert torch.equal(cache.lookup("a")[0], text_pair()[0])
    assert cache.memory_bytes == size and cache.memory_entries == 1
    cache.path_for("a").unlink()
    assert cache.lookup("a") is None
    assert cache.memory_bytes == 0


def test_clear_text_ram_retains_all_windows_and_skips_encoders(tmp_path):
    cache = texts(tmp_path, caption_mode="chunked")
    expected = cache.get("complete caption", lambda: text_pair(3))
    path = cache.path_for("complete caption")
    original_bytes = path.read_bytes()
    assert cache.memory_bytes > 0 and cache.memory_entries == 1
    cache.clear_memory()
    assert cache.memory_bytes == cache.memory_entries == 0
    assert path.read_bytes() == original_bytes
    actual = cache.get("complete caption", lambda: pytest.fail("Clearing RAM must not invoke text encoders"))
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))
    assert actual[0].shape[1] == 231
    assert 0 < cache.memory_bytes <= cache._cache.memory_limit_bytes


def test_text_cache_rejects_bad_shape_dtype_and_nonfinite_values(tmp_path):
    cache = texts(tmp_path)
    with pytest.raises(ValueError, match="shape"):
        cache.get("two windows", lambda: text_pair(2))
    with pytest.raises(ValueError, match="dtype"):
        cache.get("wrong precision", lambda: text_pair(dtype=torch.float32))
    with pytest.raises(ValueError, match="non-finite"):
        cache.get("bad", lambda: text_pair(value=float("inf")))
    assert list(tmp_path.glob("*.safetensors")) == []


def test_text_cache_rejects_corruption_and_symlink_replacement(tmp_path):
    cache = texts(tmp_path)
    cache.get("a", text_pair)
    path = cache.path_for("a")
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="Invalid frozen cache"):
        cache.lookup("a")
    path.unlink()
    other = tmp_path / "other"
    other.write_bytes(b"data")
    path.symlink_to(other)
    with pytest.raises(ValueError, match="not a regular file"):
        cache.lookup("a")


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_invalid_memory_budgets_are_rejected(tmp_path, limit):
    with pytest.raises(ValueError, match="RAM limit"):
        texts(tmp_path / "text", memory_limit_bytes=limit)
    with pytest.raises(ValueError, match="RAM limit"):
        samples(tmp_path / "sample", memory_limit_bytes=limit)
