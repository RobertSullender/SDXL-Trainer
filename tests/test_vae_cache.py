"""Frozen VAE caching preserves stochastic samples and verifies cached data."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("safetensors")
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from sdxl_trainer.training.vae_cache import FrozenVaeCache, make_frozen_vae_cache


class FakeVae:
    def __init__(self):
        self.calls = 0

    def encode(self, pixels):
        self.calls += 1
        mean = torch.nn.functional.avg_pool2d(pixels[:, :1], 2)
        logvar = torch.full_like(mean, -0.25)
        return SimpleNamespace(latent_dist=DiagonalGaussianDistribution(torch.cat([mean, logvar], dim=1)))


def entries(n=2):
    return [{"image_sha256": format(i + 1, "064x"), "width": 8, "height": 8} for i in range(n)]


def test_cache_preserves_original_sampling_and_draws_new_noise(tmp_path):
    pixels = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8) / 100
    base = FakeVae()
    torch.manual_seed(51)
    original = base.encode(pixels).latent_dist.sample()
    expected_rng = torch.get_rng_state().clone()
    cache_vae = FakeVae()
    cache = FrozenVaeCache(tmp_path / "cache", "model-one", 1, 2)
    torch.manual_seed(51)
    first = cache.posterior(cache_vae, pixels, entries()).sample()
    assert cache_vae.calls == 2
    torch.testing.assert_close(first, original, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    second = cache.posterior(cache_vae, pixels, entries()).sample()
    assert cache_vae.calls == 2
    assert not torch.equal(first, second)
    torch.manual_seed(51)
    third = cache.posterior(cache_vae, pixels, entries()).sample()
    torch.testing.assert_close(third, original, rtol=0, atol=0)


def test_latent_finiteness_check_preserves_sampling_and_rng():
    from sdxl_trainer.training.worker import sample_vae_latents
    vae=FakeVae(); pixels=torch.ones(2,3,8,8)
    posterior=vae.encode(pixels).latent_dist
    torch.manual_seed(92)
    expected=posterior.sample().detach().to('cpu') * .13025
    state=torch.get_rng_state().clone()
    torch.manual_seed(92)
    actual=sample_vae_latents(posterior,.13025,'cpu','fp32')
    assert torch.equal(actual,expected) and torch.equal(torch.get_rng_state(),state)


@pytest.mark.parametrize('value,scale', [(float('nan'),1.0),(float('inf'),1.0),(65504.,2.0)])
def test_nonfinite_fp16_latents_report_actionable_precision(value,scale):
    from sdxl_trainer.training.worker import sample_vae_latents
    class Posterior:
        calls=0
        def sample(self):
            self.calls+=1
            return torch.tensor([value],dtype=torch.float16)
    posterior=Posterior()
    with pytest.raises(FloatingPointError,match='Non-finite VAE latents.*fp16.*fp32 or bf16'):
        sample_vae_latents(posterior,scale,'cpu','fp16')
    assert posterior.calls==1


def test_nonfinite_fp32_latents_do_not_claim_fp16_overflow():
    from sdxl_trainer.training.worker import sample_vae_latents
    posterior=SimpleNamespace(sample=lambda: torch.tensor([float('nan')]))
    with pytest.raises(FloatingPointError,match='VAE precision: fp32') as error:
        sample_vae_latents(posterior,1.,'cpu','fp32')
    assert 'FP16 VAE computation can overflow' not in str(error.value)


def test_real_diffusers_vae_sliced_batch_is_bitwise_identical(tmp_path):
    from diffusers import AutoencoderKL
    torch.set_num_threads(2)
    torch.manual_seed(19)
    vae = AutoencoderKL(in_channels=3, out_channels=3, block_out_channels=(8, 8),
                        down_block_types=("DownEncoderBlock2D",) * 2,
                        up_block_types=("UpDecoderBlock2D",) * 2,
                        latent_channels=4, layers_per_block=1, norm_num_groups=4,
                        sample_size=16).eval().requires_grad_(False)
    vae.enable_slicing()
    pixels = torch.randn(2, 3, 16, 16)
    batch = [{"image_sha256": format(i + 1, "064x"), "width": 16, "height": 16} for i in range(2)]
    torch.manual_seed(32)
    with torch.no_grad():
        original = vae.encode(pixels).latent_dist.sample()
    expected_rng = torch.get_rng_state().clone()
    cache = FrozenVaeCache(tmp_path / "cache", "model-a", 4, 2)
    torch.manual_seed(32)
    with torch.no_grad():
        cached = cache.posterior(vae, pixels, batch).sample()
    torch.testing.assert_close(cached, original, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_corruption_and_symlinks_fail_closed(tmp_path):
    pixels = torch.ones(1, 3, 8, 8)
    vae = FakeVae()
    cache = FrozenVaeCache(tmp_path / "cache", "model-one", 1, 2)
    cache.posterior(vae, pixels, entries(1))
    file = next(cache.directory.glob("*.safetensors"))
    file.write_bytes(b"truncated or tampered")
    with pytest.raises(ValueError, match="Invalid frozen VAE cache"):
        cache.posterior(vae, pixels, entries(1))
    assert file.read_bytes() == b"truncated or tampered"
    file.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(b"untouched")
    file.symlink_to(outside)
    with pytest.raises(ValueError, match="regular file"):
        cache.posterior(vae, pixels, entries(1))
    assert outside.read_bytes() == b"untouched"


def test_model_or_image_fingerprint_gets_a_distinct_cache(tmp_path):
    pixels = torch.ones(1, 3, 8, 8)
    vae = FakeVae()
    a = FrozenVaeCache(tmp_path / "cache", "model-a", 1, 2)
    b = FrozenVaeCache(tmp_path / "cache", "model-b", 1, 2)
    a.posterior(vae, pixels, entries(1))
    b.posterior(vae, pixels, entries(1))
    changed = [{**entries(1)[0], "image_sha256": "a" * 64}]
    a.posterior(vae, pixels, changed)
    assert vae.calls == 3
    assert len(list(a.directory.glob("*.safetensors"))) == 3


def test_lazy_pixels_mixed_hits_and_misses_preserve_batch_samples(tmp_path):
    pixels = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8) / 100
    vae = FakeVae()
    cache = FrozenVaeCache(tmp_path / "cache", "model", 1, 2, memory_limit_bytes=1024)
    cache.posterior(vae, pixels[:1], entries(1))
    loaded = []
    def load(index):
        loaded.append(index)
        return pixels[index:index + 1]
    torch.manual_seed(41)
    expected = FakeVae().encode(pixels).latent_dist.sample()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(41)
    actual = cache.posterior(vae, load, entries()).sample()
    assert loaded == [1]
    assert vae.calls == 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    loaded.clear()
    again = cache.posterior(vae, load, entries()).sample()
    assert loaded == []
    assert not torch.equal(actual, again)


def test_ram_limit_lru_and_changed_disk_entry_are_checked(tmp_path):
    pixels = torch.ones(2, 3, 8, 8)
    vae = FakeVae()
    # One pair is two fp32 [1, 1, 4, 4] tensors = 128 bytes.
    cache = FrozenVaeCache(tmp_path / "cache", "model", 1, 2, memory_limit_bytes=128)
    cache.posterior(vae, pixels, entries())
    assert cache._memory_bytes == 128
    assert len(cache._memory) == 1
    cache.posterior(vae, pixels[:1], entries(1))  # reloads evicted entry from disk
    assert len(cache._memory) == 1
    file = cache.directory / (cache._key(entries(1)[0]) + ".safetensors")
    file.write_bytes(b"changed after RAM caching")
    with pytest.raises(ValueError, match="Invalid frozen VAE cache"):
        cache.posterior(vae, pixels[:1], entries(1))
    assert cache._memory_bytes == 0
    assert vae.calls == 2


def test_zero_ram_limit_still_uses_validated_disk_cache(tmp_path):
    pixels = torch.ones(1, 3, 8, 8)
    vae = FakeVae()
    cache = FrozenVaeCache(tmp_path / "cache", "model", 1, 2, memory_limit_bytes=0)
    cache.posterior(vae, pixels, entries(1))
    cache.posterior(vae, lambda _: pytest.fail("cached image should not be decoded"), entries(1))
    assert vae.calls == 1 and cache._memory_bytes == 0


def test_clear_ram_preserves_posterior_files_and_fresh_sampling(tmp_path):
    pixels = torch.ones(1, 3, 8, 8)
    vae = FakeVae()
    cache = FrozenVaeCache(tmp_path / "cache", "model", 1, 2, memory_limit_bytes=128)
    torch.manual_seed(71)
    expected = cache.posterior(vae, pixels, entries(1)).sample()
    expected_rng = torch.get_rng_state().clone()
    saved = {path: path.read_bytes() for path in cache.directory.glob("*.safetensors")}
    assert cache._memory_bytes == 128
    cache.clear_memory()
    assert cache._memory_bytes == 0 and len(cache._memory) == 0
    assert all(path.read_bytes() == contents for path, contents in saved.items())
    torch.manual_seed(71)
    actual = cache.posterior(vae, lambda _: pytest.fail("Clearing RAM must not decode pixels"), entries(1)).sample()
    assert vae.calls == 1
    assert torch.equal(actual, expected) and torch.equal(torch.get_rng_state(), expected_rng)
    assert cache._memory_bytes == 128


def test_trainable_vae_and_disabled_option_never_create_cache(tmp_path):
    vae = SimpleNamespace(config=SimpleNamespace(latent_channels=1, block_out_channels=[1, 1]))
    assert make_frozen_vae_cache(SimpleNamespace(train_vae=True, cache_frozen_vae=True), tmp_path, "m", vae) is None
    assert make_frozen_vae_cache(SimpleNamespace(train_vae=False, cache_frozen_vae=False), tmp_path, "m", vae) is None
    assert not (tmp_path / "working").exists()


def test_posterior_cache_and_observability_options_do_not_change_resume_identity():
    from sdxl_trainer.training.config import TrainingConfig
    from sdxl_trainer.training.worker import _config_identity
    a = TrainingConfig(frozen_vae_cache_mode="posterior", cache_frozen_vae=True,
                       profile_phases=True, monitor_gpu=True)
    b = TrainingConfig(frozen_vae_cache_mode="posterior", cache_frozen_vae=False,
                       vae_cache_ram_mib=0, profile_phases=False, monitor_gpu=False)
    assert _config_identity(a) == _config_identity(b)


def test_fixed_sample_cache_choice_changes_resume_identity_but_ram_budget_does_not():
    from sdxl_trainer.training.config import TrainingConfig
    from sdxl_trainer.training.worker import _config_identity
    cached = TrainingConfig(frozen_vae_cache_mode="sample", cache_frozen_vae=True)
    fresh = TrainingConfig(frozen_vae_cache_mode="sample", cache_frozen_vae=False)
    assert _config_identity(cached) != _config_identity(fresh)
    disk_only = TrainingConfig(frozen_vae_cache_mode="sample", cache_frozen_vae=True,
                               vae_cache_ram_mib=0, frozen_text_cache_ram_mib=0)
    assert _config_identity(cached) == _config_identity(disk_only)
