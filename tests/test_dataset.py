import json
from pathlib import Path

from PIL import Image, PngImagePlugin
import pytest

from sdxl_trainer.dataset import BUCKETS, DatasetProject, nearest_bucket, sha256, verify_finalized
from sdxl_trainer.tokens import TokenCounter


class FakeTokenizer:
    model_max_length = 77

    def __call__(self, text, **kwargs):
        return {"input_ids": [0] + list(range(len(text.split()))) + [1]}


@pytest.fixture
def counter():
    return TokenCounter(FakeTokenizer(), FakeTokenizer())


def source_image(tmp_path, name="source.png", size=(300, 200), mode="RGB", color="red"):
    source = tmp_path / name
    image = Image.new(mode, size, color)
    info = PngImagePlugin.PngInfo()
    info.add_text("private", "GPS and author metadata")
    image.save(source, pnginfo=info)
    return source


def test_nearest_bucket_uses_aspect_and_valid_dimensions():
    assert nearest_bucket(2048, 1024) == (1472, 704)
    assert nearest_bucket(200, 200) == (1024, 1024)
    assert nearest_bucket(1, 10000) == (512, 2048)
    assert all(w % 64 == h % 64 == 0 for w, h in BUCKETS)
    with pytest.raises(ValueError):
        nearest_bucket(0, 1)


def test_import_non_destructive_metadata_free_rgb_and_numbered(tmp_path):
    source = source_image(tmp_path)
    before = sha256(source)
    project = DatasetProject.create(tmp_path / "draft")
    first = project.import_images([source])[0]
    second = project.import_images([source])[0]
    assert (first.image, second.image) == ("Img-1.png", "Img-2.png")
    assert project.caption_text(1) == ""
    assert sha256(source) == before
    with Image.open(project.image_path(1)) as image:
        assert image.info == {}
        assert image.mode == "RGB"
        assert image.size == (first.width, first.height)
    reopened = DatasetProject.open(project.root)
    assert reopened.entries == project.entries


def test_alpha_composites_before_preprocessing(tmp_path):
    source = source_image(tmp_path, mode="RGBA", color=(255, 0, 0, 0))
    project = DatasetProject.create(tmp_path / "draft")
    project.import_images([source], fit="pad", background="#123456")
    with Image.open(project.image_path(1)) as image:
        assert image.getpixel((image.width // 2, image.height // 2)) == (18, 52, 86)


def test_exif_orientation_applied_before_bucket_choice(tmp_path):
    source = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (600, 300), "red")
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)
    project = DatasetProject.create(tmp_path / "draft")
    entry = project.import_images([source])[0]
    assert entry.original_size == [300, 600]
    assert entry.width < entry.height
    with Image.open(project.image_path(1)) as prepared:
        assert not prepared.getexif()


def test_bad_import_batch_rolls_back_every_file(tmp_path):
    source = source_image(tmp_path)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"invalid")
    project = DatasetProject.create(tmp_path / "draft")
    with pytest.raises(Exception):
        project.import_images([source, bad])
    assert project.entries == []
    assert list(project.image_dir.iterdir()) == []
    assert DatasetProject.open(project.root).entries == []


def test_folder_import_is_sorted_and_nonrecursive_by_default(tmp_path):
    folder = tmp_path / "originals"
    folder.mkdir()
    source_image(folder, "z.png")
    source_image(folder, "a.png")
    nested = folder / "nested"
    nested.mkdir()
    source_image(nested, "nested.png")
    (folder / "notes.txt").write_text("not an image")
    project = DatasetProject.create(tmp_path / "draft")
    rows = project.import_images([folder])
    assert [Path(row.source).name for row in rows] == ["a.png", "z.png"]


def test_finalize_copies_caption_and_protects_source_and_snapshot(tmp_path, counter):
    source = source_image(tmp_path)
    project = DatasetProject.create(tmp_path / "draft")
    project.import_images([source])
    project.save_caption(1, "A red image.")
    output = project.finalize(tmp_path / "finalized", counter)
    manifest = verify_finalized(output)
    assert set(p.name for p in output.iterdir()) == {"manifest.json", "Img-1.png", "Img-1.txt"}
    assert "source" not in manifest["entries"][0]
    assert manifest["entries"][0]["tokens"]["clip_l"] == 5
    project.save_caption(1, "A changed draft.")
    assert (output / "Img-1.txt").read_text() == "A red image."
    with pytest.raises(ValueError, match="never overwritten"):
        project.finalize(output, counter)


def test_finalize_refuses_empty_and_overlong_captions(tmp_path, counter):
    project = DatasetProject.create(tmp_path / "draft")
    project.import_images([source_image(tmp_path)])
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="empty"):
        project.finalize(destination, counter)
    project.save_caption(1, " ".join(["word"] * 76))
    with pytest.raises(ValueError, match="Shorten"):
        project.finalize(destination, counter)
    assert not destination.exists()
    assert not list(tmp_path.glob(".out.*"))


def test_finalization_requires_exact_token_counter(tmp_path):
    project = DatasetProject.create(tmp_path / "draft")
    with pytest.raises(ValueError, match="tokenizers"):
        project.finalize(tmp_path / "out", None)


def test_changed_pixels_require_recaptions_and_tampered_export_rejected(tmp_path, counter):
    project = DatasetProject.create(tmp_path / "draft")
    project.import_images([source_image(tmp_path)])
    project.save_caption(1, "A red image.")
    output = project.finalize(tmp_path / "out", counter)
    (output / "Img-1.txt").write_text("Changed!")
    with pytest.raises(ValueError, match="integrity"):
        verify_finalized(output)
    with project.image_path(1).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed after import"):
        project.finalize(tmp_path / "out2", counter)


def test_project_rejects_traversal_and_symlink_caption(tmp_path):
    project = DatasetProject.create(tmp_path / "draft")
    project.import_images([source_image(tmp_path)])
    caption = project.image_dir / "Img-1.txt"
    caption.unlink()
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched")
    caption.symlink_to(victim)
    with pytest.raises(ValueError, match="regular local"):
        project.save_caption(1, "damage")
    assert victim.read_text() == "untouched"


def test_invalid_color_profile_fails_instead_of_silently_changing_colors(tmp_path):
    source = tmp_path / "profile.png"
    Image.new("RGB", (100, 100), "red").save(source, icc_profile=b"not a profile")
    project = DatasetProject.create(tmp_path / "draft")
    with pytest.raises(ValueError, match="color profile"):
        project.import_images([source])
    assert project.entries == []


def test_existing_project_folder_never_overwritten(tmp_path):
    marker = tmp_path / "notes.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="never replaced"):
        DatasetProject.create(tmp_path)
    assert marker.read_text() == "keep"


def test_import_own_processed_images_rejected(tmp_path):
    project = DatasetProject.create(tmp_path / "draft")
    project.import_images([source_image(tmp_path)])
    with pytest.raises(ValueError, match="outside"):
        project.import_images([project.image_path(1)])
