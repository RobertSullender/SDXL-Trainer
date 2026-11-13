import json
from pathlib import Path

from PIL import Image, PngImagePlugin, UnidentifiedImageError
import pytest

from sdxl_trainer.dataset import (
    BUCKETS, DatasetProject, load_training_dataset, nearest_bucket, sha256, verify_finalized, validate_project_name,
)
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
    source = source_image(tmp_path, size=(1024, 1024))
    small = source_image(tmp_path, "small.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"invalid")
    project = DatasetProject.create(tmp_path / "draft")
    skipped = []
    with pytest.raises(UnidentifiedImageError):
        project.import_images([source, small, bad], allow_upscale=False, on_skip=skipped.append)
    assert skipped == [small]
    assert project.entries == []
    assert project.next_image_id == 1
    assert list(project.image_dir.iterdir()) == []
    assert DatasetProject.open(project.root).entries == []


@pytest.mark.parametrize(("fit", "accepted", "skipped"), [
    ("crop", ["b.png", "d.png"], ["a.png", "c.png"]),
    ("pad", ["b.png", "c.png", "d.png"], ["a.png"]),
])
def test_import_without_upscaling_respects_fit_and_keeps_ids_contiguous(tmp_path, fit, accepted, skipped):
    folder = tmp_path / "originals"
    folder.mkdir()
    sizes = [(500, 500), (1024, 1024), (1024, 1000), (1200, 1200)]
    originals = [source_image(folder, f"{name}.png", size=size) for name, size in zip("abcd", sizes)]
    original_hashes = [sha256(path) for path in originals]
    project = DatasetProject.create(tmp_path / "draft")
    skipped_sources = []

    entries = project.import_images([folder], fit=fit, allow_upscale=False, on_skip=skipped_sources.append)

    assert [Path(entry.source).name for entry in entries] == accepted
    assert [path.name for path in skipped_sources] == skipped
    assert [entry.id for entry in entries] == list(range(1, len(accepted) + 1))
    assert all((entry.width, entry.height) == (1024, 1024) for entry in entries)
    assert DatasetProject.open(project.root).next_image_id == len(accepted) + 1
    assert [sha256(path) for path in originals] == original_hashes


def test_import_all_skipped_keeps_project_unchanged(tmp_path, monkeypatch):
    project = DatasetProject.create(tmp_path / "draft")
    source = source_image(tmp_path)
    monkeypatch.setattr(project, "save", lambda: pytest.fail("All-skipped import wrote project metadata"))

    assert project.import_images([source], allow_upscale=False) == []
    assert project.next_image_id == 1
    assert not project.entries
    assert list(project.image_dir.iterdir()) == []
    assert not list(project.root.glob(".import-*"))


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


def test_partial_caption_survives_restart_and_blocks_finalize_until_edited(tmp_path, counter):
    project = DatasetProject.create(tmp_path / "draft")
    project.import_images([source_image(tmp_path)])
    project.save_generated_caption(1, "Partial caption", "Context full")
    project = DatasetProject.open(project.root)
    assert project.incomplete_captions()[0].caption_incomplete_reason == "Context full"
    # The UI frequently saves unchanged editor contents, which must not count as review.
    project.save_caption(1, "Partial caption")
    with pytest.raises(ValueError, match="Img-1.png: Context full"):
        project.finalize(tmp_path / "blocked", counter)
    assert not (tmp_path / "blocked").exists()
    project.save_caption(1, "Partial caption, reviewed and finished.")
    assert not DatasetProject.open(project.root).incomplete_captions()
    assert verify_finalized(project.finalize(tmp_path / "ready", counter))["entries"]


def test_partial_caption_status_is_safe_on_write_failure_and_complete_recaption(tmp_path, monkeypatch):
    import sdxl_trainer.dataset as dataset
    project = DatasetProject.create(tmp_path / "draft")
    project.import_images([source_image(tmp_path)])
    original = dataset._atomic_text
    def fail_caption(path, text):
        if path.name == "Img-1.txt":
            raise OSError("simulated interrupted write")
        return original(path, text)
    monkeypatch.setattr(dataset, "_atomic_text", fail_caption)
    with pytest.raises(OSError, match="simulated interrupted write"):
        project.save_generated_caption(1, "Partial", "Generation interrupted")
    assert DatasetProject.open(project.root).incomplete_captions()[0].image == "Img-1.png"
    monkeypatch.setattr(dataset, "_atomic_text", original)
    project.save_generated_caption(1, "Complete caption")
    assert not DatasetProject.open(project.root).incomplete_captions()


def test_bulk_caption_prepend_and_replace_only_edit_draft(tmp_path, counter):
    project = DatasetProject.create(tmp_path / "draft")
    source = source_image(tmp_path)
    project.import_images([source])
    project.save_caption(1, "A studio portrait.")
    exported = project.finalize_in_project(counter)
    project.import_images([source])
    project.import_images([source])
    project.save_generated_caption(2, "A red mat", "Context full")

    assert project.apply_bulk_caption("  yoga,  ", prepend=True) == 3
    assert [project.caption_text(i) for i in (1, 2, 3)] == ["yoga, A studio portrait.", "yoga, A red mat", "yoga"]
    assert project.apply_bulk_caption("yoga", prepend=True) == 0
    assert [entry.id for entry in DatasetProject.open(project.root).incomplete_captions()] == [2]
    project.save_generated_caption(3, "yoga", "Context full")

    assert project.apply_bulk_caption("yoga") == 3
    reopened = DatasetProject.open(project.root)
    assert [reopened.caption_text(i) for i in (1, 2, 3)] == ["yoga", "yoga", "yoga"]
    assert not reopened.incomplete_captions()
    assert (exported / "Img-1.txt").read_text() == "A studio portrait."


def test_bulk_caption_only_fills_empty_and_whitespace_captions(tmp_path):
    project = DatasetProject.create(tmp_path / "draft")
    for index in range(4):
        project.import_images([source_image(tmp_path, f"source-{index}.png")])
    project.save_caption(1, "A studio portrait.\n")
    project.save_generated_caption(2, "Partial description.", "Context full")
    project.save_generated_caption(3, "  \n\t", "Context full")

    assert project.count_empty_captions() == 2
    assert project.count_empty_captions(image_ids=[1, 3, 3]) == 1
    assert project.apply_bulk_caption("  yoga  ", only_empty=True, image_ids=[1, 3]) == 1
    assert project.caption_text(4) == ""
    assert project.apply_bulk_caption("  yoga  ", only_empty=True) == 1
    reopened = DatasetProject.open(project.root)
    assert [reopened.caption_text(i) for i in (1, 2, 3, 4)] == [
        "A studio portrait.\n", "Partial description.", "yoga", "yoga",
    ]
    assert [entry.id for entry in reopened.incomplete_captions()] == [2]
    assert reopened.count_empty_captions() == 0
    assert reopened.apply_bulk_caption("new text", only_empty=True) == 0


def test_bulk_caption_append_only_updates_selected_and_preserves_partial_status(tmp_path, counter):
    project = DatasetProject.create(tmp_path / "draft")
    source = source_image(tmp_path)
    project.import_images([source])
    project.save_caption(1, "A studio portrait,")
    exported = project.finalize_in_project(counter)
    for _ in range(3):
        project.import_images([source])
    project.save_generated_caption(3, "A blue mat,  \n", "Context full")
    project.save_caption(4, "Keep this caption.")

    assert project.apply_bulk_caption(" , yoga, indoors,  ", append=True, image_ids=[3, 1, 3, 2]) == 3
    assert project.apply_bulk_caption("yoga, indoors", append=True, image_ids=[1, 2, 3]) == 0
    assert [project.caption_text(i) for i in (1, 2, 3, 4)] == [
        "A studio portrait, yoga, indoors", "yoga, indoors", "A blue mat, yoga, indoors", "Keep this caption.",
    ]
    assert [entry.id for entry in DatasetProject.open(project.root).incomplete_captions()] == [3]
    assert (exported / "Img-1.txt").read_text() == "A studio portrait,"
    assert project.apply_bulk_caption("replacement", image_ids=[]) == 0
    with pytest.raises(ValueError, match="Unknown image number: 99"):
        project.apply_bulk_caption("replacement", image_ids=[1, 99])
    assert project.caption_text(1) == "A studio portrait, yoga, indoors"


def test_bulk_truncation_only_updates_selected_captions_in_draft_order(tmp_path, counter):
    project = DatasetProject.create(tmp_path / "draft")
    source = source_image(tmp_path)
    long_text = "long caption " * 40
    project.import_images([source])
    project.save_caption(1, long_text)
    exported = project.finalize_in_project(counter)
    for _ in range(4):
        project.import_images([source])
    project.save_caption(2, "Short caption.")
    project.save_generated_caption(3, long_text + "continued", "Context full")
    project.save_caption(4, "  \t\n")
    project.save_caption(5, long_text)
    calls = []

    class TrimCounter:
        def truncate(self, text):
            calls.append(text)
            if text.startswith(long_text):
                return "long caption"
            return text

    assert project.truncate_captions(TrimCounter(), image_ids=[4, 3, 2, 1, 3]) == 2
    assert calls == [long_text, "Short caption.", long_text + "continued"]
    reopened = DatasetProject.open(project.root)
    assert [reopened.caption_text(i) for i in range(1, 6)] == [
        "long caption", "Short caption.", "long caption", "  \t\n", long_text,
    ]
    assert [entry.id for entry in reopened.incomplete_captions()] == [3]
    assert (exported / "Img-1.txt").read_text() == long_text


def test_bulk_truncation_aborts_on_tokenizer_failure_before_writing(tmp_path):
    project = DatasetProject.create(tmp_path / "draft")
    source = source_image(tmp_path)
    project.import_images([source])
    project.import_images([source])
    project.save_caption(1, "First caption")
    project.save_caption(2, "Second caption")

    class FailingCounter:
        def truncate(self, text):
            if text == "Second caption":
                raise RuntimeError("Tokenizer stopped")
            return "First"

    with pytest.raises(RuntimeError, match="Tokenizer stopped"):
        project.truncate_captions(FailingCounter())
    assert [project.caption_text(i) for i in (1, 2)] == ["First caption", "Second caption"]


def test_bulk_caption_rolls_back_text_and_status_when_metadata_write_fails(tmp_path, monkeypatch):
    project = DatasetProject.create(tmp_path / "draft")
    source = source_image(tmp_path)
    project.import_images([source])
    project.import_images([source])
    project.save_caption(1, "First caption")
    project.save_generated_caption(2, "Partial caption", "Context full")

    def fail_save():
        raise OSError("simulated project write failure")

    monkeypatch.setattr(project, "save", fail_save)
    with pytest.raises(OSError, match="simulated project write failure"):
        project.apply_bulk_caption("yoga")
    for draft in (project, DatasetProject.open(project.root)):
        assert [draft.caption_text(i) for i in (1, 2)] == ["First caption", "Partial caption"]
        assert [entry.caption_incomplete_reason for entry in draft.entries] == ["", "Context full"]
    assert not list(project.root.glob(".bulk-caption-*"))


def test_project_finalizations_expand_without_changing_earlier_snapshots(tmp_path, counter):
    project = DatasetProject.create(tmp_path / "Portrait study")
    source1 = source_image(tmp_path, "first.png")
    source2 = source_image(tmp_path, "second.png", color="blue")
    original_hashes = (sha256(source1), sha256(source2))
    project.import_images([source1])
    project.save_caption(1, "First image.")
    first = project.finalize_in_project(counter)
    assert first == project.root / "finalized" / "Dataset-1"
    first_bytes = {p.name: sha256(p) for p in first.iterdir()}
    project.import_images([source2])
    project.save_caption(1, "Revised caption.")
    project.save_caption(2, "Second image.")
    second = project.finalize_in_project(counter)
    assert second == project.root / "finalized" / "Dataset-2"
    assert len(verify_finalized(first)["entries"]) == 1
    assert len(verify_finalized(second)["entries"]) == 2
    assert (first / "Img-1.txt").read_text() == "First image."
    assert (second / "Img-1.txt").read_text() == "Revised caption."
    assert {p.name: sha256(p) for p in first.iterdir()} == first_bytes
    assert (sha256(source1), sha256(source2)) == original_hashes
    assert len(DatasetProject.open(project.root).entries) == 2


@pytest.mark.parametrize("kind", ["file", "symlink", "dangling_symlink"])
def test_project_finalization_rejects_unsafe_finalized_folder(tmp_path, counter, kind):
    project = DatasetProject.create(tmp_path / "draft")
    project.import_images([source_image(tmp_path)])
    project.save_caption(1, "Reviewed caption.")
    folder = project.root / "finalized"
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("keep")
    if kind == "file":
        folder.write_text("do not change")
    else:
        folder.symlink_to(victim if kind == "symlink" else tmp_path / "missing", target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        project.finalize_in_project(counter)
    assert (victim / "keep.txt").read_text() == "keep"
    assert list(victim.iterdir()) == [victim / "keep.txt"]
    if kind == "file":
        assert folder.read_text() == "do not change"
    else:
        assert folder.is_symlink()


def test_project_finalization_rejects_symlinks_in_finalized_folder(tmp_path, counter):
    project = DatasetProject.create(tmp_path / "draft")
    project.import_images([source_image(tmp_path)])
    project.save_caption(1, "Reviewed caption.")
    folder = project.root / "finalized"
    folder.mkdir()
    (folder / "Dataset-1").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        project.finalize_in_project(counter)
    assert not (tmp_path / "outside").exists()


def test_project_finalization_skips_existing_versions_and_preserves_other_files(tmp_path, counter):
    project = DatasetProject.create(tmp_path / "draft")
    project.import_images([source_image(tmp_path)])
    project.save_caption(1, "Reviewed caption.")
    folder = project.root / "finalized"
    folder.mkdir()
    (folder / "Dataset-2").mkdir()
    (folder / "Dataset-2" / "keep.txt").write_text("keep")
    (folder / "notes.txt").write_text("notes")
    assert project.finalize_in_project(counter) == folder / "Dataset-3"
    assert (folder / "Dataset-2" / "keep.txt").read_text() == "keep"
    assert (folder / "notes.txt").read_text() == "notes"


def test_concurrent_finalization_never_replaces_a_snapshot(tmp_path, counter):
    from concurrent.futures import ThreadPoolExecutor
    project = DatasetProject.create(tmp_path / "draft")
    project.import_images([source_image(tmp_path)])
    project.save_caption(1, "Reviewed caption.")
    with ThreadPoolExecutor(max_workers=2) as executor:
        paths = list(executor.map(lambda _: project.finalize_in_project(counter), range(2)))
    assert set(paths) == {project.root / "finalized" / "Dataset-1", project.root / "finalized" / "Dataset-2"}
    assert all(len(verify_finalized(path)["entries"]) == 1 for path in paths)


def test_bucket_finalization_keeps_each_resolution_trainable_and_revisioned(tmp_path, counter):
    project = DatasetProject.create(tmp_path / "Photo-Realistic")
    square = source_image(tmp_path, "square.png", size=(120, 120))
    wide = source_image(tmp_path, "wide.png", size=(240, 120))
    project.import_images([square, wide])
    project.import_images([square])
    long_caption = " ".join(["detail"] * 176)
    for number, caption in enumerate((long_caption, "Wide caption", "Another square"), 1):
        project.save_caption(number, caption)
    legacy = project.finalize_in_project(counter)

    square_set, wide_set = project.finalize_buckets_in_project(counter)
    assert square_set == project.root / "finalized/Dataset-2/Photo-Realistic:1024X1024"
    assert wide_set == project.root / "finalized/Dataset-2/Photo-Realistic:1472X704"
    for path, dimensions, ids in ((square_set, (1024, 1024), [1, 3]), (wide_set, (1472, 704), [2])):
        manifest = verify_finalized(path)
        assert manifest["tokenizers"] == counter.metadata
        assert [entry["id"] for entry in manifest["entries"]] == ids
        assert {(entry["width"], entry["height"]) for entry in manifest["entries"]} == {dimensions}
        for entry in manifest["entries"]:
            assert sha256(path / entry["image"]) == sha256(project.image_path(entry["id"]))
            assert (path / entry["caption"]).read_text() == project.caption_text(entry["id"])
            assert "source" not in entry
    assert verify_finalized(square_set)["entries"][0]["tokens"] == counter.count(long_caption).to_dict()

    project.save_caption(1, "Edited after export")
    newest = project.finalize_buckets_in_project(counter)
    assert [path.parent.name for path in newest] == ["Dataset-3", "Dataset-3"]
    assert (square_set / "Img-1.txt").read_text() == long_caption
    assert (newest[0] / "Img-1.txt").read_text() == "Edited after export"
    assert project.finalized_datasets() == (*newest, square_set, wide_set, legacy)
    assert project.finalized_datasets(include_all=True) == (
        newest[0].parent, *newest, square_set.parent, square_set, wide_set, legacy,
    )
    assert load_training_dataset(legacy) == (verify_finalized(legacy), sha256(legacy / "manifest.json"))
    assert load_training_dataset(square_set) == (verify_finalized(square_set), sha256(square_set / "manifest.json"))


@pytest.fixture
def bucket_revision(tmp_path, counter):
    project = DatasetProject.create(tmp_path / "Portrait project")
    project.import_images([
        source_image(tmp_path, "square.png", size=(100, 100)),
        source_image(tmp_path, "wide.png", size=(200, 100)),
    ])
    project.save_caption(1, "Square caption")
    project.save_caption(2, " ".join(["detail"] * 176))
    return project, project.finalize_buckets_in_project(counter)


def test_training_revision_reuses_exported_files_and_tracks_sealed_manifests(tmp_path, bucket_revision):
    _, buckets = bucket_revision
    revision = buckets[0].parent
    (revision / "notes.txt").write_text("Unrelated notes.")
    before = {path.relative_to(revision): sha256(path) for path in revision.rglob("*") if path.is_file()}

    combined, fingerprint = load_training_dataset(revision)
    assert [entry["id"] for entry in combined["entries"]] == [1, 2]
    for bucket, combined_row in zip(buckets, combined["entries"], strict=True):
        manifest = verify_finalized(bucket)
        assert combined["tokenizers"] == manifest["tokenizers"]
        expected = manifest["entries"][0]
        assert combined_row == {**expected, "image": f"{bucket.name}/{expected['image']}",
                                "caption": f"{bucket.name}/{expected['caption']}"}
        assert sha256(revision / combined_row["image"]) == expected["image_sha256"]
        assert sha256(revision / combined_row["caption"]) == expected["caption_sha256"]
    assert {path.relative_to(revision): sha256(path) for path in revision.rglob("*") if path.is_file()} == before

    destination = tmp_path / "moved"
    destination.mkdir()
    relocated = revision.rename(destination / revision.name)
    assert load_training_dataset(relocated) == (combined, fingerprint)
    changed_manifest = relocated / buckets[1].name / "manifest.json"
    with changed_manifest.open("ab") as stream:
        stream.write(b"\n")
    assert load_training_dataset(relocated)[1] != fingerprint


@pytest.mark.parametrize("damage, message", [
    ("missing", "Invalid finalized aspect bucket"),
    ("symlink", "Aspect bucket must be a real directory"),
    ("dimensions", "Image dimensions do not match"),
    ("tokenizers", "Tokenizer metadata differs"),
    ("duplicate", "Duplicate image number across aspect buckets"),
    ("traversal", "inconsistent filenames"),
])
def test_training_revision_rejects_invalid_bucket_children(tmp_path, bucket_revision, damage, message):
    project, (_, wide) = bucket_revision
    revision = wide.parent
    manifest_path = wide / "manifest.json"
    if damage == "missing":
        manifest_path.unlink()
        assert revision not in project.finalized_datasets(include_all=True)
    elif damage == "symlink":
        outside = wide.rename(tmp_path / "outside")
        wide.symlink_to(outside, target_is_directory=True)
        assert revision not in project.finalized_datasets(include_all=True)
    elif damage == "dimensions":
        wide.rename(wide.with_name("Portrait project:1216X832"))
    else:
        manifest = json.loads(manifest_path.read_text())
        if damage == "tokenizers":
            manifest["tokenizers"]["tokenizer_sha256"] = "different"
        elif damage == "duplicate":
            row = manifest["entries"][0]
            row["id"] = 1
            for kind, suffix in (("image", ".png"), ("caption", ".txt")):
                destination = wide / ("Img-1" + suffix)
                (wide / row[kind]).rename(destination)
                row[kind] = destination.name
        else:
            manifest["entries"][0]["image"] = "../Img-2.png"
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        load_training_dataset(revision)


def test_bucket_finalization_failure_never_publishes_a_partial_revision(tmp_path, counter):
    project = DatasetProject.create(tmp_path / "draft")
    project.import_images([
        source_image(tmp_path, "square.png", size=(100, 100)),
        source_image(tmp_path, "wide.png", size=(200, 100)),
    ])
    project.save_caption(1, "Square caption")
    with pytest.raises(ValueError, match="Img-2.txt is empty"):
        project.finalize_buckets_in_project(counter)
    assert not list((project.root / "finalized").iterdir())
    assert project.finalized_datasets() == ()

    project.save_caption(2, "Wide caption")
    project.entries[1].width, project.entries[1].height = 1024, 1024
    with pytest.raises(ValueError, match="Invalid training image: Img-2.png"):
        project.finalize_buckets_in_project(counter)
    assert not list((project.root / "finalized").iterdir())


def test_bucket_and_legacy_exports_share_revision_numbers_when_concurrent(tmp_path, counter):
    from concurrent.futures import ThreadPoolExecutor

    project = DatasetProject.create(tmp_path / "draft")
    project.import_images([source_image(tmp_path)])
    project.save_caption(1, "Reviewed caption")
    with ThreadPoolExecutor(max_workers=2) as executor:
        bucket_future = executor.submit(project.finalize_buckets_in_project, counter)
        legacy_future = executor.submit(project.finalize_in_project, counter)
        bucket, = bucket_future.result()
        legacy = legacy_future.result()
    assert {bucket.parent.name, legacy.name} == {"Dataset-1", "Dataset-2"}
    assert set(project.finalized_datasets()) == {bucket, legacy}
    assert all(len(verify_finalized(path)["entries"]) == 1 for path in (bucket, legacy))


def test_bucket_export_shortens_only_long_project_name_stem(tmp_path, counter):
    name = "写" * 85
    project = DatasetProject.create(tmp_path / validate_project_name(name))
    project.import_images([source_image(tmp_path, size=(100, 100))])
    project.save_caption(1, "Reviewed caption")
    bucket, = project.finalize_buckets_in_project(counter)
    stem, suffix = bucket.name.rsplit(":", 1)
    assert name.startswith(stem) and 0 < len(stem) < len(name)
    assert suffix == "1024X1024"
    assert len(bucket.name.encode("utf-8")) <= 255
    assert verify_finalized(bucket)["entries"][0]["width"] == 1024


def test_bucket_discovery_ignores_unpublished_and_unsafe_paths(tmp_path, counter):
    project = DatasetProject.create(tmp_path / "draft")
    project.import_images([source_image(tmp_path)])
    project.save_caption(1, "Reviewed caption")
    good, = project.finalize_buckets_in_project(counter)
    finalized = project.root / "finalized"
    incomplete = finalized / "Dataset-2"
    incomplete.mkdir()
    (incomplete / "draft:1024X1024").symlink_to(good, target_is_directory=True)
    unsafe = incomplete / "draft:640X1664"
    unsafe.mkdir()
    (unsafe / "manifest.json").symlink_to(good / "manifest.json")
    (finalized / "Dataset-3").symlink_to(good, target_is_directory=True)
    staged = finalized / ".dataset-hidden" / "draft:1024X1024"
    staged.mkdir(parents=True)
    (staged / "manifest.json").write_bytes((good / "manifest.json").read_bytes())
    assert project.finalized_datasets() == (good,)


def test_finalize_refuses_empty_but_seals_full_long_caption(tmp_path, counter):
    project = DatasetProject.create(tmp_path / "draft")
    project.import_images([source_image(tmp_path)])
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="empty"):
        project.finalize(destination, counter)
    caption = " ".join(["word"] * 176)
    project.save_caption(1, caption)
    project.finalize(destination, counter)
    assert (destination / "Img-1.txt").read_text() == caption
    row = verify_finalized(destination)["entries"][0]
    assert row["tokens"]["clip_l"] == 178
    assert row["tokens"]["clip_g"] == 178
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


@pytest.mark.parametrize("name", ["", " ", ".", "..", "nested/name", "nested\\name", "NUL\0", "line\nbreak",
                                  "tab\tname", "trailing\n", "delete\x7f", "hidden\u200bname", "a" * 201, "é" * 128])
def test_invalid_project_names_are_rejected(name):
    with pytest.raises(ValueError):
        validate_project_name(name)


def test_project_name_accepts_readable_unicode_and_spaces():
    assert validate_project_name("  Portrait study – été  ") == "Portrait study – été"
