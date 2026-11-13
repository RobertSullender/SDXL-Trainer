"""Draft removal renumbers pairs while preserving source images and snapshots."""
from dataclasses import replace
import os
from pathlib import Path

from PIL import Image
import pytest

from sdxl_trainer.dataset import DatasetProject, sha256, verify_finalized
from sdxl_trainer.project_remove import remove_images
from sdxl_trainer.tokens import TokenCounter


class Tokenizer:
    model_max_length = 77

    def __call__(self, text, **kwargs):
        return {"input_ids": [0, *range(len(text.split())), 1]}


def make_project(tmp_path, count=3):
    project = DatasetProject.create(tmp_path / "project")
    sources = []
    for index in range(count):
        source = tmp_path / f"original-{index}.png"
        Image.new("RGB", (100 + index * 20, 100), (100, 20, index)).save(source)
        sources.append(source)
    project.import_images(sources)
    for entry in project.entries:
        project.save_caption(entry.id, f"Image number {entry.id}.")
    return project, sources


def test_remove_multiple_keeps_source_snapshot_and_nonremoved_caption(tmp_path):
    project, sources = make_project(tmp_path)
    source_hashes = [sha256(path) for path in sources]
    snapshot = project.finalize_in_project(TokenCounter(Tokenizer(), Tokenizer()))
    snapshot_hashes = {path.name: sha256(path) for path in snapshot.iterdir()}
    original_entry = project.entries[1]
    original_image = project.image_path(2).stat()
    project.save_generated_caption(2, "Image number 2.", "Check the rest of this caption.")
    assert remove_images(project, [3, 1, 3]) == 2
    assert project.entries == [replace(original_entry, id=1, image="Img-1.png", caption="Img-1.txt")]
    assert project.entries[0].caption_incomplete_reason == "Check the rest of this caption."
    assert project.caption_text(1) == "Image number 2."
    assert project.image_path(1).stat().st_ino == original_image.st_ino
    assert {path.name for path in project.image_dir.iterdir()} == {"Img-1.png", "Img-1.txt"}
    assert [sha256(path) for path in sources] == source_hashes
    assert {path.name: sha256(path) for path in snapshot.iterdir()} == snapshot_hashes
    assert len(verify_finalized(snapshot)["entries"]) == 3
    reopened = DatasetProject.open(project.root)
    assert reopened.entries == project.entries
    assert reopened.next_image_id == 2
    assert reopened.import_images([sources[0]])[0].id == 2


@pytest.mark.parametrize("selected, kept", [([1], [2, 3]), ([2], [1, 3]), ([3], [1, 2]), ([1, 2, 3], [])])
def test_remove_reassigns_ids_and_keeps_pair_order(tmp_path, selected, kept):
    project, sources = make_project(tmp_path)
    old_entries = project.entries
    assert remove_images(project, selected) == len(selected)

    reopened = DatasetProject.open(project.root)
    assert reopened.next_image_id == len(kept) + 1
    for new_id, old_id in enumerate(kept, 1):
        entry = reopened.entries[new_id - 1]
        assert entry.id == new_id
        assert reopened.caption_text(new_id) == f"Image number {old_id}."
        assert sha256(reopened.image_path(new_id)) == old_entries[old_id - 1].image_sha256
    assert len(list(reopened.image_dir.iterdir())) == 2 * len(kept)
    assert reopened.import_images([sources[0]])[0].id == len(kept) + 1


def make_sparse_project(tmp_path):
    project, _ = make_project(tmp_path, count=4)
    # Recreate a deletion from a version that left Img-2 missing.
    removed = project.entries.pop(1)
    (project.image_dir / removed.image).unlink()
    (project.image_dir / removed.caption).unlink()
    project.save()
    return project


def test_remove_compacts_earlier_gaps_in_old_drafts(tmp_path):
    project = make_sparse_project(tmp_path)
    assert remove_images(project, [4]) == 1
    reopened = DatasetProject.open(project.root)
    assert reopened.next_image_id == 3
    assert [entry.id for entry in reopened.entries] == [1, 2]
    assert reopened.caption_text(1) == "Image number 1."
    assert reopened.caption_text(2) == "Image number 3."


@pytest.mark.parametrize("symlink", [False, True])
def test_remove_does_not_replace_untracked_collision(tmp_path, symlink):
    project = make_sparse_project(tmp_path)
    collision = project.image_dir / "Img-2.txt"
    if symlink:
        collision.symlink_to(tmp_path / "missing-caption.txt")
    else:
        collision.write_text("Untracked caption", encoding="utf-8")
    before = (project.root / "project.json").read_bytes()

    with pytest.raises(ValueError, match="Img-2.txt already exists"):
        remove_images(project, [4])
    assert (project.root / "project.json").read_bytes() == before
    assert [entry.id for entry in project.entries] == [1, 3, 4]
    if symlink:
        assert collision.readlink() == tmp_path / "missing-caption.txt"
    else:
        assert collision.read_text(encoding="utf-8") == "Untracked caption"


def test_remove_rolls_back_files_and_manifest_on_delete_error(tmp_path, monkeypatch):
    project, _ = make_project(tmp_path)
    before = {path.name: sha256(path) for path in project.image_dir.iterdir()}
    manifest = (project.root / "project.json").read_bytes()
    real_unlink = Path.unlink

    def fail_one(path, *args, **kwargs):
        if path == project.image_dir / "Img-3.png":
            raise OSError("simulated removal failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_one)
    with pytest.raises(OSError, match="simulated removal failure"):
        remove_images(project, [1, 3])
    assert [entry.id for entry in project.entries] == [1, 2, 3]
    assert (project.root / "project.json").read_bytes() == manifest
    assert {path.name: sha256(path) for path in project.image_dir.iterdir()} == before
    assert [entry.id for entry in DatasetProject.open(project.root).entries] == [1, 2, 3]


@pytest.mark.parametrize("fail_publish", [False, True])
def test_remove_rolls_back_renumbering_if_publication_or_save_fails(tmp_path, monkeypatch, fail_publish):
    project, _ = make_project(tmp_path)
    before = {path.name: sha256(path) for path in project.image_dir.iterdir()}
    manifest = (project.root / "project.json").read_bytes()
    real_link = os.link

    def link_one(source, target, **kwargs):
        if source.name == "Img-3.txt" and target.name == "Img-2.txt":
            raise OSError("simulated commit failure")
        return real_link(source, target, **kwargs)

    def fail_save():
        raise OSError("simulated commit failure")

    if fail_publish:
        monkeypatch.setattr(os, "link", link_one)
    else:
        monkeypatch.setattr(project, "save", fail_save)
    with pytest.raises(OSError, match="simulated commit failure"):
        remove_images(project, [2])
    assert [entry.id for entry in project.entries] == [1, 2, 3]
    assert project.next_image_id == 4
    assert (project.root / "project.json").read_bytes() == manifest
    assert {path.name: sha256(path) for path in project.image_dir.iterdir()} == before
    assert not list(project.root.glob(".remove-*"))


def test_remove_rejects_unknown_ids_without_changes(tmp_path):
    project, _ = make_project(tmp_path)
    before = (project.root / "project.json").read_bytes()
    with pytest.raises(ValueError, match="Unknown image"):
        remove_images(project, [1, 99])
    assert (project.root / "project.json").read_bytes() == before
    assert len(project.entries) == 3
