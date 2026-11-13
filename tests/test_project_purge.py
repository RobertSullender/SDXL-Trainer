import os
from pathlib import Path
import shutil

import pytest

from sdxl_trainer.dataset import DatasetProject
from sdxl_trainer.project_purge import preview_project_purge, purge_project


def test_purge_removes_complete_project_but_never_external_content(tmp_path):
    original = tmp_path / 'original.png'
    original.write_bytes(b'original image')
    models = tmp_path / 'models'
    models.mkdir()
    (models / 'model.gguf').write_bytes(b'model weights')
    project = DatasetProject.create(tmp_path / 'project')
    nested = project.root / 'finalized/Dataset-1'
    nested.mkdir(parents=True)
    (nested / 'Img-1.png').write_bytes(b'sealed')
    logs = project.root / 'runs/run-1/logs'
    logs.mkdir(parents=True)
    (logs / 'session.log').write_text('training output')
    (project.root / 'external-models').symlink_to(models, target_is_directory=True)
    (project.root / 'external-original.png').symlink_to(original)
    preview = preview_project_purge(project.root)
    assert preview.path == project.root
    assert preview.files >= 5 and preview.directories >= 4 and preview.bytes > 0
    purge_project(preview)
    assert not project.root.exists()
    assert original.read_bytes() == b'original image'
    assert (models / 'model.gguf').read_bytes() == b'model weights'
    assert not list(tmp_path.glob('.sdxl-purge-*'))


def test_purge_refuses_a_project_replaced_after_preview(tmp_path):
    project = DatasetProject.create(tmp_path / 'project')
    preview = preview_project_purge(project.root)
    project.root.rename(tmp_path / 'real-project')
    other = DatasetProject.create(tmp_path / 'project')
    (other.root / 'important.txt').write_text('keep')
    with pytest.raises(ValueError, match='changed'):
        purge_project(preview)
    assert (other.root / 'important.txt').read_text() == 'keep'
    assert (tmp_path / 'real-project/project.json').exists()


def test_purge_rejects_links_fake_projects_and_repositories(tmp_path):
    project = DatasetProject.create(tmp_path / 'project')
    alias = tmp_path / 'alias'
    alias.symlink_to(project.root, target_is_directory=True)
    with pytest.raises(ValueError, match='symbolic'):
        preview_project_purge(alias)
    fake = tmp_path / 'fake'
    fake.mkdir()
    (fake / 'project.json').write_text('{"kind":"wrong","schema_version":1}')
    with pytest.raises(ValueError, match='genuine'):
        preview_project_purge(fake)
    (project.root / '.git').mkdir()
    with pytest.raises(ValueError, match='repository'):
        preview_project_purge(project.root)
    assert project.root.exists()


def test_purge_rejects_symlinked_images_and_manifest(tmp_path):
    project = DatasetProject.create(tmp_path / 'project')
    shutil.rmtree(project.image_dir)
    external = tmp_path / 'external'
    external.mkdir()
    project.image_dir.symlink_to(external, target_is_directory=True)
    with pytest.raises(OSError):
        preview_project_purge(project.root)
    project.image_dir.unlink()
    project.image_dir.mkdir()
    manifest = project.root / 'project.json'
    copied = tmp_path / 'manifest.json'
    copied.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(copied)
    with pytest.raises(OSError):
        preview_project_purge(project.root)
    assert copied.exists() and external.exists()


def test_purge_refuses_workspace_and_root_even_if_manifest_present(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    project_root = tmp_path / 'SDXL-Trainer/projects'
    DatasetProject.create(project_root)
    with pytest.raises(ValueError, match='workspace'):
        preview_project_purge(project_root)
    with pytest.raises(ValueError, match='system'):
        preview_project_purge('/')
    assert (project_root / 'project.json').exists()


def test_nested_mount_rejected_before_preview(tmp_path, monkeypatch):
    from sdxl_trainer import project_purge
    project = DatasetProject.create(tmp_path / 'project')
    (project.root / 'data').mkdir()
    monkeypatch.setattr(project_purge, '_mount_points', lambda: [project.root / 'data'])
    with pytest.raises(ValueError, match='mounted'):
        preview_project_purge(project.root)
    assert project.root.exists()


def test_parent_symlink_rejected(tmp_path):
    actual = tmp_path / 'actual'
    actual.mkdir()
    project = DatasetProject.create(actual / 'project')
    alias = tmp_path / 'alias'
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match='symbolic'):
        preview_project_purge(alias / 'project')
    assert project.root.exists()


def test_application_models_and_source_tree_are_never_purgeable(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    models = tmp_path / 'data/sdxl-trainer/models'
    project = DatasetProject.create(models / 'model-as-project')
    with pytest.raises(ValueError, match='model'):
        preview_project_purge(project.root)
    source_project = DatasetProject.create(tmp_path / 'SDXL-Trainer/src/subfolder')
    with pytest.raises(ValueError, match='source'):
        preview_project_purge(source_project.root)
    assert project.root.exists() and source_project.root.exists()


def test_swap_during_atomic_quarantine_never_deletes_the_replacement(tmp_path, monkeypatch):
    from sdxl_trainer import project_purge
    project = DatasetProject.create(tmp_path / 'project')
    preview = preview_project_purge(project.root)
    original_rename = project_purge._rename_noreplace
    swapped = False

    def race(parent_fd, source, destination):
        nonlocal swapped
        if not swapped and source == 'project':
            swapped = True
            original_rename(parent_fd, 'project', 'original')
            replacement = DatasetProject.create(tmp_path / 'project')
            (replacement.root / 'keep.txt').write_text('replacement')
        original_rename(parent_fd, source, destination)

    monkeypatch.setattr(project_purge, '_rename_noreplace', race)
    with pytest.raises(ValueError, match='changed during purge'):
        purge_project(preview)
    assert (tmp_path / 'project/keep.txt').read_text() == 'replacement'
    assert (tmp_path / 'original/project.json').exists()
    assert not list(tmp_path.glob('.sdxl-purge-*'))


def test_partial_delete_failure_restores_folder_name_and_raises(tmp_path, monkeypatch):
    from sdxl_trainer import project_purge
    project = DatasetProject.create(tmp_path / 'project')
    (project.root / 'one.txt').write_text('one')
    preview = preview_project_purge(project.root)

    def fail_midway(name, *, dir_fd):
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=dir_fd)
        try:
            os.unlink('one.txt', dir_fd=fd)
        finally:
            os.close(fd)
        raise OSError('simulated disk error')

    monkeypatch.setattr(project_purge.shutil, 'rmtree', fail_midway)
    # Monkeypatching the function replaces its safety capability attribute.
    monkeypatch.setattr(fail_midway, 'avoids_symlink_attacks', True, raising=False)
    with pytest.raises(OSError, match='disk error'):
        purge_project(preview)
    assert project.root.exists()
    assert (project.root / 'project.json').exists()
    assert not (project.root / 'one.txt').exists()
