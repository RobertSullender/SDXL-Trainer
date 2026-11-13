from pathlib import Path

from sdxl_trainer.paths import projects_dir


def test_default_projects_are_beneath_home_application_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "person"))
    assert projects_dir() == tmp_path / "person" / "SDXL-Trainer" / "projects"
    assert not projects_dir().exists()
