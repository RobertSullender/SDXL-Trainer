"""Native offscreen workflow tests; no model weights, GPU, or network required."""
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QInputDialog, QMessageBox, QPushButton,
)
from PIL import Image

from sdxl_trainer import app as desktop
from sdxl_trainer.captioning import CaptionError
from sdxl_trainer.dataset import DatasetProject, verify_finalized
from sdxl_trainer.runtime import RemoteTokenCounter
from sdxl_trainer.tokens import TokenCounts
from sdxl_trainer.training.config import TrainingConfig, fast_training_settings
from sdxl_trainer.widgets import ConfigForm


@pytest.fixture(scope="session")
def qt_app():
    return QApplication.instance() or QApplication([])


def wait_until(qt_app, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        qt_app.processEvents()
        if time.monotonic() > deadline:
            raise AssertionError("The native UI operation did not finish in time.")
        time.sleep(0.005)
    qt_app.processEvents()


@pytest.fixture
def window(qt_app, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(desktop, "projects_dir", lambda: tmp_path / "home" / "SDXL-Trainer" / "projects")
    errors = []
    qt_exceptions = []
    monkeypatch.setattr(desktop.Window, "error", lambda self, text: errors.append(text))
    monkeypatch.setattr(sys, "excepthook", lambda *args: qt_exceptions.append(args))
    result = desktop.Window()
    result.test_errors = errors
    result.show()
    qt_app.processEvents()
    yield result
    if result.task:
        result.task.cancel.set()
        wait_until(qt_app, lambda: result.task is None)
    if result.process:
        result.process.kill()
        wait_until(qt_app, lambda: result.process is None)
    result.close()
    qt_app.processEvents()
    assert not qt_exceptions, qt_exceptions


def imported_project(tmp_path, count=2):
    project = DatasetProject.create(tmp_path / "draft")
    originals = []
    for i in range(count):
        path = tmp_path / f"original-{i}.jpg"
        Image.new("RGB", (96 + i * 20, 64), (100 + i, 30, 80)).save(path)
        originals.append(path)
    project.import_images(originals)
    return project, originals


def test_application_loads_all_pages_and_all_backend_controls(window, qt_app):
    assert window.pages.count() == 6
    assert set(window.training_controls) == set(TrainingConfig.__dataclass_fields__)
    assert set(window.caption_form.controls) == set(desktop.CaptionConfig.__dataclass_fields__)
    assert window.save_shared_caption_paths_button in window.pages.widget(4).findChildren(QPushButton)
    assert window.save_caption_settings_button in window.pages.widget(1).findChildren(QPushButton)
    assert window.save_training_settings_button in window.pages.widget(2).findChildren(QPushButton)
    for index in range(6):
        window.navigation.setCurrentRow(index)
        qt_app.processEvents()
        assert window.pages.currentIndex() == index
    assert not window.test_errors


def test_component_offload_labels_round_trip_stable_recipe_values(qt_app):
    choices = {
        "none": "No component offload",
        "text_encoders_cpu": "Text encoders on CPU",
        "text_and_vae_cpu": "Text encoders and VAE on CPU",
        "text_cpu_staged_unet": "Text encoders on CPU; move UNet for VAE",
        "stage_unet_gpu_text": "Text encoders on GPU; move UNet for VAE",
    }
    for value, label in choices.items():
        # Both dataclass construction and saved-recipe loading use stable values.
        form = ConfigForm(TrainingConfig(component_offload=value), ["component_offload"])
        combo = form.controls["component_offload"]
        assert combo.currentText() == label
        assert form.values()["component_offload"] == value
        form.set_values({"component_offload": "none"})
        form.set_values({"component_offload": value})
        assert combo.currentText() == label
        assert form.values()["component_offload"] == value
        restored = TrainingConfig.from_dict(form.values())
        assert restored.component_offload == value
        form.deleteLater()
    qt_app.processEvents()


def test_inspection_reports_missing_preset_files_without_blocking_selection(window):
    from sdxl_trainer.downloader import PRESETS, RemoteFile, RepositorySnapshot
    index = next(i for i, preset in enumerate(PRESETS) if "FP16 variant" in preset["name"])
    window.preset.setCurrentIndex(index)
    preset = PRESETS[index]
    present = preset["filenames"][0]
    missing = preset["filenames"][1]
    window.show_repository(RepositorySnapshot(preset["repo_id"], "abc123", [RemoteFile(present, 1)]))
    assert missing in window.download_info.text()
    assert "Preset incomplete" in window.status.text()
    assert window.remote_files.count() == 1
    assert window.remote_files.item(0).isSelected()


def test_fp16_preset_download_selects_separate_model_and_variant(window, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from sdxl_trainer import downloader
    index = next(i for i, preset in enumerate(downloader.PRESETS) if preset.get("model_variant") == "fp16")
    preset = downloader.PRESETS[index]
    window.preset.setCurrentIndex(index)
    window.download_dir.set_value(str(tmp_path / "models"))
    snapshot = downloader.RepositorySnapshot(preset["repo_id"], "a" * 40,
        [downloader.RemoteFile(name, 1) for name in preset["filenames"]])
    window.show_repository(snapshot)
    selected = []

    class FakeDownloader:
        def __init__(self, *_args):
            pass

        def download(self, _snapshot, filenames, destination, **_kwargs):
            for name in filenames:
                path = destination / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")
            return [destination / name for name in filenames]

    monkeypatch.setattr(downloader, "HuggingFaceDownloader", FakeDownloader)
    monkeypatch.setattr(window.model_field, "set_value", lambda value: selected.append(Path(value)))
    monkeypatch.setattr(window, "run_task", lambda _description, work, complete:
        complete(work(SimpleNamespace(cancel=None, progress=SimpleNamespace(emit=lambda *_: None)))))
    window.download_selected()
    destination = tmp_path / "models" / "stabilityai--stable-diffusion-xl-base-1.0--fp16"
    assert selected == [destination]
    assert window.training_controls["model_variant"].currentText() == "fp16"
    assert window.shared_training_model.values() == {"model_path": str(destination), "model_variant": "fp16"}
    assert desktop.load_settings()["shared_training_model"] == window.shared_training_model.values()


def test_new_import_edit_switch_and_reopen_native_workflow(window, qt_app, tmp_path, monkeypatch):
    root = tmp_path / "draft"
    window.projects_location.set_value(tmp_path)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("draft", True))
    monkeypatch.setattr(desktop.file_dialogs, "get_existing_directory", lambda *a, **k: str(root))
    window.new_project()
    assert window.project.root == root
    originals = []
    original_bytes = []
    for index in range(2):
        path = tmp_path / f"source-{index}.jpg"
        Image.new("RGB", (160, 100 + index * 20), "blue").save(path)
        originals.append(str(path))
        original_bytes.append(path.read_bytes())
    monkeypatch.setattr(desktop.file_dialogs, "get_open_file_names", lambda *a, **k: (originals, ""))
    window.import_images(False)
    wait_until(qt_app, lambda: window.task is None)
    assert window.images.count() == 2 and window.entry_id == 2
    window.images.setCurrentRow(0)
    window.caption.setPlainText("First reviewed image caption.")
    window.images.setCurrentRow(1)
    assert window.project.caption_text(1) == "First reviewed image caption."
    window.caption.setPlainText("Second reviewed image caption.")
    wait_until(qt_app, lambda: window.project.caption_text(2) == "Second reviewed image caption.")
    window.open_project()
    assert window.images.count() == 2
    assert window.project.caption_text(1) == "First reviewed image caption."
    assert window.project.caption_text(2) == "Second reviewed image caption."
    assert [Path(p).read_bytes() for p in originals] == original_bytes
    assert not window.test_errors


@pytest.mark.parametrize("import_folder", [False, True])
def test_import_skips_upscaling_and_keeps_selection(window, qt_app, tmp_path, monkeypatch, import_folder):
    project, _ = imported_project(tmp_path)
    window.set_project(project)
    window.images.setCurrentRow(1)
    window.caption.setPlainText("Keep reviewing this caption.")
    sources = tmp_path / "sources"
    sources.mkdir()
    Image.new("RGB", (256, 256), "blue").save(sources / "small.png")
    monkeypatch.setattr(desktop.file_dialogs, "get_existing_directory", lambda *a, **k: str(sources))
    monkeypatch.setattr(desktop.file_dialogs, "get_open_file_names",
                        lambda *a, **k: ([str(path) for path in sorted(sources.glob("*.png"))], ""))

    assert window.allow_upscale.isChecked()
    window.allow_upscale.setChecked(False)
    window.save_settings()
    assert desktop.load_settings()["allow_import_upscale"] is False
    window.import_images(import_folder)
    wait_until(qt_app, lambda: window.task is None)
    assert window.images.count() == 2
    assert window.entry_id == 2
    assert window.caption.toPlainText() == "Keep reviewing this caption."
    assert window.status.text() == "Imported 0 images. Skipped 1 requiring upscaling."

    Image.new("RGB", (1024, 1024), "green").save(sources / "full.png")
    window.import_images(import_folder)
    wait_until(qt_app, lambda: window.task is None)
    assert window.images.count() == 3
    assert window.entry_id == 3
    assert window.status.text() == "Imported 1 image. Skipped 1 requiring upscaling."
    assert not window.test_errors


@pytest.mark.parametrize(("operation", "expected", "count"), [
    ("Set every caption", "yoga", 2),
    ("Prepend to captions", "yoga, Pending local detail", 2),
    ("Fill empty captions", "Pending local detail", 1),
    ("Append to captions", "Pending local detail, yoga", 2),
])
def test_bulk_caption_flushes_edits_and_refreshes_selected_image(
    window, qt_app, tmp_path, monkeypatch, operation, expected, count,
):
    assert not window.bulk_caption_button.isEnabled()
    project, _ = imported_project(tmp_path)
    window.set_project(project)
    window.images.setCurrentRow(1)
    window.caption.setPlainText("Pending local detail")

    def apply_caption(dialog):
        mode = dialog.findChild(QComboBox, "captionOperation")
        editor = dialog.findChild(desktop.CaptionEditor)
        buttons = dialog.findChild(QDialogButtonBox)
        apply = next(b for b in buttons.buttons() if buttons.buttonRole(b) == QDialogButtonBox.ButtonRole.AcceptRole)
        assert mode.currentText() == "Set every caption"
        assert apply.text() == "Apply to 2 images"
        assert not apply.isEnabled()
        mode.setCurrentIndex(mode.findText(operation))
        assert apply.text() == ("Apply to 1 image" if count == 1 else "Apply to 2 images")
        editor.setPlainText("yoga")
        assert apply.isEnabled()
        apply.click()
        return dialog.result()

    monkeypatch.setattr(QDialog, "exec", apply_caption)
    assert window.bulk_caption_button.isEnabled()
    window.bulk_caption_button.click()
    wait_until(qt_app, lambda: window.task is None)
    assert window.entry_id == 2
    assert window.caption.toPlainText() == expected
    assert window.save_caption()
    assert project.caption_text(1) == "yoga"
    assert project.caption_text(2) == expected
    assert window.status.text() == ("Updated 1 caption." if count == 1 else "Updated 2 captions.")
    assert not window.test_errors


def test_bulk_caption_empty_mode_disabled_when_no_captions_need_filling(window, tmp_path, monkeypatch):
    project, _ = imported_project(tmp_path, count=1)
    project.save_caption(1, "Already captioned.")
    window.set_project(project)

    def check_dialog(dialog):
        mode = dialog.findChild(QComboBox, "captionOperation")
        editor = dialog.findChild(desktop.CaptionEditor)
        buttons = dialog.findChild(QDialogButtonBox)
        apply = next(b for b in buttons.buttons() if buttons.buttonRole(b) == QDialogButtonBox.ButtonRole.AcceptRole)
        mode.setCurrentIndex(mode.findText("Fill empty captions"))
        editor.setPlainText("Shared text")
        assert apply.text() == "Apply to 0 images"
        assert not apply.isEnabled()
        mode.setCurrentIndex(mode.findText("Prepend to captions"))
        assert apply.text() == "Apply to 1 image"
        assert apply.isEnabled()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", check_dialog)
    window.bulk_caption_button.click()
    assert project.caption_text(1) == "Already captioned."
    assert not window.test_errors


def test_bulk_caption_append_to_noncontiguous_selection(window, qt_app, tmp_path, monkeypatch):
    project, _ = imported_project(tmp_path, 4)
    project.save_caption(4, "Other caption")
    window.set_project(project)
    window.caption.setPlainText("Pending local detail")
    window.images.item(2).setSelected(True)

    def apply_caption(dialog):
        mode = dialog.findChild(QComboBox, "captionOperation")
        scope = dialog.findChild(QComboBox, "captionScope")
        editor = dialog.findChild(desktop.CaptionEditor)
        buttons = dialog.findChild(QDialogButtonBox)
        apply = next(b for b in buttons.buttons() if buttons.buttonRole(b) == QDialogButtonBox.ButtonRole.AcceptRole)
        assert scope.currentText() == "Selected images (2)"
        mode.setCurrentIndex(mode.findText("Fill empty captions"))
        assert apply.text() == "Apply to 1 image"
        scope.setCurrentIndex(scope.findData("all"))
        assert scope.currentText() == "All draft images (4)"
        assert apply.text() == "Apply to 2 images"
        mode.setCurrentIndex(mode.findText("Append to captions"))
        scope.setCurrentIndex(scope.findData("selected"))
        editor.setPlainText("yoga")
        assert apply.isEnabled() and apply.text() == "Apply to 2 images"
        apply.click()
        return dialog.result()

    monkeypatch.setattr(QDialog, "exec", apply_caption)
    window.bulk_caption_button.click()
    wait_until(qt_app, lambda: window.task is None)
    assert [project.caption_text(index) for index in range(1, 5)] == [
        "Pending local detail, yoga", "", "yoga", "Other caption",
    ]
    assert window.caption.toPlainText() == "Pending local detail, yoga"
    assert [window.images.row(item) for item in window.images.selectedItems()] == [0, 2]
    assert not window.test_errors


@pytest.mark.parametrize("whole_draft", [False, True])
def test_trim_caption_shortcut_applies_to_chosen_scope(window, qt_app, tmp_path, monkeypatch, whole_draft):
    project, _ = imported_project(tmp_path)
    project.save_caption(2, "Another lengthy caption.")
    window.set_project(project)
    window.caption.setPlainText("Pending lengthy caption.")
    monkeypatch.setattr(window, "request_token_count", lambda: None)
    window.model_field.set_value(tmp_path / "local-model")
    received = []

    class Counter:
        def __init__(self, python, model):
            assert model == str(tmp_path / "local-model")

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def truncate(self, text):
            received.append(text)
            return text.split()[0]

    def apply_trim(dialog):
        mode = dialog.findChild(QComboBox, "captionOperation")
        scope = dialog.findChild(QComboBox, "captionScope")
        editor = dialog.findChild(desktop.CaptionEditor)
        buttons = dialog.findChild(QDialogButtonBox)
        apply = next(b for b in buttons.buttons() if buttons.buttonRole(b) == QDialogButtonBox.ButtonRole.AcceptRole)
        assert mode.currentText() == "Trim to 77 tokens"
        assert scope.currentText() == "Selected images (1)"
        assert editor.isHidden()
        if whole_draft:
            scope.setCurrentIndex(scope.findData("all"))
        assert apply.text() == ("Apply to 2 images" if whole_draft else "Apply to 1 image")
        assert apply.isEnabled()
        apply.click()
        return dialog.result()

    monkeypatch.setattr(desktop, "RemoteTokenCounter", Counter)
    monkeypatch.setattr(QDialog, "exec", apply_trim)
    window.trim_caption_button.click()
    wait_until(qt_app, lambda: window.task is None)
    assert received == (["Pending lengthy caption.", "Another lengthy caption."] if whole_draft else ["Pending lengthy caption."])
    assert project.caption_text(1) == window.caption.toPlainText() == "Pending"
    assert project.caption_text(2) == ("Another" if whole_draft else "Another lengthy caption.")
    assert window.status.text() == ("Trimmed 2 captions." if whole_draft else "Trimmed 1 caption.")
    assert not window.test_errors


def test_new_project_uses_home_workspace_and_readable_editable_name(window, tmp_path, monkeypatch):
    parent = tmp_path / "home" / "SDXL-Trainer" / "projects"
    prompts = []
    def choose_name(*args, **kwargs):
        prompts.append((args, kwargs))
        return "My landscape dataset", True
    monkeypatch.setattr(QInputDialog, "getText", choose_name)
    window.new_project()
    assert window.project.root == parent / "My landscape dataset"
    assert prompts[0][1]["text"] == "New project"
    assert window.purge_project_button.isEnabled()
    assert desktop.load_settings()["last_project"] == str(window.project.root)
    assert not window.test_errors


def test_new_project_cancel_and_existing_name_do_not_change_folders(window, monkeypatch):
    parent = Path(window.projects_location.value())
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Cancelled", False))
    window.new_project()
    assert not parent.exists()
    existing = parent / "My existing project"
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("Do not change")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: (existing.name, True))
    window.new_project()
    assert window.project is None
    assert (existing / "keep.txt").read_text() == "Do not change"
    assert len(window.test_errors) == 1


def test_workspace_preferences_preserve_downloads_and_model_paths(window, tmp_path, monkeypatch):
    monkeypatch.setattr(window, "request_token_count", lambda: None)
    library = tmp_path / "existing-model-library"
    library.mkdir()
    qwen = library / "qwen.gguf"
    projector = library / "mmproj.gguf"
    for path in (qwen, projector):
        path.write_bytes(b"GGUFexisting-download")
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in (qwen, projector)]
    window.shared_caption_paths["model_path"].set_value(qwen)
    window.shared_caption_paths["mmproj_path"].set_value(projector)
    window.shared_training_model.set_values({"model_path": str(library / "sdxl"), "model_variant": "fp16"})
    window.download_dir.set_value(library)
    window.projects_location.set_value(tmp_path / "custom-projects")
    window.save_settings()
    settings = desktop.load_settings()
    assert settings["shared_caption_paths"]["model_path"] == str(qwen)
    assert settings["shared_caption_paths"]["mmproj_path"] == str(projector)
    assert settings["shared_training_model"] == {"model_path": str(library / "sdxl"), "model_variant": "fp16"}
    assert "training" not in settings
    assert settings["model_library"] == str(library)
    assert settings["projects_dir"] == str(tmp_path / "custom-projects")
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in (qwen, projector)] == before


def test_managed_runtime_replaces_legacy_default_but_preserves_custom_runtime(window, qt_app, tmp_path):
    managed = desktop.data_dir() / "runtime/bin/python"
    managed.parent.mkdir(parents=True)
    managed.write_text("#!/bin/sh\nexit 0\n")
    managed.chmod(0o755)
    window.settings["runtime_python"] = "/usr/bin/python3"
    desktop.atomic_json(desktop.config_dir() / "settings.json", window.settings)
    upgraded = desktop.Window()
    assert upgraded.runtime.value() == str(managed)
    upgraded.close()
    custom = str(tmp_path / "chosen-python")
    settings = desktop.load_settings()
    settings["runtime_python"] = custom
    desktop.atomic_json(desktop.config_dir() / "settings.json", settings)
    upgraded = desktop.Window()
    assert upgraded.runtime.value() == custom
    upgraded.close()


def test_opening_empty_project_clears_old_image_and_caption(window, qt_app, tmp_path):
    project, _ = imported_project(tmp_path)
    window.set_project(project)
    assert not window.loaded_pixmap.isNull()
    window.caption.setPlainText("Saved caption.")
    empty = DatasetProject.create(tmp_path / "empty")
    window.set_project(empty)
    assert window.entry_id is None
    assert window.loaded_pixmap.isNull()
    assert not window.caption.toPlainText()
    assert "Import images" in window.preview.text()
    assert project.caption_text(1) == "Saved caption."


def test_project_caption_prompts_are_independent_and_reset_individually(window, qt_app, tmp_path):
    first, _ = imported_project(tmp_path, 1)
    second = DatasetProject.create(tmp_path / "second-project")
    system = window.caption_form.controls["system_prompt"]
    user = window.caption_form.controls["user_prompt"]
    defaults = desktop.default_prompts()
    window.set_project(first)
    assert system.toPlainText() == defaults["system_prompt"]
    system.setPlainText("Describe the first project's subject.")
    user.setPlainText("First project request.")
    window.set_project(second)
    assert system.toPlainText() == defaults["system_prompt"]
    assert user.toPlainText() == defaults["user_prompt"]
    system.setPlainText("Second project system.")
    user.setPlainText("Second project request.")
    window.set_project(first)
    assert system.toPlainText() == "Describe the first project's subject."
    assert user.toPlainText() == "First project request."
    window.reset_system_prompt.click()
    assert system.toPlainText() == defaults["system_prompt"]
    assert user.toPlainText() == "First project request."
    window.reset_user_prompt.click()
    assert user.toPlainText() == defaults["user_prompt"]
    window.set_project(second)
    assert system.toPlainText() == "Second project system."
    assert user.toPlainText() == "Second project request."
    assert "caption" not in desktop.load_settings()
    assert not window.test_errors


def test_reset_all_caption_settings_keeps_paths_and_saves_project_defaults(window, tmp_path):
    reset = window.reset_all_caption_settings_button
    assert not reset.isEnabled()
    other = DatasetProject.create(tmp_path / "other-project")
    project = DatasetProject.create(tmp_path / "reset-project")
    window.set_project(other)
    window.caption_form.set_values({"temperature": 0.61, "system_prompt": "Other project prompt."})
    assert window.save_caption_settings()
    other_settings = (other.root / "caption-prompts.json").read_bytes()

    window.set_project(project)
    assert reset.isEnabled()
    selected_paths = {
        "server_executable": str(tmp_path / "llama-server-custom"),
        "model_path": str(tmp_path / "model.gguf"),
        "mmproj_path": str(tmp_path / "projector.gguf"),
    }
    window.caption_form.set_values({
        **selected_paths,
        "trim_generated_captions": False,
        "max_tokens": 640,
        "thinking": True,
        "temperature": 0.37,
        "system_prompt": "Custom system prompt.",
        "user_prompt": "Custom user prompt.",
    })
    reset.click()

    expected = window._caption_defaults | selected_paths
    assert window.caption_form.values() == expected
    assert json.loads((project.root / "caption-prompts.json").read_text()) == expected
    assert (other.root / "caption-prompts.json").read_bytes() == other_settings
    assert not window.test_errors


def test_caption_settings_round_trip_between_projects(window, qt_app, monkeypatch):
    replies = iter([("First caption settings", True), ("Second caption settings", True)])
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: next(replies))
    save = window.save_caption_settings_button
    shared_save = window.save_shared_caption_paths_button
    assert not save.isEnabled()
    assert shared_save.isEnabled()
    settings_before = desktop.load_settings()
    save.click()
    assert desktop.load_settings() == settings_before
    shared_paths = {
        "server_executable": "/local/shared-server",
        "model_path": "/local/shared-model.gguf",
        "mmproj_path": "/local/shared-projector.gguf",
    }
    for name, path in shared_paths.items():
        window.shared_caption_paths[name].set_value(path)
    shared_save.click()
    assert desktop.load_settings() == {**settings_before, "shared_caption_paths": shared_paths}

    window.new_project()
    first = window.project.root
    assert save.isEnabled()
    current = window.caption_form.values()
    assert {name: current[name] for name in shared_paths} == shared_paths
    assert current["trim_generated_captions"] is True
    project_file = first / "caption-prompts.json"
    original_project_settings = project_file.read_bytes()
    machine_paths = {
        "server_executable": "/local/llama-server-a",
        "model_path": "/local/model-a.gguf",
        "mmproj_path": "/local/projector-a.gguf",
    }
    window.caption_form.set_values({
        **machine_paths,
        "trim_generated_captions": False,
        "temperature": 0.61,
        "max_tokens": 640,
        "system_prompt": "Describe the first project's style.",
        "user_prompt": "Describe this subject.",
    })
    expected_first = window.caption_form.values()
    shared_paths = {**shared_paths, "server_executable": "/local/updated-shared-server"}
    window.shared_caption_paths["server_executable"].set_value(shared_paths["server_executable"])
    settings_before = desktop.load_settings()
    shared_save.click()
    assert desktop.load_settings() == {**settings_before, "shared_caption_paths": shared_paths}
    assert project_file.read_bytes() == original_project_settings
    assert window.caption_form.values() == expected_first

    settings_before = desktop.load_settings()
    save.click()
    assert json.loads(project_file.read_text()) == expected_first
    assert desktop.load_settings() == settings_before

    window.new_project()
    second = window.project.root
    current = window.caption_form.values()
    assert current["trim_generated_captions"] is True
    assert current["temperature"] == desktop.CaptionConfig().temperature
    assert current["max_tokens"] == desktop.CaptionConfig().max_tokens
    assert current["system_prompt"] == desktop.default_prompts()["system_prompt"]
    assert {key: current[key] for key in shared_paths} == shared_paths
    window.caption_form.set_values({
        "server_executable": "/local/llama-server-b",
        "model_path": "/local/model-b.gguf",
        "mmproj_path": "/local/projector-b.gguf",
        "temperature": 0.37,
        "max_tokens": 980,
        "system_prompt": "Describe the second project's lighting.",
        "user_prompt": "Describe this scene.",
    })
    expected_second = window.caption_form.values()
    save.click()
    assert json.loads((second / "caption-prompts.json").read_text()) == expected_second

    window.set_project(DatasetProject.open(first))
    assert window.caption_form.values() == expected_first
    window.set_project(DatasetProject.open(second))
    assert window.caption_form.values() == expected_second
    assert {name: field.value() for name, field in window.shared_caption_paths.items()} == shared_paths

    window.caption_form.controls["trim_generated_captions"].setChecked(False)
    expected_second["trim_generated_captions"] = False
    window.close()
    restored = desktop.Window()
    try:
        assert restored.project.root == second
        assert restored.caption_form.values() == expected_second
        assert {name: field.value() for name, field in restored.shared_caption_paths.items()} == shared_paths
        restored.set_project(DatasetProject.open(first))
        assert restored.caption_form.values() == expected_first
    finally:
        restored.close()
        qt_app.processEvents()
    assert not window.test_errors


@pytest.mark.parametrize("has_local_prompts", [False, True])
def test_legacy_caption_settings_migrate_only_to_last_project(qt_app, tmp_path, monkeypatch, has_local_prompts):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    first = DatasetProject.create(tmp_path / "last-project")
    second = DatasetProject.create(tmp_path / "other-project")
    prompts = {"system_prompt": "My previous custom system", "user_prompt": "My previous custom request"}
    legacy_paths = {
        "server_executable": str(tmp_path / "configured-server"),
        "model_path": str(tmp_path / "configured-model.gguf"),
        "mmproj_path": str(tmp_path / "configured-projector.gguf"),
    }
    legacy = {**prompts, **legacy_paths, "trim_generated_captions": False, "temperature": 0.47}
    if has_local_prompts:
        prompts = {"system_prompt": "Local system prompt", "user_prompt": "Local user prompt"}
        (first.root / "caption-prompts.json").write_text(json.dumps(prompts), encoding="utf-8")
    desktop.atomic_json(desktop.config_dir() / "settings.json", {
        "last_project": str(first.root),
        "caption": legacy,
    })
    instance = desktop.Window()
    try:
        assert instance.project.root == first.root
        current = instance.caption_form.values()
        assert {key: current[key] for key in prompts} == prompts
        assert current["trim_generated_captions"] is False
        assert current["temperature"] == legacy["temperature"]
        assert {name: current[name] for name in legacy_paths} == legacy_paths
        assert {name: field.value() for name, field in instance.shared_caption_paths.items()} == legacy_paths
        instance.set_project(second)
        current = instance.caption_form.values()
        assert current["system_prompt"] == desktop.default_prompts()["system_prompt"]
        assert current["user_prompt"] == desktop.default_prompts()["user_prompt"]
        assert current["trim_generated_captions"] is True
        assert current["temperature"] == desktop.CaptionConfig().temperature
        assert {name: current[name] for name in legacy_paths} == legacy_paths
        instance.set_project(first)
        current = instance.caption_form.values()
        assert {key: current[key] for key in prompts} == prompts
        assert current["trim_generated_captions"] is False
        assert current["temperature"] == legacy["temperature"]
        settings = desktop.load_settings()
        assert "caption" not in settings
        assert settings["shared_caption_paths"] == legacy_paths
    finally:
        instance.close()
        qt_app.processEvents()


def test_remove_selected_images_from_draft_updates_preview_and_preserves_sources(window, qt_app, tmp_path, monkeypatch):
    project, originals = imported_project(tmp_path, 3)
    for entry in project.entries:
        project.save_caption(entry.id, f"Caption {entry.id}")
    expected_preview = desktop.QImageReader(str(project.image_path(2))).read()
    window.set_project(project)
    window.images.item(0).setSelected(True)
    window.images.item(2).setSelected(True)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    window.remove_selected_images()
    wait_until(qt_app, lambda: window.task is None)
    assert [e.id for e in project.entries] == [1]
    assert window.images.count() == 1
    assert window.entry_id == 1 and window.caption.toPlainText() == "Caption 2"
    assert window.loaded_pixmap.toImage() == expected_preview.convertToFormat(window.loaded_pixmap.toImage().format())
    assert window.save_caption()
    assert project.caption_text(1) == "Caption 2"
    assert all(path.exists() for path in originals)
    assert not (project.image_dir / "Img-2.png").exists()
    assert not (project.image_dir / "Img-3.txt").exists()
    assert not window.test_errors


def test_cancel_remove_keeps_images_and_caption(window, tmp_path, monkeypatch):
    project, _ = imported_project(tmp_path, 1)
    window.set_project(project)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    window.remove_selected_images()
    assert len(project.entries) == 1 and project.image_path(1).is_file()


def test_generated_caption_auto_trim_reuses_tokenizers_and_preserves_incomplete_warning(
        window, qt_app, tmp_path, monkeypatch):
    from sdxl_trainer.captioning import CaptionResult

    project, _ = imported_project(tmp_path)
    window.set_project(project)
    monkeypatch.setattr(desktop.CaptionConfig, "validate", lambda self: [])
    monkeypatch.setattr(window, "request_token_count", lambda: None)
    model = tmp_path / "local-model"
    (model / "tokenizer").mkdir(parents=True)
    (model / "tokenizer_2").mkdir()
    window.model_field.set_value(model)
    control = window.caption_form.controls["trim_generated_captions"]
    assert control.isChecked()
    assert window.save_settings()
    assert json.loads((project.root / "caption-prompts.json").read_text())["trim_generated_captions"] is True

    captions = ["Long generated text " * 80, "Partial generated text " * 80]
    shortened = ["Long generated text", "Partial generated text"]
    calls = []
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))

    class Counter:
        def __init__(self, python, path):
            assert python == window.runtime.value()
            assert path == str(model)
            calls.append("load")

        def __enter__(self):
            return self

        def __exit__(self, *_):
            calls.append("close")

        def truncate(self, caption):
            calls.append(caption)
            return shortened[captions.index(caption)]

    class Server:
        def __init__(self, config):
            self.index = 0

        def start(self, cancel):
            assert calls == ["load"]
            calls.append("start")

        def caption(self, path, cancel):
            index = self.index
            self.index += 1
            return CaptionResult(captions[index], index == 1, "Context full" if index == 1 else "")

        def stop(self):
            pass

    monkeypatch.setattr(desktop, "RemoteTokenCounter", Counter)
    monkeypatch.setattr(desktop, "CaptionServer", Server)
    window.auto_caption(True)
    wait_until(qt_app, lambda: window.task is None)
    assert not window.test_errors
    assert calls == ["load", "start", *captions, "close"]
    saved = DatasetProject.open(project.root)
    assert [saved.caption_text(i) for i in (1, 2)] == shortened
    assert window.caption.toPlainText() == shortened[0]
    assert [entry.id for entry in saved.incomplete_captions()] == [2]
    assert warnings and "Img-2.png: Context full" in warnings[0][2]


def test_generated_caption_auto_trim_off_keeps_long_caption_without_tokenizers(
        window, qt_app, tmp_path, monkeypatch):
    project, _ = imported_project(tmp_path, 1)
    window.set_project(project)
    monkeypatch.setattr(desktop.CaptionConfig, "validate", lambda self: [])
    window.caption_form.controls["trim_generated_captions"].setChecked(False)
    assert window.save_settings()
    assert json.loads((project.root / "caption-prompts.json").read_text())["trim_generated_captions"] is False
    restored = desktop.Window()
    try:
        assert not restored.caption_form.controls["trim_generated_captions"].isChecked()
    finally:
        restored.close()
        qt_app.processEvents()
    long_caption = ("Detailed generated caption. " * 80).strip()

    class Server:
        def __init__(self, config): pass
        def start(self, cancel): pass
        def caption(self, path, cancel): return long_caption
        def stop(self): pass

    def unexpected_counter(*args, **kwargs):
        pytest.fail("Captioning without auto-trim should not open tokenizers")

    monkeypatch.setattr(desktop, "RemoteTokenCounter", unexpected_counter)
    monkeypatch.setattr(desktop, "CaptionServer", Server)
    window.auto_caption(False)
    wait_until(qt_app, lambda: window.task is None)
    assert not window.test_errors
    assert project.caption_text(1) == window.caption.toPlainText() == long_caption


def test_generated_caption_auto_trim_requires_a_base_model(window, tmp_path, monkeypatch):
    project, _ = imported_project(tmp_path, 1)
    window.set_project(project)
    monkeypatch.setattr(desktop.CaptionConfig, "validate", lambda self: [])
    window.model_field.set_value("")

    window.auto_caption(False)
    assert window.task is None
    assert project.caption_text(1) == ""
    assert window.test_errors and "base model" in window.test_errors[0].lower()


@pytest.mark.parametrize(
    ("modifier", "end_row", "selected_rows", "trim"),
    [
        (Qt.KeyboardModifier.ControlModifier, 0, [0, 2], True),
        (Qt.KeyboardModifier.ShiftModifier, 1, [1, 2], False),
    ],
    ids=["ctrl-click-trim", "shift-click-full-caption"],
)
def test_caption_selected_processes_highlighted_images_in_draft_order(
        window, qt_app, tmp_path, monkeypatch, modifier, end_row, selected_rows, trim):
    project, _ = imported_project(tmp_path, 4)
    original = {entry.id: f"Existing caption {entry.id}." for entry in project.entries}
    for image_id, caption in original.items():
        project.save_caption(image_id, caption)
    monkeypatch.setattr(window, "request_token_count", lambda: None)
    monkeypatch.setattr(desktop.CaptionConfig, "validate", lambda self: [])
    window.set_project(project)
    window.model_field.set_value(tmp_path / "local-model")
    window.caption_form.controls["trim_generated_captions"].setChecked(trim)
    qt_app.processEvents()
    # Click from a later image back toward the start to test draft ordering.
    for row, keys in [(2, Qt.KeyboardModifier.NoModifier), (end_row, modifier)]:
        position = window.images.visualItemRect(window.images.item(row)).center()
        QTest.mouseClick(window.images.viewport(), Qt.MouseButton.LeftButton, keys, position)
    assert sorted(window.images.row(item) for item in window.images.selectedItems()) == selected_rows
    assert window.images.currentRow() == end_row
    targets = [project.entries[row] for row in selected_rows]
    generated = {entry.image: f"Generated {entry.image}" + " detailed description" * 80 for entry in targets}
    expected = dict(original)
    expected.update({entry.id: f"Generated {entry.image}" if trim else generated[entry.image] for entry in targets})
    server_calls, trim_calls = [], []

    class Server:
        def __init__(self, config): pass
        def start(self, cancel): server_calls.append("start")
        def caption(self, path, cancel):
            server_calls.append(path.name)
            return generated[path.name]
        def stop(self): server_calls.append("stop")

    class Counter:
        def __init__(self, python, model):
            assert trim
            assert model == str(tmp_path / "local-model")
            trim_calls.append("load")
        def __enter__(self): return self
        def __exit__(self, *_): trim_calls.append("close")
        def truncate(self, caption):
            trim_calls.append(caption)
            return caption.split(" detailed description", 1)[0]

    monkeypatch.setattr(desktop, "CaptionServer", Server)
    monkeypatch.setattr(desktop, "RemoteTokenCounter", Counter)
    button = next(button for button in window.findChildren(QPushButton) if button.text() == "Caption selected")
    button.click()
    wait_until(qt_app, lambda: window.task is None)

    assert not window.test_errors
    assert server_calls == ["start", *(entry.image for entry in targets), "stop"]
    assert trim_calls == (["load", *generated.values(), "close"] if trim else [])
    assert {entry.id: project.caption_text(entry.id) for entry in project.entries} == expected
    assert sorted(window.images.row(item) for item in window.images.selectedItems()) == selected_rows
    assert window.images.currentRow() == end_row
    assert window.caption.toPlainText() == expected[project.entries[end_row].id]


def test_caption_selected_without_highlighted_images_does_not_start(window, tmp_path, monkeypatch):
    project, _ = imported_project(tmp_path, 1)
    project.save_caption(1, "Keep this caption.")
    window.set_project(project)
    window.images.clearSelection()
    assert window.images.currentRow() == 0
    monkeypatch.setattr(desktop.CaptionConfig, "validate", lambda self: [])
    def unexpected_server(*_):
        raise AssertionError("Unexpected caption generation")
    monkeypatch.setattr(desktop, "CaptionServer", unexpected_server)

    window.auto_caption(False)

    assert window.task is None
    assert project.caption_text(1) == "Keep this caption."
    assert "select" in window.status.text().lower() and "image" in window.status.text().lower()
    assert not window.test_errors


def test_failed_caption_batch_reloads_completed_caption_before_any_save(window, qt_app, tmp_path, monkeypatch):
    project, _ = imported_project(tmp_path)
    window.set_project(project)
    window.caption_form.controls["trim_generated_captions"].setChecked(False)
    monkeypatch.setattr(desktop.CaptionConfig, "validate", lambda self: [])

    class Server:
        def __init__(self, config):
            self.calls = 0
        def start(self, cancel):
            pass
        def caption(self, path, cancel):
            self.calls += 1
            if self.calls == 2:
                raise CaptionError("Simulated second-image failure")
            return "Authoritative generated caption."
        def stop(self):
            pass

    monkeypatch.setattr(desktop, "CaptionServer", Server)
    window.auto_caption(True)
    wait_until(qt_app, lambda: window.task is None)
    assert "Simulated second-image failure" in window.test_errors[0]
    assert window.caption.toPlainText() == "Authoritative generated caption."
    window.save_caption()
    assert project.caption_text(1) == "Authoritative generated caption."
    assert project.caption_text(2) == ""


def test_caption_limit_saves_available_text_and_reports_incomplete(window, qt_app, tmp_path, monkeypatch):
    project, _ = imported_project(tmp_path, 1)
    window.set_project(project)
    window.caption_form.controls["trim_generated_captions"].setChecked(False)
    monkeypatch.setattr(desktop.CaptionConfig, "validate", lambda self: [])
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))

    class Partial(str):
        truncated = True
        reason = "Context capacity was reached."

    class Server:
        def __init__(self, config): pass
        def start(self, cancel): pass
        def caption(self, path, cancel): return Partial("Available generated caption")
        def stop(self): pass

    monkeypatch.setattr(desktop, "CaptionServer", Server)
    window.auto_caption(False)
    wait_until(qt_app, lambda: window.task is None)
    assert project.caption_text(1) == "Available generated caption"
    assert "could not finish automatically" in window.status.text()
    assert warnings and "available text was saved" in warnings[0][2].lower()
    assert "Img-1.png" in warnings[0][2] and "Context capacity was reached." in warnings[0][2]



def test_caption_batch_identifies_only_unfinished_images(window, qt_app, tmp_path, monkeypatch):
    from sdxl_trainer.captioning import CaptionResult
    project, _ = imported_project(tmp_path, 2)
    window.set_project(project)
    window.caption_form.controls["trim_generated_captions"].setChecked(False)
    monkeypatch.setattr(desktop.CaptionConfig, "validate", lambda self: [])
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))

    class Server:
        def __init__(self, config): self.calls = 0
        def start(self, cancel): pass
        def caption(self, path, cancel):
            self.calls += 1
            return CaptionResult("Saved text", self.calls == 2, "Context full" if self.calls == 2 else "")
        def stop(self): pass

    monkeypatch.setattr(desktop, "CaptionServer", Server)
    window.auto_caption(True)
    wait_until(qt_app, lambda: window.task is None)
    assert project.caption_text(1) == project.caption_text(2) == "Saved text"
    assert warnings and "Img-2.png: Context full" in warnings[0][2]
    assert "Img-1.png:" not in warnings[0][2]


@pytest.mark.parametrize("cancelled", [False, True])
def test_caption_batch_failure_keeps_prior_partial_marked_and_visible(window, qt_app, tmp_path, monkeypatch, cancelled):
    from sdxl_trainer.captioning import CaptionCancelled, CaptionError, CaptionResult
    from sdxl_trainer.dataset import DatasetProject
    project, _ = imported_project(tmp_path, 2)
    window.set_project(project)
    window.caption_form.controls["trim_generated_captions"].setChecked(False)
    monkeypatch.setattr(desktop.CaptionConfig, "validate", lambda self: [])
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    class Server:
        def __init__(self, config): self.calls = 0
        def start(self, cancel): pass
        def caption(self, path, cancel):
            self.calls += 1
            if self.calls == 1:
                return CaptionResult("Saved partial", True, "Context full")
            if cancelled:
                cancel.set()
                raise CaptionCancelled("Captioning cancelled")
            raise CaptionError("Model stopped")
        def stop(self): pass
    monkeypatch.setattr(desktop, "CaptionServer", Server)
    window.auto_caption(True)
    wait_until(qt_app, lambda: window.task is None)
    wait_until(qt_app, lambda: any("Img-1.png: Context full" in call[2] for call in warnings))
    assert DatasetProject.open(project.root).incomplete_captions()[0].image == "Img-1.png"
    assert "⚠" in window.images.item(0).text()
    assert project.caption_text(1) == "Saved partial"


def test_incomplete_warning_lists_every_image(window, qt_app, tmp_path, monkeypatch):
    project, _ = imported_project(tmp_path, 11)
    for entry in project.entries:
        project.save_generated_caption(entry.id, "partial", "Limit reached")
    window.set_project(project)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    window.finalize()
    assert all(f"Img-{i}.png: Limit reached" in warnings[0][2] for i in range(1, 12))
    assert "⚠" in window.images.item(0).text()


@pytest.mark.parametrize("select_all", [False, True], ids=["single-bucket", "all-aspects"])
def test_finalize_selects_bucket_exports_and_chooser_restores_selection(window, qt_app, tmp_path, monkeypatch, select_all):
    project, _ = imported_project(tmp_path, 2)
    for entry in project.entries:
        project.save_caption(entry.id, "A reviewed image caption.")
    window.set_project(project)
    monkeypatch.setattr(desktop.Window, "request_token_count", lambda self: None)
    window.model_field.set_value(tmp_path / "local-model")

    class Counter:
        metadata = {"max_length": 77, "special_tokens_included": True,
                    "tokenizer_sha256": "a" * 64, "tokenizer_2_sha256": "b" * 64}
        def __init__(self, python, model):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def count(self, text):
            return TokenCounts(8, 9)

    monkeypatch.setattr(desktop, "RemoteTokenCounter", Counter)
    window.finalize()
    wait_until(qt_app, lambda: window.task is None)
    datasets = project.finalized_datasets()
    assert len(datasets) == 2
    for path in datasets:
        manifest = verify_finalized(path)
        assert path.parent == project.root / "finalized/Dataset-1"
        assert manifest["tokenizers"] == Counter.metadata
        entry, = manifest["entries"]
        assert path.name == f"{project.root.name}:{entry['width']}X{entry['height']}"
    assert window.training_controls["dataset_path"].value() == str(datasets[0])
    assert "Finalized 2 bucket datasets" in window.status.text()
    revision = datasets[0].parent
    expected = revision if select_all else datasets[1]

    # Reopening a project restores its saved selection.
    window.set_project(project)
    assert window.training_controls["dataset_path"].value() == str(datasets[0])

    first_run = project.root / "runs/Run-1"
    first_run.mkdir(parents=True)
    window.training_controls["resume_from"].set_value(first_run / "checkpoint")

    def choose_dataset(_parent, _title, _label, choices, current, editable):
        assert choices == ["Dataset-1 / All aspect ratios", *(f"Dataset-1 / {path.name}" for path in datasets)]
        assert current == 1 and not editable
        return choices[0 if select_all else 2], True

    monkeypatch.setattr(QInputDialog, "getItem", choose_dataset)
    assert window.project_dataset_button.isEnabled()
    window.project_dataset_button.click()
    assert window.training_controls["dataset_path"].value() == str(expected)
    assert window.training_controls["output_dir"].value() == str(project.root / "runs/Run-2")
    assert window.training_controls["resume_from"].value() == ""
    reopened = desktop.Window()
    try:
        assert reopened.project.root == project.root
        assert reopened.training_controls["dataset_path"].value() == str(expected)
        assert reopened.project_dataset_button.isEnabled()
    finally:
        reopened.close()
    assert not window.test_errors


def test_expanding_and_finalizing_preserves_earlier_snapshot(window, qt_app, tmp_path, monkeypatch):
    project, _ = imported_project(tmp_path, 1)
    project.save_caption(1, "First caption.")
    window.set_project(project)
    monkeypatch.setattr(window, "request_token_count", lambda: None)
    window.model_field.set_value(tmp_path / "local-model")
    class Counter:
        metadata = {"max_length": 77, "special_tokens_included": True,
                    "tokenizer_sha256": "a" * 64, "tokenizer_2_sha256": "b" * 64}
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def count(self, text): return TokenCounts(8, 9)
    monkeypatch.setattr(desktop, "RemoteTokenCounter", Counter)
    window.finalize()
    wait_until(qt_app, lambda: window.task is None)
    entry = project.entries[0]
    bucket = f"{project.root.name}:{entry.width}X{entry.height}"
    first = project.root / "finalized/Dataset-1" / bucket
    saved = (first / "manifest.json").read_bytes()
    image = tmp_path / "additional.jpg"
    Image.new("RGB", (120, 80), "red").save(image)
    project.import_images([image])
    project.save_caption(2, "Second caption.")
    window.finalize()
    wait_until(qt_app, lambda: window.task is None)
    second = project.root / "finalized/Dataset-2" / bucket
    assert (first / "manifest.json").read_bytes() == saved
    assert len(verify_finalized(first)["entries"]) == 1
    assert len(verify_finalized(second)["entries"]) == 2
    assert window.training_controls["dataset_path"].value() == str(second)
    assert window.training_controls["output_dir"].value() == str(project.root / "runs/Run-1")
    assert not window.test_errors


def test_new_project_replaces_paths_from_deleted_last_project(qt_app, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    projects = tmp_path / "projects"
    previous = projects / "Deleted project"
    desktop.atomic_json(desktop.config_dir() / "settings.json", {
        "last_project": str(previous),
        "projects_dir": str(projects),
        "training": {
            "dataset_path": str(previous / "finalized/Dataset-1"),
            "output_dir": str(previous / "runs/Run-1"),
            "resume_from": str(previous / "runs/Run-1/checkpoint"),
        },
    })
    window = desktop.Window()
    try:
        assert window.project is None
        monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("Fresh project", True))
        window.new_project()
        assert window.project.root == projects / "Fresh project"
        assert window.training_controls["dataset_path"].value() == ""
        assert window.training_controls["resume_from"].value() == ""
        assert window.training_controls["output_dir"].value() == str(window.project.root / "runs/Run-1")
        assert not window.project_dataset_button.isEnabled()
        assert not previous.exists()
    finally:
        window.close()


def test_purge_requires_exact_confirmation_and_preserves_external_files(window, qt_app, tmp_path, monkeypatch):
    project, originals = imported_project(tmp_path, 1)
    window.set_project(project)
    root = project.root
    external = tmp_path / "shared-model.gguf"
    external.write_bytes(b"model data")
    (root / "outside-link").symlink_to(external)
    for filename in ("finalized/Dataset-1/manifest.json", "runs/Run-1/run.log", "logs/launches/launch.json"):
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("project data")
    monkeypatch.setattr(window, "request_token_count", lambda: None)
    window.shared_training_model.set_values({"model_path": str(external), "model_variant": "fp16"})
    window.model_field.set_value(root / "project-model")
    window.training_controls["dataset_path"].set_value(root / "finalized/Dataset-1")
    window.training_controls["output_dir"].set_value(root / "runs/Run-1")
    window.caption_form.controls["server_executable"].set_value(root / "llama-server")
    window.shared_caption_paths["server_executable"].set_value(root / "llama-server")
    window.shared_caption_paths["model_path"].set_value(external)
    window.logs.setPlainText("Project log")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("wrong name", True))
    window.purge_project()
    wait_until(qt_app, lambda: window.task is None and bool(window.test_errors))
    assert root.exists() and external.read_bytes() == b"model data"
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: (root.name, True))
    window.purge_project()
    wait_until(qt_app, lambda: window.project is None and window.task is None)
    assert not root.exists()
    assert originals[0].exists() and external.read_bytes() == b"model data"
    assert window.training_controls["dataset_path"].value() == ""
    assert window.training_controls["output_dir"].value() == ""
    assert window.model_field.value() == str(external)
    assert window.training_controls["model_variant"].currentText() == "fp16"
    assert not window.save_training_settings_button.isEnabled()
    assert window.caption_form.controls["server_executable"].value() == window._caption_defaults["server_executable"]
    assert window.shared_caption_paths["server_executable"].value() == window._caption_defaults["server_executable"]
    assert window.shared_caption_paths["model_path"].value() == str(external)
    assert not window.save_caption_settings_button.isEnabled()
    assert window.logs.toPlainText() == ""
    assert desktop.load_settings()["last_project"] == ""


def test_new_project_restores_trim_default_after_purge_in_same_window(window, qt_app, monkeypatch):
    replies = iter([("First project", True), ("First project", True), ("Fresh project", True)])
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: next(replies))
    window.new_project()
    previous = window.project.root
    trim = window.caption_form.controls["trim_generated_captions"]
    trim.setChecked(False)
    window.save_settings()
    assert json.loads((previous / "caption-prompts.json").read_text())["trim_generated_captions"] is False

    window.purge_project()
    wait_until(qt_app, lambda: window.project is None and window.task is None)
    assert not previous.exists()

    window.new_project()
    assert window.project.root.name == "Fresh project"
    assert trim.isChecked()
    assert json.loads((window.project.root / "caption-prompts.json").read_text())["trim_generated_captions"] is True
    assert not window.test_errors


def test_training_launch_records_are_stored_inside_project(window, tmp_path, monkeypatch):
    project, _ = imported_project(tmp_path, 1)
    window.set_project(project)
    monkeypatch.setattr(window, "request_token_count", lambda: None)
    window.model_field.set_value(tmp_path / "base-model")
    window.training_controls["dataset_path"].set_value(project.root / "finalized/Dataset-1")
    calls = []
    monkeypatch.setattr(window, "run_process", lambda *args: calls.append(args))
    window.train()
    assert len(calls) == 1
    args = calls[0][0]
    config = Path(args[args.index("--config") + 1])
    assert config.parent == project.root / "logs/launches"
    assert json.loads(config.read_text())["output_dir"] == str(project.root / "runs/Run-1")
    saved = json.loads((project.root / "training-settings.json").read_text())
    saved.pop("mode_profiles", None)
    assert saved == json.loads(config.read_text())
    assert "training" not in desktop.load_settings()


def test_training_settings_round_trip_between_projects(window, qt_app, tmp_path, monkeypatch):
    monkeypatch.setattr(desktop.Window, "request_token_count", lambda self: None)
    replies = iter([("First training project", True), ("Second training project", True)])
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: next(replies))
    save = window.save_training_settings_button
    assert not save.isEnabled()
    initial = desktop.load_settings()
    save.click()
    assert desktop.load_settings() == initial

    shared_model = {"model_path": str(tmp_path / "shared-model"), "model_variant": "fp16"}
    window.shared_training_model.set_values(shared_model)
    window.save_settings()
    window.new_project()
    first = window.project.root
    assert save.isEnabled()
    current = window.training_config().to_dict()
    assert {name: current[name] for name in shared_model} == shared_model
    assert current["output_dir"] == str(first / "runs/Run-1")

    mode = window.training_controls["mode"]
    lora_output = first / "runs" / "Run-1"
    lora_resume = lora_output / "checkpoints" / "step-110"
    window.training_controls["resume_from"].set_value(lora_resume)
    window.training_controls["learning_rate"].setValue(0.000076)
    window.training_controls["max_steps"].setValue(2001)
    first_lora = window.training_config().to_dict()
    mode.setCurrentText("full")
    assert window.training_config().output_dir == str(first / "runs" / "Run-2")
    assert window.training_config().resume_from == ""
    for form in window.training_forms:
        form.set_values({
            "model_path": str(tmp_path / "project-model"),
            "model_variant": "bf16",
            "dataset_path": str(tmp_path / "external-dataset"),
            "output_dir": str(tmp_path / "external-runs" / "Run-4"),
            "resume_from": str(tmp_path / "external-runs" / "prior-run" / "checkpoints" / "step-100"),
            "optimizer": "adamw",
            "seed": 137,
            "learning_rate": 0.000024,
            "max_steps": 421,
        })
    first_settings = window.training_config().to_dict()
    common_paths = {name: first_settings[name] for name in ("model_path", "model_variant", "dataset_path")}
    first_lora |= common_paths
    app_settings = desktop.load_settings()
    caption_path = first / "caption-prompts.json"
    caption_before = caption_path.read_bytes()
    window.caption_form.set_values({"temperature": 0.56})
    save.click()
    saved_first = json.loads((first / "training-settings.json").read_text())
    profiles = saved_first.pop("mode_profiles")
    assert saved_first == first_settings
    assert profiles["lora"]["learning_rate"] == 0.000076
    assert profiles["lora"]["max_steps"] == 2001
    assert profiles["lora"]["output_dir"] == str(lora_output)
    assert profiles["lora"]["resume_from"] == str(lora_resume)
    assert not (profiles["lora"].keys() & common_paths.keys())
    assert desktop.load_settings() == app_settings
    assert caption_path.read_bytes() == caption_before

    window.new_project()
    second = window.project.root
    defaults = TrainingConfig()
    config = window.training_config()
    assert config.model_path == shared_model["model_path"] and config.model_variant == shared_model["model_variant"]
    assert config.dataset_path == config.resume_from == ""
    assert config.output_dir == str(second / "runs/Run-1")
    assert config.mode == defaults.mode and config.optimizer == defaults.optimizer
    assert config.seed == defaults.seed and config.learning_rate == defaults.learning_rate
    window.training_controls["seed"].setValue(201)
    second_settings = window.training_config().to_dict()

    window.set_project(DatasetProject.open(first))
    assert window.training_config().to_dict() == first_settings
    mode.setCurrentText("lora")
    assert window.training_config().to_dict() == first_lora
    mode.setCurrentText("full")
    saved_second = json.loads((second / "training-settings.json").read_text())
    saved_second.pop("mode_profiles", None)
    assert saved_second == second_settings
    window.set_project(DatasetProject.open(second))
    assert window.training_config().to_dict() == second_settings
    window.training_controls["seed"].setValue(202)
    second_settings = window.training_config().to_dict()
    window.close()

    reopened = desktop.Window()
    try:
        assert reopened.project.root == second
        assert reopened.training_config().to_dict() == second_settings
        reopened.set_project(DatasetProject.open(first))
        assert reopened.training_config().to_dict() == first_settings
        reopened.training_controls["mode"].setCurrentText("lora")
        assert reopened.training_config().to_dict() == first_lora
        assert reopened.shared_training_model.values() == shared_model
        assert "training" not in desktop.load_settings()
    finally:
        reopened.close()
        qt_app.processEvents()
    assert not window.test_errors


@pytest.mark.parametrize(("choice", "mode", "rate"), [
    ("LoRA defaults", "lora", 1e-4),
    ("Full fine-tune defaults", "full", 1e-6),
    ("Cancel", None, None),
])
def test_reset_all_training_settings_keeps_paths_and_saves_project_defaults(
        window, tmp_path, monkeypatch, choice, mode, rate):
    monkeypatch.setattr(window, "request_token_count", lambda: None)
    reset = window.reset_all_training_settings_button
    assert reset in window.pages.widget(2).findChildren(QPushButton)
    assert not reset.isEnabled()

    other = DatasetProject.create(tmp_path / "other-project")
    project = DatasetProject.create(tmp_path / "reset-project")
    window.set_project(other)
    window.training_controls["seed"].setValue(277)
    assert window.save_training_settings()
    other_settings = other.root / "training-settings.json"
    other_before = other_settings.read_bytes()

    window.set_project(project)
    assert reset.isEnabled()
    window.caption_form.set_values({"temperature": 0.61, "trim_generated_captions": False})
    assert window.save_caption_settings()
    caption_settings = project.root / "caption-prompts.json"
    caption_before = caption_settings.read_bytes()
    settings_before = desktop.load_settings()
    retained = {
        "model_path": str(tmp_path / "custom-base"),
        "model_variant": "fp16",
        "dataset_path": str(tmp_path / "external-dataset"),
    }
    lora_output = tmp_path / "lora-output"
    full_output = tmp_path / "full-output"
    overrides = {
        **retained,
        "output_dir": str(full_output),
        "resume_from": str(full_output / "checkpoints" / "step-100"),
        "optimizer": "sgd",
        "learning_rate": 0.00036,
        "text_encoder_lr": 0.000029,
        "vae_lr": 0.000013,
        "max_steps": 421,
        "batch_size": 2,
        "train_text_encoder": True,
        "train_vae": True,
        "seed": 137,
        "caption_mode": "native",
    }
    method = window.training_controls["mode"]
    for form in window.training_forms:
        form.set_values({**retained, "output_dir": str(lora_output),
                         "resume_from": str(lora_output / "checkpoints" / "step-200"),
                         "seed": 739, "max_steps": 842, "learning_rate": 0.000025})
    lora_before = window.training_config().to_dict()
    method.setCurrentText("full")
    for form in window.training_forms:
        form.set_values(overrides)
    controls_before = window.training_config().to_dict()
    training_settings = project.root / "training-settings.json"
    training_before = training_settings.read_bytes()

    def choose_defaults(dialog):
        buttons = {button.text(): button for button in dialog.buttons()}
        assert set(buttons) == {"LoRA defaults", "Full fine-tune defaults", "Cancel"}
        buttons[choice].click()
        return dialog.result()

    monkeypatch.setattr(QMessageBox, "exec", choose_defaults)

    reset.click()

    if mode is None:
        assert window.training_config().to_dict() == controls_before
        assert training_settings.read_bytes() == training_before
    else:
        expected = (TrainingConfig().to_dict() | fast_training_settings(mode) | retained
                    | {"mode": mode, "learning_rate": rate,
                       "output_dir": str(lora_output if mode == "lora" else full_output)})
        assert window.training_config().to_dict() == expected
        saved = json.loads(training_settings.read_text())
        saved.pop("mode_profiles")
        assert saved == expected
        other_mode, other_before_reset = ("full", controls_before) if mode == "lora" else ("lora", lora_before)
        method.setCurrentText(other_mode)
        assert window.training_config().to_dict() == other_before_reset
        method.setCurrentText(mode)
        assert window.training_config().to_dict() == expected
    assert caption_settings.read_bytes() == caption_before
    assert other_settings.read_bytes() == other_before
    assert desktop.load_settings() == settings_before
    assert not window.test_errors


@pytest.mark.parametrize("has_saved_settings", [False, True])
def test_legacy_training_settings_migrate_only_to_last_project(qt_app, tmp_path, monkeypatch, has_saved_settings):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(desktop.Window, "request_token_count", lambda self: None)
    first = DatasetProject.create(tmp_path / "last-project")
    second = DatasetProject.create(tmp_path / "other-project")
    model = {"model_path": str(tmp_path / "base-model"), "model_variant": "fp16"}
    legacy = TrainingConfig(
        **model, mode="full", optimizer="adamw", dataset_path=str(tmp_path / "external-dataset"),
        output_dir=str(tmp_path / "external-runs"), resume_from=str(tmp_path / "checkpoint"),
        seed=19, max_steps=1234, learning_rate=0.00031,
    ).to_dict()
    expected = legacy
    path = first.root / "training-settings.json"
    if has_saved_settings:
        expected = {**legacy, "seed": 42, "learning_rate": 0.0002, "model_variant": "default"}
        desktop.atomic_json(path, expected)
    desktop.atomic_json(desktop.config_dir() / "settings.json", {
        "last_project": str(first.root),
        "training": legacy,
        "training_defaults_version": 1,
    })

    instance = desktop.Window()
    try:
        assert instance.project.root == first.root
        assert instance.training_config().to_dict() == expected
        saved_project = json.loads(path.read_text())
        saved_project.pop("mode_profiles", None)
        assert saved_project == expected
        method = instance.training_controls["mode"]
        method.setCurrentText("lora")
        shared = {name: expected[name] for name in ("model_path", "model_variant", "dataset_path")}
        assert instance.training_config().to_dict() == (
            TrainingConfig().to_dict() | fast_training_settings("lora") | shared
            | {"output_dir": str(first.root / "runs" / "Run-1")}
        )
        method.setCurrentText("full")
        assert instance.training_config().to_dict() == expected
        saved = desktop.load_settings()
        assert "training" not in saved
        assert saved["shared_training_model"] == model

        instance.set_project(second)
        current = instance.training_config()
        assert current.model_path == model["model_path"] and current.model_variant == model["model_variant"]
        assert current.dataset_path == current.resume_from == ""
        assert current.output_dir == str(second.root / "runs/Run-1")
        assert current.learning_rate == TrainingConfig().learning_rate
        instance.set_project(first)
        assert instance.training_config().to_dict() == expected
    finally:
        instance.close()
        qt_app.processEvents()


def test_legacy_training_profile_uses_resume_checkpoint_run_for_missing_output(window, tmp_path, monkeypatch):
    monkeypatch.setattr(window, "request_token_count", lambda: None)
    project = DatasetProject.create(tmp_path / "legacy-project")
    full_output = project.root / "runs" / "Full training"
    full_resume = full_output / "checkpoints" / "step-100"
    full_resume.mkdir(parents=True)
    (full_resume / "checkpoint.json").write_text("{}")
    active = TrainingConfig(
        model_path=str(tmp_path / "model"), model_variant="fp16", dataset_path=str(tmp_path / "dataset"),
        output_dir=str(project.root / "runs" / "LoRA training"),
    ).to_dict()
    training_file = project.root / "training-settings.json"
    desktop.atomic_json(training_file, active | {"mode_profiles": {
        "full": {"resume_from": str(full_resume), "learning_rate": 0.000004},
    }})

    window.set_project(project)
    assert window.training_config().to_dict() == active
    method = window.training_controls["mode"]
    method.setCurrentText("full")
    config = window.training_config()
    assert config.output_dir == str(full_output)
    assert config.resume_from == str(full_resume)
    assert config.learning_rate == 0.000004
    assert config.model_path == active["model_path"]
    assert config.model_variant == active["model_variant"]
    assert config.dataset_path == active["dataset_path"]
    method.setCurrentText("lora")
    assert window.training_config().to_dict() == active
    assert window.save_training_settings()
    assert json.loads(training_file.read_text())["mode_profiles"]["full"]["output_dir"] == str(full_output)
    assert not window.test_errors


@pytest.mark.parametrize("method", ["browse", "type"])
def test_selecting_resume_checkpoint_loads_run_settings_and_output(window, qt_app, tmp_path, monkeypatch, method):
    run = tmp_path / "Original run"
    checkpoint = run / "checkpoints" / "step-00000100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.json").write_text("{}")
    config = TrainingConfig(
        model_path=str(tmp_path / "Saved model"), model_variant="fp16",
        dataset_path=str(tmp_path / "Saved dataset"), output_dir=str(tmp_path / "Previous run"),
        resume_from=str(tmp_path / "Older checkpoint"), mode="full", optimizer="adamw_disk",
        learning_rate=0.000004, max_steps=2500, gradient_accumulation_steps=7, seed=123,
    )
    config.save(run / "training-config.json")
    monkeypatch.setattr(window, "request_token_count", lambda: None)
    window.navigation.setCurrentRow(2)
    resume = window.training_controls["resume_from"]
    output = window.training_controls["output_dir"]
    output.set_value(tmp_path / "Previous output")

    if method == "browse":
        monkeypatch.setattr(desktop.file_dialogs, "get_existing_directory", lambda *args, **kwargs: str(checkpoint))
        resume.findChild(QPushButton).click()
    else:
        resume.edit.setFocus()
        QTest.keyClicks(resume.edit, str(checkpoint))
        assert output.value() == str(tmp_path / "Previous output")
        QTest.keyClick(resume.edit, Qt.Key.Key_Return)
    expected = config.to_dict() | {"resume_from": str(checkpoint), "output_dir": str(run)}
    assert window.training_config().to_dict() == expected
    assert window.process is None and window.task is None

    branch = tmp_path / "New output"
    output.set_value(branch)
    resume.edit.setFocus()
    qt_app.processEvents()
    output.edit.setFocus()
    qt_app.processEvents()
    assert output.value() == str(branch)
    assert not window.test_errors


@pytest.mark.parametrize("method", ["browse", "type"])
def test_selecting_run_loads_latest_completed_checkpoint_and_settings(window, qt_app, tmp_path, monkeypatch, method):
    run = tmp_path / "Saved run"
    checkpoints = run / "checkpoints"
    older = checkpoints / "step-9"
    latest = checkpoints / "step-100"
    for folder in (older, latest, checkpoints / "step-200.tmp"):
        folder.mkdir(parents=True)
        (folder / "checkpoint.json").write_text("{}")
    (checkpoints / "step-300").mkdir()
    (checkpoints / "step-400").symlink_to(latest, target_is_directory=True)
    linked_metadata = checkpoints / "step-500"
    linked_metadata.mkdir()
    (linked_metadata / "checkpoint.json").symlink_to(latest / "checkpoint.json")
    config = TrainingConfig(
        model_path=str(tmp_path / "Saved model"), model_variant="fp16",
        dataset_path=str(tmp_path / "Saved dataset"), output_dir=str(tmp_path / "Previous output"),
        resume_from=str(older), mode="full", optimizer="adamw_disk", learning_rate=0.000004,
        max_steps=2500, gradient_accumulation_steps=7, seed=123,
    )
    config.save(run / "training-config.json")
    monkeypatch.setattr(window, "request_token_count", lambda: None)
    window.navigation.setCurrentRow(2)
    resume = window.training_controls["resume_from"]
    browse = resume.findChild(QPushButton)

    if method == "browse":
        monkeypatch.setattr(desktop.file_dialogs, "get_existing_directory", lambda *args, **kwargs: str(run))
        browse.click()
    else:
        resume.edit.setFocus()
        QTest.keyClicks(resume.edit, str(run))
        QTest.keyClick(resume.edit, Qt.Key.Key_Return)

    expected = config.to_dict() | {"resume_from": str(latest), "output_dir": str(run)}
    assert window.training_config().to_dict() == expected
    picker = window.checkpoint_step
    assert picker.isVisible()
    assert [picker.itemData(index) for index in range(picker.count())] == [str(latest), str(older)]
    assert picker.currentData() == str(latest)
    assert window.process is None and window.task is None
    output = window.training_controls["output_dir"]
    branch = tmp_path / "New output"
    output.set_value(branch)
    resume.edit.setFocus()
    qt_app.processEvents()
    output.edit.setFocus()
    qt_app.processEvents()
    assert output.value() == str(branch)

    monkeypatch.setattr(desktop.file_dialogs, "get_existing_directory", lambda *args, **kwargs: str(older))
    browse.click()
    assert window.training_config().to_dict() == expected | {"resume_from": str(older)}
    assert picker.currentData() == str(older)
    assert not window.test_errors


def test_saved_runs_lists_project_checkpoints_and_keeps_external_browse(
        window, qt_app, tmp_path, monkeypatch):
    project = DatasetProject.create(tmp_path / "First project")
    other = DatasetProject.create(tmp_path / "Other project")

    def make_run(run, steps, *, seed, mode="lora"):
        for step in steps:
            checkpoint = run / "checkpoints" / f"step-{step}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "checkpoint.json").write_text("{}")
        config = TrainingConfig(model_path=str(tmp_path / "Base model"),
                                dataset_path=str(tmp_path / "Finalized dataset"),
                                output_dir=str(run), max_steps=3000, seed=seed, mode=mode,
                                optimizer="adamw_disk" if mode == "full" else "adamw")
        config.save(run / "training-config.json")
        return config

    runs = project.root / "runs"
    first, second = runs / "Run-2", runs / "Run-10"
    first_config = make_run(first, (900, 1200), seed=73)
    make_run(second, (2500,), seed=81, mode="full")
    (first / "checkpoints" / "step-3000").mkdir()
    (runs / "Incomplete" / "checkpoints" / "step-200").mkdir(parents=True)
    external = tmp_path / "External run"
    external_config = make_run(external, (250,), seed=89)
    (runs / "Linked run").symlink_to(external, target_is_directory=True)
    other_run = other.root / "runs" / "Other saved run"
    make_run(other_run, (300,), seed=97)
    (other_run / "training-config.json").write_text("[]")
    (other_run / "checkpoints" / "step-300" / "checkpoint.json").write_text(
        json.dumps({"config": {"mode": "full"}})
    )
    monkeypatch.setattr(window, "request_token_count", lambda: None)

    window.navigation.setCurrentRow(2)
    picker = window.saved_runs
    qt_app.processEvents()
    assert not picker.isVisible()
    window.set_project(project)
    qt_app.processEvents()
    assert picker.isVisible() and picker.itemData(0) is None
    assert [picker.itemData(index) for index in range(1, picker.count())] == [str(first), str(second)]
    assert [picker.itemText(index) for index in range(1, picker.count())] == [
        "Run-2 · LoRA · Step 1,200", "Run-10 · Full Fine-Tune · Step 2,500",
    ]
    assert picker.currentData() is None

    QTest.keyClick(picker, Qt.Key.Key_Down)
    latest = str(first / "checkpoints" / "step-1200")
    older = str(first / "checkpoints" / "step-900")
    assert picker.currentData() == str(first)
    assert window.checkpoint_step.currentData() == latest
    assert window.training_config().to_dict() == first_config.to_dict() | {"resume_from": latest}
    window.checkpoint_step.setCurrentIndex(window.checkpoint_step.findData(older))
    assert window.training_config().resume_from == older
    assert picker.currentData() == str(first)
    assert picker.currentText() == "Run-2 · LoRA · Step 1,200"
    picker.activated.emit(picker.currentIndex())
    assert window.checkpoint_step.currentData() == latest

    monkeypatch.setattr(desktop.file_dialogs, "get_existing_directory", lambda *args, **kwargs: str(external))
    window.training_controls["resume_from"].findChild(QPushButton).click()
    external_checkpoint = str(external / "checkpoints" / "step-250")
    assert window.training_config().to_dict() == external_config.to_dict() | {"resume_from": external_checkpoint}
    assert window.checkpoint_step.currentData() == external_checkpoint
    assert picker.currentData() is None

    window.set_project(other)
    assert [picker.itemData(index) for index in range(picker.count())] == [None, str(other_run)]
    assert picker.itemText(1) == "Other saved run · Full Fine-Tune · Step 300"
    assert picker.currentData() is None
    window.set_project(DatasetProject.open(project.root))
    assert window.checkpoint_step.currentData() == external_checkpoint
    assert picker.currentData() is None
    assert window.process is None and window.task is None
    assert not window.test_errors


def test_saved_runs_refresh_after_new_project_checkpoint(window, qt_app, tmp_path):
    project = DatasetProject.create(tmp_path / "Project")
    window.set_project(project)
    window.navigation.setCurrentRow(2)
    qt_app.processEvents()
    picker = window.saved_runs
    assert picker.isVisible() and not picker.isEnabled()
    assert picker.count() == 1 and picker.currentData() is None

    run = project.root / "runs" / "Run-1"
    checkpoint = run / "checkpoints" / "step-100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.json").write_text("{}")
    window.process_finished(0)

    assert picker.isEnabled()
    assert [picker.itemData(index) for index in range(picker.count())] == [None, str(run)]
    assert picker.itemText(1) == "Run-1 · Unknown method · Step 100"
    assert picker.currentData() is None
    assert not window.training_controls["resume_from"].value()
    assert not window.test_errors


def test_checkpoint_dropdown_preserves_custom_output_and_only_reverts_automatic_branch(
        window, qt_app, tmp_path, monkeypatch):
    project = DatasetProject.create(tmp_path / "Project")
    run = project.root / "runs" / "Run-1"
    older = run / "checkpoints" / "step-200"
    latest = run / "checkpoints" / "step-300"
    for checkpoint in (older, latest):
        checkpoint.mkdir(parents=True)
        (checkpoint / "checkpoint.json").write_text("{}")
    config = TrainingConfig(
        model_path=str(tmp_path / "Saved model"), dataset_path=str(tmp_path / "Saved dataset"),
        output_dir=str(run), max_steps=400, learning_rate=0.000045, seed=123,
    )
    config.save(run / "training-config.json")
    monkeypatch.setattr(window, "request_token_count", lambda: None)
    monkeypatch.setattr(desktop.file_dialogs, "get_existing_directory", lambda *args, **kwargs: str(run))
    window.set_project(project)
    window.navigation.setCurrentRow(2)
    resume = window.training_controls["resume_from"]
    output = window.training_controls["output_dir"]
    picker = window.checkpoint_step
    resume.findChild(QPushButton).click()
    qt_app.processEvents()
    assert picker.currentData() == str(latest)
    window.training_controls["learning_rate"].setValue(0.000067)
    before = window.training_config().to_dict()

    QTest.keyClick(picker, Qt.Key.Key_Down)
    suggested = Path(output.value())
    assert suggested.parent == run.parent and suggested != run
    assert not suggested.exists()
    assert window.training_config().to_dict() == before | {
        "resume_from": str(older), "output_dir": str(suggested),
    }

    assert window.save_training_settings()
    window.set_project(DatasetProject.open(project.root))
    assert picker.currentData() == str(older)
    assert output.value() == str(suggested)
    QTest.keyClick(picker, Qt.Key.Key_Up)
    assert window.training_config().to_dict() == before

    suggested.mkdir()
    QTest.keyClick(picker, Qt.Key.Key_Down)
    suggested_again = output.value()
    assert Path(suggested_again).parent == run.parent
    assert not Path(suggested_again).exists()
    output.edit.selectAll()
    QTest.keyClicks(output.edit, str(tmp_path / "Custom output"))
    output.edit.selectAll()
    QTest.keyClicks(output.edit, suggested_again)
    QTest.keyClick(picker, Qt.Key.Key_Up)
    assert window.training_config().to_dict() == before | {"output_dir": suggested_again}

    QTest.keyClick(picker, Qt.Key.Key_Down)
    custom = str(tmp_path / "Other output")
    output.edit.selectAll()
    QTest.keyClicks(output.edit, custom)
    assert window.save_training_settings()
    window.set_project(DatasetProject.open(project.root))
    assert picker.currentData() == str(older)
    assert output.value() == custom
    QTest.keyClick(picker, Qt.Key.Key_Up)
    assert window.training_config().to_dict() == before | {"output_dir": custom}
    assert not window.test_errors


def test_selecting_run_without_completed_checkpoints_keeps_current_settings(window, tmp_path, monkeypatch):
    run = tmp_path / "Empty run"
    (run / "checkpoints" / "step-100.tmp").mkdir(parents=True)
    (run / "checkpoints" / "step-50").mkdir()
    TrainingConfig(mode="full", optimizer="adamw_disk", seed=456).save(run / "training-config.json")
    monkeypatch.setattr(desktop.file_dialogs, "get_existing_directory", lambda *args, **kwargs: str(run))
    resume = window.training_controls["resume_from"]
    resume.set_value(tmp_path / "Other run" / "checkpoints" / "step-1")
    window.training_controls["output_dir"].set_value(tmp_path / "New output")
    before = window.training_config().to_dict()

    resume.findChild(QPushButton).click()

    assert window.training_config().to_dict() == before
    assert "No completed checkpoints found" in window.status.text()
    assert not window.test_errors


@pytest.mark.parametrize("saved_config", [None, "invalid json", '{"mode": "unsupported"}'])
def test_resume_checkpoint_keeps_current_settings_without_readable_run_config(window, tmp_path, monkeypatch, saved_config):
    run = tmp_path / "Original run"
    checkpoint = run / "checkpoints" / "step-00000100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.json").write_text("{}")
    if saved_config is not None:
        (run / "training-config.json").write_text(saved_config)
    monkeypatch.setattr(desktop.file_dialogs, "get_existing_directory", lambda *args, **kwargs: str(checkpoint))
    window.training_controls["learning_rate"].setValue(0.000032)
    window.training_controls["seed"].setValue(123)
    before = window.training_config().to_dict()

    window.training_controls["resume_from"].findChild(QPushButton).click()

    assert window.training_config().to_dict() == before | {"resume_from": str(checkpoint), "output_dir": str(run)}
    assert "training-config.json" in window.status.text()
    assert not window.test_errors


def test_resume_browse_leaves_output_when_cancelled_or_checkpoint_is_unrecognized(window, tmp_path, monkeypatch):
    resume = window.training_controls["resume_from"]
    output = window.training_controls["output_dir"]
    branch = tmp_path / "New output"
    output.set_value(branch)
    nonstandard = tmp_path / "Other checkpoint"
    nonstandard.mkdir()
    (nonstandard / "checkpoint.json").write_text("{}")
    incomplete = tmp_path / "run" / "checkpoints" / "step-100"
    incomplete.mkdir(parents=True)
    choices = iter((str(nonstandard), str(incomplete), ""))
    monkeypatch.setattr(desktop.file_dialogs, "get_existing_directory", lambda *args, **kwargs: next(choices))

    browse = resume.findChild(QPushButton)
    browse.click()
    assert resume.value() == str(nonstandard)
    assert output.value() == str(branch)
    browse.click()
    assert resume.value() == str(incomplete)
    assert output.value() == str(branch)
    browse.click()
    assert resume.value() == str(incomplete)
    assert output.value() == str(branch)
    assert not window.test_errors


@pytest.mark.parametrize("method", ["browse", "type"])
def test_resume_load_keeps_other_method_and_overridden_output(window, tmp_path, monkeypatch, method):
    project = DatasetProject.create(tmp_path / "Project")
    window.set_project(project)
    monkeypatch.setattr(window, "request_token_count", lambda: None)
    resume = window.training_controls["resume_from"]
    output = window.training_controls["output_dir"]
    lora_resume = project.root / "runs" / "LoRA run" / "checkpoints" / "step-00000050"
    resume.set_value(lora_resume)
    output.set_value(lora_resume.parent.parent)
    window.training_controls["learning_rate"].setValue(0.000067)
    assert window.save_training_settings()
    lora_before = window.training_config().to_dict()
    training_file = project.root / "training-settings.json"
    training_before = training_file.read_bytes()

    run = project.root / "runs" / "Original run"
    checkpoint = run / "checkpoints" / "step-00000100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.json").write_text("{}")
    run_config = TrainingConfig(
        mode="full", model_path=str(tmp_path / "Saved model"), model_variant="fp16",
        dataset_path=str(tmp_path / "Saved dataset"), output_dir=str(run), learning_rate=0.000004,
        optimizer="adamw_disk",
    )
    run_config.save(run / "training-config.json")
    if method == "browse":
        monkeypatch.setattr(desktop.file_dialogs, "get_existing_directory", lambda *args, **kwargs: str(checkpoint))
        resume.findChild(QPushButton).click()
    else:
        window.navigation.setCurrentRow(2)
        resume.edit.setFocus()
        resume.edit.selectAll()
        QTest.keyClicks(resume.edit, str(checkpoint))
        QTest.keyClick(resume.edit, Qt.Key.Key_Return)
    assert output.value() == str(run)
    assert window.training_config().mode == "full"
    assert training_file.read_bytes() == training_before

    branch = project.root / "runs" / "New output"
    output.set_value(branch)
    mode = window.training_controls["mode"]
    mode.setCurrentText("lora")
    shared = {name: getattr(run_config, name) for name in ("model_path", "model_variant", "dataset_path")}
    assert window.training_config().to_dict() == lora_before | shared
    mode.setCurrentText("full")
    assert (resume.value(), output.value()) == (str(checkpoint), str(branch))
    assert window.save_training_settings()
    window.set_project(DatasetProject.open(project.root))
    assert (resume.value(), output.value()) == (str(checkpoint), str(branch))

    recipe = tmp_path / "recipe.json"
    recipe_output = tmp_path / "Recipe output"
    TrainingConfig(mode="full", output_dir=str(recipe_output), resume_from=str(checkpoint)).save(recipe)
    monkeypatch.setattr(desktop.file_dialogs, "get_open_file_name", lambda *args, **kwargs: (str(recipe), "JSON"))
    window.load_recipe()
    assert (resume.value(), output.value()) == (str(checkpoint), str(recipe_output))
    assert not window.test_errors


def test_live_gpu_and_step_timing_events_show_readable_metrics(window):
    class Process:
        def readAllStandardOutput(self):
            records = [
                {"type": "gpu", "available": True, "device_name": "RTX 5080",
                 "utilization_percent": 73.0, "memory_used_mib": 8192.0, "memory_total_mib": 16384.0,
                 "system_memory": {"mem_available_mib": 10240, "swap_used_mib": 3072,
                                   "swap_total_mib": 8192, "process_swap_mib": 512}},
                {"type": "progress", "step": 2, "total": 5, "loss": 0.25},
                {"type": "telemetry", "step": 2, "step_seconds": 12.5,
                 "phase_wall_seconds": {"vae": 1.25, "optimizer": 3.5}},
            ]
            return ("\n".join(json.dumps(record) for record in records) + "\n").encode()
    window.process = Process()
    window.process_output = window.logs
    window.process_buffer = ""
    window.read_process()
    window.process = None
    assert "RTX 5080" in window.gpu_status.text()
    assert "73%" in window.gpu_status.text()
    assert "8.0 / 16.0 GiB" in window.gpu_status.text()
    assert "Available RAM: 10.0 GiB" in window.system_memory_status.text()
    assert "system swap: 3.0 / 8.0 GiB" in window.system_memory_status.text()
    assert "worker swap: 0.5 GiB" in window.system_memory_status.text()
    assert "step 12.5s" in window.phase_status.text()
    assert "optimizer 3.5s" in window.phase_status.text()
    assert '"type": "gpu"' not in window.logs.toPlainText()


def test_memory_stays_live_when_gpu_monitor_is_unavailable(window):
    class Process:
        def readAllStandardOutput(self):
            return (json.dumps({"type": "gpu", "available": False, "reason": "nvidia-smi missing",
                                "system_memory": {"mem_available_mib": 2048, "swap_used_mib": 1024,
                                                  "swap_total_mib": 4096, "process_swap_mib": None}}) + "\n").encode()
    window.process = Process()
    window.process_output = window.logs
    window.process_buffer = ""
    window.read_process()
    window.process = None
    assert "nvidia-smi missing" in window.gpu_status.text()
    assert "Available RAM: 2.0 GiB" in window.system_memory_status.text()
    assert "system swap: 1.0 / 4.0 GiB" in window.system_memory_status.text()
    assert "worker swap: —" in window.system_memory_status.text()


def fake_token_python(tmp_path, content):
    executable = tmp_path / "token-python"
    executable.write_text(f"#!{sys.executable}\n" + content, encoding="utf-8")
    executable.chmod(0o755)
    return executable


TOKEN_PROTOCOL = '''import json, sys
print(json.dumps({'ready':True,'max_length':77,'metadata':{'tokenizer_sha256':'a'*64,'tokenizer_2_sha256':'b'*64}}),flush=True)
for line in sys.stdin:
    text=json.loads(line)['text']
    print(json.dumps({'clip_l':len(text.split())+2,'clip_g':len(text.split())+3,'max_length':77}),flush=True)
'''


def test_real_local_token_process_updates_editor_counts(window, qt_app, tmp_path):
    project, _ = imported_project(tmp_path, 1)
    project.save_caption(1, "three caption words")
    window.set_project(project)
    executable = fake_token_python(tmp_path, TOKEN_PROTOCOL)
    window.runtime.set_value(executable)
    window.model_field.set_value(tmp_path / "local-model")
    wait_until(qt_app, lambda: "CLIP-L 5 tokens" in window.tokens_label.text())
    assert "CLIP-G 6 tokens" in window.tokens_label.text()
    window.caption.setPlainText("changed caption")
    wait_until(qt_app, lambda: "CLIP-L 4 tokens" in window.tokens_label.text())
    assert "CLIP-G 5 tokens" in window.tokens_label.text()
    window.caption.setPlainText(" ".join(["word"] * 155))
    wait_until(qt_app, lambda: "CLIP-L 157 tokens" in window.tokens_label.text())
    assert "CLIP-G 158 tokens" in window.tokens_label.text()
    assert "3 × 77-token windows" in window.tokens_label.text()
    window.stop_tokens()
    assert window.token_process is None
    assert not window.test_errors


def test_remote_counter_preserves_metadata_and_real_protocol(tmp_path):
    executable = fake_token_python(tmp_path, TOKEN_PROTOCOL)
    with RemoteTokenCounter(str(executable), "local-model", startup_timeout=2) as counter:
        assert counter.count("a small caption") == TokenCounts(5, 6)
        assert counter.metadata["tokenizer_sha256"] == "a" * 64
        process = counter.process
    assert process.poll() is not None


def test_remote_counter_silent_runtime_times_out_and_does_not_leak_process(tmp_path):
    executable = fake_token_python(tmp_path, "import time\ntime.sleep(30)\n")
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        RemoteTokenCounter(str(executable), "local-model", startup_timeout=0.1)
    assert time.monotonic() - started < 2


def test_missing_dictionary_does_not_raise_in_editor(qt_app):
    from sdxl_trainer.widgets import CaptionEditor
    class Unavailable:
        available = False
        def misspellings(self, text):
            raise AssertionError("Unavailable dictionaries must not be called by the highlighter.")
    editor = CaptionEditor(Unavailable())
    editor.setPlainText("A caption with an unavailable dictionary.")
    editor.highlighter.rehighlight()
    qt_app.processEvents()
    editor.deleteLater()
