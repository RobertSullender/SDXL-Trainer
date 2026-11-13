"""Fast preferences preserve experiment/caption data and large lists stay lazy."""
import copy
import json

import pytest

pytest.importorskip("PySide6")
from PIL import Image

from sdxl_trainer import app as desktop
from sdxl_trainer.dataset import DatasetEntry, DatasetProject
from sdxl_trainer.paths import migrate_training_defaults
from sdxl_trainer.training.config import TrainingConfig, fast_training_settings
from test_ui import imported_project, qt_app, wait_until, window


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_fast_preferences_migration_changes_only_performance_fields(mode):
    caption = {"temperature": 0.2, "system_prompt": "Keep my exact caption instructions."}
    training = {
        "mode": mode, "model_path": "/models/base", "dataset_path": "/project/finalized/Dataset-1",
        "output_dir": "/project/runs/Run-1", "resume_from": "", "learning_rate": 0.0005,
        "text_encoder_lr": 0.0005, "vae_lr": 0.0005, "max_steps": 32200,
        "save_every": 32200, "batch_size": 1, "gradient_accumulation_steps": 4,
        "seed": 123, "max_grad_norm": 10.0, "optimizer": "adamw_disk",
    }
    settings = {"caption": caption, "last_project": "/project", "training": training,
                "model_library": "/models", "runtime_python": "/runtime/python"}
    before = copy.deepcopy(settings)
    assert migrate_training_defaults(settings)
    expected = {**before, "training": {**training, **fast_training_settings(mode)},
                "training_defaults_version": 1}
    assert settings == expected
    assert settings["caption"] == caption
    assert settings["training"]["caption_mode"] == "chunked"
    # A later custom selection survives subsequent application launches.
    settings["training"]["optimizer"] = "sgd"
    assert not migrate_training_defaults(settings)
    assert settings["training"]["optimizer"] == "sgd"


def test_startup_persists_only_fast_preferences_migration(qt_app, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    saved = {"caption": {"temperature": 0.37, "system_prompt": "Exact legacy prompt"},
             "training": {"mode": "full", "learning_rate": 0.0005, "max_steps": 32200},
             "model_library": str(tmp_path / "models")}
    desktop.atomic_json(desktop.config_dir() / "settings.json", saved)
    instance = desktop.Window()
    try:
        persisted = desktop.load_settings()
        assert persisted["caption"] == saved["caption"]
        assert persisted["model_library"] == saved["model_library"]
        assert persisted["training"]["learning_rate"] == 0.0005
        assert persisted["training"]["max_steps"] == 32200
        assert persisted["training"]["optimizer"] == "paged_adamw8bit"
        assert persisted["training_defaults_version"] == 1
    finally:
        instance.close()
        qt_app.processEvents()


def test_fast_button_preserves_project_captions_caption_settings_and_experiment_fields(window, tmp_path):
    project, _ = imported_project(tmp_path, 1)
    project.save_caption(1, "The complete original caption stays here.")
    window.set_project(project)
    window.caption_form.set_values({
        "system_prompt": "Describe exactly what is visible.",
        "trim_generated_captions": False,
        "temperature": 0.61,
    })
    window.shared_caption_paths["model_path"].set_value(tmp_path / "shared-caption-model.gguf")
    window.save_settings()
    protected = {
        "model_path": str(tmp_path / "base"), "dataset_path": str(project.root / "finalized/Dataset-1"),
        "output_dir": str(project.root / "runs/Run-7"), "resume_from": "",
        "learning_rate": 0.0005, "text_encoder_lr": 0.0005, "vae_lr": 0.0005,
        "max_steps": 32200, "save_every": 32200, "seed": 123,
        "batch_size": 1, "gradient_accumulation_steps": 4, "max_grad_norm": 10.0,
    }
    window.training_controls["mode"].setCurrentText("full")
    for form in window.training_forms:
        form.set_values({**protected, "optimizer": "adamw_disk"})
    prompt_bytes = (project.root / "caption-prompts.json").read_bytes()
    caption_bytes = (project.image_dir / "Img-1.txt").read_bytes()
    project_bytes = (project.root / "project.json").read_bytes()
    app_settings = desktop.load_settings()
    window.fast_training_button.click()
    values = window.training_config().to_dict()
    for key, value in protected.items():
        assert values[key] == value, key
    for key, value in fast_training_settings("full").items():
        assert values[key] == value, key
    assert (project.root / "caption-prompts.json").read_bytes() == prompt_bytes
    assert (project.image_dir / "Img-1.txt").read_bytes() == caption_bytes
    assert (project.root / "project.json").read_bytes() == project_bytes
    assert window.caption_form.controls["system_prompt"].toPlainText() == "Describe exactly what is visible."
    assert not window.caption_form.controls["trim_generated_captions"].isChecked()
    assert window.caption_form.controls["temperature"].value() == 0.61
    saved = json.loads((project.root / "training-settings.json").read_text())
    saved.pop("mode_profiles", None)
    assert saved == values
    assert desktop.load_settings() == app_settings


def test_mode_switch_restores_remembered_controls_and_keeps_selected_files(window, tmp_path, monkeypatch):
    monkeypatch.setattr(window, "request_token_count", lambda: None)
    mode = window.training_controls["mode"]
    files = {
        "model_path": str(tmp_path / "model"),
        "model_variant": "fp16",
        "dataset_path": str(tmp_path / "dataset"),
    }
    lora_edits = {
        "output_dir": str(tmp_path / "lora-run"),
        "resume_from": str(tmp_path / "lora-run" / "checkpoint"),
        "optimizer": "sgd",
        "learning_rate": 0.0003,
        "text_encoder_lr": 0.000043,
        "vae_lr": 0.000017,
        "max_steps": 4000,
        "batch_size": 2,
        "train_text_encoder": True,
        "train_vae": True,
        "seed": 213,
        "caption_mode": "native",
    }
    assert mode.currentText() == "lora"
    assert window.training_config().learning_rate == 1e-4
    for form in window.training_forms:
        form.set_values(files | lora_edits)
    lora = window.training_config().to_dict()

    mode.setCurrentText("full")
    full_defaults = TrainingConfig().to_dict() | fast_training_settings("full") | files | {
        "mode": "full", "learning_rate": 1e-6,
    }
    assert window.training_config().to_dict() == full_defaults
    full_edits = {"learning_rate": 0.000004, "max_steps": 137, "optimizer": "adamw_disk",
                  "output_dir": str(tmp_path / "full-run"),
                  "resume_from": str(tmp_path / "full-run" / "checkpoint-full")}
    for form in window.training_forms:
        form.set_values(full_edits)
    full = window.training_config().to_dict()

    mode.setCurrentText("lora")
    assert window.training_config().to_dict() == lora
    mode.setCurrentText("full")
    assert window.training_config().to_dict() == full


def test_loading_recipe_restores_its_method_settings_and_keeps_other_method(window, tmp_path, monkeypatch):
    monkeypatch.setattr(window, "request_token_count", lambda: None)
    recipe = tmp_path / "recipe.json"
    paths = {"model_path": str(tmp_path / "model"), "model_variant": "fp16",
             "dataset_path": str(tmp_path / "dataset")}
    config = TrainingConfig(mode="full", optimizer="adamw_disk", learning_rate=0.000004, max_steps=3500,
                            output_dir=str(tmp_path / "full-run"), **paths)
    config.save(recipe)
    monkeypatch.setattr(desktop.file_dialogs, "get_open_file_name", lambda *args, **kwargs: (str(recipe), "JSON"))
    mode = window.training_controls["mode"]
    assert mode.currentText() == "lora"
    window.training_controls["learning_rate"].setValue(0.00023)
    window.training_controls["output_dir"].set_value(tmp_path / "lora-run")
    lora = window.training_config().to_dict()

    window.load_recipe()

    assert window.training_config().to_dict() == config.to_dict()
    mode.setCurrentText("lora")
    assert window.training_config().to_dict() == lora | paths
    mode.setCurrentText("full")
    assert window.training_config().to_dict() == config.to_dict()
    assert not window.test_errors


@pytest.mark.parametrize(("mode", "optimizer"), [
    ("lora", "adamw"),
    ("full", "paged_adamw8bit"),
])
def test_fast_button_uses_selected_mode_without_a_project(window, mode, optimizer):
    assert window.project is None
    assert window.fast_training_button.isEnabled()
    window.training_controls["mode"].setCurrentText(mode)
    optimizer_control = window.training_controls["optimizer"]
    optimizer_control.setCurrentIndex(optimizer_control.findData("sgd"))
    window.training_controls["learning_rate"].setValue(0.0005)
    settings_path = desktop.config_dir() / "settings.json"
    assert not settings_path.exists()
    before = window.training_config().to_dict()

    window.fast_training_button.click()

    assert window.training_config().to_dict() == before | fast_training_settings(mode)
    assert optimizer_control.currentData() == optimizer
    assert not settings_path.exists()
    assert not window.test_errors


def test_large_image_list_loads_only_visible_scaled_thumbnails(window, qt_app, tmp_path, monkeypatch):
    project = DatasetProject.create(tmp_path / "large-project")
    for number in range(1, 121):
        image = f"Img-{number}.png"
        caption = f"Img-{number}.txt"
        Image.new("RGB", (512, 256), (number, 30, 80)).save(project.image_dir / image)
        (project.image_dir / caption).write_text(f"Caption {number}")
        project.entries.append(DatasetEntry(number, image, caption, 512, 256, [512, 256], [0, 0],
                                           "crop", str(tmp_path / "original.png"), "test"))
    project.next_image_id = 121
    project.save()
    loaded = []
    original = window._load_image_thumbnail

    def tracked(path):
        loaded.append(path.name)
        return original(path)

    monkeypatch.setattr(window, "_load_image_thumbnail", tracked)
    window.set_project(project)
    assert not loaded, "Opening the list must not decode all image thumbnails synchronously"
    wait_until(qt_app, lambda: bool(loaded) and not window.thumbnail_timer.isActive())
    assert len(loaded) < 20
    icon = window.images.item(0).icon()
    assert not icon.isNull()
    assert all(size.width() <= 74 and size.height() <= 74 for size in icon.availableSizes())
    window.images.scrollToItem(window.images.item(119))
    wait_until(qt_app, lambda: "Img-120.png" in loaded and not window.thumbnail_timer.isActive())
    assert len(loaded) < 35
    assert not window.images.item(119).icon().isNull()
    assert project.caption_text(120) == "Caption 120"


def test_hidden_dataset_page_does_not_decode_thumbnails(window, qt_app, tmp_path, monkeypatch):
    project, _ = imported_project(tmp_path, 2)
    loaded = []
    original = window._load_image_thumbnail
    monkeypatch.setattr(window, "_load_image_thumbnail", lambda path: loaded.append(path) or original(path))
    window.navigation.setCurrentRow(2)
    window.set_project(project)
    qt_app.processEvents()
    assert loaded == []
    window.navigation.setCurrentRow(0)
    wait_until(qt_app, lambda: len(loaded) == 2)
