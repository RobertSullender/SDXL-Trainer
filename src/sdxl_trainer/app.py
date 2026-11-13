"""Native, offline desktop UI. GPU workloads live in separate managed processes."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import uuid

from PySide6.QtCore import Qt, QLockFile, QPoint, QProcess, QProcessEnvironment, QSignalBlocker, QSize, QTimer
from PySide6.QtGui import QIcon, QImageReader, QPixmap, QCloseEvent
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSplitter, QStackedWidget,
    QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from . import __version__, file_dialogs
from .captioning import CaptionConfig, CaptionServer
from .dataset import DatasetProject, validate_project_name
from .paths import atomic_json, config_dir, data_dir, load_settings, migrate_training_defaults, offline_environment, projects_dir
from .project_purge import preview_project_purge, purge_project
from .project_prompts import (
    PROMPT_KEYS, default_prompts, load_project_caption_settings, migrate_legacy_prompts,
    save_project_caption_settings,
)
from .project_remove import remove_images
from .project_training import (
    SHARED_TRAINING_FIELDS, load_project_training_profiles, save_project_training_settings, saved_project_runs,
)
from .runtime import RemoteTokenCounter
from .spelling import SpellChecker
from .training.config import (
    DEFAULT_UNET_LEARNING_RATES, TrainingConfig, checkpoint_run_folder, fast_training_settings,
    has_checkpoint_metadata, preflight, run_checkpoints,
)
from .widgets import CaptionEditor, ConfigForm, PathField, Task
from .theme import FILE_DIALOG_OPTIONS, apply_theme


ROOT = Path(__file__).resolve().parent
CAPTION_PATH_FIELDS = ("server_executable", "model_path", "mmproj_path")
TRAINING_MODEL_FIELDS = ("model_path", "model_variant")



def button(text, callback, primary=False):
    widget = QPushButton(text)
    if primary:
        widget.setObjectName("primary")
    widget.clicked.connect(callback)
    return widget


def note(text):
    label = QLabel(text)
    label.setObjectName("muted")
    label.setWordWrap(True)
    return label


def scroll(widget):
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    return area


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        if "training" in self.settings and migrate_training_defaults(self.settings):
            # Persist only the preferences migration before loading any project;
            # caption prompts and draft images do not participate in this save.
            atomic_json(config_dir() / "settings.json", self.settings)
        self._legacy_caption_project = self.settings.get("last_project", "")
        legacy = self.settings.get("caption", {})
        self._legacy_caption = dict(legacy) if isinstance(legacy, dict) else {}
        legacy_training = self.settings.get("training", {})
        self._legacy_training = dict(legacy_training) if isinstance(legacy_training, dict) else {}
        self._legacy_training_project = self.settings.get("last_project", "") if self._legacy_training else ""
        self.project = None
        self.entry_id = None
        self.task = None
        self.process = None
        self._completed_training = False
        self.token_process = None
        self.token_buffer = ""
        self.token_ready = False
        self.token_queue = []
        self.token_generation = 0
        self.loaded_pixmap = QPixmap()
        self.checker = SpellChecker()
        self.setWindowTitle("SDXL-Trainer")
        self.setWindowIcon(QIcon(str(ROOT / "assets/sdxl-trainer.svg")))
        self.resize(1320, 900)
        self.setMinimumSize(1020, 740)
        apply_theme(QApplication.instance())
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        self.content = QWidget()
        horizontal = QHBoxLayout(self.content)
        horizontal.setContentsMargins(0, 0, 0, 0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(218)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(18, 26, 18, 20)
        logo = QLabel()
        logo.setPixmap(QIcon(str(ROOT / "assets/sdxl-trainer.svg")).pixmap(52, 52))
        side.addWidget(logo)
        brand = QLabel("SDXL-Trainer")
        brand.setObjectName("brand")
        side.addWidget(brand)
        side.addWidget(note("Your images. Your machine."))
        side.addSpacing(25)
        self.navigation = QListWidget()
        self.navigation.setObjectName("navigation")
        self.navigation.addItems(["01   Dataset", "02   Caption settings", "03   Training", "04   Checkpoint export", "05   Models & runtime", "06   Offline wiki"])
        side.addWidget(self.navigation, 1)
        side.addWidget(note(f"v{__version__} · Local workspace\nNo telemetry · No cloud training"))
        self.pages = QStackedWidget()
        horizontal.addWidget(sidebar)
        horizontal.addWidget(self.pages, 1)
        self.build_dataset()
        self.build_caption_settings()
        self.build_training()
        self.build_export()
        self.build_models()
        self.build_wiki()
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.navigation.currentRowChanged.connect(self._queue_thumbnail_load)
        self.navigation.setCurrentRow(0)
        outer.addWidget(self.content, 1)
        bottom = QHBoxLayout()
        bottom.setContentsMargins(18, 6, 18, 10)
        self.status = note("Ready. Create or open a dataset project to begin.")
        self.activity = QProgressBar()
        self.activity.setTextVisible(False)
        self.activity.setFixedWidth(150)
        self.activity.setRange(0, 1)
        self.activity.setValue(0)
        self.cancel_button = button("Cancel task", self.cancel_task)
        self.cancel_button.setEnabled(False)
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.activity)
        bottom.addWidget(self.cancel_button)
        outer.addLayout(bottom)
        self.setCentralWidget(central)
        self.caption_timer = QTimer(self)
        self.caption_timer.setSingleShot(True)
        self.caption_timer.setInterval(450)
        self.caption_timer.timeout.connect(self.save_caption)
        self.caption.textChanged.connect(self.caption_changed)
        self.model_field.changed.connect(self.model_changed)
        self.runtime.changed.connect(self.model_changed)
        if self.settings.get("last_project"):
            try:
                self.set_project(DatasetProject.open(self.settings["last_project"]))
            except (OSError, ValueError):
                pass

    def page(self, title, subtitle):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 25, 28, 20)
        label = QLabel(title)
        label.setObjectName("title")
        layout.addWidget(label)
        layout.addWidget(note(subtitle))
        layout.addSpacing(12)
        self.pages.addWidget(page)
        return page, layout

    def build_dataset(self):
        _, layout = self.page("Build a dataset", "Prepare pixels first. Caption what you see. Finalize a separate, verified training snapshot.")
        row = QHBoxLayout()
        row.addWidget(button("New project", self.new_project, True))
        row.addWidget(button("Open project", self.open_project))
        self.purge_project_button = button("Fully purge project…", self.purge_project)
        self.purge_project_button.setEnabled(False)
        self.purge_project_button.setToolTip("Permanently remove this project and everything inside its folder, after confirmation.")
        row.addWidget(self.purge_project_button)
        row.addStretch()
        row.addWidget(button("Import files…", lambda: self.import_images(False)))
        row.addWidget(button("Import folder…", lambda: self.import_images(True)))
        layout.addLayout(row)
        options = QHBoxLayout()
        options.addWidget(QLabel("Fit to nearest SDXL bucket"))
        self.fit = QComboBox()
        self.fit.addItem("Crop to fill", "crop")
        self.fit.addItem("Pad to fit", "pad")
        options.addWidget(self.fit)
        options.addWidget(QLabel("Fill color"))
        self.background = QLineEdit("#000000")
        self.background.setMaximumWidth(110)
        options.addWidget(self.background)
        self.recursive = QCheckBox("Include subfolders")
        options.addWidget(self.recursive)
        self.allow_upscale = QCheckBox("Allow upscaling")
        self.allow_upscale.setChecked(self.settings.get("allow_import_upscale", True) is not False)
        self.allow_upscale.setToolTip("Clear this to skip images that need enlarging to fit the selected crop or pad setting.")
        options.addWidget(self.allow_upscale)
        options.addStretch()
        layout.addLayout(options)
        self.project_label = note("No project open · Original images are never modified")
        layout.addWidget(self.project_label)
        splitter = QSplitter()
        self.images = QListWidget()
        self.images.setIconSize(QSize(74, 74))
        self.images.setUniformItemSizes(True)
        self._thumbnail_ids = set()
        self.thumbnail_timer = QTimer(self)
        self.thumbnail_timer.setSingleShot(True)
        self.thumbnail_timer.timeout.connect(self._load_visible_thumbnails)
        self.images.verticalScrollBar().valueChanged.connect(self._queue_thumbnail_load)
        self.images.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.images.setToolTip("Use Ctrl or Shift to select multiple images for caption edits or removal.")
        self.images.currentRowChanged.connect(self.select_image)
        splitter.addWidget(self.images)
        right = QWidget()
        panel = QVBoxLayout(right)
        panel.setContentsMargins(12, 0, 0, 0)
        self.preview = QLabel("Import images to see the exact training crop")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(360, 240)
        self.preview.setStyleSheet("background:#090a10; border-radius:10px; color:#777d9b;")
        panel.addWidget(self.preview, 1)
        self.image_info = note("RGB PNG · Metadata removed · 64-pixel bucket grid")
        panel.addWidget(self.image_info)
        caption_row = QHBoxLayout()
        caption_row.addWidget(QLabel("Training caption"))
        self.bulk_caption_button = button("Bulk caption…", self.bulk_caption)
        self.bulk_caption_button.setEnabled(False)
        self.bulk_caption_button.setToolTip("Edit captions for selected images or the whole draft.")
        caption_row.addWidget(self.bulk_caption_button)
        self.trim_caption_button = button("Trim to 77…", lambda: self.bulk_caption(trim=True))
        self.trim_caption_button.setEnabled(False)
        self.trim_caption_button.setToolTip("Choose captions to shorten to the native 77-token SDXL window.")
        caption_row.addWidget(self.trim_caption_button)
        caption_row.addStretch()
        self.tokens_label = note("CLIP-L — tokens · CLIP-G — tokens · — windows")
        caption_row.addWidget(self.tokens_label)
        panel.addLayout(caption_row)
        self.caption = CaptionEditor(self.checker)
        panel.addWidget(self.caption)
        self.spelling_label = note("Spelling: English (offline). Right-click an underlined word for suggestions." if self.checker.available else f"Spelling unavailable: {self.checker.reason}")
        panel.addWidget(self.spelling_label)
        actions = QHBoxLayout()
        caption_selected = button("Caption selected", lambda: self.auto_caption(False))
        caption_selected.setToolTip("Generate new captions for all selected images, replacing their current captions.")
        actions.addWidget(caption_selected)
        actions.addWidget(button("Caption all empty", lambda: self.auto_caption(True)))
        self.remove_images_button = button("Remove selected…", self.remove_selected_images)
        self.remove_images_button.setToolTip("Remove selected image/caption pairs and renumber the remaining draft images.")
        actions.addWidget(self.remove_images_button)
        actions.addStretch()
        actions.addWidget(button("Finalize dataset…", self.finalize, True))
        panel.addLayout(actions)
        splitter.addWidget(right)
        splitter.setSizes([250, 750])
        layout.addWidget(splitter, 1)

    def build_caption_settings(self):
        _, layout = self.page("Caption settings", "Qwen3.5-9B GGUF + matching vision projector. The managed server unloads after captioning to release VRAM.")
        config = CaptionConfig()
        bundled = Path("/opt/sdxl-trainer/runtime/llama/llama-server")
        development = ROOT.parents[1] / "build/llama-runtime/llama-server"
        if bundled.is_file() or development.is_file():
            config.server_executable = str(bundled if bundled.is_file() else development)
        self._caption_defaults = asdict(config)
        self.caption_form = ConfigForm(config)
        self.caption_form.setEnabled(False)
        self.prompt_timer = QTimer(self)
        self.prompt_timer.setSingleShot(True)
        self.prompt_timer.setInterval(450)
        self.prompt_timer.timeout.connect(self.save_project_caption_form)
        for name in PROMPT_KEYS:
            control = self.caption_form.controls[name]
            control.textChanged.connect(self.prompt_timer.start)
        layout.addWidget(scroll(self.caption_form), 1)
        prompt_actions = QHBoxLayout()
        self.reset_system_prompt = button("Reset system prompt", lambda: self.reset_project_prompt("system_prompt"))
        self.reset_user_prompt = button("Reset user prompt", lambda: self.reset_project_prompt("user_prompt"))
        self.reset_all_caption_settings_button = button("Reset all caption settings", self.reset_all_caption_settings)
        self.reset_all_caption_settings_button.setToolTip(
            "Restore the default prompts, generation controls, and 77-token trimming for the selected models."
        )
        for reset in (self.reset_system_prompt, self.reset_user_prompt, self.reset_all_caption_settings_button):
            reset.setEnabled(False)
            prompt_actions.addWidget(reset)
        prompt_actions.addStretch()
        layout.addLayout(prompt_actions)
        layout.addWidget(note("Open or create a project to edit its caption settings. Configure shared model paths in Models & runtime."))
        self.save_caption_settings_button = button("Save caption settings", self.save_caption_settings, True)
        self.save_caption_settings_button.setEnabled(False)
        layout.addWidget(self.save_caption_settings_button)

    def build_training(self):
        training_page, layout = self.page("Train SDXL", "Only finalized datasets can train. Every control below is passed to the training worker; hover for its explanation.")
        tabs = QTabWidget()
        self.training_forms = []
        config = TrainingConfig()
        self._training_defaults = config.to_dict()
        self._suggested_branch_outputs = {}
        initial_values = self._legacy_training | self._saved_training_model()
        groups = list(dict.fromkeys(f.metadata.get("group", "Training") for f in fields(config)))
        for group in groups:
            names = [f.name for f in fields(config) if f.metadata.get("group", "Training") == group]
            form = ConfigForm(config, names)
            form.set_values(initial_values)
            if group == "Files":
                self.project_dataset_button = button("Choose project dataset…", self.choose_project_dataset)
                self.project_dataset_button.setEnabled(False)
                self.project_dataset_button.setToolTip("Choose one bucket or all aspect ratios from an export revision.")
                form.controls["dataset_path"].layout().addWidget(self.project_dataset_button)
                self.checkpoint_step = QComboBox()
                self.checkpoint_step.setToolTip("Choose a completed step from the selected run.")
                self._checkpoint_form_layout = form.layout()
                self.saved_runs = QComboBox()
                self.saved_runs.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                self.saved_runs.setMinimumContentsLength(24)
                self.saved_runs.setToolTip("Choose a project run to load its latest checkpoint. Use Browse for other runs.")
                resume_row, _ = self._checkpoint_form_layout.getWidgetPosition(form.controls["resume_from"])
                self._checkpoint_form_layout.insertRow(resume_row, "Saved runs", self.saved_runs)
                self._checkpoint_form_layout.addRow("Checkpoint step", self.checkpoint_step)
                self._checkpoint_form_layout.setRowVisible(self.saved_runs, False)
                self._checkpoint_form_layout.setRowVisible(self.checkpoint_step, False)
            tabs.addTab(scroll(form), group)
            self.training_forms.append(form)
        self.training_controls = {name: widget for form in self.training_forms for name, widget in form.controls.items()}
        self._training_mode = self.training_controls["mode"].currentText()
        self._training_mode_profiles = {}
        self.training_controls["mode"].currentTextChanged.connect(self.training_mode_changed)
        resume = self.training_controls["resume_from"]
        self._committed_resume = resume.value()
        resume.changed.connect(self.track_resume_value)
        resume.path_selected.connect(self.load_resume_selection)
        resume.edit.editingFinished.connect(self.finish_resume_edit)
        self.saved_runs.activated.connect(self.select_saved_run)
        self.pages.currentChanged.connect(lambda index: self.refresh_saved_runs()
                                          if self.pages.widget(index) is training_page else None)
        self.checkpoint_step.currentIndexChanged.connect(self.select_checkpoint_step)
        output = self.training_controls["output_dir"]
        output.changed.connect(self.clear_suggested_branch_output)
        output.path_selected.connect(self.clear_suggested_branch_output)
        self.model_field = self.training_controls["model_path"]
        layout.addWidget(tabs, 1)
        layout.addWidget(note("VAE adaptation is experimental: reconstruction + KL + a frozen-base latent anchor. Review reconstructions before relying on it. Full training on 16 GB requires measured validation."))
        row = QHBoxLayout()
        row.addWidget(button("Load recipe…", self.load_recipe))
        row.addWidget(button("Save recipe…", self.save_recipe))
        self.fast_training_button = button("Fast training defaults", self.apply_fast_training_defaults)
        self.fast_training_button.setToolTip(
            "Apply the fast profile for the selected training method. LoRA uses fused GPU AdamW. "
            "Full fine-tuning uses paged 8-bit AdamW. Keep paths, learning rates, steps, and seed."
        )
        row.addWidget(self.fast_training_button)
        row.addStretch()
        self.reset_all_training_settings_button = button("Reset all training settings", self.reset_all_training_settings)
        self.reset_all_training_settings_button.setToolTip(
            "Choose LoRA or full fine-tune defaults. Keep the model, file variant, dataset, and output paths."
        )
        self.reset_all_training_settings_button.setEnabled(False)
        row.addWidget(self.reset_all_training_settings_button)
        self.save_training_settings_button = button("Save training settings", self.save_training_settings)
        self.save_training_settings_button.setEnabled(False)
        row.addWidget(self.save_training_settings_button)
        layout.addLayout(row)
        run_actions = QHBoxLayout()
        run_actions.addStretch()
        run_actions.addWidget(button("Check configuration", self.check_training))
        self.start_training = button("Start training", self.train, True)
        run_actions.addWidget(self.start_training)
        layout.addLayout(run_actions)
        self.training_progress = QProgressBar()
        self.training_progress.setTextVisible(False)
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        layout.addWidget(self.training_progress)
        self.gpu_status = note("GPU: waiting for training")
        self.gpu_status.setToolTip("Device-wide utilization and VRAM sampled locally by NVIDIA; other applications are included.")
        self.system_memory_status = note("System memory: waiting for training")
        self.system_memory_status.setToolTip("Available RAM and system swap include all processes. Worker swap is only the training worker process.")
        self.phase_status = note("Step timings will appear after an optimizer update.")
        self.phase_status.setToolTip("Wall-clock phase times can overlap asynchronous GPU work. CUDA stream intervals can include idle time.")
        layout.addWidget(self.gpu_status)
        layout.addWidget(self.system_memory_status)
        layout.addWidget(self.phase_status)
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(2500)
        self.logs.setMinimumHeight(130)
        self.logs.setMaximumHeight(220)
        layout.addWidget(self.logs)

    def build_export(self):
        _, layout = self.page("Export a checkpoint", "Training saves a complete merged Diffusers pipeline. Convert that result to a conventional SDXL safetensors checkpoint.")
        form = QFormLayout()
        self.export_model = PathField()
        self.export_output = PathField(kind="save")
        form.addRow("Trained Diffusers folder", self.export_model)
        form.addRow("Output .safetensors", self.export_output)
        layout.addLayout(form)
        layout.addWidget(note("The checkpoint includes UNet, both text encoders, and VAE. Export preserves component weight precision and requires disk space for the combined file. The original Diffusers result remains available."))
        layout.addWidget(button("Export merged checkpoint", self.export_checkpoint, True))
        layout.addStretch()

    def build_models(self):
        from .downloader import PRESETS
        _, layout = self.page("Models & runtime", "Internet is used only by the download actions on this page. Training, captioning, editing, and documentation work locally.")
        tabs = QTabWidget()
        model_page = QWidget()
        form = QVBoxLayout(model_page)
        self.preset = QComboBox()
        self.preset.addItems([p["name"] for p in PRESETS] + ["Custom Hugging Face repository"])
        self.repo = QLineEdit(PRESETS[0]["repo_id"])
        self.preset.currentIndexChanged.connect(lambda i: self.repo.setText(PRESETS[i]["repo_id"]) if i < len(PRESETS) else None)
        self.download_dir = PathField(self.settings.get("model_library", str(data_dir() / "models")))
        self.hf_token = QLineEdit()
        self.hf_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token.setPlaceholderText("Optional token for gated models; held only in memory")
        fields_layout = QFormLayout()
        fields_layout.addRow("Preset", self.preset)
        fields_layout.addRow("Repository", self.repo)
        fields_layout.addRow("Model library", self.download_dir)
        fields_layout.addRow("Hugging Face token", self.hf_token)
        form.addLayout(fields_layout)
        form.addWidget(button("Connect & inspect repository", self.inspect_repository, True))
        self.remote_files = QListWidget()
        self.remote_files.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        form.addWidget(self.remote_files, 1)
        self.download_info = note("No connection has been made. Inspect to see file sizes and pin a repository revision.")
        form.addWidget(self.download_info)
        form.addWidget(button("Download selected files", self.download_selected, True))
        tabs.addTab(model_page, "Model downloads")
        runtime_page = QWidget()
        rt = QVBoxLayout(runtime_page)
        rt.addWidget(note("The GUI stays small by keeping PyTorch in a separate local Python environment. Select that environment's Python executable. The bundled llama.cpp server needs no separate installation."))
        selected_runtime = self.settings.get("runtime_python", sys.executable)
        managed_runtime = data_dir() / "runtime" / "bin" / "python"
        if selected_runtime in {"", sys.executable, "/usr/bin/python3"} and managed_runtime.is_file() and os.access(managed_runtime, os.X_OK):
            selected_runtime = str(managed_runtime)
        self.runtime = PathField(selected_runtime, kind="file")
        rtf = QFormLayout()
        rtf.addRow("Training Python", self.runtime)
        self.projects_location = PathField(self.settings.get("projects_dir", str(projects_dir())))
        self.projects_location.setToolTip("New named projects are created inside this folder. Existing projects and model files are not moved.")
        rtf.addRow("Projects folder", self.projects_location)
        rt.addLayout(rtf)
        rt.addWidget(note("Shared SDXL model for new projects"))
        self.shared_training_model = ConfigForm(TrainingConfig(), TRAINING_MODEL_FIELDS)
        self.shared_training_model.set_values(self._saved_training_model())
        rt.addWidget(self.shared_training_model)
        rt.addWidget(button("Inspect runtime", self.diagnose, True))
        self.runtime_info = QPlainTextEdit()
        self.runtime_info.setReadOnly(True)
        rt.addWidget(self.runtime_info, 1)
        rt.addWidget(button("Save settings", self.save_settings))
        tabs.addTab(runtime_page, "Local runtime")
        tabs.addTab(self.build_shared_caption_paths(), "Shared caption paths")
        layout.addWidget(tabs, 1)
        self.snapshot = None

    def _saved_training_model(self):
        saved = self.settings.get("shared_training_model", self._legacy_training)
        if not isinstance(saved, dict):
            saved = self._legacy_training
        return {name: saved[name] if isinstance(saved.get(name), str) else self._training_defaults[name]
                for name in TRAINING_MODEL_FIELDS}

    def build_shared_caption_paths(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(note("New projects start with these paths. Each project can save its own selections in Caption settings."))
        saved = self.settings.get("shared_caption_paths")
        legacy = not isinstance(saved, dict)
        if legacy:
            saved = self._legacy_caption
        paths = {name: saved[name] if isinstance(saved.get(name), str) else self._caption_defaults[name]
                 for name in CAPTION_PATH_FIELDS}
        engine = paths["server_executable"]
        bundled_engine = self._caption_defaults["server_executable"]
        if legacy and engine in {"", "llama-server"} and not shutil.which(engine) and Path(bundled_engine).is_file():
            paths["server_executable"] = bundled_engine

        labels = {"model_path": "Qwen model (GGUF)", "mmproj_path": "Vision projector (mmproj)",
                  "server_executable": "llama.cpp server"}
        form = QFormLayout()
        self.shared_caption_paths = {}
        for name, label in labels.items():
            field = PathField(paths[name], kind="file")
            self.shared_caption_paths[name] = field
            form.addRow(label, field)
        layout.addLayout(form)
        self.save_shared_caption_paths_button = button("Save shared caption paths", self.save_shared_caption_paths, True)
        layout.addWidget(self.save_shared_caption_paths_button)
        layout.addStretch()
        return page

    def build_wiki(self):
        _, layout = self.page("Offline wiki", "A local guide to this application, from first import to checkpoint export.")
        splitter = QSplitter()
        self.wiki_topics = QListWidget()
        self.wiki_browser = QTextBrowser()
        self.wiki_browser.document().setDefaultStyleSheet("a { color: #a8bfff; } th { background-color: #29263c; } td, th { padding: 6px; }")
        self.wiki_browser.setOpenExternalLinks(False)
        self.wiki_browser.setOpenLinks(False)
        self.wiki_browser.anchorClicked.connect(self.wiki_link)
        self.wiki_files = sorted((ROOT / "wiki").glob("*.md"))
        for path in self.wiki_files:
            self.wiki_topics.addItem(path.stem.replace("-", " ").title())
        self.wiki_topics.currentRowChanged.connect(self.show_wiki)
        splitter.addWidget(self.wiki_topics)
        splitter.addWidget(self.wiki_browser)
        splitter.setSizes([210, 750])
        layout.addWidget(splitter, 1)
        if self.wiki_files:
            self.wiki_topics.setCurrentRow(0)

    def show_wiki(self, row):
        if 0 <= row < len(self.wiki_files):
            self.wiki_browser.setMarkdown(self.wiki_files[row].read_text(encoding="utf-8"))

    def wiki_link(self, url):
        path = ROOT / "wiki" / url.toString().split("#")[0]
        if path in self.wiki_files:
            self.wiki_topics.setCurrentRow(self.wiki_files.index(path))
        else:
            self.status.setText("External references are listed for attribution. The wiki does not open network links.")

    def error(self, message):
        self.status.setText(message[:180])
        QMessageBox.warning(self, "SDXL-Trainer", message)

    def save_settings(self):
        if not self.save_project_caption_form() or not self.save_project_training_form():
            return False
        self.settings.update(runtime_python=self.runtime.value(),
                             projects_dir=self.projects_location.value(), model_library=self.download_dir.value(),
                             allow_import_upscale=self.allow_upscale.isChecked(),
                             last_project=str(self.project.root) if self.project else "")
        self._save_app_settings(shared_caption_paths=self._shared_caption_path_values(),
                                shared_training_model=self.shared_training_model.values())
        self.status.setText("Settings saved locally.")
        return True

    def _save_app_settings(self, **changes):
        settings = self.settings | changes
        if not self._legacy_caption_project:
            settings.pop("caption", None)
        if not self._legacy_training_project:
            settings.pop("training", None)
        atomic_json(config_dir() / "settings.json", settings)
        self.settings = settings

    def save_shared_caption_paths(self):
        try:
            self._save_app_settings(shared_caption_paths=self._shared_caption_path_values())
        except (OSError, ValueError) as exc:
            self.error(f"Could not save shared caption paths: {exc}")
            return False
        self.status.setText("Shared caption paths saved.")
        return True

    def save_caption_settings(self):
        if not self.project:
            return False
        if not self.save_project_caption_form():
            return False
        self.status.setText(f"Caption settings saved for {self.project.root.name}.")
        return True

    def save_project_caption_form(self):
        self.prompt_timer.stop()
        if not self.project:
            return True
        try:
            save_project_caption_settings(self.project, self.caption_form.values())
            return True
        except Exception as exc:
            self.error(f"Could not save this project's caption settings: {exc}")
            return False

    def save_training_settings(self):
        if not self.project or not self.save_project_training_form():
            return False
        self.status.setText(f"Training settings saved for {self.project.root.name}.")
        return True

    def reset_all_training_settings(self):
        if not self.project:
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Reset all training settings")
        dialog.setText("Choose the defaults to apply to this project.")
        dialog.setInformativeText(
            f"LoRA uses fused GPU AdamW and a {DEFAULT_UNET_LEARNING_RATES['lora']:.4f} UNet learning rate.\n"
            f"Full fine-tuning uses paged 8-bit AdamW and a {DEFAULT_UNET_LEARNING_RATES['full']:.6f} UNet learning rate.\n\n"
            "The chosen method's output is kept and its resume selection is cleared. "
            "The shared model and dataset and the other method's settings are kept."
        )
        lora = dialog.addButton("LoRA defaults", QMessageBox.ButtonRole.AcceptRole)
        full = dialog.addButton("Full fine-tune defaults", QMessageBox.ButtonRole.AcceptRole)
        cancel = dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(cancel)
        dialog.setEscapeButton(cancel)
        dialog.exec()
        choice = dialog.clickedButton()
        modes = {lora: "lora", full: "full"}
        mode = modes.get(choice)
        dialog.deleteLater()
        if mode is None:
            return
        current = self.training_config().to_dict()
        profile = current if mode == self._training_mode else self._training_mode_profiles.get(mode, {})
        values = self._training_preset(mode) | self._shared_training_values(current)
        if "output_dir" in profile:
            values["output_dir"] = profile["output_dir"]
        self._suggested_branch_outputs.pop(mode, None)
        self._select_training_mode(mode, values)
        if "output_dir" not in profile:
            self._suggest_run()
        if self.save_project_training_form():
            label = "LoRA" if mode == "lora" else "Full fine-tune"
            self.status.setText(f"{label} defaults saved for {self.project.root.name}.")

    def save_project_training_form(self):
        if not self.project:
            return True
        try:
            save_project_training_settings(
                self.project, self.training_config(), mode_profiles=self._training_mode_profiles,
                suggested_branch_outputs=self._suggested_branch_outputs,
            )
            return True
        except (OSError, ValueError) as exc:
            self.error(f"Could not save this project's training settings: {exc}")
            return False

    def reset_project_prompt(self, name):
        if self.project and name in PROMPT_KEYS:
            self.caption_form.controls[name].setPlainText(default_prompts()[name])

    def reset_all_caption_settings(self):
        if not self.project:
            return
        self._set_project_caption_form(self._caption_defaults | self._caption_path_values(), True)
        if self.save_project_caption_form():
            self.status.setText(f"Caption settings reset and saved for {self.project.root.name}.")

    def _caption_path_values(self):
        return {name: self.caption_form.controls[name].value() for name in CAPTION_PATH_FIELDS}

    def _shared_caption_path_values(self):
        return {name: field.value() for name, field in self.shared_caption_paths.items()}

    def _set_project_caption_form(self, values, enabled):
        for name in PROMPT_KEYS:
            self.caption_form.controls[name].blockSignals(True)
        self.caption_form.set_values(values)
        for name in PROMPT_KEYS:
            control = self.caption_form.controls[name]
            control.blockSignals(False)
        self.caption_form.setEnabled(enabled)
        self.reset_system_prompt.setEnabled(enabled)
        self.reset_user_prompt.setEnabled(enabled)
        self.reset_all_caption_settings_button.setEnabled(enabled)
        self.save_caption_settings_button.setEnabled(enabled)

    def _set_project_training_values(self, values):
        previous_model = self.model_field.value()
        output = self.training_controls["output_dir"]
        resume = self.training_controls["resume_from"]
        with (QSignalBlocker(self.model_field), QSignalBlocker(self.training_controls["mode"]),
              QSignalBlocker(output), QSignalBlocker(resume)):
            for form in self.training_forms:
                form.set_values(values)
        self._training_mode = self.training_controls["mode"].currentText()
        self._committed_resume = resume.value()
        self.refresh_checkpoint_steps()
        if self.model_field.value() != previous_model:
            self.stop_tokens()

    def set_project(self, project, *, created=False):
        if not self.save_caption() or not self.save_project_caption_form() or not self.save_project_training_form():
            return
        saved = load_project_caption_settings(project)
        saved_training, mode_profiles, branch_outputs = load_project_training_profiles(project)
        values = self._caption_defaults | self._shared_caption_path_values()
        legacy_project = self._legacy_caption_project
        if not created and legacy_project and Path(legacy_project).expanduser().resolve() == project.root:
            values.update({key: value for key, value in self._legacy_caption.items()
                           if key in self._caption_defaults and key not in PROMPT_KEYS and key not in CAPTION_PATH_FIELDS})
            values.update(migrate_legacy_prompts(self._legacy_caption))
        values.update(saved or {})
        self._legacy_caption = {}
        self._legacy_caption_project = ""
        training = self._training_defaults | self.shared_training_model.values()
        if saved_training is not None:
            training = saved_training.to_dict()
        elif (not created and self._legacy_training_project
              and Path(self._legacy_training_project).expanduser().resolve() == project.root):
            training.update({name: value for name, value in self._legacy_training.items() if name in training})
        self._legacy_training = {}
        self._legacy_training_project = ""
        self.project = project
        self._set_project_caption_form(values, True)
        self._training_mode_profiles = mode_profiles
        self._suggested_branch_outputs = branch_outputs
        self.refresh_saved_runs()
        self._set_project_training_values(training)
        self.purge_project_button.setEnabled(True)
        self.save_training_settings_button.setEnabled(True)
        self.reset_all_training_settings_button.setEnabled(True)
        self.entry_id = None
        self.settings["last_project"] = str(project.root)
        datasets = project.finalized_datasets(include_all=True)
        self.project_dataset_button.setEnabled(bool(datasets))
        if saved_training is None:
            if not self.training_controls["dataset_path"].value():
                first_bucket = next((path for path in datasets if (path / "manifest.json").is_file()), "")
                self._select_project_dataset(first_bucket)
            self._suggest_run()
        self.refresh_images()
        self.save_settings()

    @staticmethod
    def _inside_project(value, root):
        if not value:
            return False
        path = Path(os.path.abspath(os.path.expanduser(value)))
        return path == root or root in path.parents

    def _suggest_run(self):
        if not self.project:
            return
        field = self.training_controls["output_dir"]
        if field.value():
            return
        root = self.project.root / "runs"
        if root.is_symlink():
            return
        reserved = {Path(os.path.abspath(os.path.expanduser(profile["output_dir"])))
                    for profile in self._training_mode_profiles.values() if profile.get("output_dir")}
        number = 1
        while True:
            candidate = root / f"Run-{number}"
            if candidate not in reserved and not candidate.exists() and not candidate.is_symlink():
                field.set_value(candidate)
                return
            number += 1

    def _select_project_dataset(self, path):
        dataset = self.training_controls["dataset_path"]
        if dataset.value() == str(path):
            return
        dataset.set_value(path)
        self.training_controls["resume_from"].set_value("")
        self.clear_suggested_branch_output()
        output = self.training_controls["output_dir"]
        if self._inside_project(output.value(), self.project.root / "runs"):
            current = Path(output.value()).expanduser()
            if current.exists() or current.is_symlink():
                output.set_value("")

    def choose_project_dataset(self):
        if not self.project:
            return
        datasets = self.project.finalized_datasets(include_all=True)
        if not datasets:
            self.project_dataset_button.setEnabled(False)
            self.status.setText("Finalize images in this project to create training datasets.")
            return
        folder = self.project.root / "finalized"
        labels = []
        for path in datasets:
            label = " / ".join(path.relative_to(folder).parts)
            if path.parent == folder and not (path / "manifest.json").is_file():
                label += " / All aspect ratios"
            labels.append(label)
        current = self.training_controls["dataset_path"].value()
        index = next((i for i, path in enumerate(datasets) if str(path) == current), 0)
        selected, accepted = QInputDialog.getItem(self, "Choose project dataset", "Finalized dataset", labels, index, False)
        if accepted:
            path = datasets[labels.index(selected)]
            self._select_project_dataset(path)
            self._suggest_run()
            if self.save_settings():
                self.status.setText(f"Selected {selected} for training.")

    def new_project(self):
        parent = Path(self.projects_location.value() or projects_dir()).expanduser().resolve()
        default_name = "New project"
        number = 2
        while (parent / default_name).exists() or (parent / default_name).is_symlink():
            default_name = f"New project {number}"
            number += 1
        name, accepted = QInputDialog.getText(self, "New project", f"Project name:\nCreate inside {parent}", text=default_name)
        if not accepted:
            return
        try:
            destination = parent / validate_project_name(name)
            if destination.exists() or destination.is_symlink():
                raise ValueError("A project or folder already uses that name. Choose another name or open the existing project.")
            self.set_project(DatasetProject.create(destination), created=True)
        except Exception as exc:
            self.error(str(exc))

    def purge_project(self):
        if not self.project or self.task or self.process:
            return
        if not self.save_project_caption_form():
            return
        self._pending_purge = None
        self.run_task("Checking project contents…", lambda task: preview_project_purge(self.project.root),
                      lambda preview: setattr(self, "_pending_purge", preview), cancellable=False)
        if self.task:
            self.task.finished.connect(self._confirm_project_purge)

    def _confirm_project_purge(self):
        preview = getattr(self, "_pending_purge", None)
        self._pending_purge = None
        if preview is None:
            return
        prompt = (f"Permanently delete {preview.path} and everything inside?\n"
                  f"{preview.files} files · {preview.directories} folders · {preview.bytes / 2**30:.2f} GiB\n\n"
                  "This cannot be undone. Finalized datasets, runs, checkpoints, and logs inside this folder will be deleted. "
                  "Original images, shared models, and older outputs outside the folder are kept.\n\n"
                  f"Type the project name exactly to confirm: {preview.path.name}")
        text, accepted = QInputDialog.getText(self, "Fully purge project", prompt)
        if not accepted:
            self.status.setText("Project purge cancelled.")
            return
        if text != preview.path.name:
            self.error("The name did not match. The project was not deleted.")
            return
        def work(task):
            try:
                purge_project(preview)
            except Exception as exc:
                raise RuntimeError(f"Purge failed and removal may be partial. Inspect the project folder: {exc}") from exc
        def done(_):
            old = preview.path
            for field in (self.export_model, self.export_output):
                if self._inside_project(field.value(), old):
                    field.set_value("")
            for name, field in self.shared_caption_paths.items():
                if self._inside_project(field.value(), old):
                    field.set_value(self._caption_defaults[name])
            self.project = None
            if self._inside_project(self.shared_training_model.controls["model_path"].value(), old):
                self.shared_training_model.set_values({name: self._training_defaults[name]
                                                       for name in TRAINING_MODEL_FIELDS})
            self.prompt_timer.stop()
            self._set_project_caption_form(self._caption_defaults | self._shared_caption_path_values(), False)
            self._training_mode_profiles.clear()
            self._suggested_branch_outputs.clear()
            self.refresh_saved_runs()
            self._set_project_training_values(self._training_defaults | self.shared_training_model.values())
            self.entry_id = None
            self.purge_project_button.setEnabled(False)
            self.project_dataset_button.setEnabled(False)
            self.save_training_settings_button.setEnabled(False)
            self.reset_all_training_settings_button.setEnabled(False)
            self.refresh_images()
            self.project_label.setText("No project open · Original images are never modified")
            self.logs.clear()
            self.training_progress.setValue(0)
            self.save_settings()
            self.status.setText(f"Purged project {old.name}.")
        self.run_task("Permanently removing the confirmed project…", work, done, cancellable=False)

    def open_project(self):
        path = file_dialogs.get_existing_directory(self, "Open dataset project", self.projects_location.value(), options=FILE_DIALOG_OPTIONS)
        if path:
            try:
                self.set_project(DatasetProject.open(path))
            except Exception as exc:
                self.error(str(exc))

    def remove_selected_images(self):
        if not self.project or self.task or self.process:
            return
        selected = {item.data(Qt.ItemDataRole.UserRole) for item in self.images.selectedItems()}
        if not selected:
            self.status.setText("Select one or more images to remove.")
            return
        count = len(selected)
        answer = QMessageBox.question(
            self, "Remove draft images",
            f"Remove {count} selected image{'s' if count != 1 else ''} and their captions from this draft?\n\n"
            "The remaining draft images and captions will be renumbered. Original files and finalized datasets are kept.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if not self.save_caption():
            return
        project = self.project
        current_row = max(0, self.images.currentRow())
        next_row = sum(entry.id not in selected for entry in project.entries[:current_row])

        def done(removed):
            # A reused ID can now belong to a different image.
            self.entry_id = None
            next_id = min(next_row + 1, len(project.entries))
            self.refresh_images(next_id or None)
            self.status.setText(f"Removed {removed} draft image{'s' if removed != 1 else ''}. Draft numbering updated.")

        self.run_task("Removing images and updating draft numbering…", lambda task: remove_images(project, selected),
                      done, cancellable=False)

    def refresh_images(self, select_id=None):
        has_images = bool(self.project and self.project.entries)
        self.bulk_caption_button.setEnabled(has_images)
        self.trim_caption_button.setEnabled(has_images)
        self.thumbnail_timer.stop()
        self._thumbnail_ids.clear()
        self.images.blockSignals(True)
        self.images.clear()
        if self.project:
            for entry in self.project.entries:
                item = QListWidgetItem(self._image_item_text(entry))
                item.setData(Qt.ItemDataRole.UserRole, entry.id)
                # Reserve stable rows while thumbnails are loaded on demand.
                # File-backed QIcons can decode/cache every full-size image at
                # once, making a large dataset slow to open and memory hungry.
                item.setSizeHint(QSize(200, 86))
                self.images.addItem(item)
            self.project_label.setText(f"{self.project.root} · {len(self.project.entries)} images")
        self.images.blockSignals(False)
        if self.images.count():
            index = next((i for i, e in enumerate(self.project.entries) if e.id == select_id), 0)
            self.images.setCurrentRow(index)
        else:
            self.entry_id = None
            self.loaded_pixmap = QPixmap()
            self.preview.clear()
            self.preview.setText("Import images to see the exact training crop")
            self.image_info.setText("RGB PNG · Metadata removed · 64-pixel bucket grid")
            self.caption.blockSignals(True)
            self.caption.clear()
            self.caption.blockSignals(False)
            self.token_generation += 1
            self.tokens_label.setText("CLIP-L — tokens · CLIP-G — tokens · — windows")
        self._queue_thumbnail_load()

    def _queue_thumbnail_load(self, *_):
        if hasattr(self, "thumbnail_timer"):
            self.thumbnail_timer.start(0)

    def _load_image_thumbnail(self, path):
        reader = QImageReader(str(path))
        size = reader.size()
        if size.isValid():
            reader.setScaledSize(size.scaled(self.images.iconSize(), Qt.AspectRatioMode.KeepAspectRatio))
        return QIcon(QPixmap.fromImage(reader.read()))

    def _load_visible_thumbnails(self):
        if self.task or not self.project or not self.images.isVisible() or not self.images.count():
            return
        viewport = self.images.viewport()
        if viewport.height() <= 0:
            return
        first = self.images.indexAt(QPoint(1, 1)).row()
        last = self.images.indexAt(QPoint(1, viewport.height() - 1)).row()
        first = max(0, first - 2)
        last = min(self.images.count() - 1, (last if last >= 0 else self.images.count() - 1) + 2)
        loaded = 0
        for row in range(first, last + 1):
            item = self.images.item(row)
            entry_id = item.data(Qt.ItemDataRole.UserRole)
            if entry_id in self._thumbnail_ids:
                continue
            item.setIcon(self._load_image_thumbnail(self.project.image_path(entry_id)))
            self._thumbnail_ids.add(entry_id)
            loaded += 1
            if loaded == 4:
                # Bound synchronous decoding so scrolling/caption editing can
                # proceed between small batches, even on slower local disks.
                self.thumbnail_timer.start(0)
                break

    def select_image(self, row):
        self.save_caption()
        if not self.project or row < 0:
            return
        entry = self.project.entries[row]
        self.entry_id = entry.id
        self.token_generation += 1
        # Draft renumbering can reuse a filename for different pixels.
        reader = QImageReader(str(self.project.image_path(entry.id)))
        self.loaded_pixmap = QPixmap.fromImage(reader.read())
        self.update_preview()
        self._show_image_info(entry)
        self.caption.blockSignals(True)
        self.caption.setPlainText(self.project.caption_text(entry.id))
        self.caption.blockSignals(False)
        self.request_token_count()

    def reload_captions(self):
        # Clear the editor identity before reloading saved captions. The list
        # keeps its selection and scroll position because its rows are reused.
        for row, entry in enumerate(self.project.entries):
            self.images.item(row).setText(self._image_item_text(entry))
        self.entry_id = None
        self.select_image(self.images.currentRow())

    @staticmethod
    def _image_item_text(entry):
        flag = "⚠ " if entry.caption_incomplete_reason else ""
        return f"{flag}{entry.image}\n{entry.width} × {entry.height}"

    def _show_image_info(self, entry):
        message = f"{entry.image} · {entry.width} × {entry.height} · {entry.fit} · RGB PNG · Metadata removed"
        if entry.caption_incomplete_reason:
            message += " · Incomplete: edit or re-caption before finalizing"
        self.image_info.setText(message)

    def update_preview(self):
        if not self.loaded_pixmap.isNull():
            self.preview.setPixmap(self.loaded_pixmap.scaled(self.preview.size() - QSize(18, 18), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "preview"):
            self.update_preview()
        self._queue_thumbnail_load()

    def showEvent(self, event):
        super().showEvent(event)
        self._queue_thumbnail_load()

    def caption_changed(self):
        self.caption_timer.start()
        self.token_generation += 1
        self.tokens_label.setText("CLIP-L … tokens · CLIP-G … tokens · … windows")

    def save_caption(self):
        if self.project is not None and self.entry_id is not None:
            self.caption_timer.stop()
            try:
                self.project.save_caption(self.entry_id, self.caption.toPlainText())
                entry = self.project._entry(self.entry_id)
                for row in range(self.images.count()):
                    item = self.images.item(row)
                    if item.data(Qt.ItemDataRole.UserRole) == entry.id:
                        item.setText(self._image_item_text(entry))
                        break
                self._show_image_info(entry)
                self.request_token_count()
            except Exception as exc:
                self.error(str(exc))
                return False
        return True

    def model_changed(self, *_):
        self.stop_tokens()
        self.request_token_count()

    def stop_tokens(self):
        process = self.token_process
        self.token_process = None
        if process:
            process.kill()
            process.waitForFinished(1000)
            process.deleteLater()
        self.token_buffer = ""
        self.token_ready = False
        self.token_queue = []

    def process_environment(self):
        environment = QProcessEnvironment()
        for name, value in offline_environment().items():
            environment.insert(name, value)
        return environment

    def request_token_count(self):
        if not hasattr(self, "runtime") or not self.model_field.value():
            self.tokens_label.setText("Select a base model in Training for exact counts")
            return
        if not self.token_process:
            process = QProcess(self)
            self.token_process = process
            process.setProcessEnvironment(self.process_environment())
            process.readyReadStandardOutput.connect(lambda: self.read_tokens(process))
            process.errorOccurred.connect(lambda _: self.token_failed(process, "Tokenizer runtime unavailable; see Models & runtime"))
            process.finished.connect(lambda *_: self.token_failed(process, "Tokenizer stopped; check runtime and base model"))
            process.start(self.runtime.value(), ["-m", "sdxl_trainer.runtime", "tokens", "--model", self.model_field.value()])
        elif self.token_ready:
            self.token_queue.append(self.token_generation)
            self.token_process.write((json.dumps({"text": self.caption.toPlainText()}) + "\n").encode())

    def token_failed(self, process, message):
        if process is self.token_process:
            self.tokens_label.setText(message)
            self.token_process = None
            self.token_ready = False
            self.token_queue = []
            self.token_buffer = ""
            process.deleteLater()

    def read_tokens(self, process=None):
        process = self.token_process if process is None else process
        if process is None or process is not self.token_process:
            return
        self.token_buffer += bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self.token_buffer:
            line, self.token_buffer = self.token_buffer.split("\n", 1)
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if "error" in value:
                self.tokens_label.setText(value["error"][:120])
            elif value.get("ready"):
                self.token_ready = True
                self.request_token_count()
            else:
                generation = self.token_queue.pop(0) if self.token_queue else -1
                if generation == self.token_generation:
                    a, b = value["clip_l"], value["clip_g"]
                    windows = max(1, (max(a, b) - 2 + 74) // 75)
                    self.tokens_label.setText(f"CLIP-L {a} tokens · CLIP-G {b} tokens · {windows} × 77-token windows")
                    self.tokens_label.setStyleSheet("color:#9ebeff")

    def busy(self, active, text="Ready."):
        self.content.setEnabled(not active)
        self.cancel_button.setEnabled(active)
        self.activity.setRange(0, 0 if active else 1)
        self.activity.setValue(0)
        self.status.setText(text)

    def run_task(self, text, function, done=None, cancellable=True, *, on_finished=None):
        if self.task or self.process:
            return
        self.save_caption()
        self.busy(True, text)
        self.cancel_button.setEnabled(cancellable)
        task = Task(function, self)
        self.task = task
        task.progress.connect(self.status.setText)
        task.failed.connect(self.error)
        if done:
            task.result.connect(done)
        if on_finished:
            task.finished.connect(on_finished)
        task.finished.connect(self.task_finished)
        task.start()

    def task_finished(self):
        task = self.task
        self.task = None
        self.busy(False, self.status.text())
        self._queue_thumbnail_load()
        if task:
            task.deleteLater()

    def import_images(self, folder):
        if not self.project:
            self.error("Create or open a dataset project first.")
            return
        if folder:
            path = file_dialogs.get_existing_directory(self, "Import original image folder", options=FILE_DIALOG_OPTIONS)
            paths = [path] if path else []
        else:
            paths, _ = file_dialogs.get_open_file_names(self, "Import original images", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)", options=FILE_DIALOG_OPTIONS)
        if not paths:
            return
        project = self.project
        fit, background, recursive = self.fit.currentData(), self.background.text(), self.recursive.isChecked()
        allow_upscale = self.allow_upscale.isChecked()

        def work(task):
            skipped = []
            imported = project.import_images(paths, fit, background, recursive,
                                             allow_upscale=allow_upscale, on_skip=skipped.append)
            return imported, len(skipped)

        def done(result):
            imported, skipped = result
            if imported:
                self.refresh_images(imported[-1].id)
            count = len(imported)
            message = f"Imported {count} image{'s' if count != 1 else ''}."
            if skipped:
                message += f" Skipped {skipped} requiring upscaling."
            self.status.setText(message)

        self.run_task("Preparing clean PNG images…", work, done, cancellable=False)

    def bulk_caption(self, *, trim=False):
        if self.task or self.process or not self.project or not self.project.entries:
            return
        if not self.save_caption():
            return
        project = self.project
        selected = tuple(item.data(Qt.ItemDataRole.UserRole) for item in self.images.selectedItems())
        model, python = self.model_field.value(), self.runtime.value()
        dialog = QDialog(self)
        dialog.setWindowTitle("Bulk caption")
        layout = QVBoxLayout(dialog)
        fields = QFormLayout()
        mode = QComboBox(dialog)
        mode.setObjectName("captionOperation")
        mode.addItem("Set every caption", "set")
        mode.addItem("Prepend to captions", "prepend")
        mode.addItem("Fill empty captions", "empty")
        mode.addItem("Append to captions", "append")
        mode.addItem("Trim to 77 tokens", "trim")
        fields.addRow("Operation", mode)
        scope = QComboBox(dialog)
        scope.setObjectName("captionScope")
        scope.addItem(f"All draft images ({len(project.entries)})", "all")
        scope.addItem(f"Selected images ({len(selected)})", "selected")
        if len(selected) > 1 or (trim and selected):
            scope.setCurrentIndex(scope.findData("selected"))
        if trim:
            mode.setCurrentIndex(mode.findData("trim"))
        fields.addRow("Apply to", scope)
        layout.addLayout(fields)
        detail = note("")
        layout.addWidget(detail)
        editor = CaptionEditor(self.checker)
        editor.setPlaceholderText("Enter a tag or caption…")
        layout.addWidget(editor)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, dialog)
        apply_button = buttons.addButton("Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        apply_button.setObjectName("primary")
        descriptions = {
            "set": "Replace the captions for these images.",
            "prepend": "Add the text before existing captions, separated by a comma. Fill empty captions with the text.",
            "empty": "Fill captions that are empty or contain only whitespace.",
            "append": "Add the text after existing captions, separated by a comma. Fill empty captions with the text.",
            "trim": "Shorten overlong captions to fit both SDXL tokenizers within 77 tokens, including start and end tokens.",
        }
        empty_counts = {}
        target_count = 0

        def update_apply():
            ready = bool(model) if mode.currentData() == "trim" else bool(editor.toPlainText().strip())
            apply_button.setEnabled(target_count > 0 and ready)

        def update_operation():
            nonlocal target_count
            choice = mode.currentData()
            target_scope = scope.currentData()
            selected_only = target_scope == "selected"
            target_count = len(selected) if selected_only else len(project.entries)
            description = descriptions[choice]
            if choice == "empty" and target_count:
                if target_scope not in empty_counts:
                    try:
                        count = project.count_empty_captions(image_ids=selected if selected_only else None)
                        empty_counts[target_scope] = count, "All images in this scope already have captions."
                    except (OSError, ValueError) as exc:
                        empty_counts[target_scope] = 0, f"Could not count empty captions. {exc}"
                target_count, message = empty_counts[target_scope]
                if not target_count:
                    description = message
            elif choice == "trim" and not model:
                description = "Choose a local Diffusers base model in Training to load both SDXL tokenizers."
            detail.setText(description)
            editor.setVisible(choice != "trim")
            apply_button.setText(f"Apply to {target_count} image{'s' if target_count != 1 else ''}")
            update_apply()

        mode.currentIndexChanged.connect(update_operation)
        scope.currentIndexChanged.connect(update_operation)
        editor.textChanged.connect(update_apply)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        update_operation()
        dialog.resize(530, dialog.sizeHint().height())
        if trim:
            scope.setFocus()
        else:
            editor.setFocus()
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        text, choice = editor.toPlainText(), mode.currentData()
        image_ids = selected if scope.currentData() == "selected" else None
        dialog.deleteLater()
        if not accepted:
            return

        def done(changed):
            self.reload_captions()
            verb = "Trimmed" if choice == "trim" else "Updated"
            self.status.setText(f"{verb} {changed} caption{'s' if changed != 1 else ''}.")

        def work(task):
            if choice == "trim":
                with RemoteTokenCounter(python, model) as counter:
                    return project.truncate_captions(counter, image_ids=image_ids)
            return project.apply_bulk_caption(text, prepend=choice == "prepend", only_empty=choice == "empty",
                                              append=choice == "append", image_ids=image_ids)

        action = "Fitting captions to 77 tokens…" if choice == "trim" else "Updating draft captions…"
        self.run_task(action, work, done, cancellable=False)

    def auto_caption(self, all_empty):
        if self.task or self.process:
            return
        project = self.project
        if not project or not project.entries:
            self.error("Import an image first.")
            return
        selected_ids = {item.data(Qt.ItemDataRole.UserRole) for item in self.images.selectedItems()}
        if not all_empty and not selected_ids:
            self.status.setText("Select one or more images to caption.")
            return
        if not self.save_caption() or not self.save_project_caption_form():
            return
        if all_empty:
            entries = [entry for entry in project.entries if not project.caption_text(entry.id).strip()]
        else:
            entries = [entry for entry in project.entries if entry.id in selected_ids]
        if not entries:
            self.status.setText("Every image already has a caption. Use Caption selected to replace existing captions.")
            return
        config = CaptionConfig(**self.caption_form.values())
        errors = config.validate()
        if errors:
            self.error("\n".join(errors))
            return
        model, python = self.model_field.value(), self.runtime.value()
        if config.trim_generated_captions and not model:
            self.error("Choose a local Diffusers base model in Training to trim generated captions with both SDXL tokenizers.")
            return
        completed = False

        def work(task):
            with ExitStack() as resources:
                counter = None
                if config.trim_generated_captions:
                    counter = resources.enter_context(RemoteTokenCounter(python, model))
                    task.progress.emit("Loading local caption model…")
                server = CaptionServer(config)
                resources.callback(server.stop)
                server.start(task.cancel)
                for i, entry in enumerate(entries):
                    task.progress.emit(f"Captioning {entry.image} · {i + 1}/{len(entries)}")
                    text = server.caption(project.image_path(entry.id), task.cancel)
                    reason = ""
                    if getattr(text, "truncated", False):
                        reason = " ".join(str(getattr(text, "reason", "") or "Generation did not finish.").split())[:180]
                    if counter is not None:
                        text = counter.truncate(text)
                    project.save_generated_caption(entry.id, text, reason)
                return len(entries)

        def complete(count):
            nonlocal completed
            completed = True
            pending = project.incomplete_captions()
            label = "image" if count == 1 else "images"
            if pending:
                self.status.setText(f"Captioned {count} {label}; {len(pending)} could not finish automatically. Review or continue those captions.")
                self._warn_incomplete_captions()
            else:
                self.status.setText(f"Captioned {count} {label}. Review spelling and both token counts.")

        def refresh_after_captioning():
            # A cancelled/failed batch may still have saved earlier images. Reload
            # those authoritative captions before any later editor save can occur.
            self.reload_captions()
            if not completed and project.incomplete_captions():
                self._warn_incomplete_captions()

        action = "Loading both local SDXL tokenizers…" if config.trim_generated_captions else "Loading local caption model…"
        self.run_task(action, work, complete, on_finished=refresh_after_captioning)

    def _warn_incomplete_captions(self):
        pending = self.project.incomplete_captions()
        if pending:
            details = "\n".join(f"{entry.image}: {entry.caption_incomplete_reason}" for entry in pending)
            QMessageBox.warning(self, "Incomplete captions", f"The available text was saved for {len(pending)} incomplete caption(s). Edit or re-caption each one before finalizing:\n\n{details}")

    def finalize(self):
        if not self.project or not self.project.entries:
            self.error("Import and caption images before finalizing.")
            return
        if not self.save_caption():
            return
        if self.project.incomplete_captions():
            self._warn_incomplete_captions()
            return
        if not self.model_field.value():
            self.error("Choose the local Diffusers base model in Training so both tokenizer counts can be verified.")
            return
        model, python = self.model_field.value(), self.runtime.value()
        def work(task):
            with RemoteTokenCounter(python, model) as counter:
                return self.project.finalize_buckets_in_project(counter)
        def done(paths):
            self.project_dataset_button.setEnabled(True)
            self._select_project_dataset(paths[0])
            self._suggest_run()
            count = len(paths)
            label = "dataset" if count == 1 else "datasets"
            if self.save_settings():
                self.status.setText(f"Finalized {count} bucket {label} in {paths[0].parent.name}. Selected {paths[0].name} for training.")
        self.run_task("Verifying images and captions and exporting separate buckets…", work, done, cancellable=False)

    def training_config(self):
        values = {key: value for form in self.training_forms for key, value in form.values().items()}
        return TrainingConfig.from_dict(values)

    def track_resume_value(self, _):
        field = self.training_controls["resume_from"]
        if not field.edit.isModified():
            self._committed_resume = field.value()
            self.refresh_checkpoint_steps()

    def finish_resume_edit(self):
        field = self.training_controls["resume_from"]
        if field.edit.isModified():
            field.edit.setModified(False)
            previous = self._committed_resume
            self._committed_resume = field.value()
            self.refresh_checkpoint_steps()
            self.load_resume_selection(field.value(), previous)

    def load_resume_selection(self, checkpoint, previous_resume):
        if not checkpoint:
            self.refresh_checkpoint_steps()
            return
        run = checkpoint_run_folder(checkpoint)
        resume = self.training_controls["resume_from"]
        try:
            if run is None:
                run = Path(checkpoint).expanduser()
                checkpoints = run / "checkpoints"
                if not (checkpoints.exists() or checkpoints.is_symlink() or (run / "training-config.json").is_file()):
                    return
                checkpoints = run_checkpoints(run)
                if not checkpoints:
                    resume.set_value(previous_resume)
                    self.status.setText("No completed checkpoints found in the selected run.")
                    return
                checkpoint = str(checkpoints[0][1])
                resume.set_value(checkpoint)
            elif not has_checkpoint_metadata(Path(checkpoint).expanduser()):
                return
        except (OSError, RuntimeError, ValueError) as exc:
            resume.set_value(previous_resume)
            self.status.setText(f"Could not read the selected checkpoint or run. {exc}")
            return
        try:
            settings_path = run.expanduser() / "training-config.json"
            if settings_path.is_symlink() or not settings_path.is_file():
                raise ValueError("The run has no regular training settings file.")
            config = TrainingConfig.load(settings_path)
            config.resume_from = checkpoint
            config.output_dir = str(run)
            errors = config.validate()
            if errors:
                raise ValueError(errors[0])
        except (OSError, RuntimeError, ValueError, OverflowError) as exc:
            self._suggested_branch_outputs.pop(self._training_mode, None)
            self.training_controls["output_dir"].set_value(run)
            self.refresh_checkpoint_steps()
            self.status.setText(f"Run output selected. Could not load training-config.json. {exc}")
            return

        self._suggested_branch_outputs.pop(config.mode, None)
        self._select_training_mode(config.mode, config.to_dict(), previous_resume=previous_resume)
        self.request_token_count()
        self.status.setText("Saved training settings loaded. You can change Run output before starting.")

    def refresh_checkpoint_steps(self):
        resume = self.training_controls["resume_from"].value()
        run = checkpoint_run_folder(resume)
        entries = []
        selected = ""
        if run is not None:
            try:
                selected = str(Path(resume).expanduser())
                entries = run_checkpoints(run.expanduser())
            except (OSError, RuntimeError, ValueError):
                pass
        picker = self.checkpoint_step
        with QSignalBlocker(picker):
            picker.clear()
            for row, (step, checkpoint) in enumerate(entries):
                label = f"Step {step:,}" if row else f"Step {step:,} (latest)"
                picker.addItem(label, str(checkpoint))
            index = picker.findData(selected)
            picker.setCurrentIndex(index)
        self._checkpoint_form_layout.setRowVisible(picker, index >= 0)
        self.sync_saved_run(run if index >= 0 else None)

    def refresh_saved_runs(self):
        runs = []
        if self.project:
            try:
                runs = saved_project_runs(self.project)
            except OSError:
                pass
        with QSignalBlocker(self.saved_runs):
            self.saved_runs.clear()
            self.saved_runs.addItem("Choose a project run…" if runs else "No saved runs with checkpoints")
            methods = {"lora": "LoRA", "full": "Full Fine-Tune"}
            for run in runs:
                method = methods.get(run.mode, "Unknown method")
                label = f"{run.path.name} · {method} · Step {run.latest_step:,}"
                self.saved_runs.addItem(label, str(run.path))
        self.saved_runs.setEnabled(bool(runs))
        self._checkpoint_form_layout.setRowVisible(self.saved_runs, self.project is not None)
        self.sync_saved_run(checkpoint_run_folder(self.training_controls["resume_from"].value()))

    def sync_saved_run(self, run):
        index = -1
        if run is not None:
            try:
                index = self.saved_runs.findData(os.path.abspath(run.expanduser()))
            except (OSError, RuntimeError, ValueError):
                pass
        with QSignalBlocker(self.saved_runs):
            self.saved_runs.setCurrentIndex(max(index, 0))

    def select_saved_run(self, index):
        run = self.saved_runs.itemData(index)
        if run is not None:
            self.load_resume_selection(run, self._committed_resume)
        self.sync_saved_run(checkpoint_run_folder(self.training_controls["resume_from"].value()))
        if run is not None and self.saved_runs.currentData() != run:
            self.refresh_saved_runs()

    def clear_suggested_branch_output(self, *_):
        self._suggested_branch_outputs.pop(self._training_mode, None)

    def select_checkpoint_step(self, index):
        if index < 0:
            return
        picker = self.checkpoint_step
        checkpoint = Path(picker.itemData(index))
        run = checkpoint.parent.parent
        output = self.training_controls["output_dir"]
        suggested = self._suggested_branch_outputs.get(self._training_mode)
        try:
            if not has_checkpoint_metadata(checkpoint):
                self.refresh_checkpoint_steps()
                self.status.setText("The selected checkpoint is no longer available.")
                return
            if index == 0 and output.value() == suggested:
                output.set_value(run)
            elif index > 0 and output.value() and Path(output.value()).expanduser().resolve() == run.resolve():
                branch = self._unused_branch_output(run)
                output.set_value(branch)
                self._suggested_branch_outputs[self._training_mode] = str(branch)
        except (OSError, RuntimeError, ValueError) as exc:
            self.refresh_checkpoint_steps()
            self.status.setText(f"Could not select the checkpoint. {exc}")
            return
        self.training_controls["resume_from"].set_value(checkpoint)
        if index > 0 and output.value() == self._suggested_branch_outputs.get(self._training_mode):
            self.status.setText("Older checkpoint selected. A new Run output was suggested.")
        else:
            self.status.setText(f"{checkpoint.name} selected.")

    def _unused_branch_output(self, run):
        base = f"{run.name}-branch"
        if len(os.fsencode(base)) > 240:
            base = "Run-branch"
        reserved = {Path(os.path.abspath(os.path.expanduser(profile["output_dir"])))
                    for profile in self._training_mode_profiles.values() if profile.get("output_dir")}
        candidate = run.with_name(base)
        number = 2
        while candidate.absolute() in reserved or candidate.exists() or candidate.is_symlink():
            candidate = run.with_name(f"{base}-{number}")
            number += 1
        return candidate

    def training_mode_changed(self, mode):
        profile = self._training_mode_profiles.get(mode, {})
        values = self._training_preset(mode) | profile | self._shared_training_values(self.training_config().to_dict())
        self._select_training_mode(mode, values)
        if "output_dir" not in profile:
            self._suggest_run()
        label = "LoRA" if mode == "lora" else "Full fine-tune"
        self.status.setText(f"{label} training settings selected.")

    def _training_preset(self, mode):
        return self._training_defaults | fast_training_settings(mode) | {
            "mode": mode, "learning_rate": DEFAULT_UNET_LEARNING_RATES[mode],
        }

    @staticmethod
    def _shared_training_values(current):
        return {name: current[name] for name in SHARED_TRAINING_FIELDS}

    def _select_training_mode(self, mode, values, *, previous_resume=None):
        if mode != self._training_mode:
            current = self.training_config().to_dict()
            if previous_resume is not None:
                current["resume_from"] = previous_resume
            self._training_mode_profiles[self._training_mode] = {
                name: value for name, value in current.items()
                if name != "mode" and name not in SHARED_TRAINING_FIELDS
            }
        self._training_mode_profiles.pop(mode, None)
        self._set_project_training_values(values)

    def apply_fast_training_defaults(self):
        mode = self.training_controls["mode"].currentText()
        values = fast_training_settings(mode)
        for form in self.training_forms:
            form.set_values(values)
        if self.save_project_training_form():
            profile = "Fast full fine-tune settings applied with paged 8-bit AdamW." if mode == "full" else (
                "Fast LoRA settings applied with fused GPU AdamW."
            )
            self.status.setText(f"{profile} Learning rates and steps kept.")

    def load_recipe(self):
        path, _ = file_dialogs.get_open_file_name(self, "Load training recipe", "", "JSON (*.json)", options=FILE_DIALOG_OPTIONS)
        if path:
            try:
                config = TrainingConfig.load(path)
                self._suggested_branch_outputs.pop(config.mode, None)
                self._select_training_mode(config.mode, config.to_dict())
                self.request_token_count()
            except Exception as exc:
                self.error(str(exc))

    def save_recipe(self):
        path, _ = file_dialogs.get_save_file_name(self, "Save training recipe", "recipe.json", "JSON (*.json)", options=FILE_DIALOG_OPTIONS)
        if path:
            try:
                self.training_config().save(path)
            except Exception as exc:
                self.error(str(exc))

    def check_training(self):
        config = self.training_config()
        self.run_task("Checking dataset integrity and training configuration…", lambda task: preflight(config),
                      lambda result: self.logs.setPlainText("\n".join([*("ERROR: " + e for e in result.errors), *result.warnings, f"Disk estimate: {result.estimated_disk_gib:.1f} GiB", "Configuration valid. Runtime/GPU are checked when training starts." if result.ok else "Resolve the errors above."])), cancellable=False)

    def train(self):
        config = self.training_config()
        errors = config.validate()
        if errors:
            self.error("\n".join(errors))
            return
        if not self.save_settings():
            return
        path = config_dir() / "launches" / f"{uuid.uuid4().hex}.json"
        if self.project:
            logs = self.project.root / "logs"
            launch_dir = logs / "launches"
            if logs.is_symlink() or launch_dir.is_symlink():
                self.error("Project logs folders must not be symbolic links.")
                return
            path = launch_dir / f"{uuid.uuid4().hex}.json"
        atomic_json(path, config.to_dict())
        self.logs.clear()
        self._completed_training = False
        self.gpu_status.setText("GPU: starting monitor…" if config.monitor_gpu else "GPU monitoring disabled.")
        self.system_memory_status.setText("System memory: starting monitor…" if config.monitor_gpu else "System memory monitoring disabled.")
        self.phase_status.setText("Step timings will appear after an optimizer update." if config.profile_phases else "Phase timings disabled.")
        self.training_progress.setRange(0, config.max_steps)
        self.training_progress.setValue(0)
        self.run_process(["-m", "sdxl_trainer.training.worker", "--config", str(path)], "Training: validating local runtime…", self.logs)

    def run_process(self, arguments, description, output):
        if self.task or self.process:
            return
        self.busy(True, description)
        self.process_output = output
        self.process_buffer = ""
        process = QProcess(self)
        self.process = process
        process.setProcessEnvironment(self.process_environment())
        process.readyReadStandardOutput.connect(self.read_process)
        process.readyReadStandardError.connect(lambda: output.appendPlainText(bytes(process.readAllStandardError()).decode("utf-8", errors="replace")))
        process.errorOccurred.connect(self.process_error)
        process.finished.connect(self.process_finished)
        process.start(self.runtime.value(), arguments)

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.process_output.appendPlainText("Could not run selected Python executable. Check Models & runtime.")
            self.process_finished(-1)

    def read_process(self):
        if self.process is None:
            return
        self.process_buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self.process_buffer:
            line, self.process_buffer = self.process_buffer.split("\n", 1)
            try:
                event = json.loads(line)
            except ValueError:
                self.process_output.appendPlainText(line)
                continue
            if isinstance(event, dict) and event.get("type") == "gpu":
                if event.get("available"):
                    used = event.get("memory_used_mib")
                    total = event.get("memory_total_mib")
                    utilization = event.get("utilization_percent")
                    memory = f"{used / 1024:.1f} / {total / 1024:.1f} GiB" if isinstance(used, (int, float)) and isinstance(total, (int, float)) else "—"
                    percent = f"{utilization:.0f}%" if isinstance(utilization, (int, float)) else "—"
                    self.gpu_status.setText(f"GPU: {event.get('device_name') or 'NVIDIA'} · utilization {percent} · VRAM {memory}")
                else:
                    self.gpu_status.setText("GPU monitoring unavailable: " + str(event.get("reason", "NVIDIA metrics unavailable"))[:120])
                system = event.get("system_memory")
                if isinstance(system, dict):
                    def gib(value):
                        return f"{value / 1024:.1f} GiB" if isinstance(value, (int, float)) else "—"
                    swap_used = system.get("swap_used_mib")
                    swap_total = system.get("swap_total_mib")
                    swap = f"{swap_used / 1024:.1f} / {swap_total / 1024:.1f} GiB" if isinstance(swap_used, (int, float)) and isinstance(swap_total, (int, float)) else "—"
                    self.system_memory_status.setText(f"Available RAM: {gib(system.get('mem_available_mib'))} · system swap: {swap} · worker swap: {gib(system.get('process_swap_mib'))}")
                continue
            self.process_output.appendPlainText(json.dumps(event, ensure_ascii=False))
            if isinstance(event, dict):
                if event.get("type") == "telemetry":
                    self.show_step_metrics(event)
                    continue
                if event.get("type") == "progress":
                    self.training_progress.setValue(int(event.get("step", 0)))
                    self.status.setText(f"Step {event.get('step')} · loss {event.get('loss', '—')}")
                    metrics = event.get("metrics")
                    if isinstance(metrics, dict):
                        self.show_step_metrics(metrics)
                elif event.get("type") == "complete":
                    path = event.get("output_dir") or event.get("path")
                    if path and (Path(path) / "model_index.json").is_file():
                        self.export_model.set_value(path)
                        self._completed_training = True
                elif event.get("message"):
                    self.status.setText(str(event["message"])[:180])

    def show_step_metrics(self, metrics):
        phases = metrics.get("phase_wall_seconds") or {}
        parts = [f"{name.replace('_', ' ')} {value:.1f}s" for name, value in phases.items() if isinstance(value, (int, float))]
        total = metrics.get("step_seconds")
        if isinstance(total, (int, float)):
            parts.insert(0, f"step {total:.1f}s")
        if parts:
            self.phase_status.setText("Timings (wall time; GPU work may overlap): " + " · ".join(parts))

    def process_finished(self, code, *_):
        process = self.process
        self.process = None
        if code == 0 and self._completed_training and self.project:
            output = self.training_controls["output_dir"]
            if self._inside_project(output.value(), self.project.root / "runs"):
                output.set_value("")
                self._suggest_run()
                self.save_settings()
        self._completed_training = False
        self.refresh_saved_runs()
        self.refresh_checkpoint_steps()
        self.busy(False, "Task completed." if code == 0 else f"Task exited with code {code}. See the log for details.")
        if process:
            process.deleteLater()

    def export_checkpoint(self):
        source, target = self.export_model.value(), self.export_output.value()
        if not source or not target:
            self.error("Choose the trained Diffusers folder and a new .safetensors output file.")
            return
        self.navigation.setCurrentRow(2)
        self.run_process(["-m", "sdxl_trainer.training.export", "--model_path", source, "--output_path", target], "Exporting merged checkpoint…", self.logs)

    def diagnose(self):
        self.runtime_info.clear()
        self.run_process(["-m", "sdxl_trainer.runtime", "diagnose"], "Inspecting local Python runtime…", self.runtime_info)

    def inspect_repository(self):
        from .downloader import HuggingFaceDownloader
        repo, token = self.repo.text(), self.hf_token.text() or None
        self.run_task("Connecting to Hugging Face to inspect selected repository…", lambda task: HuggingFaceDownloader(True, token).inspect(repo, cancel_event=task.cancel), self.show_repository)

    def show_repository(self, snapshot):
        from .downloader import PRESETS
        self.snapshot = snapshot
        self.remote_files.clear()
        index = self.preset.currentIndex()
        preset = PRESETS[index] if index < len(PRESETS) and PRESETS[index]["repo_id"] == snapshot.repo_id else {}
        wanted = {value for value in (set(preset.get("filenames", [])) |
                                      {preset.get("model_filename"), preset.get("mmproj_filename")}) if value}
        available = {remote.path for remote in snapshot.files}
        for remote in snapshot.files:
            item = QListWidgetItem(f"{remote.path}    {remote.size / 2**30:.3f} GiB")
            item.setData(Qt.ItemDataRole.UserRole, remote.path)
            self.remote_files.addItem(item)
            item.setSelected(remote.path in wanted)
        missing = sorted(wanted - available)
        message = f"{snapshot.repo_id} · pinned revision {snapshot.revision}"
        if missing:
            message += "\nPreset files not found in this repository: " + ", ".join(missing)
        self.download_info.setText(message)
        self.status.setText("Preset incomplete; review the missing files and selection." if missing else
                            "Repository inspected. Review selection, then download explicitly.")

    def download_selected(self):
        from .downloader import HuggingFaceDownloader, PRESETS, model_library_destination
        if self.snapshot is None:
            self.error("Inspect the repository first.")
            return
        filenames = [item.data(Qt.ItemDataRole.UserRole) for item in self.remote_files.selectedItems()]
        if not filenames or not self.download_dir.value():
            self.error("Select files and a local model library folder.")
            return
        snapshot = self.snapshot
        index = self.preset.currentIndex()
        preset = PRESETS[index] if index < len(PRESETS) and PRESETS[index]["repo_id"] == snapshot.repo_id else {}
        destination = model_library_destination(self.download_dir.value(), snapshot.repo_id, preset)
        token = self.hf_token.text() or None
        def work(task):
            def progress(filename, done, total):
                task.progress.emit(f"Downloading {filename}: {done / 2**30:.2f} / {total / 2**30:.2f} GiB")
            return HuggingFaceDownloader(True, token).download(snapshot, filenames, destination, progress=progress, cancel_event=task.cancel)
        def complete(paths):
            self.hf_token.clear()
            for path in paths:
                if str(path).endswith(".gguf"):
                    key = "mmproj_path" if "mmproj" in Path(path).name.lower() else "model_path"
                    self.shared_caption_paths[key].set_value(path)
                    if self.project:
                        self.caption_form.controls[key].set_value(path)
            if (destination / "model_index.json").is_file():
                self.model_field.set_value(destination)
                self.shared_training_model.controls["model_path"].set_value(destination)
                if preset.get("kind") == "sdxl" and all((destination / name).is_file() for name in preset.get("filenames", [])):
                    self.training_controls["model_variant"].setCurrentText(preset["model_variant"])
                    self.shared_training_model.controls["model_variant"].setCurrentText(preset["model_variant"])
            self.save_settings()
            self.status.setText(f"Downloaded and verified {len(paths)} files. Local model paths selected.")
        self.run_task("Downloading only the files you selected…", work, complete)

    def cancel_task(self):
        if self.task:
            self.task.cancel.set()
            self.status.setText("Cancelling… current operation will stop safely.")
        elif self.process and self.process.processId():
            os.kill(self.process.processId(), signal.SIGINT)
            self.status.setText("Stop requested. Training saves a resumable checkpoint at the next optimizer boundary.")
        self.cancel_button.setEnabled(False)

    def closeEvent(self, event: QCloseEvent):
        if self.task or self.process:
            self.error("A task is still running. Use Cancel task and let it finish before closing.")
            event.ignore()
            return
        self.save_caption()
        self.save_settings()
        self.stop_tokens()
        self.checker.close()
        event.accept()


def main():
    os.environ.update({name: value for name, value in offline_environment().items() if name in
                       {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK", "TOKENIZERS_PARALLELISM"}})
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)
    app = QApplication(sys.argv)
    apply_theme(app)
    app.setApplicationName("SDXL-Trainer")
    app.setOrganizationName("SDXL-Trainer")
    app.setDesktopFileName("sdxl-trainer")
    # Keep two desktop instances from concurrently modifying the same draft/run.
    data_dir().mkdir(parents=True, exist_ok=True)
    instance_lock = QLockFile(str(data_dir() / "desktop.lock"))
    if not instance_lock.tryLock(100):
        QMessageBox.information(None, "SDXL-Trainer", "SDXL-Trainer is already running. Use its existing window.")
        return 0
    window = Window()
    window.show()
    return app.exec()
