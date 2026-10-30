"""Native, offline desktop UI. GPU workloads live in separate managed processes."""
from __future__ import annotations

from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import signal
import sys
import uuid

from PySide6.QtCore import Qt, QLockFile, QProcess, QProcessEnvironment, QSize, QTimer
from PySide6.QtGui import QIcon, QPixmap, QCloseEvent
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSplitter, QStackedWidget,
    QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from . import __version__
from .captioning import CaptionConfig, CaptionServer
from .dataset import DatasetProject
from .paths import atomic_json, config_dir, data_dir, load_settings, offline_environment
from .runtime import RemoteTokenCounter
from .spelling import SpellChecker
from .training.config import TrainingConfig, preflight
from .widgets import CaptionEditor, ConfigForm, PathField, Task


ROOT = Path(__file__).resolve().parent
STYLE = """
QWidget { background: #11121a; color: #e7e8f4; font-family: 'Ubuntu', 'Noto Sans', sans-serif; font-size: 14px; }
QMainWindow { background: #11121a; }
QLabel#brand { font-size: 21px; font-weight: 700; }
QLabel#title { font-size: 26px; font-weight: 700; }
QLabel#muted { color: #9c9fb8; }
QFrame#sidebar { background: #0b0c12; border-right: 1px solid #282a3d; }
QFrame#sidebar QLabel, QFrame#sidebar QWidget { background: transparent; }
QPushButton { background: #242637; border: 1px solid #383b54; border-radius: 7px; padding: 9px 14px; }
QPushButton:hover { background: #34324f; border-color: #8871cf; }
QPushButton:pressed { background: #454065; }
QPushButton:disabled { color: #686b80; border-color: #242637; }
QPushButton#primary { background: #7353cf; border-color: #997ae8; color: white; font-weight: 600; }
QLineEdit, QPlainTextEdit, QTextBrowser, QSpinBox, QDoubleSpinBox, QComboBox {
 background: #191b28; border: 1px solid #34374e; border-radius: 6px; padding: 7px; selection-background-color: #5c4c97; }
QLineEdit:focus, QPlainTextEdit:focus { border-color: #9b7aec; }
QListWidget { background: #151621; border: 1px solid #292c40; border-radius: 8px; outline: none; }
QListWidget::item { padding: 10px; border-radius: 5px; }
QListWidget::item:selected { background: #343052; color: #d9cbff; }
QListWidget#navigation { background: transparent; border: none; }
QListWidget#navigation::item { padding: 14px 12px; margin-bottom: 5px; }
QTabWidget::pane { border: 1px solid #303349; border-radius: 6px; }
QTabBar::tab { background: #1b1d2b; padding: 10px 15px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #c5a9ff; border-bottom-color: #a479ff; }
QCheckBox { spacing: 8px; }
QProgressBar { background: #202234; border: none; border-radius: 4px; min-height: 7px; max-height: 7px; }
QProgressBar::chunk { background: #9874f5; border-radius: 4px; }
QScrollArea { border: none; }
QScrollBar:vertical { background: #13141f; width: 10px; }
QScrollBar::handle:vertical { background: #383852; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QToolTip { background: #29263c; color: #f0e8ff; border: 1px solid #8d73ca; padding: 6px; }
QSplitter::handle { background: #2c2e44; }
"""


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
        self.project = None
        self.entry_id = None
        self.task = None
        self.process = None
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
        self.setStyleSheet(STYLE)
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
        options.addStretch()
        layout.addLayout(options)
        self.project_label = note("No project open · Original images are never modified")
        layout.addWidget(self.project_label)
        splitter = QSplitter()
        self.images = QListWidget()
        self.images.setIconSize(QSize(74, 74))
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
        caption_row.addStretch()
        self.tokens_label = note("CLIP-L — / 77    CLIP-G — / 77")
        self.tokens_label.setWordWrap(False)
        caption_row.addWidget(self.tokens_label)
        panel.addLayout(caption_row)
        self.caption = CaptionEditor(self.checker)
        panel.addWidget(self.caption)
        self.spelling_label = note("Spelling: English (offline). Right-click an underlined word for suggestions." if self.checker.available else f"Spelling unavailable: {self.checker.reason}")
        panel.addWidget(self.spelling_label)
        actions = QHBoxLayout()
        actions.addWidget(button("Caption selected", lambda: self.auto_caption(False)))
        actions.addWidget(button("Caption all empty", lambda: self.auto_caption(True)))
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
        self.caption_form = ConfigForm(config)
        self.caption_form.set_values(self.settings.get("caption", {}))
        layout.addWidget(scroll(self.caption_form), 1)
        layout.addWidget(button("Save caption settings", self.save_settings, True))

    def build_training(self):
        _, layout = self.page("Train SDXL", "Only finalized datasets can train. Every control below is passed to the training worker; hover for its explanation.")
        tabs = QTabWidget()
        self.training_forms = []
        config = TrainingConfig()
        groups = list(dict.fromkeys(f.metadata.get("group", "Training") for f in fields(config)))
        for group in groups:
            names = [f.name for f in fields(config) if f.metadata.get("group", "Training") == group]
            form = ConfigForm(config, names)
            form.set_values(self.settings.get("training", {}))
            tabs.addTab(scroll(form), group)
            self.training_forms.append(form)
        self.training_controls = {name: widget for form in self.training_forms for name, widget in form.controls.items()}
        self.model_field = self.training_controls["model_path"]
        layout.addWidget(tabs, 1)
        layout.addWidget(note("VAE adaptation is experimental: reconstruction + KL + a frozen-base latent anchor. Review reconstructions before relying on it. Full training on 16 GB requires measured validation."))
        row = QHBoxLayout()
        row.addWidget(button("Load recipe…", self.load_recipe))
        row.addWidget(button("Save recipe…", self.save_recipe))
        row.addWidget(button("Check configuration", self.check_training))
        row.addStretch()
        self.start_training = button("Start training", self.train, True)
        row.addWidget(self.start_training)
        layout.addLayout(row)
        self.training_progress = QProgressBar()
        self.training_progress.setTextVisible(False)
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        layout.addWidget(self.training_progress)
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
        layout.addWidget(note("The checkpoint includes UNet, both text encoders, and VAE. Export preserves fp32 weights and requires disk space for the combined file. The original Diffusers result remains available."))
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
        self.download_dir = PathField(str(data_dir() / "models"))
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
        self.runtime = PathField(self.settings.get("runtime_python", sys.executable), kind="file")
        rtf = QFormLayout()
        rtf.addRow("Training Python", self.runtime)
        rt.addLayout(rtf)
        rt.addWidget(button("Inspect runtime", self.diagnose, True))
        self.runtime_info = QPlainTextEdit()
        self.runtime_info.setReadOnly(True)
        rt.addWidget(self.runtime_info, 1)
        rt.addWidget(button("Save settings", self.save_settings))
        tabs.addTab(runtime_page, "Local runtime")
        layout.addWidget(tabs, 1)
        self.snapshot = None

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
        self.settings.update(runtime_python=self.runtime.value(), caption=self.caption_form.values(),
                             training=self.training_config().to_dict(),
                             last_project=str(self.project.root) if self.project else "")
        atomic_json(config_dir() / "settings.json", self.settings)
        self.status.setText("Settings saved locally.")

    def set_project(self, project):
        self.save_caption()
        self.project = project
        self.entry_id = None
        self.settings["last_project"] = str(project.root)
        self.refresh_images()
        self.save_settings()

    def new_project(self):
        parent = QFileDialog.getExistingDirectory(self, "Choose an empty folder for the new dataset project")
        if parent:
            try:
                self.set_project(DatasetProject.create(parent))
            except Exception as exc:
                self.error(str(exc))

    def open_project(self):
        path = QFileDialog.getExistingDirectory(self, "Open dataset project")
        if path:
            try:
                self.set_project(DatasetProject.open(path))
            except Exception as exc:
                self.error(str(exc))

    def refresh_images(self, select_id=None):
        self.images.blockSignals(True)
        self.images.clear()
        if self.project:
            for entry in self.project.entries:
                item = QListWidgetItem(f"{entry.image}\n{entry.width} × {entry.height}")
                item.setData(Qt.ItemDataRole.UserRole, entry.id)
                item.setIcon(QIcon(str(self.project.image_path(entry.id))))
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
            self.tokens_label.setText("CLIP-L — / 77    CLIP-G — / 77")

    def select_image(self, row):
        self.save_caption()
        if not self.project or row < 0:
            return
        entry = self.project.entries[row]
        self.entry_id = entry.id
        self.token_generation += 1
        self.loaded_pixmap = QPixmap(str(self.project.image_path(entry.id)))
        self.update_preview()
        self.image_info.setText(f"{entry.image} · {entry.width} × {entry.height} · {entry.fit} · RGB PNG · Metadata removed")
        self.caption.blockSignals(True)
        self.caption.setPlainText(self.project.caption_text(entry.id))
        self.caption.blockSignals(False)
        self.request_token_count()

    def update_preview(self):
        if not self.loaded_pixmap.isNull():
            self.preview.setPixmap(self.loaded_pixmap.scaled(self.preview.size() - QSize(18, 18), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "preview"):
            self.update_preview()

    def caption_changed(self):
        self.caption_timer.start()
        self.token_generation += 1
        self.tokens_label.setText("CLIP-L … / 77    CLIP-G … / 77")

    def save_caption(self):
        if self.project is not None and self.entry_id is not None:
            self.caption_timer.stop()
            try:
                self.project.save_caption(self.entry_id, self.caption.toPlainText())
                self.request_token_count()
            except Exception as exc:
                self.error(str(exc))

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
                    self.tokens_label.setText(f"CLIP-L {a} / 77    CLIP-G {b} / 77 (includes start/end)")
                    self.tokens_label.setStyleSheet("color:#fb8798" if max(a, b) > 77 else "color:#9ebeff")

    def busy(self, active, text="Ready."):
        self.content.setEnabled(not active)
        self.cancel_button.setEnabled(active)
        self.activity.setRange(0, 0 if active else 1)
        self.activity.setValue(0)
        self.status.setText(text)

    def run_task(self, text, function, done=None, cancellable=True):
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
        task.finished.connect(self.task_finished)
        task.start()

    def task_finished(self):
        task = self.task
        self.task = None
        self.busy(False, self.status.text())
        if task:
            task.deleteLater()

    def import_images(self, folder):
        if not self.project:
            self.error("Create or open a dataset project first.")
            return
        if folder:
            path = QFileDialog.getExistingDirectory(self, "Import original image folder")
            paths = [path] if path else []
        else:
            paths, _ = QFileDialog.getOpenFileNames(self, "Import original images", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)")
        if not paths:
            return
        fit, background, recursive = self.fit.currentData(), self.background.text(), self.recursive.isChecked()
        self.run_task("Preparing clean PNG images…", lambda task: self.project.import_images(paths, fit, background, recursive),
                      lambda result: (self.refresh_images(result[-1].id if result else None), self.status.setText(f"Imported {len(result)} images. Originals preserved.")), cancellable=False)

    def auto_caption(self, all_empty):
        if self.task or self.process:
            return
        if not self.project or self.entry_id is None:
            self.error("Import an image first.")
            return
        self.save_caption()
        config = CaptionConfig(**self.caption_form.values())
        errors = config.validate()
        if errors:
            self.error("\n".join(errors))
            return
        entries = [e for e in self.project.entries if not self.project.caption_text(e.id).strip()] if all_empty else [e for e in self.project.entries if e.id == self.entry_id]
        selected = self.entry_id
        if not entries:
            self.status.setText("Every image already has a caption. Select an image to replace its caption explicitly.")
            return
        def work(task):
            server = CaptionServer(config)
            try:
                server.start(task.cancel)
                for i, entry in enumerate(entries):
                    task.progress.emit(f"Captioning {entry.image} · {i + 1}/{len(entries)}")
                    text = server.caption(self.project.image_path(entry.id), task.cancel)
                    self.project.save_caption(entry.id, text)
                return len(entries)
            finally:
                server.stop()
        def complete(count):
            self.status.setText(f"Captioned {count} images. Review spelling and both token counts.")
        def refresh_after_captioning():
            # A cancelled/failed batch may still have saved earlier images. Reload
            # those authoritative captions before any later editor save can occur.
            self.entry_id = None
            self.refresh_images(selected)
        self.run_task("Loading local caption model…", work, complete)
        if self.task:
            self.task.finished.connect(refresh_after_captioning)

    def finalize(self):
        if not self.project or not self.project.entries:
            self.error("Import and caption images before finalizing.")
            return
        if not self.model_field.value():
            self.error("Choose the local Diffusers base model in Training so both tokenizer counts can be verified.")
            return
        parent = QFileDialog.getExistingDirectory(self, "Choose the parent folder for the finalized dataset")
        if not parent:
            return
        name, accepted = QInputDialog.getText(self, "Name the finalized dataset", "New folder name:",
                                             text=self.project.root.name + "-finalized")
        if not accepted:
            return
        name = name.strip()
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            self.error("Choose a simple new folder name, without path separators.")
            return
        destination = str(Path(parent) / name)
        model, python = self.model_field.value(), self.runtime.value()
        def work(task):
            with RemoteTokenCounter(python, model) as counter:
                return self.project.finalize(destination, counter)
        def done(path):
            self.training_controls["dataset_path"].set_value(path)
            self.status.setText(f"Finalized dataset verified and selected for training: {path}")
            self.save_settings()
        self.run_task("Verifying images, captions, both tokenizers and dataset hashes…", work, done, cancellable=False)

    def training_config(self):
        values = {key: value for form in self.training_forms for key, value in form.values().items()}
        return TrainingConfig.from_dict(values)

    def load_recipe(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load training recipe", "", "JSON (*.json)")
        if path:
            try:
                config = TrainingConfig.load(path)
                for form in self.training_forms:
                    form.set_values(config.to_dict())
            except Exception as exc:
                self.error(str(exc))

    def save_recipe(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save training recipe", "recipe.json", "JSON (*.json)")
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
        self.save_settings()
        path = config_dir() / "launches" / f"{uuid.uuid4().hex}.json"
        atomic_json(path, config.to_dict())
        self.logs.clear()
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
            self.process_output.appendPlainText(json.dumps(event, ensure_ascii=False))
            if isinstance(event, dict):
                if event.get("type") == "progress":
                    self.training_progress.setValue(int(event.get("step", 0)))
                    self.status.setText(f"Step {event.get('step')} · loss {event.get('loss', '—')}")
                elif event.get("type") == "complete":
                    path = event.get("output_dir") or event.get("path")
                    if path and (Path(path) / "model_index.json").is_file():
                        self.export_model.set_value(path)
                elif event.get("message"):
                    self.status.setText(str(event["message"])[:180])

    def process_finished(self, code, *_):
        process = self.process
        self.process = None
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
        wanted = set(preset.get("filenames", [])) | {preset.get("model_filename"), preset.get("mmproj_filename")}
        for remote in snapshot.files:
            item = QListWidgetItem(f"{remote.path}    {remote.size / 2**30:.3f} GiB")
            item.setData(Qt.ItemDataRole.UserRole, remote.path)
            self.remote_files.addItem(item)
            item.setSelected(remote.path in wanted)
        self.download_info.setText(f"{snapshot.repo_id} · pinned revision {snapshot.revision}")
        self.status.setText("Repository inspected. Review selection, then download explicitly.")

    def download_selected(self):
        from .downloader import HuggingFaceDownloader
        if self.snapshot is None:
            self.error("Inspect the repository first.")
            return
        filenames = [item.data(Qt.ItemDataRole.UserRole) for item in self.remote_files.selectedItems()]
        if not filenames or not self.download_dir.value():
            self.error("Select files and a local model library folder.")
            return
        snapshot = self.snapshot
        destination = Path(self.download_dir.value()) / snapshot.repo_id.replace("/", "--")
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
                    self.caption_form.controls[key].set_value(path)
            if (destination / "model_index.json").is_file():
                self.model_field.set_value(destination)
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
    app = QApplication(sys.argv)
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
