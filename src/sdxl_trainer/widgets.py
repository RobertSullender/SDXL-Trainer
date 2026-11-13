from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import re
import threading

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QSyntaxHighlighter, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLineEdit, QMenu, QPlainTextEdit, QPushButton, QSpinBox, QWidget,
)

from . import file_dialogs
from .theme import FILE_DIALOG_OPTIONS


class Task(QThread):
    result = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function
        self.cancel = threading.Event()

    def run(self):
        try:
            value = self.function(self)
            self.result.emit(value)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class PathField(QWidget):
    changed = Signal(str)
    path_selected = Signal(str, str)

    def __init__(self, value="", kind="directory", parent=None):
        super().__init__(parent)
        self.kind = kind
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(str(value))
        self.edit.setPlaceholderText("Choose a local path…")
        button = QPushButton("Browse…")
        button.setFixedWidth(94)
        button.clicked.connect(self.browse)
        self.edit.textChanged.connect(self.changed)
        layout.addWidget(self.edit, 1)
        layout.addWidget(button)

    def browse(self):
        if self.kind == "directory":
            path = file_dialogs.get_existing_directory(self, "Choose folder", self.value(), options=FILE_DIALOG_OPTIONS)
        elif self.kind == "save":
            path, _ = file_dialogs.get_save_file_name(self, "Save checkpoint", self.value(), "Safetensors (*.safetensors)", options=FILE_DIALOG_OPTIONS)
        else:
            path, _ = file_dialogs.get_open_file_name(self, "Choose local file", self.value(), options=FILE_DIALOG_OPTIONS)
        if path:
            previous = self.value()
            self.set_value(path)
            self.path_selected.emit(self.value(), previous)

    def value(self):
        return self.edit.text().strip()

    def set_value(self, value):
        self.edit.setText(str(value))


CHOICES = {
    "mode": ["lora", "full"], "precision": ["bf16", "fp32"],
    "device": ["cuda", "cpu"], "optimizer": ["adamw", "adamw_disk", "sgd"],
    "lr_scheduler": ["constant", "linear", "cosine"],
    "flash_attention": ["auto", "on", "off"],
}

# The values are part of saved recipes and checkpoint identities; keep them
# stable while giving the user plain-language choices in the interface.
CHOICE_LABELS = {
    "optimizer": {
        "adamw_disk": "Disk AdamW · FP32 moments / masters",
        "paged_adamw8bit": "Paged AdamW · 8-bit moments (direct weights)",
        "adamw": "AdamW · device memory",
        "sgd": "SGD · device memory",
    },
    "component_offload": {
        "none": "No component offload",
        "text_encoders_cpu": "Text encoders on CPU",
        "text_and_vae_cpu": "Text encoders and VAE on CPU",
        "text_cpu_staged_unet": "Text encoders on CPU; move UNet for VAE",
        "stage_unet_gpu_text": "Text encoders on GPU; move UNet for VAE",
    },
}


class CompactDoubleSpinBox(QDoubleSpinBox):
    """Keep the configured decimal precision without visually padding every value."""

    def textFromValue(self, value):
        text = super().textFromValue(value)
        decimal = self.locale().decimalPoint()
        return text.rstrip("0").rstrip(decimal) if decimal in text else text


class ConfigForm(QWidget):
    """Every public backend dataclass field gets an editable, typed control."""

    def __init__(self, config, names=None, parent=None):
        super().__init__(parent)
        self.controls = {}
        layout = QFormLayout(self)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for field in fields(config):
            name = field.name
            if names is not None and name not in names:
                continue
            value = getattr(config, name)
            label = field.metadata.get("label", name.replace("_", " ").capitalize())
            help_text = field.metadata.get("help", name.replace("_", " "))
            if isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
            elif "choices" in field.metadata or name in CHOICES:
                widget = QComboBox()
                choices = field.metadata.get("choices", CHOICES.get(name, []))
                labels = CHOICE_LABELS.get(name)
                if labels is None:
                    widget.addItems(choices)
                    widget.setCurrentText(str(value))
                else:
                    for choice in choices:
                        widget.addItem(labels.get(choice, choice), choice)
                    index = widget.findData(value)
                    if index >= 0:
                        widget.setCurrentIndex(index)
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(int(field.metadata.get("min", -1 if name in {"seed", "gpu_layers", "repeat_last_n"} else 0)),
                                min(2_000_000_000, int(field.metadata.get("max", 2_000_000_000))))
                widget.setValue(value)
            elif isinstance(value, float):
                widget = CompactDoubleSpinBox()
                widget.setDecimals(int(field.metadata.get("decimals", 10)))
                widget.setRange(float(field.metadata.get("min", -1e6)), float(field.metadata.get("max", 1e6)))
                widget.setSingleStep(0.00001 if "lr" in name or "rate" in name else 0.05)
                widget.setValue(value)
            elif name in {"system_prompt", "user_prompt"}:
                widget = QPlainTextEdit(str(value))
                widget.setMinimumHeight(110)
                widget.setMaximumHeight(170)
                # A blank editor still shows what its project-local reset uses.
                from .captioning import CaptionConfig
                widget.setPlaceholderText(getattr(CaptionConfig(), name))
            elif name.endswith(("_path", "_dir", "_from", "_executable")):
                kind = "file" if name in {"server_executable", "mmproj_path"} else "directory"
                if name == "model_path" and hasattr(config, "mmproj_path"):
                    kind = "file"
                widget = PathField(value or "", kind=kind)
            else:
                widget = QLineEdit("" if value is None else str(value))
            widget.setToolTip(help_text)
            layout.addRow(label, widget)
            self.controls[name] = widget

    def values(self):
        result = {}
        for name, widget in self.controls.items():
            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QComboBox):
                value = widget.currentData() if name in CHOICE_LABELS else widget.currentText()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                value = widget.value()
            elif isinstance(widget, QPlainTextEdit):
                value = widget.toPlainText()
            elif isinstance(widget, PathField):
                value = widget.value()
            else:
                value = widget.text()
            result[name] = value
        return result

    def set_values(self, values):
        for name, value in values.items():
            widget = self.controls.get(name)
            if widget is None:
                continue
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                if name in CHOICE_LABELS:
                    index = widget.findData(value)
                    if index >= 0:
                        widget.setCurrentIndex(index)
                else:
                    widget.setCurrentText(str(value))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QPlainTextEdit):
                widget.setPlainText(str(value))
            elif isinstance(widget, PathField):
                widget.set_value(value or "")
            else:
                widget.setText(str(value))


class SpellHighlighter(QSyntaxHighlighter):
    def __init__(self, document, checker):
        super().__init__(document)
        self.checker = checker
        self.mark = QTextCharFormat()
        self.mark.setUnderlineColor(QColor("#fb8798"))
        self.mark.setUnderlineStyle(QTextCharFormat.UnderlineStyle.SpellCheckUnderline)

    def highlightBlock(self, text):
        if not self.checker.available:
            return
        for start, length, _ in self.checker.misspellings(text):
            # Qt offsets count UTF-16 code units, Python offsets count codepoints.
            qt_start = len(text[:start].encode("utf-16-le")) // 2
            qt_length = len(text[start:start + length].encode("utf-16-le")) // 2
            self.setFormat(qt_start, qt_length, self.mark)


class CaptionEditor(QPlainTextEdit):
    def __init__(self, checker, parent=None):
        super().__init__(parent)
        self.checker = checker
        self.highlighter = SpellHighlighter(self.document(), checker)
        self.setPlaceholderText("Caption the processed image. Edits are saved automatically.")
        self.setMinimumHeight(130)

    def contextMenuEvent(self, event):
        menu = self.createStandardContextMenu()
        cursor = self.cursorForPosition(event.pos())
        cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        word = cursor.selectedText()
        if word and self.checker.available and not self.checker.check(word):
            suggestions = self.checker.suggest(word)[:8]
            spelling = menu.addMenu("Spelling suggestions")
            for suggestion in suggestions:
                action = spelling.addAction(suggestion)
                action.triggered.connect(lambda checked=False, s=suggestion, c=QTextCursor(cursor): c.insertText(s))
            if not suggestions:
                spelling.addAction("No suggestions").setEnabled(False)
        menu.exec(event.globalPos())
