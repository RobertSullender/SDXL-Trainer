"""Exercise the real widget-based dialogs without a desktop or native file picker."""
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow

from sdxl_trainer import file_dialogs


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_dialog_window_controls_and_state_transitions(app, tmp_path):
    parent = QMainWindow()
    dialog = file_dialogs._dialog(parent, "Open project", str(tmp_path), QFileDialog.FileMode.Directory)
    try:
        assert dialog.testOption(QFileDialog.Option.DontUseNativeDialog)
        assert dialog.testOption(QFileDialog.Option.ShowDirsOnly)
        assert dialog.parentWidget() is None
        assert dialog.windowType() == Qt.WindowType.Window
        assert dialog.windowFlags() & Qt.WindowType.CustomizeWindowHint
        assert dialog.windowFlags() & Qt.WindowType.WindowMinimizeButtonHint
        assert dialog.windowFlags() & Qt.WindowType.WindowMaximizeButtonHint
        dialog.show()
        app.processEvents()
        dialog.showMinimized()
        app.processEvents()
        assert dialog.isMinimized()
        dialog.showNormal()
        app.processEvents()
        assert not dialog.isMinimized() and not dialog.isMaximized()
        dialog.showMaximized()
        app.processEvents()
        assert dialog.isMaximized()
        dialog.showNormal()
        app.processEvents()
        assert not dialog.isMaximized()
    finally:
        dialog.close()
        parent.close()


def test_real_directory_selection_and_cancel(app, tmp_path):
    root = tmp_path / "projects"
    root.mkdir()
    def accept():
        dialog = app.activeModalWidget()
        assert isinstance(dialog, QFileDialog)
        dialog.accept()
    QTimer.singleShot(0, accept)
    assert file_dialogs.get_existing_directory(None, "Open", str(root)) == str(root)
    QTimer.singleShot(0, lambda: app.activeModalWidget().reject())
    assert file_dialogs.get_existing_directory(None, "Open", str(root)) == ""


def test_parentless_browser_still_uses_application_modality(app, tmp_path):
    parent = QMainWindow()
    parent.show()
    observed = []
    def inspect():
        dialog = app.activeModalWidget()
        observed.append((dialog is not None, dialog.parentWidget() is None, dialog.isModal(),
                         dialog.windowModality() == Qt.WindowModality.ApplicationModal))
        dialog.reject()
    QTimer.singleShot(0, inspect)
    try:
        assert file_dialogs.get_existing_directory(parent, "Open", str(tmp_path)) == ""
        assert observed == [(True, True, True, True)]
    finally:
        parent.close()


def test_real_file_and_save_selection(app, tmp_path):
    file = tmp_path / "image.png"
    file.write_bytes(b"image")
    def select():
        dialog = app.activeModalWidget()
        dialog.selectFile(str(file))
        dialog.accept()
    QTimer.singleShot(0, select)
    selected, chosen_filter = file_dialogs.get_open_file_name(None, "Open", str(tmp_path), "Images (*.png)")
    assert selected == str(file)
    assert chosen_filter == "Images (*.png)"
    target = tmp_path / "model.safetensors"
    def save():
        dialog = app.activeModalWidget()
        assert dialog.acceptMode() == QFileDialog.AcceptMode.AcceptSave
        dialog.selectFile(str(target))
        dialog.accept()
    QTimer.singleShot(0, save)
    selected, chosen_filter = file_dialogs.get_save_file_name(None, "Save", str(target), "Safetensors (*.safetensors)")
    assert selected == str(target)
    assert chosen_filter == "Safetensors (*.safetensors)"
