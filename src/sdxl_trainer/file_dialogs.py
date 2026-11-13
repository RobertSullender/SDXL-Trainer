"""Application-themed file dialogs with normal desktop window controls."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFileDialog

from .theme import FILE_DIALOG_OPTIONS


WINDOW_FLAGS = (Qt.WindowType.Window | Qt.WindowType.CustomizeWindowHint |
                Qt.WindowType.WindowTitleHint | Qt.WindowType.WindowSystemMenuHint |
                Qt.WindowType.WindowMinMaxButtonsHint | Qt.WindowType.WindowCloseButtonHint)


def _dialog(parent, title, directory, file_mode, name_filter="", options=FILE_DIALOG_OPTIONS,
            *, save=False):
    # A parented modal window is attached to its parent by GNOME/Mutter, which
    # prevents independent minimization on Wayland. The dialog is application
    # modal via exec() but has its own top-level desktop window.
    dialog = QFileDialog(None, WINDOW_FLAGS)
    if parent is not None:
        dialog.setWindowIcon(parent.window().windowIcon())
    # Must be set before other properties or showing the dialog.
    dialog.setOptions(options | FILE_DIALOG_OPTIONS)
    dialog.setWindowTitle(title)
    dialog.setFileMode(file_mode)
    if file_mode == QFileDialog.FileMode.Directory:
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
    if save:
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
    if name_filter:
        dialog.setNameFilter(name_filter)
    if directory:
        path = Path(directory).expanduser()
        if path.is_dir():
            dialog.setDirectory(str(path))
        else:
            if str(path.parent) != ".":
                dialog.setDirectory(str(path.parent))
            dialog.selectFile(path.name)
    return dialog


def get_existing_directory(parent=None, title="Select folder", directory="", *, options=FILE_DIALOG_OPTIONS):
    dialog = _dialog(parent, title, directory, QFileDialog.FileMode.Directory, options=options)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        selected = dialog.selectedFiles()
        return selected[0] if selected else ""
    return ""


def get_open_file_name(parent=None, title="Select file", directory="", name_filter="", *, options=FILE_DIALOG_OPTIONS):
    dialog = _dialog(parent, title, directory, QFileDialog.FileMode.ExistingFile, name_filter, options)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        selected = dialog.selectedFiles()
        return (selected[0] if selected else "", dialog.selectedNameFilter())
    return "", ""


def get_open_file_names(parent=None, title="Select files", directory="", name_filter="", *, options=FILE_DIALOG_OPTIONS):
    dialog = _dialog(parent, title, directory, QFileDialog.FileMode.ExistingFiles, name_filter, options)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.selectedFiles(), dialog.selectedNameFilter()
    return [], ""


def get_save_file_name(parent=None, title="Save file", directory="", name_filter="", *, options=FILE_DIALOG_OPTIONS):
    dialog = _dialog(parent, title, directory, QFileDialog.FileMode.AnyFile, name_filter, options, save=True)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        selected = dialog.selectedFiles()
        return (selected[0] if selected else "", dialog.selectedNameFilter())
    return "", ""
