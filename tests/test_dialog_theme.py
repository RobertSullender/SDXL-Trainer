"""File browsers share the desktop's dark Qt theme without changing the OS."""
import os
from pathlib import Path
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PySide6")
from PySide6.QtGui import QPalette
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QFileDialog, QStyle, QStyleOptionButton, QTreeView, QWidget

from sdxl_trainer.theme import FILE_DIALOG_OPTIONS, STYLE, apply_theme
from sdxl_trainer import file_dialogs
from sdxl_trainer.widgets import PathField


@pytest.fixture(scope="module")
def themed_app():
    app = QApplication.instance() or QApplication([])
    previous_palette = app.palette()
    previous_stylesheet = app.styleSheet()
    previous_style = app.style().name()
    apply_theme(app)
    yield app
    app.setStyle(previous_style or "Fusion")
    app.setPalette(previous_palette)
    app.setStyleSheet(previous_stylesheet)


def test_dark_theme_applies_to_new_file_browser_and_its_file_tree(themed_app, tmp_path):
    (tmp_path / "source-images").mkdir()
    (tmp_path / "caption.txt").write_text("A local caption.")
    dialog = QFileDialog()
    dialog.setOptions(FILE_DIALOG_OPTIONS)
    dialog.setWindowTitle("Import original images")
    dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
    dialog.setViewMode(QFileDialog.ViewMode.Detail)
    dialog.setDirectory(str(tmp_path))
    dialog.resize(960, 620)
    dialog.show()
    try:
        deadline = time.monotonic() + 3
        tree = dialog.findChild(QTreeView, "treeView")
        while tree.model().rowCount(tree.rootIndex()) < 2 and time.monotonic() < deadline:
            themed_app.processEvents()
            time.sleep(0.01)
        themed_app.processEvents()
        assert dialog.testOption(QFileDialog.Option.DontUseNativeDialog)
        assert dialog.directory().absolutePath() == str(tmp_path)
        assert tree.model().rowCount(tree.rootIndex()) >= 2
        assert themed_app.palette().color(QPalette.ColorRole.Window).name() == "#11121a"
        assert tree.palette().color(QPalette.ColorRole.Base).name() == "#151621"
        assert tree.palette().color(QPalette.ColorRole.Text).name() == "#e7e8f4"
        assert tree.header().palette().color(QPalette.ColorRole.Window).lightness() < 80
        # Verify the rendered browser, not just palette declarations: an empty
        # region of the file list should remain dark under the active Qt style.
        rendered = tree.viewport().grab().toImage()
        assert rendered.pixelColor(rendered.width() - 8, rendered.height() - 8).lightness() < 80
    finally:
        dialog.close()
        dialog.deleteLater()
        themed_app.processEvents()


@pytest.mark.parametrize("kind,method,returned", [
    ("directory", "get_existing_directory", "/selected/project"),
    ("file", "get_open_file_name", ("/selected/model.gguf", "")),
    ("save", "get_save_file_name", ("/selected/model.safetensors", "")),
])
def test_all_path_browsers_request_themed_dialog_and_preserve_selection(themed_app, monkeypatch,
                                                                      kind, method, returned):
    calls = []

    def choose(*args, **kwargs):
        calls.append((args, kwargs))
        return returned

    monkeypatch.setattr(file_dialogs, method, choose)
    field = PathField("/starting/location", kind=kind)
    changes = []
    field.changed.connect(changes.append)
    field.browse()
    expected = returned[0] if isinstance(returned, tuple) else returned
    assert field.value() == expected and changes == [expected]
    assert calls[0][0][2] == "/starting/location"
    assert calls[0][1]["options"] & QFileDialog.Option.DontUseNativeDialog
    field.deleteLater()


def test_cancelled_path_browse_preserves_existing_value(themed_app, monkeypatch):
    monkeypatch.setattr(file_dialogs, "get_open_file_name", lambda *a, **k: ("", ""))
    field = PathField("/existing/model.gguf", kind="file")
    field.browse()
    assert field.value() == "/existing/model.gguf"
    field.deleteLater()


def test_theme_changes_only_qt_application_palette(themed_app):
    widget = QWidget()
    widget.ensurePolished()
    assert themed_app.styleSheet() == STYLE
    assert widget.palette().color(QPalette.ColorRole.Window).name() == "#11121a"
    assert themed_app.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text).name() == "#686b80"
    widget.deleteLater()


def test_folder_navigation_icons_remain_visible_on_dark_buttons(themed_app):
    for standard in (QStyle.StandardPixmap.SP_FileDialogBack, QStyle.StandardPixmap.SP_FileDialogToParent):
        image = themed_app.style().standardIcon(standard).pixmap(24, 24).toImage()
        opaque_pixels = [image.pixelColor(x, y) for x in range(image.width()) for y in range(image.height())
                         if image.pixelColor(x, y).alpha() > 150]
        assert opaque_pixels
        assert sum(color.lightness() for color in opaque_pixels) / len(opaque_pixels) > 150


def test_checkbox_rendering_and_keyboard_toggle_are_visible(themed_app):
    checkbox = QCheckBox("Train both text encoders")
    checkbox.resize(260, 40)
    checkbox.show()
    checkbox.setFocus()
    themed_app.processEvents()
    try:
        option = QStyleOptionButton()
        option.initFrom(checkbox)
        option.text = checkbox.text()
        indicator = checkbox.style().subElementRect(QStyle.SubElement.SE_CheckBoxIndicator, option, checkbox)
        assert indicator.width() >= 20 and indicator.height() >= 20
        unchecked = checkbox.grab().toImage()
        border = unchecked.pixelColor(indicator.x() + 1, indicator.center().y())
        assert border.lightness() > 120  # Unchecked box remains visible on a dark background.
        assert not checkbox.isChecked()

        QTest.keyClick(checkbox, Qt.Key.Key_Space)
        themed_app.processEvents()
        assert checkbox.isChecked()
        checked = checkbox.grab().toImage()
        pixels = [checked.pixelColor(x, y) for x in range(indicator.left(), indicator.right() + 1)
                  for y in range(indicator.top(), indicator.bottom() + 1)]
        assert sum(c.red() > 245 and c.green() > 245 and c.blue() > 245 for c in pixels) >= 3
        assert checked.pixelColor(indicator.x() + 4, indicator.y() + 4) != unchecked.pixelColor(indicator.x() + 4, indicator.y() + 4)

        checkbox.setEnabled(False)
        QTest.keyClick(checkbox, Qt.Key.Key_Space)
        assert checkbox.isChecked()  # Disabled controls do not respond to the shortcut.
    finally:
        checkbox.close()
        checkbox.deleteLater()
        themed_app.processEvents()
