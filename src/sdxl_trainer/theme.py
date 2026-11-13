"""Application-local dark colors, including Qt's built-in file browsers.

The system's desktop theme is untouched. Non-native dialogs intentionally use the
same Qt rendering, palette and stylesheet as the rest of SDXL-Trainer.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QProxyStyle, QStyle, QStyleFactory


FILE_DIALOG_OPTIONS = QFileDialog.Option.DontUseNativeDialog

STYLE = """
QWidget { background: #11121a; color: #e7e8f4; font-family: 'Ubuntu', 'Noto Sans', sans-serif; font-size: 14px; }
QMainWindow, QDialog, QFileDialog { background: #11121a; }
QLabel#brand { font-size: 21px; font-weight: 700; }
QLabel#title { font-size: 26px; font-weight: 700; }
QLabel#muted { color: #9c9fb8; }
QFrame#sidebar { background: #0b0c12; border-right: 1px solid #282a3d; }
QFrame#sidebar QLabel, QFrame#sidebar QWidget { background: transparent; }
QPushButton, QToolButton { background: #242637; border: 1px solid #383b54; border-radius: 7px; padding: 9px 14px; }
QPushButton:hover, QToolButton:hover { background: #34324f; border-color: #8871cf; }
QPushButton:pressed, QToolButton:pressed, QToolButton:checked { background: #454065; }
QPushButton:disabled, QToolButton:disabled { color: #686b80; border-color: #242637; }
QPushButton#primary { background: #7353cf; border-color: #997ae8; color: white; font-weight: 600; }
QPushButton:default { border-color: #997ae8; }
QLineEdit, QPlainTextEdit, QTextBrowser, QSpinBox, QDoubleSpinBox, QComboBox {
 background: #191b28; border: 1px solid #34374e; border-radius: 6px; padding: 7px; selection-background-color: #5c4c97; }
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus { border-color: #9b7aec; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled { color: #686b80; }
QAbstractItemView { background: #151621; alternate-background-color: #191b28; color: #e7e8f4;
 selection-background-color: #343052; selection-color: #d9cbff; outline: none; }
QListWidget, QListView, QTreeView, QTableView { background: #151621; border: 1px solid #292c40; border-radius: 8px; outline: none; }
QListWidget::item { padding: 10px; border-radius: 5px; }
QListWidget::item:selected, QListView::item:selected, QTreeView::item:selected, QTableView::item:selected {
 background: #343052; color: #d9cbff; }
QListWidget#navigation { background: transparent; border: none; }
QListWidget#navigation::item { padding: 14px 12px; margin-bottom: 5px; }
QComboBox QAbstractItemView { background: #191b28; color: #e7e8f4; border: 1px solid #51466e;
 selection-background-color: #343052; selection-color: #d9cbff; }
QHeaderView { background: #1b1d2b; }
QHeaderView::section { background: #242637; color: #c9cbe0; border: none;
 border-right: 1px solid #34374e; border-bottom: 1px solid #34374e; padding: 8px; }
QTableCornerButton::section { background: #242637; border: 1px solid #34374e; }
QFileDialog QToolButton { padding: 6px; min-width: 22px; min-height: 22px; }
QFileDialog QListView::item, QFileDialog QTreeView::item { padding: 4px; }
QFileDialog QListView#sidebar { background: #0f1018; }
QFileDialog QSplitter::handle { background: #282a3d; }
QMenuBar, QMenu { background: #191b28; color: #e7e8f4; }
QMenu { border: 1px solid #34374e; padding: 4px; }
QMenu::item { padding: 7px 24px 7px 12px; }
QMenu::item:selected { background: #343052; color: #d9cbff; }
QMenu::item:disabled { color: #686b80; }
QMenu::separator { height: 1px; background: #34374e; margin: 4px 8px; }
QTabWidget::pane { border: 1px solid #303349; border-radius: 6px; }
QTabBar::tab { background: #1b1d2b; padding: 10px 15px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #c5a9ff; border-bottom-color: #a479ff; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator {
 width: 20px; height: 20px; border: 2px solid #a7a9c3;
 border-radius: 5px; background: #242637;
}
QCheckBox::indicator:hover { border-color: #d3c3ff; background: #34324f; }
QCheckBox::indicator:checked {
 background: #7353cf; border-color: #c1abff; image: url("@CHECK_ICON@");
}
QCheckBox::indicator:checked:hover { background: #8969e5; border-color: #e2d7ff; }
QCheckBox::indicator:focus { border-color: #e4d8ff; }
QCheckBox::indicator:disabled { background: #242637; border-color: #55596e; }
QCheckBox::indicator:checked:disabled { background: #47405e; border-color: #77718b; }
QProgressBar { background: #202234; border: none; border-radius: 4px; min-height: 7px; max-height: 7px; }
QProgressBar::chunk { background: #9874f5; border-radius: 4px; }
QScrollArea { border: none; }
QScrollBar:vertical { background: #13141f; width: 10px; }
QScrollBar::handle:vertical { background: #383852; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { background: #13141f; height: 10px; }
QScrollBar::handle:horizontal { background: #383852; border-radius: 5px; min-width: 30px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
QToolTip { background: #29263c; color: #f0e8ff; border: 1px solid #8d73ca; padding: 6px; }
QSplitter::handle { background: #2c2e44; }
""".replace("@CHECK_ICON@", str(Path(__file__).parent / "assets" / "checkbox-check.svg").replace("\\", "/").replace('"', r'\"'))


class DarkFusionStyle(QProxyStyle):
    """Use visible navigation chevrons where Fusion otherwise supplies black icons."""

    def __init__(self):
        super().__init__(QStyleFactory.create("Fusion"))
        self.setObjectName("Fusion")

    def standardIcon(self, standard_icon, option=None, widget=None):
        standard = QStyle.StandardPixmap
        arrows = {
            standard.SP_ArrowBack: [(15, 5), (8, 12), (15, 19)],
            standard.SP_FileDialogBack: [(15, 5), (8, 12), (15, 19)],
            standard.SP_ArrowForward: [(9, 5), (16, 12), (9, 19)],
            standard.SP_ArrowUp: [(5, 15), (12, 8), (19, 15)],
            standard.SP_FileDialogToParent: [(5, 15), (12, 8), (19, 15)],
        }
        if standard_icon not in arrows:
            return super().standardIcon(standard_icon, option, widget)
        icon = QIcon()
        for mode, color in ((QIcon.Mode.Normal, "#c9cbe0"), (QIcon.Mode.Disabled, "#686b80")):
            pixmap = QPixmap(24, 24)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor(color), 2.2, Qt.PenStyle.SolidLine,
                                Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            path = QPainterPath()
            points = arrows[standard_icon]
            path.moveTo(*points[0])
            for point in points[1:]:
                path.lineTo(*point)
            painter.drawPath(path)
            painter.end()
            icon.addPixmap(pixmap, mode)
        return icon


def apply_theme(app: QApplication) -> None:
    """Apply deterministic dark styling only to this application's Qt widgets."""
    app.setStyle(DarkFusionStyle())
    palette = QPalette()
    colors = {
        QPalette.ColorRole.Window: "#11121a",
        QPalette.ColorRole.WindowText: "#e7e8f4",
        QPalette.ColorRole.Base: "#191b28",
        QPalette.ColorRole.AlternateBase: "#151621",
        QPalette.ColorRole.Text: "#e7e8f4",
        QPalette.ColorRole.Button: "#242637",
        QPalette.ColorRole.ButtonText: "#e7e8f4",
        QPalette.ColorRole.BrightText: "#ffffff",
        QPalette.ColorRole.Highlight: "#7353cf",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.Link: "#9ebeff",
        QPalette.ColorRole.LinkVisited: "#c5a9ff",
        QPalette.ColorRole.ToolTipBase: "#29263c",
        QPalette.ColorRole.ToolTipText: "#f0e8ff",
        QPalette.ColorRole.PlaceholderText: "#777d9b",
        QPalette.ColorRole.Light: "#42465e",
        QPalette.ColorRole.Midlight: "#34374e",
        QPalette.ColorRole.Dark: "#0b0c12",
        QPalette.ColorRole.Mid: "#282a3d",
        QPalette.ColorRole.Shadow: "#090a10",
        QPalette.ColorRole.Accent: "#9874f5",
    }
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText, QPalette.ColorRole.PlaceholderText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#686b80"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor("#343052"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor("#9c9fb8"))
    app.setPalette(palette)
    app.setStyleSheet(STYLE)
