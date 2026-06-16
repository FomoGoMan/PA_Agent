"""Apply the global application theme."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QApplication

_QSS_PATH = Path(__file__).with_name("dark.qss")


def apply_theme(app: QApplication, light: bool = False) -> None:
    """Load theme and set application-wide palette hints."""
    name = "light.qss" if light else "dark.qss"
    qss_path = Path(__file__).with_name(name)
    if qss_path.is_file():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
    app.setStyle("Fusion")
