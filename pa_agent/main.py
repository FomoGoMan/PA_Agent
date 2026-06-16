"""Application entry point for PA Agent."""
from __future__ import annotations

import logging
import sys

from PyQt6.QtWidgets import QApplication

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    light_mode = "--light" in argv
    if light_mode:
        argv.remove("--light")

    app = QApplication(argv)
    app.setApplicationName("PA Agent")

    from pa_agent.gui.theme import apply_theme
    apply_theme(app, light=light_mode)

    logger.info("PA Agent starting up (light=%s)", light_mode)

    from pa_agent.app_context import AppContext
    ctx = AppContext.bootstrap()

    if ctx.settings is not None:
        from pa_agent.util.logging import update_api_key
        update_api_key(ctx.settings.provider.api_key)

    from pa_agent.gui.main_window import MainWindow
    window = MainWindow(ctx)
    window.show()

    logger.info("Main window shown")
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
