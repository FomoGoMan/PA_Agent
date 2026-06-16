"""Application entry point for PA Agent."""
from __future__ import annotations

import logging
import sys

from PyQt6.QtWidgets import QApplication

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)

    # Parse CLI arguments
    light_mode = "--light" in argv
    if light_mode:
        argv.remove("--light")

    silent_mode = "--silent" in argv
    if silent_mode:
        argv.remove("--silent")

    auto_start = "--auto-start" in argv
    if auto_start:
        argv.remove("--auto-start")

    # Parse --interval=<ms>
    custom_interval = None
    for arg in list(argv):
        if arg.startswith("--interval="):
            try:
                custom_interval = int(arg.split("=", 1)[1])
                argv.remove(arg)
            except (ValueError, IndexError):
                pass

    app = QApplication(argv)
    app.setApplicationName("PA Agent")

    from pa_agent.app_context import AppContext
    ctx = AppContext.bootstrap()

    # Check theme setting
    settings_light = False
    if ctx.settings is not None:
        settings_light = getattr(ctx.settings.general, "theme_light", False)

    from pa_agent.gui.theme import apply_theme
    apply_theme(app, light=light_mode or settings_light)

    logger.info("PA Agent starting up (light=%s, silent=%s, auto_start=%s)",
                light_mode or settings_light, silent_mode, auto_start)

    if ctx.settings is not None:
        from pa_agent.util.logging import update_api_key
        update_api_key(ctx.settings.provider.api_key)

        # Apply custom interval if provided
        if custom_interval is not None:
            ctx.settings.general.refresh_interval_ms = custom_interval
            logger.info("Using custom interval: %d ms", custom_interval)

    from pa_agent.gui.main_window import MainWindow
    window = MainWindow(ctx, silent=silent_mode, auto_start=auto_start)

    if not silent_mode:
        window.show()

    logger.info("Main window shown" if not silent_mode else "Silent mode active")
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
