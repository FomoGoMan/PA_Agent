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

    # 启动后发送 QQ 上线通知
    if auto_start:
        from pa_agent.notify.channels.qq_channel import start_bot, get_bot_loop, get_notifier
        import threading
        import asyncio
        import time

        def _delayed_notify():
            print("QQ_NOTIFY: 通知线程开始", flush=True)
            time.sleep(6)  # 等待机器人连接
            from pa_agent.notify.channels.base import TradeSignal
            from datetime import datetime

            loop = get_bot_loop()
            print(f"QQ_NOTIFY: Bot loop = {loop}", flush=True)
            if loop is None:
                logger.warning("QQ bot loop 未初始化")
                return

            async def _send():
                print("QQ_NOTIFY: _send 协程开始", flush=True)
                try:
                    notifier = get_notifier()
                    print(f"QQ_NOTIFY: notifier status: enabled={notifier.is_enabled()}, running={notifier._running}", flush=True)
                    logger.info(f"QQ notifier: enabled={notifier.is_enabled()}, running={notifier._running}")
                    if notifier.is_enabled() and notifier._running:
                        signal = TradeSignal(
                            symbol="", direction="", entry_price=0.0,
                            stop_loss=0.0, take_profit=0.0, confidence=0,
                            summary="[PA Agent] 已启动运行，持续跟踪分析中...",
                            timeframe="", timestamp=datetime.now()
                        )
                        print("QQ_NOTIFY: 准备发送消息", flush=True)
                        result = await notifier.send(signal)
                        print(f"QQ_NOTIFY: 发送结果 = {result}", flush=True)
                        if result:
                            logger.info("QQ 上线通知已发送")
                        else:
                            logger.warning("QQ 上线通知发送失败")
                    else:
                        logger.warning("QQ bot 未就绪，跳过通知")
                except Exception as e:
                    print(f"QQ_NOTIFY: 异常: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    logger.error(f"QQ 上线通知异常: {e}")

            # 使用 bot 的事件循环从外部线程调度协程
            asyncio.run_coroutine_threadsafe(_send(), loop)
            logger.info("QQ 通知任务已调度到 bot 事件循环")

        start_bot()  # 启动 QQ 机器人
        t = threading.Thread(target=_delayed_notify, daemon=True)
        t.start()
        print(f"QQ_NOTIFY: 通知线程已启动，auto_start={auto_start}", flush=True)

    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
