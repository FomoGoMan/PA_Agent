"""QQ notification channel using botpy SDK with WebSocket connection."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from pa_agent.notify.channels.base import BaseNotifier, TradeSignal

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "qq.json"

# Check if botpy is available
try:
    import botpy
    from botpy import Intents
    from botpy.client import Client
    import aiohttp
    QQ_AVAILABLE = True
except ImportError:
    QQ_AVAILABLE = False
    logger.warning("QQ notification: botpy not installed. Run: pip install qq-botpy")


class QQNotifier(BaseNotifier):
    """QQ notification channel using botpy SDK with WebSocket connection.

    Maintains a persistent WebSocket connection with automatic heartbeat
    and reconnection support.
    """

    name = "qq"
    display_name = "QQ"

    def __init__(self) -> None:
        self._config: dict = {}
        self._client: Client | None = None
        self._running = False
        self._reconnect_delay = 5
        self._load_config()

    def _load_config(self) -> None:
        """Load QQ configuration from qq.json."""
        if _CONFIG_PATH.exists():
            try:
                self._config = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
                logger.info("QQ config loaded from %s", _CONFIG_PATH)
            except Exception as exc:
                logger.warning("Failed to load QQ config: %s", exc)
                self._config = {}
        else:
            self._config = {}
            logger.debug("QQ config file not found at %s", _CONFIG_PATH)

    def is_enabled(self) -> bool:
        """Check if QQ notification is enabled."""
        return bool(self._config.get("enabled", False))

    async def start(self) -> None:
        """Start the QQ bot with WebSocket connection and auto-reconnect."""
        if not QQ_AVAILABLE:
            logger.error("QQ notification: botpy not installed. Run: pip install qq-botpy")
            return

        app_id = self._config.get("app_id", "")
        secret = self._config.get("secret", "")

        if not app_id or not secret:
            logger.error("QQ notification: app_id or secret not configured")
            return

        self._running = True

        # Create botpy Client
        intents = Intents(public_messages=True, direct_message=True)

        class _Bot(Client):
            async def on_ready(self):
                logger.info("QQ bot ready: %s", self.robot.name)

            async def on_c2c_message_create(self, message):
                pass  # We only send, don't receive

            async def on_group_at_message_create(self, message):
                pass  # We only send, don't receive

            async def on_direct_message_create(self, message):
                pass  # We only send, don't receive

        self._client = _Bot(intents=intents, ext_handlers=False)
        logger.info("QQ bot starting with WebSocket connection...")

        # Run bot with reconnect loop
        while self._running:
            try:
                await self._client.start(appid=app_id, secret=secret)
            except Exception as e:
                logger.warning("QQ bot error: %s", e)
            if self._running:
                logger.info("Reconnecting QQ bot in %d seconds...", self._reconnect_delay)
                await asyncio.sleep(self._reconnect_delay)

    async def stop(self) -> None:
        """Stop the QQ bot and cleanup."""
        self._running = False
        if self._client:
            with suppress(Exception):
                await self._client.close()
        self._client = None
        logger.info("QQ bot stopped")

    async def send(self, signal: TradeSignal) -> bool:
        """Send a trade signal to QQ.

        Sends to configured notify_qq (private) and/or notify_group.
        """
        if not self.is_enabled():
            logger.debug("QQ notification disabled")
            return False

        if self._client is None or not self._running:
            logger.warning("QQ bot not connected, cannot send message")
            return False

        # 如果信号没有有效的交易数据（品种为空），发送纯文本摘要
        if signal.symbol or signal.direction:
            message = signal.format_simple()
        else:
            message = signal.summary
        results = []

        notify_qq = self._config.get("notify_qq", "")
        if notify_qq:
            ok = await self._send_c2c_message(notify_qq, message)
            results.append(ok)

        notify_group = self._config.get("notify_group", "")
        if notify_group:
            ok = await self._send_group_message(notify_group, message)
            results.append(ok)

        return any(results) if results else False

    async def _send_c2c_message(self, openid: str, content: str) -> bool:
        """Send C2C (private) message to user."""
        if self._client is None:
            return False

        try:
            await self._client.api.post_c2c_message(
                openid=openid,
                msg_type=0,
                content=content,
            )
            logger.info("QQ C2C message sent to %s", openid)
            return True
        except Exception as exc:
            logger.error("QQ C2C message error: %s", exc)
            return False

    async def _send_group_message(self, group_openid: str, content: str) -> bool:
        """Send group message."""
        if self._client is None:
            return False

        try:
            await self._client.api.post_group_message(
                group_openid=group_openid,
                msg_type=0,
                content=content,
            )
            logger.info("QQ group message sent to %s", group_openid)
            return True
        except Exception as exc:
            logger.error("QQ group message error: %s", exc)
            return False


# Global notifier instance
_notifier: QQNotifier | None = None
_bot_task: asyncio.Task | None = None
_bot_loop: asyncio.AbstractEventLoop | None = None


async def _start_bot_async() -> None:
    """Start the QQ bot in background."""
    global _notifier, _bot_loop
    if _notifier is None:
        _notifier = QQNotifier()
    if _notifier.is_enabled():
        await _notifier.start()


def start_bot() -> None:
    """Start the QQ bot in a background thread."""
    global _bot_task, _bot_loop
    if _bot_task is None or _bot_task.done():
        _bot_loop = asyncio.new_event_loop()
        # Run in background thread
        import threading
        t = threading.Thread(target=lambda: _bot_loop.run_until_complete(_start_bot_async()), daemon=True)
        t.start()


def get_bot_loop() -> asyncio.AbstractEventLoop | None:
    """Get the bot's event loop for scheduling coroutines from other threads."""
    return _bot_loop


async def stop_bot_async() -> None:
    """Stop the QQ bot."""
    global _notifier
    if _notifier:
        await _notifier.stop()


def get_notifier() -> QQNotifier:
    """Get the global QQ notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = QQNotifier()
    return _notifier