"""Base notifier interface for multi-channel notifications."""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Trading signal notification data."""

    symbol: str
    direction: str  # "long" / "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: int  # 0-100
    summary: str
    order_type: str = "market"  # "limit" / "breakout" / "market"
    timeframe: str = "1h"
    timestamp: datetime = field(default_factory=datetime.now)
    extra: dict[str, Any] = field(default_factory=dict)

    def format_simple(self) -> str:
        """Format as simple text message."""
        direction_zh = "多" if self.direction == "long" else "空"
        sl_diff = self.entry_price - self.stop_loss
        tp_diff = self.take_profit - self.entry_price

        return (
            f"[PA Agent 信号通知]\n"
            f"品种: {self.symbol} {direction_zh}\n"
            f"入场: {self.entry_price:.2f}\n"
            f"止损: {self.stop_loss:.2f} ({sl_diff:+.2f})\n"
            f"止盈: {self.take_profit:.2f} ({tp_diff:+.2f})\n"
            f"置信度: {self.confidence}%\n"
            f"类型: {self.order_type}"
        )


class BaseNotifier(ABC):
    """Abstract base class for notification channels."""

    name: str = "base"
    display_name: str = "Base"

    @abstractmethod
    async def send(self, signal: TradeSignal) -> bool:
        """Send a trade signal notification.

        Parameters
        ----------
        signal : TradeSignal
            The trade signal to send.

        Returns
        -------
        bool
            True if sent successfully, False otherwise.
        """
        ...

    async def send_text(self, text: str, **kwargs) -> bool:
        """Send a plain text message.

        Default implementation calls send() with a minimal TradeSignal.
        Override for custom text handling.
        """
        signal = TradeSignal(
            symbol="",
            direction="",
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            confidence=0,
            summary=text,
            timeframe="",
        )
        return await self.send(signal)


@dataclass
class OutboundMessage:
    """Internal message format for the dispatcher."""

    channel: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0


class MessageDispatcher:
    """Async message dispatcher supporting multiple notification channels.

    Uses a queue-based architecture similar to nanobot's MessageBus.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._channels: dict[str, BaseNotifier] = {}
        self._running = False
        self._task: asyncio.Task[None] | None = None

    def register(self, channel: BaseNotifier) -> None:
        """Register a notification channel."""
        self._channels[channel.name] = channel
        logger.info("Registered notification channel: %s (%s)", channel.name, channel.display_name)

    def unregister(self, name: str) -> None:
        """Unregister a notification channel."""
        if name in self._channels:
            del self._channels[name]
            logger.info("Unregistered notification channel: %s", name)

    async def send(self, channel: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Queue a message for async delivery."""
        msg = OutboundMessage(
            channel=channel,
            content=content,
            metadata=metadata or {},
        )
        await self._queue.put(msg)

    async def start(self) -> None:
        """Start the dispatcher worker."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._worker())
        logger.info("Message dispatcher started")

    async def stop(self) -> None:
        """Stop the dispatcher worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Message dispatcher stopped")

    async def _worker(self) -> None:
        """Main dispatcher worker loop."""
        while self._running:
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            channel = self._channels.get(msg.channel)
            if channel is None:
                logger.warning("No such channel registered: %s", msg.channel)
                continue

            try:
                signal = TradeSignal(
                    symbol=msg.metadata.get("symbol", ""),
                    direction=msg.metadata.get("direction", ""),
                    entry_price=msg.metadata.get("entry_price", 0.0),
                    stop_loss=msg.metadata.get("stop_loss", 0.0),
                    take_profit=msg.metadata.get("take_profit", 0.0),
                    confidence=msg.metadata.get("confidence", 0),
                    summary=msg.content,
                    order_type=msg.metadata.get("order_type", "market"),
                    timeframe=msg.metadata.get("timeframe", "1h"),
                )
                success = await channel.send(signal)
                if success:
                    logger.info("Message sent via %s: %s", msg.channel, msg.content[:50])
                else:
                    logger.warning("Failed to send message via %s", msg.channel)
            except Exception as exc:
                logger.error("Error sending message via %s: %s", msg.channel, exc)


# Global dispatcher instance
_dispatcher: MessageDispatcher | None = None


def get_dispatcher() -> MessageDispatcher:
    """Get the global message dispatcher instance."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = MessageDispatcher()
    return _dispatcher