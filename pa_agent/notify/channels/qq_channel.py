"""QQ notification channel using QQ Open Platform HTTP API."""
from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from pa_agent.notify.channels.base import BaseNotifier, TradeSignal

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "qq.json"
_API_BASE = "https://api.sgroup.qq.com"


class QQNotifier(BaseNotifier):
    """QQ notification channel using QQ Open Platform API.

    Supports sending messages to users (C2C) and groups via HTTP API.
    Requires a QQ Official Bot with app_id and secret.
    """

    name = "qq"
    display_name = "QQ"

    def __init__(self) -> None:
        self._config: dict = {}
        self._access_token: str | None = None
        self._token_expires_at: float = 0
        self._load_config()

    def _load_config(self) -> None:
        """Load QQ configuration from nanobot config or local qq.json."""
        # Try nanobot config first (~/.nanobot/config.json)
        nanobot_config = Path.home() / ".nanobot" / "config.json"
        if nanobot_config.exists():
            try:
                data = json.loads(nanobot_config.read_text(encoding="utf-8"))
                qq_conf = data.get("qq", {})
                if qq_conf.get("enabled"):
                    self._config = {
                        "enabled": True,
                        "app_id": qq_conf.get("appId", ""),
                        "secret": qq_conf.get("secret", ""),
                        "notify_qq": "",
                        "notify_group": "",
                    }
                    logger.info("QQ config loaded from nanobot config (~/.nanobot/config.json)")
                    return
            except Exception as exc:
                logger.debug("Failed to load nanobot config: %s", exc)

        # Fallback to local qq.json
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

    def _get_access_token(self) -> str | None:
        """Get or refresh access token."""
        import time

        # Return cached token if still valid
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token

        app_id = self._config.get("app_id", "")
        secret = self._config.get("secret", "")

        if not app_id or not secret:
            logger.warning("QQ: app_id or secret not configured")
            return None

        token_url = f"{_API_BASE}/oauth2/authorize_robot_token"
        try:
            resp = requests.post(
                token_url,
                json={
                    "app_id": app_id,
                    "client_secret": secret,
                    "grant_type": "client_credentials",
                },
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            result = resp.json()
            if result.get("code") == 0:
                self._access_token = result.get("access_token")
                expires_in = result.get("expires_in", 7200)
                self._token_expires_at = time.time() + expires_in
                return self._access_token
            else:
                logger.error("QQ: failed to get access token: %s", result)
                return None
        except Exception as exc:
            logger.error("QQ: failed to get access token: %s", exc)
            return None

    async def send(self, signal: TradeSignal) -> bool:
        """Send a trade signal to QQ.

        Sends to configured notify_qq (private) and/or notify_group.
        """
        if not self.is_enabled():
            logger.debug("QQ notification disabled")
            return False

        token = self._get_access_token()
        if not token:
            logger.error("QQ: cannot send without access token")
            return False

        message = signal.format_simple()
        results = []

        notify_qq = self._config.get("notify_qq", "")
        if notify_qq:
            ok = await self._send_c2c_message(token, notify_qq, message)
            results.append(ok)

        notify_group = self._config.get("notify_group", "")
        if notify_group:
            ok = await self._send_group_message(token, notify_group, message)
            results.append(ok)

        return any(results) if results else False

    async def _send_c2c_message(self, token: str, openid: str, content: str) -> bool:
        """Send C2C (private) message to user."""
        import aiohttp

        url = f"{_API_BASE}/v2/c2c_messages"
        payload = {
            "receive_id": openid,
            "receive_id_type": "openid",
            "msg_type": 0,
            "content": json.dumps({"text": content}),
            "msg_seq": int(openid[-8:], 16) if openid else 1,
        }
        try:
            async with aiohttp.ClientSession() as session:
                resp = await session.post(
                    url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"QQBot {token}",
                    },
                    timeout=10.0,
                )
                result = await resp.json()
                if result.get("code") == 0:
                    logger.info("QQ C2C message sent to %s", openid)
                    return True
                else:
                    logger.error("QQ C2C message failed: %s", result)
                    return False
        except Exception as exc:
            logger.error("QQ C2C message error: %s", exc)
            return False

    async def _send_group_message(self, token: str, group_openid: str, content: str) -> bool:
        """Send group message."""
        import aiohttp

        url = f"{_API_BASE}/v2/group_messages"
        payload = {
            "receive_id": group_openid,
            "receive_id_type": "group_openid",
            "msg_type": 0,
            "content": json.dumps({"text": content}),
            "msg_seq": int(group_openid[-8:], 16) if group_openid else 1,
        }
        try:
            async with aiohttp.ClientSession() as session:
                resp = await session.post(
                    url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"QQBot {token}",
                    },
                    timeout=10.0,
                )
                result = await resp.json()
                if result.get("code") == 0:
                    logger.info("QQ group message sent to %s", group_openid)
                    return True
                else:
                    logger.error("QQ group message failed: %s", result)
                    return False
        except Exception as exc:
            logger.error("QQ group message error: %s", exc)
            return False


# Global notifier instance
_notifier: QQNotifier | None = None


def get_notifier() -> QQNotifier:
    """Get the global QQ notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = QQNotifier()
    return _notifier