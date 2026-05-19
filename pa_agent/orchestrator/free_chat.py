"""FreeChatSession — post-analysis free-chat session.

Maintains a conversation history anchored to a completed two-stage
AnalysisRecord and sends follow-up messages to the DeepSeek API.

Design reference: design.md §B.17
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from pa_agent.ai.deepseek_client import DeepSeekClient
    from pa_agent.ai.prompt_assembler import PromptAssembler
    from pa_agent.ai.session_ledger import SessionTokenLedger
    from pa_agent.config.settings import Settings
    from pa_agent.records.pending_writer import PendingWriter

from pa_agent.ai.deepseek_client import AIReply
from pa_agent.records.schema import AnalysisRecord, FollowupTurn
from pa_agent.util.threading import CancelToken
from pa_agent.util.timefmt import now_local_ms

logger = logging.getLogger(__name__)


def _derive_record_id(record: AnalysisRecord) -> str:
    """Derive the record basename (without extension) from an AnalysisRecord.

    Uses the same logic as ``_build_basename`` in pending_writer.py.
    """
    from datetime import datetime, timezone

    ms = record.meta.timestamp_local_ms
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone()
    ts_str = dt.strftime("%Y-%m-%d_%H-%m-%S")
    symbol = record.meta.symbol
    timeframe = record.meta.timeframe
    return f"{ts_str}_{symbol}_{timeframe}"


def _strip_reasoning(message: dict) -> dict:
    """Return a copy of *message* without the ``reasoning_content`` key."""
    return {k: v for k, v in message.items() if k != "reasoning_content"}


class FreeChatSession:
    """Manages a free-chat conversation anchored to a completed analysis.

    Parameters
    ----------
    base_record:
        The fully completed AnalysisRecord from the two-stage pipeline.
    client:
        DeepSeekClient instance for API calls.
    assembler:
        PromptAssembler (kept for future use; system prompt is taken
        directly from base_record.stage2_messages[0]).
    pending_writer:
        PendingWriter for appending FollowupTurn entries to the JSONL
        sidecar file.
    ledger:
        SessionTokenLedger for accumulating token usage and cost.
    settings:
        Optional Settings object; used for ``reasoning_effort`` forwarding.
    kline_snapshot_fn:
        Optional callable that returns the latest closed K-line data as a
        text table string.  Called on each ``send()`` so the AI always
        sees the most recent market data.
    """

    #: When True, ``reasoning_content`` is preserved in assistant messages
    #: sent back to the API (for future tool-call scenarios).
    keep_reasoning_in_resend: bool = False

    def __init__(
        self,
        base_record: AnalysisRecord,
        client: "DeepSeekClient",
        assembler: "PromptAssembler",
        pending_writer: "PendingWriter",
        ledger: "SessionTokenLedger",
        settings: Optional["Settings"] = None,
        kline_snapshot_fn: Optional[Callable[[], str]] = None,
    ) -> None:
        self._base_record = base_record
        self._client = client
        self._assembler = assembler
        self._pending_writer = pending_writer
        self._ledger = ledger
        self._settings = settings
        self._kline_snapshot_fn = kline_snapshot_fn

        # Turn counter — incremented before each send so the first turn is 1.
        self._turn: int = 0

        # Full history including reasoning_content (for UI display and
        # persistence).  Each entry is a plain dict with at least
        # ``role`` and ``content``; assistant entries also carry
        # ``reasoning_content``.
        self._history_full: list[dict] = []

        # Derived record ID used as the JSONL sidecar basename.
        self._record_id: str = _derive_record_id(base_record)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def history_full(self) -> list[dict]:
        """Read-only view of the full message history (includes reasoning)."""
        return list(self._history_full)

    @property
    def record_id(self) -> str:
        """The record basename used for the JSONL sidecar file."""
        return self._record_id

    def send(
        self,
        user_text: str,
        cancel_token: CancelToken,
        on_reasoning_token: "Callable[[str], None] | None" = None,
        on_content_token: "Callable[[str], None] | None" = None,
    ) -> AIReply:
        """Send *user_text* to the AI and return the reply.

        Steps
        -----
        1. Build ``history_for_api`` from:
           - Stage-2 system prompt (from ``base_record.stage2_messages[0]``)
           - Stage-2 user message (``base_record.stage2_messages[-1]``)
           - Stage-2 assistant response (from ``base_record.stage2_response``,
             with ``reasoning_content`` stripped unless
             ``keep_reasoning_in_resend`` is True)
           - All previous free-chat turns (user + assistant, same stripping
             rule)
           - New user message
        2. Call ``client.chat(history_for_api, cancel_token=cancel_token)``.
        3. Append to ``_history_full`` (with ``reasoning_content`` preserved).
        4. Call ``ledger.add(reply.usage)`` and
           ``pending_writer.append_followup(record_id, turn)``.
        5. Return the AIReply.

        When *cancel_token* is already set before the call, a
        ``FollowupTurn`` with ``cancelled=True`` is persisted and the
        ``CancelledError`` is re-raised.
        """
        self._turn += 1
        turn_number = self._turn

        # ── 1. Build history_for_api ──────────────────────────────────────────
        history_for_api: list[dict] = []

        # System prompt — taken from stage2_messages[0]
        stage2_messages = self._base_record.stage2_messages
        if stage2_messages:
            system_msg = stage2_messages[0]
            history_for_api.append({"role": "system", "content": system_msg["content"]})

        # Stage-2 user message — last message in stage2_messages
        if len(stage2_messages) >= 2:
            user_msg_s2 = stage2_messages[-1]
            history_for_api.append({"role": "user", "content": user_msg_s2["content"]})

        # Stage-2 assistant response
        stage2_response = self._base_record.stage2_response or {}
        stage2_content = stage2_response.get("content", "")
        stage2_reasoning = stage2_response.get("reasoning_content", "")

        assistant_s2: dict = {"role": "assistant", "content": stage2_content}
        if self.keep_reasoning_in_resend and stage2_reasoning:
            assistant_s2["reasoning_content"] = stage2_reasoning
        history_for_api.append(assistant_s2)

        # Previous free-chat turns from history_full
        for msg in self._history_full:
            if msg["role"] == "user":
                history_for_api.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                if self.keep_reasoning_in_resend and msg.get("reasoning_content"):
                    history_for_api.append({
                        "role": "assistant",
                        "content": msg["content"],
                        "reasoning_content": msg["reasoning_content"],
                    })
                else:
                    history_for_api.append({
                        "role": "assistant",
                        "content": msg["content"],
                    })

        # New user message — prepend latest K-line snapshot if available
        user_content = user_text
        if self._kline_snapshot_fn is not None:
            try:
                kline_table = self._kline_snapshot_fn()
                if kline_table:
                    user_content = (
                        f"## 最新已收盘K线数据\n\n{kline_table}\n\n"
                        f"---\n\n{user_text}"
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("kline_snapshot_fn failed: %s", exc)

        history_for_api.append({"role": "user", "content": user_content})

        # ── 2. Resolve reasoning_effort ───────────────────────────────────────
        reasoning_effort = "max"
        if self._settings is not None:
            reasoning_effort = getattr(
                self._settings.provider, "reasoning_effort", "max"
            )

        # ── 3. Check cancellation before API call ─────────────────────────────
        from pa_agent.ai.deepseek_client import CancelledError

        if cancel_token.is_set():
            # Persist a cancelled turn and re-raise
            cancelled_turn = FollowupTurn(
                turn=turn_number,
                ts_ms=now_local_ms(),
                user=user_text,
                ai_content="",
                ai_reasoning=None,
                usage={
                    "prompt_tokens": 0,
                    "cached_prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                cancelled=True,
            )
            self._pending_writer.append_followup(self._record_id, cancelled_turn)
            raise CancelledError("FreeChatSession.send cancelled before API call")

        # ── 4. Call the API (streaming) ───────────────────────────────────────
        try:
            reply = self._client.stream_chat(
                history_for_api,
                on_reasoning_token=on_reasoning_token,
                on_content_token=on_content_token,
                cancel_token=cancel_token,
                reasoning_effort=reasoning_effort,
            )
        except CancelledError:
            # Persist a cancelled turn and re-raise
            cancelled_turn = FollowupTurn(
                turn=turn_number,
                ts_ms=now_local_ms(),
                user=user_text,
                ai_content="",
                ai_reasoning=None,
                usage={
                    "prompt_tokens": 0,
                    "cached_prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                cancelled=True,
            )
            self._pending_writer.append_followup(self._record_id, cancelled_turn)
            raise

        # ── 5. Append to history_full (with reasoning preserved) ──────────────
        self._history_full.append({"role": "user", "content": user_text})
        self._history_full.append({
            "role": "assistant",
            "content": reply.content,
            "reasoning_content": reply.reasoning_content,
        })

        # ── 6. Accumulate usage in ledger ─────────────────────────────────────
        self._ledger.add(reply.usage)

        # ── 7. Persist the followup turn ──────────────────────────────────────
        usage_dict = {
            "prompt_tokens": reply.usage.prompt_tokens,
            "cached_prompt_tokens": reply.usage.cached_prompt_tokens,
            "completion_tokens": reply.usage.completion_tokens,
            "total_tokens": reply.usage.total_tokens,
        }
        followup_turn = FollowupTurn(
            turn=turn_number,
            ts_ms=now_local_ms(),
            user=user_text,
            ai_content=reply.content,
            ai_reasoning=reply.reasoning_content or None,
            usage=usage_dict,
            cancelled=False,
        )
        self._pending_writer.append_followup(self._record_id, followup_turn)

        logger.debug(
            "FreeChatSession.send: turn=%d tokens=%d/%d",
            turn_number,
            reply.usage.prompt_tokens,
            reply.usage.completion_tokens,
        )

        return reply
