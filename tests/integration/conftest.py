"""Shared test infrastructure for TwoStageOrchestrator integration tests."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from pa_agent.data.base import KlineBar, KlineFrame, IndicatorBundle
from pa_agent.orchestrator.exception_counter import ExceptionCounter


# ── Valid JSON payloads ───────────────────────────────────────────────────────

VALID_STAGE1 = {
    "cycle_position": "normal_channel",
    "direction": "bullish",
    "diagnosis_confidence": 75,
    "market_phase": "stable",
    "detected_patterns": [],
    "key_signals": ["signal1"],
    "htf_context": "bullish trend",
    "entry_setup": "buy on pullback",
    "strategy_files_needed": ["上涨通道分析识别.txt"],
}

VALID_STAGE2 = {
    "decision": {
        "order_direction": "做多",
        "order_type": "限价单",
        "entry_price": 2000.0,
        "take_profit_price": 2050.0,
        "stop_loss_price": 1980.0,
        "reasoning": "Strong bullish signal",
        "diagnosis_confidence": 75,
        "diagnosis_confidence_reasoning": "周期位置明确，趋势方向清晰",
        "trade_confidence": 70,
        "trade_confidence_reasoning": "入场信号明确，风险回报比合理",
        "key_factors": ["factor1"],
        "watch_points": ["watch1"],
        "risk_assessment": "low risk",
        "invalidation_condition": "break below 1980",
    },
    "diagnosis_summary": {
        "cycle_position": "normal_channel",
        "direction": "bullish",
        "key_signals": ["signal1"],
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_reply(content_dict: dict) -> MagicMock:
    """Build a mock AIReply from a content dict."""
    reply = MagicMock()
    reply.content = json.dumps(content_dict)
    reply.raw = {"content": reply.content}
    reply.usage = MagicMock()
    reply.usage.prompt_tokens = 100
    reply.usage.completion_tokens = 50
    reply.usage.cached_prompt_tokens = 0
    reply.usage.total_tokens = 150
    return reply


def make_frame() -> KlineFrame:
    """Build a minimal KlineFrame for testing."""
    bars = tuple(
        KlineBar(
            seq=i + 1,
            ts_open=1000 - i * 60000,
            open=2000.0,
            high=2010.0,
            low=1990.0,
            close=2005.0,
            volume=100.0,
            closed=(i > 0),
        )
        for i in range(5)
    )
    indicators = IndicatorBundle(
        ema20=tuple([2000.0] * 5),
        atr14=tuple([10.0] * 5),
    )
    return KlineFrame(
        symbol="XAUUSD",
        timeframe="1h",
        bars=bars,
        snapshot_ts_local_ms=1700000000000,
        indicators=indicators,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def frame():
    return make_frame()


@pytest.fixture
def exc_counter(tmp_path):
    counter = ExceptionCounter(state_path=tmp_path / "exception_state.json")
    counter.load()
    return counter


@pytest.fixture
def pending_writer():
    return MagicMock()


@pytest.fixture
def assembler():
    mock = MagicMock()
    mock.build_stage1.return_value = [{"role": "system", "content": "test"}]
    mock.build_stage2.return_value = [{"role": "system", "content": "test"}]
    return mock


@pytest.fixture
def exp_reader():
    mock = MagicMock()
    mock.read_top5.return_value = []
    return mock
