"""JSON schemas for Stage 1 and Stage 2 AI outputs."""
from __future__ import annotations

# ── Stage 1 schema ────────────────────────────────────────────────────────────

STAGE1_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "cycle_position",
        "direction",
        "diagnosis_confidence",
        "market_phase",
        "detected_patterns",
        "key_signals",
        "htf_context",
        "entry_setup",
        "strategy_files_needed",
    ],
    "properties": {
        "cycle_position": {
            "type": "string",
            "enum": [
                "spike", "micro_channel", "tight_channel", "normal_channel",
                "broad_channel", "trending_tr", "trading_range", "extreme_tr", "unknown",
            ],
        },
        "alternative_cycle_position": {"type": ["string", "null"]},
        "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
        "diagnosis_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "spike_stage": {
            "type": ["string", "null"],
            "enum": ["active", "ending", "transitioning", None],
        },
        "market_phase": {"type": "string", "enum": ["stable", "transitioning"]},
        "transition_risk": {
            "type": ["string", "null"],
            "enum": ["high", "medium", "low", None],
        },
        "detected_patterns": {"type": "array", "items": {"type": "string"}},
        "key_signals": {"type": "array", "items": {"type": "string"}},
        "htf_context": {"type": "string"},
        "entry_setup": {"type": "string"},
        "strategy_files_needed": {"type": "array", "items": {"type": "string"}},
        "risk_warning": {"type": "string"},
    },
    "allOf": [
        # spike / micro_channel require spike_stage to be non-null
        {
            "if": {
                "properties": {
                    "cycle_position": {"enum": ["spike", "micro_channel"]}
                },
                "required": ["cycle_position"],
            },
            "then": {
                "properties": {
                    "spike_stage": {"type": "string", "enum": ["active", "ending", "transitioning"]}
                },
                "required": ["spike_stage"],
            },
        },
        # transitioning market_phase requires transition_risk to be non-null
        {
            "if": {
                "properties": {"market_phase": {"const": "transitioning"}},
                "required": ["market_phase"],
            },
            "then": {
                "properties": {
                    "transition_risk": {"type": "string", "enum": ["high", "medium", "low"]}
                },
                "required": ["transition_risk"],
            },
        },
    ],
    "additionalProperties": True,
}


# ── Stage 2 schema ────────────────────────────────────────────────────────────

_DECISION_BASE: dict = {
    "type": "object",
    "required": [
        "order_type",
        "reasoning",
        "diagnosis_confidence",
        "diagnosis_confidence_reasoning",
        "trade_confidence",
        "trade_confidence_reasoning",
        "key_factors",
        "watch_points",
        "risk_assessment",
    ],
    "properties": {
        "order_direction": {"type": ["string", "null"]},
        "order_type": {
            "type": "string",
            "enum": ["限价单", "突破单", "市价单", "不下单"],
        },
        "entry_price": {"type": ["number", "null"]},
        "take_profit_price": {"type": ["number", "null"]},
        "stop_loss_price": {"type": ["number", "null"]},
        "reasoning": {"type": "string"},
        "diagnosis_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "diagnosis_confidence_reasoning": {"type": "string"},
        "trade_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "trade_confidence_reasoning": {"type": "string"},
        "key_factors": {"type": "array", "items": {"type": "string"}},
        "watch_points": {"type": "array", "items": {"type": "string"}},
        "risk_assessment": {"type": "string"},
        "invalidation_condition": {"type": ["string", "null"]},
    },
    "allOf": [
        # 不下单 → all price fields and direction must be null
        {
            "if": {
                "properties": {"order_type": {"const": "不下单"}},
                "required": ["order_type"],
            },
            "then": {
                "properties": {
                    "entry_price": {"type": "null"},
                    "take_profit_price": {"type": "null"},
                    "stop_loss_price": {"type": "null"},
                    "order_direction": {"type": "null"},
                },
            },
        },
        # 有下单 → price fields must be numbers, direction must be 做多/做空
        {
            "if": {
                "properties": {
                    "order_type": {"enum": ["限价单", "突破单", "市价单"]}
                },
                "required": ["order_type"],
            },
            "then": {
                "properties": {
                    "entry_price": {"type": "number"},
                    "take_profit_price": {"type": "number"},
                    "stop_loss_price": {"type": "number"},
                    "order_direction": {"type": "string", "enum": ["做多", "做空"]},
                },
                "required": ["entry_price", "take_profit_price", "stop_loss_price", "order_direction"],
            },
        },
    ],
    "additionalProperties": True,
}

STAGE2_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["decision", "diagnosis_summary"],
    "properties": {
        "decision": _DECISION_BASE,
        "diagnosis_summary": {
            "type": "object",
            "required": ["cycle_position", "direction", "key_signals"],
            "properties": {
                "cycle_position": {"type": "string"},
                "direction": {"type": "string"},
                "key_signals": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    "additionalProperties": True,
}
