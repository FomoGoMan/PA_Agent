"""Signal pre-filter: skip LLM analysis when market conditions are unfavorable."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pa_agent.data.base import KlineFrame

logger = logging.getLogger(__name__)

# ── Default threshold constants ─────────────────────────────────────────────

DEFAULT_ATR_MIN: float = 0.5
DEFAULT_CHANNEL_WIDTH_ATR: float = 0.5
DEFAULT_TREND_ATR_SLOPE: float = 0.3
DEFAULT_OVERLAP_THRESHOLD: float = 0.65


@dataclass(frozen=True)
class PreFilterResult:
    """Result of signal pre-filter evaluation."""

    ok: bool
    reason: str
    skipped_by: str | None  # e.g. "atr_too_low"
    detail: str | None  # e.g. "ATR=0.23 < min=0.50"


def check_signal_filter(
    frame: KlineFrame,
    *,
    atr_min: float = DEFAULT_ATR_MIN,
    channel_width_atr: float = DEFAULT_CHANNEL_WIDTH_ATR,
    trend_atr_slope: float = DEFAULT_TREND_ATR_SLOPE,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> PreFilterResult:
    """Evaluate 3 pre-filter rules on the KlineFrame.

    Rules (evaluated in order, short-circuits on first failure):
      1. ATR threshold  — skip if ATR too low (low volatility)
      2. Channel width  — skip if channel too narrow
      3. Trend detection — skip if market is choppy (high overlap, flat EMA)

    Parameters
    ----------
    frame : KlineFrame
        The analysis frame with indicators attached.
    atr_min : float
        Minimum ATR in price units.
    channel_width_atr : float
        Minimum channel width as ATR ratio.
    trend_atr_slope : float
        Minimum EMA slope in ATR units (×period).
    overlap_threshold : float
        Maximum allowed overlap ratio before marking as choppy.

    Returns
    -------
    PreFilterResult
        ok=True if analysis should proceed, ok=False if skipped.
    """
    bars = frame.bars
    indicators = frame.indicators
    atr14 = indicators.atr14 if indicators else ()
    ema20 = indicators.ema20 if indicators else ()

    # ── Rule 1: ATR threshold ────────────────────────────────────────────────
    try:
        if atr14 and len(atr14) >= 1 and not math.isnan(atr14[0]):
            atr_val = float(atr14[0])
            if atr_val < atr_min:
                return PreFilterResult(
                    ok=False,
                    reason=f"ATR={atr_val:.4f} < 阈值{atr_min}，波动过低，跳过分析",
                    skipped_by="atr_too_low",
                    detail=f"ATR={atr_val:.4f} < min={atr_min}",
                )
    except (TypeError, ValueError):
        pass

    # ── Rule 2: Channel width ───────────────────────────────────────────────
    try:
        if bars and atr14 and len(atr14) >= 1 and not math.isnan(atr14[0]):
            atr_val = float(atr14[0])
            if atr_val > 0:
                # Use last 20 bars for channel width
                recent_bars = list(bars)[:20]
                highs = [float(b.high) for b in recent_bars if hasattr(b, "high")]
                lows = [float(b.low) for b in recent_bars if hasattr(b, "low")]
                if highs and lows:
                    channel = max(highs) - min(lows)
                    ratio = channel / atr_val
                    if ratio < channel_width_atr:
                        return PreFilterResult(
                            ok=False,
                            reason=f"通道宽度{ratio:.2f}×ATR < 阈值{channel_width_atr}，通道过窄，跳过分析",
                            skipped_by="channel_too_narrow",
                            detail=f"channel={channel:.4f}, ATR={atr_val:.4f}, ratio={ratio:.2f}",
                        )
    except (TypeError, ValueError):
        pass

    # ── Rule 3: Trend detection (choppy market) ──────────────────────────────
    try:
        if ema20 and len(ema20) >= 2 and not math.isnan(ema20[0]):
            # Compare K1 to K10 EMA for slope
            k = min(10, len(ema20) - 1)
            if k >= 1 and not math.isnan(ema20[k]):
                slope = float(ema20[0]) - float(ema20[k])

                # Get ATR for scaling threshold
                slope_thr = trend_atr_slope
                if atr14 and len(atr14) >= 1 and not math.isnan(atr14[0]):
                    slope_thr = trend_atr_slope * float(atr14[0])

                if abs(slope) < slope_thr:
                    # Check overlap ratio as secondary choppy indicator
                    overlap = _mean_overlap_ratio(bars, 8) if bars else None
                    if overlap is not None and overlap > overlap_threshold:
                        return PreFilterResult(
                            ok=False,
                            reason=f"市场震荡（EMA斜率{slope:.4f}平缓 + 重叠率{overlap:.2f}>0.65），跳过分析",
                            skipped_by="choppy_market",
                            detail=f"slope={slope:.4f}, overlap={overlap:.2f}",
                        )
    except (TypeError, ValueError):
        pass

    return PreFilterResult(ok=True, reason="", skipped_by=None, detail=None)


def _mean_overlap_ratio(bars: tuple, W: int = 8) -> float | None:
    """Compute mean overlap_prev_ratio for adjacent bar pairs.

    High overlap ratio indicates a choppy/ranging market.
    """
    window = list(bars)[:W]
    ratios: list[float] = []
    for i in range(len(window) - 1):
        try:
            cur_h = max(float(window[i].high), float(window[i].low))
            cur_l = min(float(window[i].high), float(window[i].low))
            prv_h = max(float(window[i + 1].high), float(window[i + 1].low))
            prv_l = min(float(window[i + 1].high), float(window[i + 1].low))
            overlap = max(0.0, min(cur_h, prv_h) - max(cur_l, prv_l))
            union = max(cur_h, prv_h) - min(cur_l, prv_l)
            if union > 0:
                ratios.append(overlap / union)
        except (TypeError, ValueError, AttributeError):
            continue
    if len(ratios) < 2:
        return None
    return sum(ratios) / len(ratios)