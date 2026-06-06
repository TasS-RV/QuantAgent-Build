"""
Hard-coded trading constraints & biases (issue #8, Objective 2).

Deterministic rules layered on top of the quantitative/LLM signal — the
"psychology" guardrails Tasin described: e.g. don't buy into a volatility spike,
lean into confirmed breakouts. Every rule is a pure function of (signal, market
context) so it is fully reproducible and unit-testable, and the set applied is
configurable.

A market context is a plain dict, e.g.:
    {"vix": 34.2, "is_bullish_breakout": True, "is_bearish_breakdown": False,
     "price": 101.3, "sma200": 95.0}

apply_constraints() runs the enabled rules in order and returns the adjusted
(direction, confidence). direction is -1 / 0 / +1; confidence is clipped [0,1].
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

DEFAULT_VIX_THRESHOLD = 30.0
BREAKOUT_CONFIDENCE_BOOST = 0.25


def _clip_conf(c: float) -> float:
    return max(0.0, min(1.0, c))


def vix_gate(direction: int, confidence: float, ctx: Dict,
             threshold: float = DEFAULT_VIX_THRESHOLD) -> Tuple[int, float]:
    """No new LONGs while VIX is above `threshold` (risk-off in high vol)."""
    vix = ctx.get("vix")
    if vix is not None and vix > threshold and direction == 1:
        return 0, 0.0
    return direction, confidence


def breakout_bias(direction: int, confidence: float, ctx: Dict,
                  boost: float = BREAKOUT_CONFIDENCE_BOOST) -> Tuple[int, float]:
    """Lean into confirmed breakouts: force/strengthen LONG on a bullish
    breakout, SHORT on a bearish breakdown."""
    if ctx.get("is_bullish_breakout"):
        return 1, _clip_conf(max(confidence, 0.0) + boost)
    if ctx.get("is_bearish_breakdown"):
        return -1, _clip_conf(max(confidence, 0.0) + boost)
    return direction, confidence


def trend_regime_filter(direction: int, confidence: float, ctx: Dict) -> Tuple[int, float]:
    """Never fight a clear 200-day trend (mirror of the backtest trend filter,
    usable in the live decision path)."""
    price, sma200 = ctx.get("price"), ctx.get("sma200")
    if price is not None and sma200 is not None:
        if direction == -1 and price > sma200:
            return 0, 0.0
        if direction == 1 and price < sma200:
            return 0, 0.0
    return direction, confidence


# Registry of named rules. apply_constraints uses this order by default.
RULES: Dict[str, Callable[[int, float, Dict], Tuple[int, float]]] = {
    "vix_gate": vix_gate,
    "breakout_bias": breakout_bias,
    "trend_regime_filter": trend_regime_filter,
}

DEFAULT_RULES: List[str] = ["vix_gate", "breakout_bias"]


def apply_constraints(direction: int, confidence: float, ctx: Optional[Dict] = None,
                      rules: Optional[List[str]] = None) -> Tuple[int, float]:
    """
    Apply the enabled hard constraints in order. Returns (direction, confidence).
    With no context (or no rules) the signal passes through unchanged.
    """
    if not ctx:
        return direction, _clip_conf(confidence)
    for name in (rules if rules is not None else DEFAULT_RULES):
        fn = RULES.get(name)
        if fn is not None:
            direction, confidence = fn(direction, confidence, ctx)
            if direction == 0:
                confidence = 0.0
    return direction, _clip_conf(confidence)


def detect_breakout(kline_data: dict, lookback: int = 20) -> Dict:
    """
    Lightweight, dependency-free breakout detector for building a market context.
    Bullish breakout = last close above the prior `lookback`-bar high;
    bearish breakdown = last close below the prior `lookback`-bar low.
    """
    closes = list(kline_data.get("Close", []))
    highs = list(kline_data.get("High", closes))
    lows = list(kline_data.get("Low", closes))
    if len(closes) < lookback + 1:
        return {"is_bullish_breakout": False, "is_bearish_breakdown": False}
    prior_high = max(highs[-lookback - 1:-1])
    prior_low = min(lows[-lookback - 1:-1])
    last = closes[-1]
    return {
        "is_bullish_breakout": last > prior_high,
        "is_bearish_breakdown": last < prior_low,
    }
