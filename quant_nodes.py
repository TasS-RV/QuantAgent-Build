"""
Pure-math LangGraph nodes — zero LLM calls.

Replaces the LLM-based perception agents with deterministic mathematical equivalents:
  - quant_indicator_node : TA-Lib composite (RSI / MACD / Stoch / WillR / ROC)
  - quant_trend_node     : Linear-regression channel (scipy)
  - quant_pattern_node   : TA-Lib CDL candlestick pattern aggregation

All three nodes write to the same state keys as the LLM agents so the downstream
quant decision node (decision_agent_quant.py) works unchanged.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import talib

from indicator_agent import quantify_indicators_from_kline
from trend_agent import quantify_trend_from_kline


# ─── Indicator node ───────────────────────────────────────────────────────────

def quant_indicator_node(state: dict) -> dict:
    """Compute TA-Lib indicators mathematically. No LLM."""
    metrics = quantify_indicators_from_kline(state["kline_data"])
    report = json.dumps(
        {"llm_analysis": "quant-only", "quantitative_metrics": metrics}, indent=2
    )
    return {"indicator_report": report, "messages": []}


# ─── Trend node ───────────────────────────────────────────────────────────────

def quant_trend_node(state: dict) -> dict:
    """Compute linear-regression channel mathematically. No LLM."""
    metrics = quantify_trend_from_kline(state["kline_data"])
    report = json.dumps(
        {"llm_analysis": "quant-only", "quantitative_metrics": metrics}, indent=2
    )
    return {"trend_report": report, "messages": []}


# ─── Pattern node (TA-Lib CDL) ────────────────────────────────────────────────

def _kline_to_ohlc(kline_data: dict):
    if isinstance(kline_data, dict) and "Open" in kline_data:
        o = np.array(kline_data["Open"],  dtype=float)
        h = np.array(kline_data["High"],  dtype=float)
        l = np.array(kline_data["Low"],   dtype=float)
        c = np.array(kline_data["Close"], dtype=float)
    else:
        rows = list(kline_data)
        o = np.array([r["Open"]  for r in rows], dtype=float)
        h = np.array([r["High"]  for r in rows], dtype=float)
        l = np.array([r["Low"]   for r in rows], dtype=float)
        c = np.array([r["Close"] for r in rows], dtype=float)
    return o, h, l, c


# TA-Lib CDL functions to evaluate.
# Each returns an array where: 100 = bullish, -100 = bearish, 0 = none.
_CDL_PATTERNS = [
    ("Engulfing",          lambda o,h,l,c: talib.CDLENGULFING(o,h,l,c)),
    ("Hammer",             lambda o,h,l,c: talib.CDLHAMMER(o,h,l,c)),
    ("InvertedHammer",     lambda o,h,l,c: talib.CDLINVERTEDHAMMER(o,h,l,c)),
    ("MorningStar",        lambda o,h,l,c: talib.CDLMORNINGSTAR(o,h,l,c)),
    ("Piercing",           lambda o,h,l,c: talib.CDLPIERCING(o,h,l,c)),
    ("ThreeWhiteSoldiers", lambda o,h,l,c: talib.CDLTHREEWHITESOLDIERS(o,h,l,c)),
    ("ShootingStar",       lambda o,h,l,c: talib.CDLSHOOTINGSTAR(o,h,l,c)),
    ("EveningStar",        lambda o,h,l,c: talib.CDLEVENINGSTAR(o,h,l,c)),
    ("DarkCloudCover",     lambda o,h,l,c: talib.CDLDARKCLOUDCOVER(o,h,l,c)),
    ("ThreeBlackCrows",    lambda o,h,l,c: talib.CDLTHREEBLACKCROWS(o,h,l,c)),
    ("Harami",             lambda o,h,l,c: talib.CDLHARAMI(o,h,l,c)),
    ("Marubozu",           lambda o,h,l,c: talib.CDLMARUBOZU(o,h,l,c)),
    ("Doji",               lambda o,h,l,c: talib.CDLDOJI(o,h,l,c)),
    ("Engulfing",          lambda o,h,l,c: talib.CDLENGULFING(o,h,l,c)),
    ("AbandonedBaby",      lambda o,h,l,c: talib.CDLABANDONEDBABY(o,h,l,c)),
]

_MAX_SCORE = len(_CDL_PATTERNS) * 100  # theoretical max (all bullish)


def quantify_candlestick_patterns(kline_data: dict, lookback: int = 5) -> dict:
    """
    Run TA-Lib CDL pattern functions over the last `lookback` candles and
    aggregate into a single directional signal.

    Returns the same JSON schema that the LLM pattern agent produces so the
    decision agent can parse it identically.
    """
    o, h, l, c = _kline_to_ohlc(kline_data)
    if len(c) < 10:
        return {
            "macro_pattern_name": "None",
            "direction": 0,
            "confidence_score": 0.0,
            "justification": "Insufficient candles for pattern detection.",
        }

    triggered_names: list[str] = []
    total_score = 0

    for name, fn in _CDL_PATTERNS:
        try:
            result = fn(o, h, l, c)
            recent = result[-lookback:]
            for val in recent:
                if val != 0:
                    triggered_names.append(name)
                    total_score += int(val)
                    break  # count each pattern once per lookback window
        except Exception:
            continue

    if not triggered_names:
        return {
            "macro_pattern_name": "None",
            "direction": 0,
            "confidence_score": 0.0,
            "justification": "No candlestick patterns detected.",
        }

    direction = 1 if total_score > 0 else (-1 if total_score < 0 else 0)
    # Scale confidence: more patterns firing and higher aggregate = higher confidence
    confidence = round(min(abs(total_score) / _MAX_SCORE * 3, 1.0), 3)  # ×3 to spread range

    unique_names = list(dict.fromkeys(triggered_names))  # preserve order, dedupe
    primary = unique_names[0]
    label = (
        primary if len(unique_names) == 1
        else f"{primary} + {len(unique_names)-1} more"
    )
    justification = f"TA-Lib CDL: {', '.join(unique_names[:4])}. Aggregate score={total_score}."

    return {
        "macro_pattern_name": label,
        "direction": direction,
        "confidence_score": confidence,
        "justification": justification,
    }


def quant_pattern_node(state: dict) -> dict:
    """TA-Lib CDL pattern aggregation. No LLM."""
    result = quantify_candlestick_patterns(state["kline_data"])
    return {"pattern_report": json.dumps(result), "messages": []}
