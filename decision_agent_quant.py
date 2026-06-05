"""
Mathematical decision agent — no LLM calls.
Combines the quantitative outputs of the indicator, trend, and pattern agents
to produce a BUY / HOLD / SELL / SHORT signal with price target and stop-loss.

Signal wiring
─────────────
  indicator_signal  = quantitative_metrics.final_indicator_signal  ∈ [-1, 1]
  trend_signal      = quantitative_metrics.normalized_signal        ∈ [-1, 1]
  pattern_signal    = direction × confidence_score                  ∈ [-1, 1]

  combined = w_ind·ind + w_trend·trend + w_pat·pattern              ∈ [-1, 1]

  combined ≥ BUY_THRESH              → BUY
  combined ≤ SHORT_THRESH (if ok)   → SHORT
  combined ≤ SELL_THRESH             → SELL
  otherwise                          → HOLD

Stop-loss is ATR-based; target uses the nearer S/R level from the trend agent.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS: Dict[str, float] = {
    "indicator": 0.40,
    "trend":     0.40,
    "pattern":   0.20,
}

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "buy":   0.15,   # combined ≥ this  → BUY
    "sell": -0.15,   # combined ≤ this  → SELL
    "short": -0.35,  # combined ≤ this (and allow_short) → SHORT
}


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class TradeDecision:
    ticker:             str
    decision:           str           # BUY | HOLD | SELL | SHORT
    combined_signal:    float         # [-1, 1]
    signal_strength:    str           # Strong | Moderate | Weak
    current_price:      float
    entry_price:        Optional[float]
    unrealized_pnl_pct: Optional[float]
    target_price:       float
    stop_loss:          float
    risk_reward_ratio:  float
    atr:                float
    signal_breakdown:   dict
    decision_rationale: str


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _parse_json_from_text(text: str) -> Optional[dict]:
    """Extract the first JSON object from a string; handles markdown fences."""
    if not text:
        return None
    text = text.strip()
    # Direct parse (clean JSON strings from indicator/trend agents)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Strip ```json ... ``` fences
    fenced = re.sub(r"```(?:json)?\s*", "", text).replace("```", "")
    try:
        return json.loads(fenced)
    except (json.JSONDecodeError, TypeError):
        pass
    # Grab first {...} block (handles LLM prose wrapping)
    match = re.search(r"\{.*?\}", fenced, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _compute_atr(kline_data: dict, period: int = 14) -> float:
    """Average True Range from kline_data (dict-of-lists or list-of-dicts)."""
    try:
        if isinstance(kline_data, dict) and "High" in kline_data:
            high  = np.array(kline_data["High"],  dtype=float)
            low   = np.array(kline_data["Low"],   dtype=float)
            close = np.array(kline_data["Close"], dtype=float)
        else:
            rows  = list(kline_data)
            high  = np.array([r["High"]  for r in rows], dtype=float)
            low   = np.array([r["Low"]   for r in rows], dtype=float)
            close = np.array([r["Close"] for r in rows], dtype=float)

        prev_close = np.concatenate([[close[0]], close[:-1]])
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - prev_close),
            np.abs(low  - prev_close),
        ])
        return float(pd.Series(tr).rolling(period, min_periods=1).mean().iloc[-1])
    except Exception:
        return 0.0


def _last_close(kline_data: dict) -> float:
    try:
        if isinstance(kline_data, dict) and "Close" in kline_data:
            return float(kline_data["Close"][-1])
        return float(list(kline_data)[-1]["Close"])
    except Exception:
        return 0.0


# ─── Signal extractors ────────────────────────────────────────────────────────

def _extract_indicator_signal(report: str) -> Tuple[float, dict]:
    """indicator_report → (final_indicator_signal, component_signals)."""
    data = _parse_json_from_text(report)
    if data is None:
        return 0.0, {}
    qm    = data.get("quantitative_metrics", data)
    final = float(qm.get("final_indicator_signal", 0.0))
    comps = qm.get("component_signals", {})
    return float(np.clip(final, -1.0, 1.0)), comps


def _extract_trend_signal(report: str) -> Tuple[float, dict]:
    """trend_report → (normalized_signal, level dict)."""
    data = _parse_json_from_text(report)
    if data is None:
        return 0.0, {}
    qm     = data.get("quantitative_metrics", data)
    signal = float(qm.get("normalized_signal", 0.0))
    return float(np.clip(signal, -1.0, 1.0)), {
        "trend_direction":  qm.get("trend_direction", "Unknown"),
        "slope":            qm.get("slope", 0.0),
        "channel_position": qm.get("channel_position", 0.5),
        "support_level":    qm.get("support_level"),
        "resistance_level": qm.get("resistance_level"),
        "current_price":    qm.get("current_price"),
    }


def _extract_pattern_signal(report: str) -> Tuple[float, dict]:
    """pattern_report (LLM JSON) → (direction × confidence, detail dict)."""
    data = _parse_json_from_text(report)
    if data is None:
        return 0.0, {"pattern_name": "Parse error", "confidence": 0.0}
    direction  = int(data.get("direction", 0))
    confidence = float(data.get("confidence_score", 0.0))
    signal     = float(direction) * float(np.clip(confidence, 0.0, 1.0))
    return float(np.clip(signal, -1.0, 1.0)), {
        "pattern_name": data.get("macro_pattern_name", "None"),
        "direction":    direction,
        "confidence":   confidence,
    }


# ─── Core decision function ───────────────────────────────────────────────────

def make_trade_decision(
    state:              dict,
    weights:            Optional[dict] = None,
    thresholds:         Optional[dict] = None,
    atr_multiplier_sl:  float = 2.0,
    risk_reward_target: float = 2.0,
    allow_short:        bool  = True,
) -> TradeDecision:
    """
    Compute a trade decision from a LangGraph state dict.

    Required state keys: indicator_report, trend_report, pattern_report, kline_data
    Optional state keys: stock_name, entry_price
    """
    w = {**DEFAULT_WEIGHTS,    **(weights    or {})}
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    ind_signal,  ind_comps   = _extract_indicator_signal(state.get("indicator_report", ""))
    trend_signal, trend_info = _extract_trend_signal(state.get("trend_report", ""))
    pat_signal,  pat_info    = _extract_pattern_signal(state.get("pattern_report", ""))

    combined = float(np.clip(
        w["indicator"] * ind_signal
        + w["trend"]   * trend_signal
        + w["pattern"] * pat_signal,
        -1.0, 1.0,
    ))

    abs_sig  = abs(combined)
    strength = "Strong" if abs_sig >= 0.50 else "Moderate" if abs_sig >= 0.25 else "Weak"

    # Current price: prefer trend agent's computed value
    current_price = trend_info.get("current_price") or _last_close(state.get("kline_data", {}))
    support       = trend_info.get("support_level")    or current_price * 0.97
    resistance    = trend_info.get("resistance_level") or current_price * 1.03

    atr         = _compute_atr(state.get("kline_data", {}))
    sl_distance = atr * atr_multiplier_sl if atr > 0 else current_price * 0.02

    # --- Decision + price levels ---
    if combined >= t["buy"]:
        decision     = "BUY"
        stop_loss    = current_price - sl_distance
        target_price = min(float(resistance), current_price + sl_distance * risk_reward_target)
    elif combined <= t["short"] and allow_short:
        decision     = "SHORT"
        stop_loss    = current_price + sl_distance
        target_price = max(float(support), current_price - sl_distance * risk_reward_target)
    elif combined <= t["sell"]:
        decision     = "SELL"
        stop_loss    = current_price - sl_distance * 0.5   # already exiting
        target_price = current_price - sl_distance * risk_reward_target
    else:
        decision     = "HOLD"
        stop_loss    = current_price - sl_distance
        # Defensive target: nearest S/R in signal direction
        target_price = float(resistance) if combined >= 0 else float(support)

    # --- P&L if in a position ---
    entry_price        = state.get("entry_price")
    unrealized_pnl_pct = None
    if entry_price and float(entry_price) > 0 and current_price > 0:
        unrealized_pnl_pct = round(
            (current_price - float(entry_price)) / float(entry_price) * 100, 2
        )
        # Protective override: deep loss while holding → force SELL
        if decision == "HOLD" and unrealized_pnl_pct < -5.0:
            decision     = "SELL"
            stop_loss    = float(entry_price) * 0.95
            target_price = current_price

    denom = abs(current_price - stop_loss)
    risk_reward_ratio = (
        round(abs(target_price - current_price) / denom, 2) if denom > 1e-9 else 0.0
    )

    rationale = (
        f"combined={combined:+.3f} [{strength}] | "
        f"ind={ind_signal:+.3f} | trend={trend_signal:+.3f} | pat={pat_signal:+.3f} | "
        f"trend_dir={trend_info.get('trend_direction', '?')} | "
        f"pattern={pat_info.get('pattern_name', 'None')} "
        f"({pat_info.get('confidence', 0):.0%} conf)"
    )

    return TradeDecision(
        ticker             = state.get("stock_name", "UNKNOWN"),
        decision           = decision,
        combined_signal    = round(combined, 4),
        signal_strength    = strength,
        current_price      = round(current_price, 4),
        entry_price        = round(float(entry_price), 4) if entry_price else None,
        unrealized_pnl_pct = unrealized_pnl_pct,
        target_price       = round(target_price, 4),
        stop_loss          = round(stop_loss, 4),
        risk_reward_ratio  = risk_reward_ratio,
        atr                = round(atr, 4),
        signal_breakdown   = {
            "indicator_signal":      round(ind_signal, 4),
            "trend_signal":          round(trend_signal, 4),
            "pattern_signal":        round(pat_signal, 4),
            "indicator_components":  ind_comps,
            "trend_details":         trend_info,
            "pattern_details":       pat_info,
        },
        decision_rationale = rationale,
    )


# ─── LangGraph node factory ───────────────────────────────────────────────────

def create_quant_decision_node(
    weights:            Optional[dict] = None,
    thresholds:         Optional[dict] = None,
    atr_multiplier_sl:  float = 2.0,
    risk_reward_target: float = 2.0,
    allow_short:        bool  = True,
):
    """
    Returns a LangGraph-compatible callable node.
    No LLM is instantiated or called.

    Usage in graph_setup.py:
        node = create_quant_decision_node(allow_short=False, atr_multiplier_sl=1.5)
        graph.add_node("Decision Maker", node)
    """
    def quant_decision_node(state: dict) -> dict:
        trade = make_trade_decision(
            state,
            weights            = weights,
            thresholds         = thresholds,
            atr_multiplier_sl  = atr_multiplier_sl,
            risk_reward_target = risk_reward_target,
            allow_short        = allow_short,
        )
        return {"final_trade_decision": json.dumps(asdict(trade), indent=2)}

    return quant_decision_node
