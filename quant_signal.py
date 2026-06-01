"""
Standalone deterministic quant decision — no LLM, no LangGraph.

Consolidates the math-only signal pipeline so it can be reused outside the
LangGraph runtime (e.g. by the hourly portfolio scheduler, issue #3, and any
backtest validation). Given OHLC kline data it computes:

  * trend signal     — linear-regression channel (scipy)           [always on]
  * indicator signal — TA-Lib composite (RSI/MACD/Stoch/WillR/ROC)  [if talib]
  * pattern signal   — TA-Lib CDL candlestick aggregation           [if talib]

then feeds them to decision_agent_quant.make_trade_decision.

If TA-Lib is unavailable the indicator/pattern signals degrade to 0.0 (with a
one-time warning) and the decision falls back to a trend-only view, so the
module runs anywhere numpy/pandas/scipy are present.
"""

from __future__ import annotations

import json
import warnings
from typing import Optional

from decision_agent_quant import DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS, make_trade_decision

_TALIB_WARNED = False


def _talib_signals(kline_data: dict):
    """
    Return (indicator_report_json, pattern_report_json), or (None, None) if the
    TA-Lib-backed indicator/pattern signals can't be computed (TA-Lib missing or
    computation fails) — the caller then falls back to a trend-only decision.
    """
    global _TALIB_WARNED
    try:
        from indicator_agent import quantify_indicators_from_kline
        from quant_nodes import quantify_candlestick_patterns

        ind = quantify_indicators_from_kline(kline_data)
        pat = quantify_candlestick_patterns(kline_data)
        return json.dumps({"quantitative_metrics": ind}), json.dumps(pat)
    except Exception:
        if not _TALIB_WARNED:
            warnings.warn("TA-Lib indicator/pattern signals unavailable — using "
                          "trend-only signal (indicator/pattern contributions = 0).",
                          RuntimeWarning)
            _TALIB_WARNED = True
        return None, None


def compute_quant_decision(
    symbol: str,
    kline_data: dict,
    entry_price: Optional[float] = None,
    weights: Optional[dict] = None,
    thresholds: Optional[dict] = None,
    atr_multiplier_sl: float = 2.0,
    risk_reward_target: float = 2.0,
    allow_short: bool = True,
):
    """
    Compute a deterministic TradeDecision for one symbol from its kline data.

    kline_data: dict-of-lists with keys Open/High/Low/Close (Datetime optional).
    Returns a decision_agent_quant.TradeDecision.
    """
    # Trend (scipy) — always available
    from trend_agent import quantify_trend_from_kline
    trend = quantify_trend_from_kline(kline_data)
    trend_report = json.dumps({"quantitative_metrics": trend})

    ind_report, pat_report = _talib_signals(kline_data)
    if ind_report is None:
        ind_report = json.dumps({"quantitative_metrics": {"final_indicator_signal": 0.0}})
        pat_report = json.dumps({"macro_pattern_name": "None", "direction": 0,
                                 "confidence_score": 0.0})

    state = {
        "stock_name": symbol,
        "kline_data": kline_data,
        "indicator_report": ind_report,
        "trend_report": trend_report,
        "pattern_report": pat_report,
    }
    if entry_price is not None:
        state["entry_price"] = entry_price

    return make_trade_decision(
        state,
        weights=weights or DEFAULT_WEIGHTS,
        thresholds=thresholds or DEFAULT_THRESHOLDS,
        atr_multiplier_sl=atr_multiplier_sl,
        risk_reward_target=risk_reward_target,
        allow_short=allow_short,
    )
