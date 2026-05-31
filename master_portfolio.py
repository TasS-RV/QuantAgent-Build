#!/usr/bin/env python
"""
QuantAgent Master Portfolio Script
===================================
Runs the full LangGraph pipeline (Indicator → Pattern → Trend → Quant Decision)
across a configurable portfolio of tickers and prints a decision table.

Edit PORTFOLIO and DECISION_CONFIG below, then run:

    python master_portfolio.py
    python master_portfolio.py --provider anthropic --api-key sk-ant-...
    python master_portfolio.py --no-short --rr-target 1.5 --output results.json

Portfolio config per ticker
───────────────────────────
    entry_price   : float | None  — your average cost (None = flat / no position)
    lookback_days : int           — calendar days of history to fetch
    timeframe     : str           — yfinance interval (1d, 1h, 15m, 5m, 1wk …)

Note on yfinance limits
───────────────────────
    1m / 2m  → max 7 days         5m / 15m / 30m / 90m → max 60 days
    1h / 60m → max ~730 days      1d / 5d / 1wk / 1mo  → up to 10 years
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────────────────────────────────────
#  PORTFOLIO — edit this
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO: Dict[str, dict] = {
    # "TICKER": {
    #     "entry_price":   float or None,   ← avg cost; None = not in position
    #     "lookback_days": int,              ← days of history to fetch
    #     "timeframe":     str,              ← yfinance interval string
    # },
    "AAPL":    {"entry_price": 189.50, "lookback_days": 120, "timeframe": "1d"},
    "TSLA":    {"entry_price":   None, "lookback_days":  90, "timeframe": "1d"},
    "NVDA":    {"entry_price": 870.00, "lookback_days":  60, "timeframe": "1d"},
    "BTC-USD": {"entry_price":   None, "lookback_days":  90, "timeframe": "1d"},
}


# ─────────────────────────────────────────────────────────────────────────────
#  DECISION ENGINE PARAMETERS — edit this
# ─────────────────────────────────────────────────────────────────────────────

DECISION_CONFIG: dict = {
    # Signal weights — must sum to 1.0
    "weights": {
        "indicator": 0.40,   # RSI / MACD / Stoch / WillR / ROC composite
        "trend":     0.40,   # linear-regression channel position + slope
        "pattern":   0.20,   # LLM vision: direction × confidence
    },
    # Decision thresholds on the combined signal ∈ [-1, 1]
    "thresholds": {
        "buy":   0.15,   # combined ≥ this  → BUY
        "sell": -0.15,   # combined ≤ this  → SELL
        "short": -0.35,  # combined ≤ this (and allow_short) → SHORT
    },
    "atr_multiplier_sl":  2.0,   # stop-loss = current_price ± ATR × this
    "risk_reward_target": 2.0,   # target = SL_distance × this
    "allow_short":        True,  # set False for long-only portfolios
}


# ─────────────────────────────────────────────────────────────────────────────
#  LLM CONFIG — edit or pass via CLI
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_LLM_PROVIDER = "openai"   # openai | anthropic | qwen | minimax | google

# Max data yfinance returns per timeframe
_YFINANCE_MAX_DAYS: Dict[str, int] = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "90m": 60,
    "60m": 730, "1h": 730,
    "1d": 3650, "5d": 3650, "1wk": 3650, "1mo": 3650,
}


# ─────────────────────────────────────────────────────────────────────────────
#  DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_kline_data(ticker: str, lookback_days: int, timeframe: str) -> dict:
    """
    Download OHLCV from Yahoo Finance.
    Returns a dict-of-lists with keys: Datetime, Open, High, Low, Close, Volume.
    This is the format expected by indicator_agent, trend_agent, and pattern_agent.
    """
    max_days = _YFINANCE_MAX_DAYS.get(timeframe, 3650)
    if lookback_days > max_days:
        print(
            f"  [WARN] {ticker}: {timeframe} only supports {max_days} days — "
            f"capping from {lookback_days}."
        )
        lookback_days = max_days

    end   = datetime.today()
    start = end - timedelta(days=lookback_days)

    df = yf.download(
        ticker,
        start    = start.strftime("%Y-%m-%d"),
        end      = end.strftime("%Y-%m-%d"),
        interval = timeframe,
        progress = False,
        auto_adjust = True,
    )

    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker!r}")

    # yfinance ≥ 0.2 may return MultiIndex columns even for single tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df.reset_index()

    # Normalise the date column name
    date_col = next(
        (c for c in df.columns if c in ("Date", "Datetime", "index")), None
    )
    if date_col is None:
        raise ValueError(f"Cannot find date column in yfinance output: {list(df.columns)}")
    df = df.rename(columns={date_col: "Datetime"})

    df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]].dropna()
    df["Datetime"] = df["Datetime"].astype(str)

    return df.to_dict(orient="list")


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

_DECISION_BADGE = {
    "BUY":   "BUY  ",
    "HOLD":  "HOLD ",
    "SELL":  "SELL ",
    "SHORT": "SHORT",
}


def _fmt_price(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "—"


def _fmt_pnl(v: Optional[float]) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"


def print_summary_table(decisions: List[dict]) -> None:
    rows = []
    for d in decisions:
        rows.append({
            "Ticker":    d["ticker"],
            "Decision":  _DECISION_BADGE.get(d["decision"], d["decision"]),
            "Signal":    f"{d['combined_signal']:+.3f}",
            "Strength":  d["signal_strength"],
            "Price":     _fmt_price(d["current_price"]),
            "Entry":     _fmt_price(d["entry_price"]),
            "PnL%":      _fmt_pnl(d["unrealized_pnl_pct"]),
            "Target":    _fmt_price(d["target_price"]),
            "StopLoss":  _fmt_price(d["stop_loss"]),
            "R:R":       f"1:{d['risk_reward_ratio']:.1f}",
            "ATR":       _fmt_price(d["atr"]),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def print_signal_breakdown(decisions: List[dict]) -> None:
    for d in decisions:
        bd  = d["signal_breakdown"]
        td  = bd.get("trend_details", {})
        pat = bd.get("pattern_details", {})
        ind = bd.get("indicator_components", {})
        print(f"\n  {d['ticker']} — {d['decision_rationale']}")
        print(
            f"    Indicator  ({bd['indicator_signal']:+.3f}): "
            + "  ".join(f"{k}={v:+.3f}" for k, v in ind.items() if isinstance(v, float))
        )
        print(
            f"    Trend      ({bd['trend_signal']:+.3f}): "
            f"{td.get('trend_direction', '?')} | "
            f"support={_fmt_price(td.get('support_level'))} | "
            f"resistance={_fmt_price(td.get('resistance_level'))} | "
            f"chan_pos={td.get('channel_position', 0):.2f}"
        )
        print(
            f"    Pattern    ({bd['pattern_signal']:+.3f}): "
            f"{pat.get('pattern_name', 'None')}  "
            f"conf={pat.get('confidence', 0):.0%}  "
            f"dir={pat.get('direction', 0):+d}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QuantAgent — portfolio-level analysis with quant decision node",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--provider",
        default=DEFAULT_LLM_PROVIDER,
        choices=["openai", "anthropic", "qwen", "minimax", "minimax_cn", "google"],
        help="LLM provider for perception agents (pattern / trend vision analysis)",
    )
    p.add_argument("--api-key",   default=None, help="API key for the chosen provider")
    p.add_argument("--no-short",  action="store_true", help="Disable SHORT signals (long-only)")
    p.add_argument("--rr-target", type=float,   default=None, help="Risk:reward target ratio")
    p.add_argument("--atr-mult",  type=float,   default=None, help="ATR multiplier for stop-loss")
    p.add_argument("--output",    default=None, help="Save JSON results to this file path")
    p.add_argument(
        "--breakdown",
        action="store_true",
        help="Print per-ticker signal breakdown after the summary table",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> List[dict]:
    args = parse_args()

    # --- Build LLM config ---
    _key_map = {
        "openai":     "api_key",
        "anthropic":  "anthropic_api_key",
        "qwen":       "qwen_api_key",
        "minimax":    "minimax_api_key",
        "minimax_cn": "minimax_cn_api_key",
    }
    llm_config: dict = {
        "agent_llm_provider": args.provider,
        "graph_llm_provider": args.provider,
    }
    if args.api_key:
        llm_config[_key_map.get(args.provider, "api_key")] = args.api_key

    # --- Merge decision config with CLI overrides ---
    dec_cfg = {**DECISION_CONFIG}
    if args.no_short:
        dec_cfg["allow_short"] = False
    if args.rr_target is not None:
        dec_cfg["risk_reward_target"] = args.rr_target
    if args.atr_mult is not None:
        dec_cfg["atr_multiplier_sl"] = args.atr_mult

    # --- Import here to avoid slow startup until we need it ---
    from trading_graph import TradingGraph

    print("=" * 72)
    print(f"  QuantAgent Portfolio  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Provider: {args.provider}  |  Tickers: {len(PORTFOLIO)}")
    print("=" * 72)

    # Initialise TradingGraph once (shared LLM instances across tickers)
    trading_graph = TradingGraph(config=llm_config)

    # Compile graph with the pure-math quant decision node
    compiled_graph = trading_graph.graph_setup.set_graph_quant(
        weights            = dec_cfg.get("weights"),
        thresholds         = dec_cfg.get("thresholds"),
        atr_multiplier_sl  = dec_cfg["atr_multiplier_sl"],
        risk_reward_target = dec_cfg["risk_reward_target"],
        allow_short        = dec_cfg["allow_short"],
    )

    all_results: List[dict] = []
    failed: List[str] = []

    for ticker, cfg in PORTFOLIO.items():
        timeframe     = cfg.get("timeframe", "1d")
        lookback_days = cfg.get("lookback_days", 90)
        entry_price   = cfg.get("entry_price")

        print(f"\n[{ticker}]  {lookback_days}d × {timeframe}  |  "
              f"entry={'%.4f' % entry_price if entry_price else '—'}")

        # 1. Fetch data
        try:
            kline_data = fetch_kline_data(ticker, lookback_days, timeframe)
        except Exception as exc:
            print(f"  ERROR fetching data: {exc}")
            failed.append(ticker)
            continue

        n = len(kline_data.get("Close", []))
        print(f"  {n} candles loaded.  Running pipeline …")

        # 2. Build initial LangGraph state
        initial_state = {
            "kline_data":  kline_data,
            "time_frame":  timeframe,
            "stock_name":  ticker,
            "entry_price": entry_price,   # picked up by quant decision node
            "messages":    [],
        }

        # 3. Run graph
        try:
            final_state = compiled_graph.invoke(initial_state)
        except Exception as exc:
            print(f"  ERROR running pipeline: {exc}")
            failed.append(ticker)
            continue

        # 4. Parse decision
        raw = final_state.get("final_trade_decision", "{}")
        try:
            decision_dict = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  ERROR: could not parse final_trade_decision JSON.")
            print(f"  Raw (first 300 chars): {raw[:300]}")
            failed.append(ticker)
            continue

        all_results.append(decision_dict)

        # Quick per-ticker line
        dec  = decision_dict.get("decision", "?")
        sig  = decision_dict.get("combined_signal", 0)
        strg = decision_dict.get("signal_strength", "?")
        cp   = decision_dict.get("current_price", 0)
        tp   = decision_dict.get("target_price", 0)
        sl   = decision_dict.get("stop_loss", 0)
        pnl  = decision_dict.get("unrealized_pnl_pct")
        pnl_str = _fmt_pnl(pnl)
        print(
            f"  {_DECISION_BADGE.get(dec, dec)}  signal={sig:+.3f} [{strg}]  "
            f"price={cp:.4f}  target={tp:.4f}  SL={sl:.4f}  PnL={pnl_str}"
        )

    # ── Summary table ──────────────────────────────────────────────────────
    if all_results:
        print("\n" + "=" * 72)
        print("  PORTFOLIO SUMMARY")
        print("=" * 72)
        print_summary_table(all_results)

        if args.breakdown:
            print("\n" + "=" * 72)
            print("  SIGNAL BREAKDOWN")
            print("=" * 72)
            print_signal_breakdown(all_results)

    if failed:
        print(f"\n  Skipped (errors): {', '.join(failed)}")

    # ── JSON export ────────────────────────────────────────────────────────
    if args.output and all_results:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\n  Results saved → {args.output}")

    print()
    return all_results


if __name__ == "__main__":
    main()
