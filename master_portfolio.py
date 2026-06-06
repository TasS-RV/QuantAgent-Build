#!/usr/bin/env python
"""
QuantAgent Master Portfolio Script
===================================
Runs the full LangGraph pipeline (Indicator → Pattern → Trend → Quant Decision)
across a configurable portfolio of tickers, prints a decision table, and
optionally sends a trade-card summary to Telegram.

Quick start
───────────
    python master_portfolio.py                  # demo portfolio, no LLM, no Telegram
    python master_portfolio.py --t212           # pull live T212 positions
    python master_portfolio.py --t212 --demo    # pull T212 practice positions
    python master_portfolio.py --telegram       # enable Telegram send this run
    python master_portfolio.py --provider google --llm   # Gemini LLM mode

Portfolio source (edit config below or pass CLI flags)
───────────────────────────────────────────────────────
    USE_T212=False   →  demo_portfolio.json  (static file you edit)
    USE_T212=True    →  live T212 positions (requires T212 API key in .env.keys)

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
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────────────────────────────────────
#  PORTFOLIO SOURCE — edit this
# ─────────────────────────────────────────────────────────────────────────────

# Set True  → pull open positions from Trading 212 (requires .env.keys keys).
# Set False → load from PORTFOLIO_FILE  (demo_portfolio.json by default).
USE_T212: bool = False

# When USE_T212=True: True = T212 practice account, False = live account.
T212_DEMO: bool = False

# Static portfolio file used when USE_T212=False.
# Format: { "TICKER": { "entry_price", "quantity", "lookback_days",
#                        "timeframe", "allocation_pct" }, ... }
PORTFOLIO_FILE: Path = Path(__file__).resolve().parent / "demo_portfolio.json"

# Default analysis settings for T212-sourced tickers
# (T212 doesn't tell us lookback / timeframe; these fill in the gap).
T212_DEFAULTS: dict = {"lookback_days": 90, "timeframe": "1d"}


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

# ─────────────────────────────────────────────────────────────────────────────
#  TELEGRAM — edit this
# ─────────────────────────────────────────────────────────────────────────────

# Send a trade-card summary to Telegram after each run.
# Override at runtime with --telegram / --no-telegram CLI flags.
TELEGRAM_ENABLED: bool = True

# ─────────────────────────────────────────────────────────────────────────────
#  LLM CONFIG — edit or pass via CLI
# ─────────────────────────────────────────────────────────────────────────────

# Set True to run the fully deterministic quant pipeline with zero LLM calls.
# Overridden at runtime by the --no-llm CLI flag.
NO_LLM_MODE = True

# Set True to save per-ticker charts after each run.
# Output: charts/<TICKER>/pattern.png, trend.png, indicators.html
SAVE_CHARTS = True
CHARTS_DIR  = "charts"

DEFAULT_LLM_PROVIDER = "featherless"   # openai | anthropic | qwen | minimax | google | featherless

GEMINI_KEY_FILE = Path(__file__).resolve().parent.parent / "Gemini_API.txt"
DEFAULT_GEMINI_AGENT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_GEMINI_GRAPH_MODEL = "gemini-3.1-flash-lite"

FEATHERLESS_KEY_FILE = Path(__file__).resolve().parent.parent / "Featherless_API.txt"
# Any model from https://api.featherless.ai/v1/models — use HuggingFace model ID format.
# Examples: "meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct",
#           "deepseek-ai/DeepSeek-V3-0324", "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_FEATHERLESS_AGENT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_FEATHERLESS_GRAPH_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Max data yfinance returns per timeframe
_YFINANCE_MAX_DAYS: Dict[str, int] = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "90m": 60,
    "60m": 730, "1h": 730,
    "1d": 3650, "5d": 3650, "1wk": 3650, "1mo": 3650,
}


# ─────────────────────────────────────────────────────────────────────────────
#  LLM CONFIG HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_google_api_key() -> Optional[str]:
    """
    Load Gemini API key. Priority order:
      1. .env.keys  (key name: Gemini_API_key)
      2. GOOGLE_API_KEY environment variable
      3. Playground/Gemini_API.txt  (legacy flat file)
    """
    try:
        from env_keys import get_key
        val = get_key("Gemini_API_key", "GOOGLE_API_KEY")
        if val:
            return val
    except ImportError:
        pass
    if GEMINI_KEY_FILE.is_file():
        key = GEMINI_KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            return key
    return os.environ.get("GOOGLE_API_KEY")


def load_featherless_api_key() -> Optional[str]:
    """Load Featherless API key from Playground/Featherless_API.txt or FEATHERLESS_API_KEY env var."""
    if FEATHERLESS_KEY_FILE.is_file():
        key = FEATHERLESS_KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            return key
    return os.environ.get("FEATHERLESS_API_KEY")


def build_llm_config(provider: str, api_key: Optional[str] = None) -> dict:
    """Build TradingGraph LLM config for the chosen provider."""
    _key_map = {
        "openai":       "api_key",
        "anthropic":    "anthropic_api_key",
        "qwen":         "qwen_api_key",
        "minimax":      "minimax_api_key",
        "minimax_cn":   "minimax_cn_api_key",
        "google":       "google_api_key",
        "featherless":  "featherless_api_key",
    }

    llm_config: dict = {
        "agent_llm_provider": provider,
        "graph_llm_provider": provider,
    }

    if provider == "google":
        llm_config["agent_llm_model"] = DEFAULT_GEMINI_AGENT_MODEL
        llm_config["graph_llm_model"] = DEFAULT_GEMINI_GRAPH_MODEL
        resolved_key = api_key or load_google_api_key()
        if resolved_key:
            llm_config["google_api_key"] = resolved_key
            os.environ["GOOGLE_API_KEY"] = resolved_key
    elif provider == "featherless":
        llm_config["agent_llm_model"] = DEFAULT_FEATHERLESS_AGENT_MODEL
        llm_config["graph_llm_model"] = DEFAULT_FEATHERLESS_GRAPH_MODEL
        resolved_key = api_key or load_featherless_api_key()
        if resolved_key:
            llm_config["featherless_api_key"] = resolved_key
            os.environ["FEATHERLESS_API_KEY"] = resolved_key
    elif api_key:
        llm_config[_key_map.get(provider, "api_key")] = api_key

    return llm_config


# ─────────────────────────────────────────────────────────────────────────────
#  PORTFOLIO LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_portfolio(
    use_t212: bool = USE_T212,
    t212_demo: bool = T212_DEMO,
) -> dict:
    """
    Return the portfolio as::

        {yf_symbol: {"entry_price", "quantity", "lookback_days",
                     "timeframe", "allocation_pct", [is_etf]}}

    use_t212=True  → fetch open positions from Trading 212 API.
    use_t212=False → read from PORTFOLIO_FILE (demo_portfolio.json).
    """
    if use_t212:
        from data_exchange.trading212_client import Trading212Client

        client    = Trading212Client(demo=t212_demo)
        positions = client.get_portfolio()

        total_val = sum(p.market_value for p in positions)

        result = {}
        for pos in positions:
            alloc = (pos.market_value / total_val * 100) if total_val > 0 else 0.0
            result[pos.yf_symbol] = {
                "entry_price":    pos.average_price if pos.average_price > 0 else None,
                "quantity":       pos.quantity,
                "lookback_days":  T212_DEFAULTS["lookback_days"],
                "timeframe":      T212_DEFAULTS["timeframe"],
                "allocation_pct": round(alloc, 2),
                "is_etf":         pos.is_etf,
            }
        account_label = "practice" if t212_demo else "live"
        print(f"  [T212] {len(result)} position(s) loaded ({account_label} account).")
        return result

    # ── Static demo file ──────────────────────────────────────────────────
    with open(PORTFOLIO_FILE, encoding="utf-8") as fh:
        data = json.load(fh)
    # Strip metadata keys (leading "_")
    return {k: v for k, v in data.items() if not k.startswith("_")}


def build_trade_card(decision: dict, portfolio_entry: dict) -> dict:
    """
    Enrich a pipeline decision dict with portfolio metadata and a
    stop-limit price.

    Stop-limit convention
    ─────────────────────
    BUY          → stop_limit = stop_loss × 0.995  (fill guaranteed below SL)
    SELL / SHORT → stop_limit = stop_loss × 1.005  (fill guaranteed above SL)
    """
    card   = dict(decision)
    sl     = decision.get("stop_loss")    or 0.0
    tp     = decision.get("target_price") or 0.0
    cp     = decision.get("current_price") or 0.0
    action = decision.get("decision", "HOLD").upper()

    # Stop-limit price
    if action == "BUY" and sl > 0:
        card["stop_limit"] = round(sl * 0.995, 4)
    elif action in ("SELL", "SHORT") and sl > 0:
        card["stop_limit"] = round(sl * 1.005, 4)
    else:
        card["stop_limit"] = None

    # % distance from current price
    if cp > 0:
        card["tp_pct"] = round((tp - cp) / cp * 100, 2) if tp else None
        card["sl_pct"] = round((sl - cp) / cp * 100, 2) if sl else None
    else:
        card["tp_pct"] = None
        card["sl_pct"] = None

    # Portfolio metadata
    card["quantity"]       = portfolio_entry.get("quantity",       0)
    card["allocation_pct"] = portfolio_entry.get("allocation_pct", 0.0)
    card["is_etf"]         = portfolio_entry.get("is_etf",         False)

    return card


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


def format_portfolio_telegram(trade_cards: List[dict], run_time: str) -> str:
    """Format a list of trade cards as a Telegram Markdown message."""
    _icon = {"BUY": "🟢", "SELL": "🔴", "SHORT": "🔴", "HOLD": "⚪"}

    lines: List[str] = [
        "📊 *QuantAgent Portfolio Report*",
        f"_{run_time}_",
        "",
    ]

    for card in trade_cards:
        ticker = card.get("ticker", "?")
        action = card.get("decision", "HOLD").upper()
        icon   = _icon.get(action, "⚪")
        sig    = card.get("combined_signal", 0)
        cp     = card.get("current_price")
        tp     = card.get("target_price")
        sl     = card.get("stop_loss")
        stl    = card.get("stop_limit")
        tp_pct = card.get("tp_pct")
        sl_pct = card.get("sl_pct")
        alloc  = card.get("allocation_pct", 0.0)
        qty    = card.get("quantity",        0)
        pnl    = card.get("unrealized_pnl_pct")
        is_etf = card.get("is_etf",          False)

        etf_tag = "  _(ETF)_" if is_etf else ""
        lines.append(f"{icon} *{ticker}*{etf_tag}  —  {action}   `{sig:+.3f}`")

        if cp:
            lines.append(
                f"  Price `{cp:.4f}`  |  Alloc `{alloc:.1f}%`  |  Qty `{qty}`"
            )
        if pnl is not None:
            sign = "+" if pnl >= 0 else ""
            lines.append(f"  PnL `{sign}{pnl:.1f}%`")
        if tp:
            if tp_pct is not None:
                sign_tp = "+" if tp_pct >= 0 else ""
                pct_tag = f"  `({sign_tp}{tp_pct:.1f}%)`"
            else:
                pct_tag = ""
            lines.append(f"  🎯 Target  `{tp:.4f}`{pct_tag}")
        if sl:
            if sl_pct is not None:
                sign_sl = "+" if sl_pct >= 0 else ""
                pct_tag = f"  `({sign_sl}{sl_pct:.1f}%)`"
            else:
                pct_tag = ""
            lines.append(f"  🛡️ Stop-Loss  `{sl:.4f}`{pct_tag}")
        if stl:
            lines.append(f"  📌 Stop-Limit  `{stl:.4f}`")
        lines.append("")

    return "\n".join(lines)


def _save_ticker_charts(ticker: str, kline_data: dict) -> None:
    """Generate and save pattern, trend, and indicator charts for a ticker."""
    import charts as _charts
    from indicator_agent import visualize_indicator_signals, _kline_to_dataframe

    out = Path(CHARTS_DIR) / ticker
    out.mkdir(parents=True, exist_ok=True)

    try:
        _charts.generate_kline_image(kline_data, save_path=str(out / "pattern.png"))
        print(f"  Chart: {out / 'pattern.png'}")
    except Exception as e:
        print(f"  [WARN] Pattern chart failed: {e}")

    try:
        _charts.generate_trend_image(kline_data, save_path=str(out / "trend.png"))
        print(f"  Chart: {out / 'trend.png'}")
    except Exception as e:
        print(f"  [WARN] Trend chart failed: {e}")

    try:
        df = _kline_to_dataframe(kline_data)
        visualize_indicator_signals(
            df, ticker,
            save_path=str(out / "indicators.html"),
            show_plot=False,
        )
        print(f"  Chart: {out / 'indicators.html'}")
    except Exception as e:
        print(f"  [WARN] Indicator chart failed: {e}")


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
        choices=["openai", "anthropic", "qwen", "minimax", "minimax_cn", "google", "featherless"],
        help="LLM provider for perception agents (pattern / trend vision analysis)",
    )
    p.add_argument("--api-key",   default=None, help="API key for the chosen provider")
    p.add_argument(
        "--no-llm",
        action="store_true",
        default=NO_LLM_MODE,
        help="Run fully deterministic quant pipeline (zero LLM calls)",
    )
    p.add_argument(
        "--llm",
        dest="no_llm",
        action="store_false",
        help="Use LLM agents (overrides NO_LLM_MODE=True in config)",
    )
    p.add_argument("--no-short",  action="store_true", help="Disable SHORT signals (long-only)")
    p.add_argument("--rr-target", type=float,   default=None, help="Risk:reward target ratio")
    p.add_argument("--atr-mult",  type=float,   default=None, help="ATR multiplier for stop-loss")
    p.add_argument("--output",    default=None, help="Save JSON results to this file path")
    p.add_argument(
        "--breakdown",
        action="store_true",
        help="Print per-ticker signal breakdown after the summary table",
    )

    # ── Portfolio source ───────────────────────────────────────────────────
    p.add_argument(
        "--t212",
        action="store_true",
        dest="t212",
        default=USE_T212,
        help="Pull portfolio from Trading 212 API (requires .env.keys keys)",
    )
    p.add_argument(
        "--no-t212",
        action="store_false",
        dest="t212",
        help="Use demo_portfolio.json (default)",
    )
    p.add_argument(
        "--t212-demo",
        action="store_true",
        dest="t212_demo",
        default=T212_DEMO,
        help="Use T212 practice account instead of live account",
    )

    # ── Telegram ───────────────────────────────────────────────────────────
    p.add_argument(
        "--telegram",
        action="store_true",
        dest="telegram",
        default=TELEGRAM_ENABLED,
        help="Send trade-card summary to Telegram after the run",
    )
    p.add_argument(
        "--no-telegram",
        action="store_false",
        dest="telegram",
        help="Disable Telegram for this run",
    )

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> List[dict]:
    args = parse_args()

    # --- Load portfolio (CLI flags override module-level config) ---
    portfolio = load_portfolio(use_t212=args.t212, t212_demo=args.t212_demo)

    # --- Build LLM config ---
    llm_config = build_llm_config(args.provider, args.api_key)

    # --- Merge decision config with CLI overrides ---
    dec_cfg = {**DECISION_CONFIG}
    if args.no_short:
        dec_cfg["allow_short"] = False
    if args.rr_target is not None:
        dec_cfg["risk_reward_target"] = args.rr_target
    if args.atr_mult is not None:
        dec_cfg["atr_multiplier_sl"] = args.atr_mult

    from graph_setup import SetGraph
    from graph_util import TechnicalTools

    print("=" * 72)
    print(f"  QuantAgent Portfolio  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if args.no_llm:
        print(f"  Mode: NO-LLM (fully deterministic)  |  Tickers: {len(portfolio)}")
    else:
        print(f"  Provider: {args.provider}  |  Tickers: {len(portfolio)}")
    print("=" * 72)

    if args.no_llm:
        # Fully deterministic — no API key required, no LLM calls at all.
        graph_setup = SetGraph(None, None, TechnicalTools())
        compiled_graph = graph_setup.set_graph_full_quant(
            weights            = dec_cfg.get("weights"),
            thresholds         = dec_cfg.get("thresholds"),
            atr_multiplier_sl  = dec_cfg["atr_multiplier_sl"],
            risk_reward_target = dec_cfg["risk_reward_target"],
            allow_short        = dec_cfg["allow_short"],
        )
    else:
        from trading_graph import TradingGraph
        trading_graph = TradingGraph(config=llm_config)
        trading_graph.ensure_initialized()
        compiled_graph = trading_graph.graph_setup.set_graph_quant(
            weights            = dec_cfg.get("weights"),
            thresholds         = dec_cfg.get("thresholds"),
            atr_multiplier_sl  = dec_cfg["atr_multiplier_sl"],
            risk_reward_target = dec_cfg["risk_reward_target"],
            allow_short        = dec_cfg["allow_short"],
        )

    all_results: List[dict] = []
    failed: List[str] = []

    for ticker, cfg in portfolio.items():
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

        # Build enriched trade card and collect result
        all_results.append(build_trade_card(decision_dict, cfg))

        # 5. Generate charts
        if SAVE_CHARTS:
            _save_ticker_charts(ticker, kline_data)

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

    # ── Telegram ──────────────────────────────────────────────────────────
    if args.telegram and all_results:
        from data_exchange.Tele_bot import send_message
        print("\n  Sending portfolio report to Telegram …")
        msg = format_portfolio_telegram(
            all_results, datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        ok = send_message(msg)
        if not ok:
            print("  [WARN] Telegram send failed — check keys in .env.keys")

    print()
    return all_results


if __name__ == "__main__":
    main()
