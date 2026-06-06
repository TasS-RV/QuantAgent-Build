"""Backtest the multi-agent pipeline across a configurable set of symbols and date range."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

# Shim sqlite for environments without _sqlite3 (yfinance caches via sqlite).
try:
    import sqlite3  # noqa: F401
except ImportError:
    import pysqlite3

    sys.modules["sqlite3"] = pysqlite3

import pandas as pd
import yfinance as yf
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm

# LLM-mode dependencies (pandas_ta + OpenAI Agents SDK).
# Imported lazily so --no-llm works without these installed.
try:
    import indicators as _indicators_mod  # noqa: F401  (warms up pandas-ta)
    from quant_agents import run_pipeline_async
    _LLM_DEPS_AVAILABLE = True
except ImportError:
    _LLM_DEPS_AVAILABLE = False

from backtest_engine import Signal, SimConfig, simulate, run_with_baselines, trades_to_frame


# ─────────────────────────────────────────────────────────────────────────────
#  BACKTEST CONFIG — edit this or override via CLI flags
# ─────────────────────────────────────────────────────────────────────────────

# SYMBOLS: list[str] = [
#     "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
#     "META", "AVGO", "TSLA", "BRK-B", "LLY",
# ]


SYMBOLS: list[str] = [
    "NVDA", 
]

# Rolling context window: bars of OHLC history fed to agents at each decision point.
# e.g. 60 daily bars ≈ 3 months of context per decision.
WINDOW_BARS: int = 30

# ── Date range ────────────────────────────────────────────────────────────────
# START_DATE : first eligible decision date (YYYY-MM-DD).
#              None = derive from LOOKBACK_PERIOD relative to today.
# END_DATE   : last  eligible decision date (YYYY-MM-DD).
#              None = today.
#
# When START_DATE is set, data is fetched from
#   (START_DATE − WINDOW_BARS × 2 calendar-day buffer)  →  END_DATE
# so the rolling-window agents always have full context at the first decision.
#
# Example: backtest Dec 2024 – Feb 2025 with a 1-month rolling window:
#   START_DATE  = "2024-12-01"
#   END_DATE    = "2025-02-01"
#   WINDOW_BARS = 21          # ~1 trading month
#   → data fetched from ~2024-10-01 to 2025-02-01
START_DATE: str | None = None   # e.g. "2024-12-01"
END_DATE:   str | None = None   # e.g. "2025-02-01"

# Fallback when START_DATE is None: how far back from END_DATE to fetch data.
# yfinance period string — "1y", "6mo", "2y", "5y", etc.
LOOKBACK_PERIOD: str = "6mo"

# ── Cadence ───────────────────────────────────────────────────────────────────
# How often trade decisions are made (pandas resample alias).
#   "D"     = every trading day
#   "W-FRI" = weekly, on Friday close  (default)
#   "2W"    = bi-weekly
#   "M"     = monthly
CADENCE: str = "W-Fri"

# ── Out-of-sample split ───────────────────────────────────────────────────────
# Set to a YYYY-MM-DD string to get separate in-sample / out-of-sample metrics.
# Useful for detecting overfitting.
OOS_START: str | None = None   # e.g. "2025-01-01"

# ── LLM Provider (used when NOT running --no-llm) ────────────────────────────
# Provider for LLM-assisted signal collection via the LangGraph pipeline.
# Supports the same providers as master_portfolio.py:
#   "google"    → Gemini (auto-loads key from ../Gemini_API.txt)
#   "openai"    → OpenAI via LangGraph
#   "anthropic" → Claude via LangGraph
#
# Usage: python backtest.py --llm --provider google
DEFAULT_LLM_PROVIDER: str = "google"

GEMINI_KEY_FILE = Path(__file__).resolve().parent.parent / "Gemini_API.txt"
DEFAULT_GEMINI_AGENT_MODEL: str = "gemini-3.1-flash-lite"
DEFAULT_GEMINI_GRAPH_MODEL: str = "gemini-3.1-flash-lite"

# ─────────────────────────────────────────────────────────────────────────────

# Approximate rebalance periods per year, by cadence — used to annualise Sharpe.
_CADENCE_PPY = {
    "D": 252.0, "DAILY": 252.0,
    "W-FRI": 52.0, "W": 52.0, "2W": 26.0,
    "M": 12.0, "ME": 12.0,
}


def _periods_per_year(cadence: str) -> float:
    return _CADENCE_PPY.get(cadence.upper(), 52.0)


# ─────────────────────────────────────────────────────────────────────────────
#  LLM CONFIG HELPERS  (mirrors master_portfolio.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_google_api_key() -> str | None:
    """
    Load Gemini API key. Priority order:
      1. .env.keys  (key name: Gemini_API_key)
      2. GOOGLE_API_KEY environment variable
      3. ../Gemini_API.txt  (legacy flat file)
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


def build_llm_config(provider: str, api_key: str | None = None) -> dict:
    """
    Build a TradingGraph LLM config dict for the chosen provider.
    Mirrors master_portfolio.py's build_llm_config() exactly.
    """
    _key_map = {
        "openai":    "api_key",
        "anthropic": "anthropic_api_key",
        "google":    "google_api_key",
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
    elif api_key:
        llm_config[_key_map.get(provider, "api_key")] = api_key
    return llm_config


# ---------- Data ----------

def _normalize_yf(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Datetime"})
    keep = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    return df[[c for c in keep if c in df.columns]]


def fetch_history(
    symbol: str,
    *,
    start: str | None = None,
    end: str | None = None,
    period: str = "5y",
    window_bars: int = WINDOW_BARS,
) -> pd.DataFrame:
    """
    Fetch daily OHLC bars for `symbol`.

    start / end (YYYY-MM-DD)
        When `start` is given the data is fetched from
        ``start − warmup_buffer`` to ``end`` (or today), where the buffer
        is ``window_bars × 2`` calendar days — enough to guarantee
        ``window_bars`` full trading bars exist before the first decision.
        This means if you want a 2-month backtest starting Dec 2024 with a
        60-bar window, data is automatically pulled from ~Sep 2024.

    period
        yfinance relative-period string ("1y", "6mo", "5y" …).
        Used only when ``start`` is None.
    """
    if start:
        warmup_days = math.ceil(window_bars * 2)
        fetch_start = (
            pd.Timestamp(start) - pd.Timedelta(days=warmup_days)
        ).strftime("%Y-%m-%d")
        fetch_end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        raw = yf.download(
            symbol, start=fetch_start, end=fetch_end,
            interval="1d", auto_adjust=True, progress=False,
        )
    else:
        raw = yf.download(
            symbol, period=period,
            interval="1d", auto_adjust=True, progress=False,
        )

    if raw.empty:
        raise RuntimeError(f"No data for {symbol}")
    df = _normalize_yf(raw)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df.reset_index(drop=True)


def rebalance_dates(
    df: pd.DataFrame,
    cadence: str = "W-FRI",
    warmup_bars: int = WINDOW_BARS,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[int]:
    """
    Return the bar indices at which trade decisions are made.

    start_date / end_date (YYYY-MM-DD)
        Restrict decisions to this date range.  Data outside this range
        is still used for rolling-window context (warmup).
    """
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    if cadence.upper() in ("D", "DAILY"):
        idxs = list(range(warmup_bars, len(df) - 1))
    else:
        s = df.set_index("Datetime")["Close"]
        period_ends = s.resample(cadence).last().dropna().index
        idx_map = {dt: i for i, dt in enumerate(df["Datetime"].tolist())}
        idxs = [
            idx_map[dt]
            for dt in period_ends
            if dt in idx_map
            and idx_map[dt] >= warmup_bars
            and idx_map[dt] < len(df) - 1
        ]

    # Filter to [start_date, end_date] if specified
    if start_date or end_date:
        dates = df["Datetime"].dt.date.astype(str).tolist()
        idxs = [
            i for i in idxs
            if (start_date is None or dates[i] >= start_date)
            and (end_date   is None or dates[i] <= end_date)
        ]

    return idxs


def window_to_kline(df: pd.DataFrame, end_idx: int, window: int = WINDOW_BARS) -> dict:
    start = max(0, end_idx - window + 1)
    sub = df.iloc[start : end_idx + 1].copy()
    return {
        "Datetime": sub["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "Open":  sub["Open"].astype(float).tolist(),
        "High":  sub["High"].astype(float).tolist(),
        "Low":   sub["Low"].astype(float).tolist(),
        "Close": sub["Close"].astype(float).tolist(),
    }


# ---------- Signal collection (LLM, non-deterministic) ----------

async def _signal_for_bar(
    symbol: str,
    df: pd.DataFrame,
    end_idx: int,
    window_bars: int = WINDOW_BARS,
) -> tuple[int, int, float, float, str]:
    """Run the LLM pipeline at one decision bar."""
    kline = window_to_kline(df, end_idx, window=window_bars)
    try:
        result = await run_pipeline_async(symbol, kline, timeframe="1d")
        direction = {"LONG": 1, "SHORT": -1, "HOLD": 0}.get(result.decision.decision, 0)
        return (
            end_idx,
            direction,
            float(result.decision.confidence),
            float(result.decision.risk_reward_ratio),
            result.decision.justification,
        )
    except Exception as e:
        return end_idx, 0, 0.0, 2.0, f"ERROR: {type(e).__name__}: {e}"


async def collect_signals(
    symbol: str,
    df: pd.DataFrame,
    rebalance_idxs: list[int],
    concurrency: int = 4,
    window_bars: int = WINDOW_BARS,
) -> list[Signal]:
    """Collect signals via the LLM pipeline at each rebalance bar (expensive)."""
    if not _LLM_DEPS_AVAILABLE:
        raise ImportError(
            "LLM signal collection requires pandas_ta and the OpenAI Agents SDK. "
            "Run with --no-llm to use the pure-quant pipeline instead."
        )
    sem = asyncio.Semaphore(concurrency)

    async def one(idx: int):
        async with sem:
            return await _signal_for_bar(symbol, df, idx, window_bars=window_bars)

    raw = await atqdm.gather(
        *[one(i) for i in rebalance_idxs],
        desc=f"  {symbol:<6} signals",
        unit="bar",
        leave=False,
        dynamic_ncols=True,
    )
    raw.sort(key=lambda r: r[0])
    return [
        Signal(
            decision_idx=idx,
            direction=direction,
            confidence=conf,
            risk_reward_ratio=rr,
            date=str(df["Datetime"].iloc[idx].date()),
            note=just[:200],
        )
        for idx, direction, conf, rr, just in raw
    ]


# ---------- Signal collection (pure-quant, deterministic) ----------

def collect_signals_quant(
    symbol: str,
    df: pd.DataFrame,
    rebalance_idxs: list[int],
    weights: dict | None = None,
    thresholds: dict | None = None,
    window_bars: int = WINDOW_BARS,
) -> list[Signal]:
    """
    Collect signals using the pure-quant pipeline — zero LLM calls.

    Runs quant_indicator_node → quant_trend_node → quant_pattern_node →
    make_trade_decision on each rebalance bar, mirroring exactly what
    master_portfolio.py does with --no-llm / set_graph_full_quant.

    Returns the same List[Signal] as collect_signals() so backtest_universe()
    can switch between the two transparently.
    """
    from quant_pipeline.quant_nodes import quant_indicator_node, quant_trend_node, quant_pattern_node
    from quant_pipeline.decision_agent_quant import make_trade_decision

    _DIR = {"BUY": 1, "SELL": -1, "SHORT": -1, "HOLD": 0}
    signals: list[Signal] = []

    for idx in tqdm(
        rebalance_idxs,
        desc=f"  {symbol:<6} quant",
        unit="bar",
        leave=False,
        dynamic_ncols=True,
    ):
        kline = window_to_kline(df, idx, window=window_bars)
        state: dict = {
            "kline_data": kline,
            "time_frame": "1d",
            "stock_name": symbol,
            "messages":   [],
        }
        state.update(quant_indicator_node(state))
        state.update(quant_trend_node(state))
        state.update(quant_pattern_node(state))

        trade = make_trade_decision(state, weights=weights, thresholds=thresholds)
        direction = _DIR.get(trade.decision, 0)

        signals.append(Signal(
            decision_idx      = idx,
            direction         = direction,
            confidence        = round(abs(trade.combined_signal), 4),
            risk_reward_ratio = trade.risk_reward_ratio if trade.risk_reward_ratio > 0 else 2.0,
            date              = str(df["Datetime"].iloc[idx].date()),
            note              = trade.decision_rationale[:200],
        ))

    return signals


# ---------- Signal collection (LangGraph / TradingGraph — any provider) ----------

def collect_signals_langgraph(
    symbol: str,
    df: pd.DataFrame,
    rebalance_idxs: list[int],
    llm_config: dict,
    window_bars: int = WINDOW_BARS,
    dec_cfg: dict | None = None,
) -> list[Signal]:
    """
    Collect signals via the LangGraph / TradingGraph pipeline at each rebalance bar.

    Supports any LLM provider (Google/Gemini, OpenAI, Anthropic …) — just pass
    the llm_config dict produced by build_llm_config().  This mirrors exactly
    what master_portfolio.py does in LLM mode (set_graph_quant path).

    dec_cfg is the optional decision-engine config dict (weights, thresholds,
    atr_multiplier_sl, risk_reward_target, allow_short); defaults match
    master_portfolio.py's DECISION_CONFIG if omitted.
    """
    from trading_graph import TradingGraph

    dec_cfg = dec_cfg or {
        "weights":            {"indicator": 0.40, "trend": 0.40, "pattern": 0.20},
        "thresholds":         {"buy": 0.15, "sell": -0.15, "short": -0.35},
        "atr_multiplier_sl":  2.0,
        "risk_reward_target": 2.0,
        "allow_short":        True,
    }

    trading_graph = TradingGraph(config=llm_config)
    trading_graph.ensure_initialized()
    compiled_graph = trading_graph.graph_setup.set_graph_quant(
        weights            = dec_cfg.get("weights"),
        thresholds         = dec_cfg.get("thresholds"),
        atr_multiplier_sl  = dec_cfg["atr_multiplier_sl"],
        risk_reward_target = dec_cfg["risk_reward_target"],
        allow_short        = dec_cfg["allow_short"],
    )

    _DIR = {"BUY": 1, "SELL": -1, "SHORT": -1, "HOLD": 0}
    signals: list[Signal] = []

    for idx in tqdm(
        rebalance_idxs,
        desc=f"  {symbol:<6} llm({llm_config.get('agent_llm_provider','?')})",
        unit="bar",
        leave=False,
        dynamic_ncols=True,
    ):
        kline = window_to_kline(df, idx, window=window_bars)
        state: dict = {
            "kline_data":  kline,
            "time_frame":  "1d",
            "stock_name":  symbol,
            "entry_price": None,
            "messages":    [],
        }
        try:
            final_state = compiled_graph.invoke(state)
            raw = final_state.get("final_trade_decision", "{}")
            decision_dict = json.loads(raw) if isinstance(raw, str) else raw
            dec       = decision_dict.get("decision", "HOLD")
            direction = _DIR.get(dec, 0)
            conf      = float(decision_dict.get("combined_signal", 0.0))
            rr        = float(decision_dict.get("risk_reward_ratio", 2.0))
            note      = decision_dict.get("decision_rationale", "")[:200]
        except Exception as e:
            direction, conf, rr, note = 0, 0.0, 2.0, f"ERROR: {type(e).__name__}: {e}"

        signals.append(Signal(
            decision_idx      = idx,
            direction         = direction,
            confidence        = round(abs(conf), 4),
            risk_reward_ratio = rr if rr > 0 else 2.0,
            date              = str(df["Datetime"].iloc[idx].date()),
            note              = note,
        ))

    return signals


# ---------- Backtest orchestration ----------

async def backtest_universe(
    symbols: Iterable[str],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    period: str = "5y",
    cadence: str = "W-FRI",
    window_bars: int = WINDOW_BARS,
    concurrency: int = 4,
    out_dir: Path | str = "backtest_results",
    cfg: SimConfig | None = None,
    oos_start: str | None = None,
    use_llm: bool = True,
    quant_weights: dict | None = None,
    quant_thresholds: dict | None = None,
    no_charts: bool = False,
    llm_config: dict | None = None,
) -> dict:
    """
    Collect signals per symbol then run the deterministic backtest engine
    (risk overlays + costs) and trivial baselines for comparison.

    start_date / end_date
        Restrict the decision window to this date range (YYYY-MM-DD).
        Data before start_date is still fetched as rolling-window warmup.
        If start_date is None the full `period` of data is used.

    window_bars
        Bars of OHLC context fed to the agents at each decision point.

    use_llm
        True  → signals via LangGraph / TradingGraph (uses llm_config provider).
        False → signals via pure-quant pipeline (quant_nodes.py +
                decision_agent_quant.py), zero LLM calls, matches
                master_portfolio.py --no-llm exactly.

    llm_config
        Provider config dict produced by build_llm_config().
        Only used when use_llm=True.  If None, auto-built from DEFAULT_LLM_PROVIDER.

    oos_start
        Optional YYYY-MM-DD — report separate metrics on decisions
        on/after this date to detect overfitting.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg or SimConfig(periods_per_year=_periods_per_year(cadence))

    if use_llm and llm_config is None:
        llm_config = build_llm_config(DEFAULT_LLM_PROVIDER)

    all_signals: dict[str, list[Signal]] = {}
    all_trades:  dict[str, list[dict]]   = {}
    histories:   dict[str, pd.DataFrame] = {}
    summary_rows:  list[dict] = []
    baseline_rows: list[dict] = []

    symbols = list(symbols)
    if use_llm:
        provider = (llm_config or {}).get("agent_llm_provider", DEFAULT_LLM_PROVIDER)
        mode_label = f"LLM agents ({provider})"
    else:
        mode_label = "pure-quant (no LLM)"
    date_range = f"{start_date or 'all'} -> {end_date or 'today'}"

    print(f"  Signal mode : {mode_label}")
    print(f"  Symbols     : {', '.join(symbols)}")
    print(f"  Date range  : {date_range}  |  cadence: {cadence}  |  window: {window_bars} bars")
    print()

    sym_bar = tqdm(symbols, desc="symbols", unit="sym", dynamic_ncols=True)
    for symbol in sym_bar:
        sym_bar.set_postfix_str(f"fetching {symbol}")
        df = fetch_history(
            symbol,
            start=start_date,
            end=end_date,
            period=period,
            window_bars=window_bars,
        )
        idxs = rebalance_dates(
            df,
            cadence=cadence,
            warmup_bars=window_bars,
            start_date=start_date,
            end_date=end_date,
        )
        sym_bar.set_postfix_str(f"{symbol}: {len(df)}b / {len(idxs)} decisions")
        histories[symbol] = df

        if use_llm:
            signals = collect_signals_langgraph(
                symbol, df, idxs,
                llm_config=llm_config,
                window_bars=window_bars,
            )
        else:
            signals = collect_signals_quant(
                symbol, df, idxs,
                weights=quant_weights,
                thresholds=quant_thresholds,
                window_bars=window_bars,
            )
        all_signals[symbol] = signals

        sim    = simulate(symbol, df, signals, cfg)
        trades = sim["trades"]
        m      = sim["metrics"]
        bl     = run_with_baselines(symbol, df, signals, idxs, cfg)

        trades_df = trades_to_frame(trades)
        trades_df.to_csv(out_dir / f"{symbol}_trades.csv", index=False)
        all_trades[symbol] = trades_df.to_dict(orient="records")

        row = {
            "symbol":             symbol,
            "n_trades":           m["n_trades"],
            "n_long":             m["n_long"],
            "n_short":            m["n_short"],
            "agent_total_return": m["total_return"],
            "buy_hold_return":    bl["buy_hold_return"],
            "excess_return":      m["total_return"] - bl["buy_hold_return"],
            "sharpe_annual":      m["sharpe_annual"],
            "win_rate":           m["win_rate"],
            "max_drawdown":       m["max_drawdown"],
        }
        if oos_start:
            oos_sigs = [s for s in signals if s.date and s.date >= oos_start]
            oos_m = simulate(symbol, df, oos_sigs, cfg)["metrics"]
            row["oos_total_return"] = oos_m["total_return"]
            row["oos_sharpe"]       = oos_m["sharpe_annual"]
            row["oos_win_rate"]     = oos_m["win_rate"]
        summary_rows.append(row)

        baseline_rows.append({
            "symbol":            symbol,
            "agent":             m["total_return"],
            "buy_hold":          bl["buy_hold_return"],
            "always_long":       bl["always_long"]["total_return"],
            "sma_trend_follower":bl["sma_trend_follower"]["total_return"],
            "random":            bl["random"]["total_return"],
            "beats_sma_baseline":m["total_return"] > bl["sma_trend_follower"]["total_return"],
        })

        sym_bar.write(
            f"[{symbol}] {m['n_trades']} trades "
            f"({m['n_long']}L/{m['n_short']}S) | agent {m['total_return']:+.1%} "
            f"vs B&H {bl['buy_hold_return']:+.1%} "
            f"vs 200dma {bl['sma_trend_follower']['total_return']:+.1%}"
        )

    summary_df  = pd.DataFrame(summary_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    summary_df .to_csv(out_dir / "summary.csv",   index=False)
    baseline_df.to_csv(out_dir / "baselines.csv", index=False)

    combined = {
        sym: [
            {
                "decision_idx":     s.decision_idx,
                "direction":        s.direction,
                "confidence":       s.confidence,
                "risk_reward_ratio":s.risk_reward_ratio,
                "date":             s.date,
                "note":             s.note,
            }
            for s in sigs
        ]
        for sym, sigs in all_signals.items()
    }
    (out_dir / "all_signals.json").write_text(json.dumps(combined, indent=2))
    (out_dir / "all_trades.json" ).write_text(json.dumps(all_trades, indent=2))
    for sym, df in histories.items():
        df.to_csv(out_dir / f"{sym}_history.csv", index=False)

    if not baseline_df.empty:
        n_beat = int(baseline_df["beats_sma_baseline"].sum())
        print(f"\nAgent beats 200dma trend-follower on {n_beat}/{len(baseline_df)} symbols.")
        if n_beat <= len(baseline_df) / 2:
            print("[WARN] Agent does NOT cleanly beat the dumb 200dma baseline -- "
                  "do not add complexity until it does (see docs/BACKTEST_IMPROVEMENT_ROADMAP.md).")

    # Auto-generate charts
    if not no_charts:
        try:
            from visualize import render as _render
            chart_title = (
                f"QuantAgent ({mode_label})  |  "
                f"{date_range}  |  cadence: {cadence}  |  window: {window_bars} bars"
            )
            print(f"\nGenerating charts in {out_dir} ...")
            _render(out_dir, title=chart_title)
        except Exception as e:
            print(f"[WARN] Chart generation failed: {e}")

    return {
        "summary":   summary_df,
        "baselines": baseline_df,
        "signals":   all_signals,
        "histories": histories,
    }


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(
        description="QuantAgent backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── What to test ──────────────────────────────────────────────────────────
    p.add_argument("--symbols", nargs="+", default=SYMBOLS,
                   help="Tickers to backtest")

    # ── Date range ────────────────────────────────────────────────────────────
    p.add_argument(
        "--start", default=START_DATE, metavar="YYYY-MM-DD",
        help=(
            "First eligible decision date. Data is fetched from "
            "(start - window_bars*2 calendar days) for warmup. "
            "If omitted, --period is used instead."
        ),
    )
    p.add_argument(
        "--end", default=END_DATE, metavar="YYYY-MM-DD",
        help="Last eligible decision date. Defaults to today.",
    )
    p.add_argument(
        "--period", default=LOOKBACK_PERIOD,
        help=(
            "yfinance relative-period string used when --start is not set "
            "(e.g. '1y', '6mo', '5y'). Ignored when --start is given."
        ),
    )

    # ── Window + cadence ──────────────────────────────────────────────────────
    p.add_argument(
        "--window-bars", type=int, default=WINDOW_BARS,
        help=(
            "Bars of OHLC history fed to agents at each decision point "
            "(rolling context window). e.g. 21 ≈ 1 trading month, "
            "60 ≈ 3 months, 126 ≈ 6 months."
        ),
    )
    p.add_argument(
        "--cadence", default=CADENCE,
        help=(
            "How often trade decisions are made. "
            "Pandas resample alias: D=daily, W-FRI=weekly, 2W=bi-weekly, M=monthly."
        ),
    )

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--out", default="backtest_results",
                   help="Output directory for CSVs and JSON")
    p.add_argument("--oos-start", default=OOS_START, metavar="YYYY-MM-DD",
                   help="Report separate out-of-sample metrics for decisions on/after this date")

    # ── Risk overlays ─────────────────────────────────────────────────────────
    p.add_argument("--no-trend-filter", action="store_true",
                   help="Disable the 200dma trend filter (default: on)")
    p.add_argument("--no-stops", action="store_true",
                   help="Disable ATR stop/target exits; hold to next rebalance")
    p.add_argument("--confidence-gate", type=float, default=0.0,
                   help="Drop signals below this confidence (e.g. 0.75); 0 disables")
    p.add_argument("--atr-mult", type=float, default=1.5,
                   help="ATR multiple for stop distance")
    p.add_argument("--commission", type=float, default=0.0002,
                   help="Commission per side (fraction)")
    p.add_argument("--slippage", type=float, default=0.0005,
                   help="Slippage per side (fraction)")
    p.add_argument("--no-short", action="store_true",
                   help="Long-only mode (drop SHORT signals)")

    # ── LLM / quant mode ──────────────────────────────────────────────────────
    p.add_argument("--no-charts", action="store_true",
                   help="Skip chart generation after the backtest completes")
    p.add_argument("--concurrency", type=int, default=4,
                   help="(LLM mode only) parallel API calls per symbol")

    llm_group = p.add_mutually_exclusive_group()
    llm_group.add_argument(
        "--no-llm", action="store_true",
        help=(
            "Use the pure-quant pipeline (zero LLM calls). "
            "Runs quant_indicator_node -> quant_trend_node -> quant_pattern_node -> "
            "make_trade_decision, matching master_portfolio.py --no-llm exactly. "
            "No API key required. (default)"
        ),
    )
    llm_group.add_argument(
        "--llm", dest="no_llm", action="store_false",
        help=(
            "Use the LangGraph LLM pipeline (overrides --no-llm default). "
            "Requires --provider and a valid API key."
        ),
    )
    p.set_defaults(no_llm=True)

    p.add_argument(
        "--provider", default=DEFAULT_LLM_PROVIDER,
        choices=["google", "openai", "anthropic", "qwen", "minimax"],
        help="LLM provider to use when --llm is set (default: %(default)s)",
    )
    p.add_argument(
        "--api-key", default=None, metavar="KEY",
        help=(
            "API key for the chosen provider. "
            "For 'google', auto-loaded from ../Gemini_API.txt if not provided."
        ),
    )

    args = p.parse_args()

    cfg = SimConfig(
        commission=args.commission,
        slippage=args.slippage,
        trend_filter=not args.no_trend_filter,
        confidence_gate=args.confidence_gate,
        use_atr_stops=not args.no_stops,
        atr_mult=args.atr_mult,
        allow_short=not args.no_short,
        periods_per_year=_periods_per_year(args.cadence),
    )

    use_llm = not args.no_llm
    llm_config = build_llm_config(args.provider, args.api_key) if use_llm else None

    if use_llm:
        provider = args.provider
        print(f"  LLM mode    : {provider}  ({DEFAULT_GEMINI_AGENT_MODEL if provider == 'google' else 'default model'})")

    asyncio.run(backtest_universe(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        period=args.period,
        cadence=args.cadence,
        window_bars=args.window_bars,
        concurrency=args.concurrency,
        out_dir=args.out,
        cfg=cfg,
        oos_start=args.oos_start,
        use_llm=use_llm,
        no_charts=args.no_charts,
        llm_config=llm_config,
    ))


if __name__ == "__main__":
    main()
