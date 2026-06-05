"""Backtest the multi-agent pipeline against the top 10 USA stocks over 5 years."""

from __future__ import annotations

import argparse
import asyncio
import json
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

import indicators  # noqa: F401  (warms up pandas-ta)
from quant_agents import run_pipeline_async
from backtest_engine import Signal, SimConfig, simulate, run_with_baselines, trades_to_frame


TOP10_USA = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "AVGO", "TSLA", "BRK-B", "LLY",
]

WINDOW_BARS = 60  # bars of history fed to the agent at each decision point

# Approximate rebalance periods per year, by cadence — used to annualize Sharpe.
_CADENCE_PPY = {"D": 252.0, "DAILY": 252.0, "W-FRI": 52.0, "W": 52.0, "M": 12.0, "ME": 12.0}


def _periods_per_year(cadence: str) -> float:
    return _CADENCE_PPY.get(cadence.upper(), 52.0)


# ---------- Data ----------

def _normalize_yf(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Datetime"})
    keep = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    return df[[c for c in keep if c in df.columns]]


def fetch_history(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    raw = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"No data for {symbol}")
    df = _normalize_yf(raw)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df.reset_index(drop=True)


def rebalance_dates(df: pd.DataFrame, cadence: str = "W-FRI", warmup_bars: int = WINDOW_BARS) -> list[int]:
    """Indices of df where we make a trade decision (close of that bar)."""
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    if cadence.upper() in ("D", "DAILY"):
        return list(range(warmup_bars, len(df) - 1))
    s = df.set_index("Datetime")["Close"]
    period_ends = s.resample(cadence).last().dropna().index
    idx_map = {dt: i for i, dt in enumerate(df["Datetime"].tolist())}
    out = []
    for dt in period_ends:
        if dt in idx_map and idx_map[dt] >= warmup_bars and idx_map[dt] < len(df) - 1:
            out.append(idx_map[dt])
    return out


def window_to_kline(df: pd.DataFrame, end_idx: int, window: int = WINDOW_BARS) -> dict:
    start = max(0, end_idx - window + 1)
    sub = df.iloc[start : end_idx + 1].copy()
    return {
        "Datetime": sub["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "Open": sub["Open"].astype(float).tolist(),
        "High": sub["High"].astype(float).tolist(),
        "Low": sub["Low"].astype(float).tolist(),
        "Close": sub["Close"].astype(float).tolist(),
    }


# ---------- Signal collection (LLM, non-deterministic) ----------

async def _signal_for_bar(symbol: str, df: pd.DataFrame, end_idx: int) -> tuple[int, int, float, float, str]:
    """Run the LLM pipeline at one decision bar. Returns (idx, direction, conf, rr, justification)."""
    kline = window_to_kline(df, end_idx)
    try:
        result = await run_pipeline_async(symbol, kline, timeframe="1d")
        direction = {"LONG": 1, "SHORT": -1, "HOLD": 0}.get(result.decision.decision, 0)
        return (end_idx, direction, float(result.decision.confidence),
                float(result.decision.risk_reward_ratio), result.decision.justification)
    except Exception as e:
        return end_idx, 0, 0.0, 2.0, f"ERROR: {type(e).__name__}: {e}"


async def collect_signals(
    symbol: str,
    df: pd.DataFrame,
    rebalance_idxs: list[int],
    concurrency: int = 4,
) -> list[Signal]:
    """Collect the agent's raw signals at each rebalance bar (the expensive LLM step)."""
    sem = asyncio.Semaphore(concurrency)

    async def one(idx: int):
        async with sem:
            return await _signal_for_bar(symbol, df, idx)

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


# ---------- Backtest orchestration (deterministic engine: backtest_engine.py) ----------

async def backtest_universe(
    symbols: Iterable[str],
    period: str = "5y",
    cadence: str = "W-FRI",
    concurrency: int = 4,
    out_dir: Path | str = "backtest_results",
    cfg: SimConfig | None = None,
    oos_start: str | None = None,
) -> dict:
    """
    Collect agent signals per symbol, then run the deterministic backtest engine
    (with risk overlays + costs) and trivial baselines for comparison.

    oos_start: optional 'YYYY-MM-DD' — also reports agent metrics on the
    out-of-sample slice (decisions on/after that date) to detect overfitting.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg or SimConfig(periods_per_year=_periods_per_year(cadence))

    all_signals: dict[str, list[Signal]] = {}
    all_trades: dict[str, list[dict]] = {}
    histories: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []
    baseline_rows: list[dict] = []

    symbols = list(symbols)
    sym_bar = tqdm(symbols, desc="symbols", unit="sym", dynamic_ncols=True)
    for symbol in sym_bar:
        sym_bar.set_postfix_str(f"fetching {symbol}")
        df = fetch_history(symbol, period=period)
        idxs = rebalance_dates(df, cadence=cadence)
        sym_bar.set_postfix_str(f"{symbol}: {len(df)}b / {len(idxs)} decisions")
        histories[symbol] = df

        signals = await collect_signals(symbol, df, idxs, concurrency=concurrency)
        all_signals[symbol] = signals

        sim = simulate(symbol, df, signals, cfg)
        trades = sim["trades"]
        m = sim["metrics"]
        bl = run_with_baselines(symbol, df, signals, idxs, cfg)

        # Persist per-symbol trade log
        trades_df = trades_to_frame(trades)
        trades_df.to_csv(out_dir / f"{symbol}_trades.csv", index=False)
        all_trades[symbol] = trades_df.to_dict(orient="records")

        row = {
            "symbol": symbol,
            "n_trades": m["n_trades"],
            "n_long": m["n_long"],
            "n_short": m["n_short"],
            "agent_total_return": m["total_return"],
            "buy_hold_return": bl["buy_hold_return"],
            "excess_return": m["total_return"] - bl["buy_hold_return"],
            "sharpe_annual": m["sharpe_annual"],
            "win_rate": m["win_rate"],
            "max_drawdown": m["max_drawdown"],
        }
        # Out-of-sample slice
        if oos_start:
            oos_sigs = [s for s in signals if s.date and s.date >= oos_start]
            oos_m = simulate(symbol, df, oos_sigs, cfg)["metrics"]
            row["oos_total_return"] = oos_m["total_return"]
            row["oos_sharpe"] = oos_m["sharpe_annual"]
            row["oos_win_rate"] = oos_m["win_rate"]
        summary_rows.append(row)

        baseline_rows.append({
            "symbol": symbol,
            "agent": m["total_return"],
            "buy_hold": bl["buy_hold_return"],
            "always_long": bl["always_long"]["total_return"],
            "sma_trend_follower": bl["sma_trend_follower"]["total_return"],
            "random": bl["random"]["total_return"],
            "beats_sma_baseline": m["total_return"] > bl["sma_trend_follower"]["total_return"],
        })

        sym_bar.write(
            f"[{symbol}] {m['n_trades']} trades "
            f"({m['n_long']}L/{m['n_short']}S) | agent {m['total_return']:+.1%} "
            f"vs B&H {bl['buy_hold_return']:+.1%} vs 200dma {bl['sma_trend_follower']['total_return']:+.1%}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(out_dir / "baselines.csv", index=False)

    # Persist signals + histories for the visualizer / vectorbt validation
    combined = {
        sym: [
            {"decision_idx": s.decision_idx, "direction": s.direction,
             "confidence": s.confidence, "risk_reward_ratio": s.risk_reward_ratio,
             "date": s.date, "note": s.note}
            for s in sigs
        ]
        for sym, sigs in all_signals.items()
    }
    (out_dir / "all_signals.json").write_text(json.dumps(combined, indent=2))
    # all_trades.json: simulated trades per symbol (consumed by visualize.py)
    (out_dir / "all_trades.json").write_text(json.dumps(all_trades, indent=2))
    for sym, df in histories.items():
        df.to_csv(out_dir / f"{sym}_history.csv", index=False)

    # Console verdict vs the 200dma baseline (roadmap pass/fail gate)
    if not baseline_df.empty:
        n_beat = int(baseline_df["beats_sma_baseline"].sum())
        print(f"\nAgent beats 200dma trend-follower on {n_beat}/{len(baseline_df)} symbols.")
        if n_beat <= len(baseline_df) / 2:
            print("⚠️  Agent does NOT cleanly beat the dumb 200dma baseline — "
                  "do not add complexity until it does (see docs/BACKTEST_IMPROVEMENT_ROADMAP.md).")

    return {"summary": summary_df, "baselines": baseline_df,
            "signals": all_signals, "histories": histories}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=TOP10_USA)
    parser.add_argument("--period", default="5y")
    parser.add_argument("--cadence", default="W-FRI",
                        help="Pandas resample alias (W-FRI=weekly Friday, M=monthly). Use 'D' for daily.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", default="backtest_results")
    # Risk overlays (roadmap Tier 1 & 2)
    parser.add_argument("--no-trend-filter", action="store_true",
                        help="Disable the 200dma trend filter (default: on)")
    parser.add_argument("--no-stops", action="store_true",
                        help="Disable ATR stop/target exits; hold to next rebalance")
    parser.add_argument("--confidence-gate", type=float, default=0.0,
                        help="Drop signals below this confidence (e.g. 0.75); 0 disables")
    parser.add_argument("--atr-mult", type=float, default=1.5, help="ATR multiple for stop distance")
    parser.add_argument("--commission", type=float, default=0.0002, help="Commission per side")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage per side")
    parser.add_argument("--no-short", action="store_true", help="Long-only (drop SHORT signals)")
    parser.add_argument("--oos-start", default=None,
                        help="Report out-of-sample metrics for decisions on/after YYYY-MM-DD")
    args = parser.parse_args()

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

    asyncio.run(backtest_universe(
        symbols=args.symbols,
        period=args.period,
        cadence=args.cadence,
        concurrency=args.concurrency,
        out_dir=args.out,
        cfg=cfg,
        oos_start=args.oos_start,
    ))


if __name__ == "__main__":
    main()
