"""Backtest the multi-agent pipeline against the top 10 USA stocks over 5 years."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, asdict
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

import indicators  # noqa: F401  (warms up pandas-ta)
from quant_agents import run_pipeline_async


TOP10_USA = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "AVGO", "TSLA", "BRK-B", "LLY",
]

WINDOW_BARS = 60  # bars of history fed to the agent at each decision point


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


# ---------- Backtest engine ----------

@dataclass
class Trade:
    symbol: str
    decision_date: str
    direction: int           # +1 long, -1 short
    entry_price: float
    exit_price: float
    exit_date: str
    confidence: float
    pnl_pct: float           # signed return on the position
    justification: str


async def _decision_for_bar(symbol: str, df: pd.DataFrame, end_idx: int) -> tuple[int, float, str]:
    kline = window_to_kline(df, end_idx)
    try:
        result = await run_pipeline_async(symbol, kline, timeframe="1d")
        direction = 1 if result.decision.decision == "LONG" else -1
        return direction, float(result.decision.confidence), result.decision.justification
    except Exception as e:
        return 0, 0.0, f"ERROR: {type(e).__name__}: {e}"


async def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    rebalance_idxs: list[int],
    concurrency: int = 4,
) -> list[Trade]:
    sem = asyncio.Semaphore(concurrency)

    async def decide(idx: int) -> tuple[int, int, float, str]:
        async with sem:
            d, c, j = await _decision_for_bar(symbol, df, idx)
            return idx, d, c, j

    tasks = [decide(i) for i in rebalance_idxs]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda r: r[0])

    trades: list[Trade] = []
    for k, (idx, direction, conf, just) in enumerate(results):
        if direction == 0:
            continue
        # Enter at next bar's open, exit at the bar following the next rebalance point's close.
        entry_idx = idx + 1
        exit_idx = results[k + 1][0] + 1 if k + 1 < len(results) else len(df) - 1
        if entry_idx >= len(df) or exit_idx >= len(df) or exit_idx <= entry_idx:
            continue
        entry = float(df["Open"].iloc[entry_idx])
        exit_ = float(df["Open"].iloc[exit_idx])
        pnl = direction * (exit_ - entry) / entry
        trades.append(Trade(
            symbol=symbol,
            decision_date=str(df["Datetime"].iloc[idx].date()),
            direction=direction,
            entry_price=entry,
            exit_price=exit_,
            exit_date=str(df["Datetime"].iloc[exit_idx].date()),
            confidence=conf,
            pnl_pct=pnl,
            justification=just[:300],
        ))
    return trades


async def backtest_universe(
    symbols: Iterable[str],
    period: str = "5y",
    cadence: str = "W-FRI",
    concurrency: int = 4,
    out_dir: Path | str = "backtest_results",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_trades: dict[str, list[Trade]] = {}
    histories: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        print(f"[{symbol}] fetching {period} history...", flush=True)
        df = fetch_history(symbol, period=period)
        idxs = rebalance_dates(df, cadence=cadence)
        print(f"[{symbol}] {len(df)} bars, {len(idxs)} decision points", flush=True)
        histories[symbol] = df

        trades = await backtest_symbol(symbol, df, idxs, concurrency=concurrency)
        all_trades[symbol] = trades

        # Persist per-symbol trade log
        pd.DataFrame([asdict(t) for t in trades]).to_csv(
            out_dir / f"{symbol}_trades.csv", index=False
        )
        print(f"[{symbol}] {len(trades)} trades, mean PnL/trade = "
              f"{(sum(t.pnl_pct for t in trades) / len(trades) * 100 if trades else 0):.3f}%", flush=True)

    # Build a unified summary
    summary_rows = []
    for sym, trades in all_trades.items():
        df = histories[sym]
        bh_ret = (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1.0
        if trades:
            returns = pd.Series([t.pnl_pct for t in trades])
            equity = (1 + returns).cumprod()
            total_ret = equity.iloc[-1] - 1.0
            sharpe = (returns.mean() / returns.std() * (52 ** 0.5)) if returns.std() else 0.0
            win_rate = float((returns > 0).mean())
            mdd = float(((equity.cummax() - equity) / equity.cummax()).max())
        else:
            total_ret = 0.0
            sharpe = 0.0
            win_rate = 0.0
            mdd = 0.0
        summary_rows.append({
            "symbol": sym,
            "n_trades": len(trades),
            "agent_total_return": total_ret,
            "buy_hold_return": float(bh_ret),
            "excess_return": total_ret - float(bh_ret),
            "sharpe_annual": sharpe,
            "win_rate": win_rate,
            "max_drawdown": mdd,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    # Persist combined trades JSON for the visualizer
    combined = {sym: [asdict(t) for t in trades] for sym, trades in all_trades.items()}
    (out_dir / "all_trades.json").write_text(json.dumps(combined, indent=2))

    # Persist histories for the visualizer
    for sym, df in histories.items():
        df.to_csv(out_dir / f"{sym}_history.csv", index=False)

    return {"summary": summary_df, "trades": all_trades, "histories": histories}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=TOP10_USA)
    parser.add_argument("--period", default="5y")
    parser.add_argument("--cadence", default="W-FRI",
                        help="Pandas resample alias (W-FRI=weekly Friday, M=monthly). Use 'D' for daily.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", default="backtest_results")
    args = parser.parse_args()

    asyncio.run(backtest_universe(
        symbols=args.symbols,
        period=args.period,
        cadence=args.cadence,
        concurrency=args.concurrency,
        out_dir=args.out,
    ))


if __name__ == "__main__":
    main()
