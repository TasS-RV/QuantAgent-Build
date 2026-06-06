"""
Classic quant strategy lab — pure math, zero LLM, vectorized + event-driven.

Implements well-documented trend / momentum strategies and backtests them with a
position simulator that supports **trailing-stop exits** (let winners run) — the
key capability the original signal/horizon engine lacked.

Strategies
----------
  • Donchian breakout (Turtle)      — buy N-day high breakout, exit M-day low
  • Moving-average crossover         — golden/death cross
  • Time-series momentum (CTA)       — sign of trailing L-day return
  • Dual momentum (rotation)         — hold the strongest symbol, cash when all weak

Exit modes
----------
  • "regime"   — exit only when the entry signal reverses (or flips for long-short)
  • "trailing" — ATR ratchet trailing stop *plus* regime reversal, whichever first

Direction modes
---------------
  • "long_flat"  — long or cash only (recommended for secular bull names)
  • "long_short" — also short on bearish regime

Reuses SimConfig / SimTrade / compute_metrics / buy_hold_return / atr from
backtest_engine.py so all metrics are directly comparable to the prior 12 configs.

Run:
    python strategy_lab.py                  # full matrix, all symbols
    python strategy_lab.py --symbols GOOGL  # single symbol
    python strategy_lab.py --out strategy_results
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

# Shim sqlite for environments without _sqlite3 (yfinance caches via sqlite).
try:
    import sqlite3  # noqa: F401
except ImportError:  # pragma: no cover
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3

import numpy as np
import pandas as pd
import yfinance as yf

from backtest_engine import (
    SimConfig,
    SimTrade,
    atr,
    buy_hold_return,
    compute_metrics,
    sma,
    trades_to_frame,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

SYMBOLS = ["GOOGL", "XOM", "JNJ"]
START_DATE = "2015-01-01"
END_DATE: str | None = None
ATR_TRAIL_MULT = 3.0           # trailing stop = 3 × ATR(14)
TRADING_DAYS_PER_YEAR = 252


# ─────────────────────────────────────────────────────────────────────────────
#  Data
# ─────────────────────────────────────────────────────────────────────────────

def fetch_history(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    """Daily OHLC, auto-adjusted. Pulls extra warmup before `start`."""
    fetch_start = (
        (pd.Timestamp(start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
        if start else None
    )
    raw = yf.download(
        symbol,
        start=fetch_start,
        end=end,
        period=None if start else "max",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError(f"No data for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index().rename(columns={raw.reset_index().columns[0]: "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    keep = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy signal generators  →  target exposure per bar, in {-1, 0, +1}
#  target[i] is decided using data through bar i (causal); executed at bar i+1 open.
# ─────────────────────────────────────────────────────────────────────────────

def donchian_target(df: pd.DataFrame, entry_n: int, exit_n: int,
                    allow_short: bool) -> np.ndarray:
    """Turtle breakout. Long on N-high breakout, exit on M-low; symmetric short."""
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    close = df["Close"].astype(float).values
    n = len(df)

    # Prior-window channels (shift by 1 so the breakout bar isn't part of its own channel)
    hh_entry = pd.Series(high).rolling(entry_n).max().shift(1).values
    ll_exit = pd.Series(low).rolling(exit_n).min().shift(1).values
    ll_entry = pd.Series(low).rolling(entry_n).min().shift(1).values
    hh_exit = pd.Series(high).rolling(exit_n).max().shift(1).values

    target = np.zeros(n)
    state = 0
    for i in range(n):
        if np.isnan(hh_entry[i]) or np.isnan(ll_exit[i]):
            target[i] = 0
            continue
        if state == 1:
            if close[i] < ll_exit[i]:
                state = 0
        elif state == -1:
            if close[i] > hh_exit[i]:
                state = 0
        if state == 0:
            if close[i] > hh_entry[i]:
                state = 1
            elif allow_short and close[i] < ll_entry[i]:
                state = -1
        target[i] = state
    return target


def ma_cross_target(df: pd.DataFrame, fast: int, slow: int,
                    allow_short: bool) -> np.ndarray:
    """Golden/death cross. Long when fast MA > slow MA."""
    close = df["Close"].astype(float)
    f = close.rolling(fast).mean().values
    s = close.rolling(slow).mean().values
    target = np.where(f > s, 1.0, (-1.0 if allow_short else 0.0))
    target[np.isnan(f) | np.isnan(s)] = 0.0
    return target


def tsmom_target(df: pd.DataFrame, lookback: int, allow_short: bool,
                 ma_filter: int = 0) -> np.ndarray:
    """Time-series momentum: sign of trailing L-day return, optional MA confirmation."""
    close = df["Close"].astype(float)
    mom = (close / close.shift(lookback) - 1.0).values
    target = np.where(mom > 0, 1.0, (-1.0 if allow_short else 0.0))
    if ma_filter:
        ma = close.rolling(ma_filter).mean().values
        # only long above MA, only short below MA
        long_ok = close.values > ma
        target = np.where((target > 0) & ~long_ok, 0.0, target)
        target = np.where((target < 0) & long_ok, 0.0, target)
    target[np.isnan(mom)] = 0.0
    return target


# ─────────────────────────────────────────────────────────────────────────────
#  Event-driven position simulator (supports trailing stops)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_target(symbol: str, df: pd.DataFrame, target: np.ndarray,
                    cfg: SimConfig, *, exit_mode: str = "trailing",
                    atr_mult: float = ATR_TRAIL_MULT,
                    warmup: int = 0) -> List[SimTrade]:
    """
    Walk bars, holding the position dictated by `target`, executing at next-bar open.

    exit_mode:
      "regime"   — exit only when target changes
      "trailing" — additionally exit on an ATR ratchet trailing stop
    After a trailing-stop exit, suppress re-entry in the same direction until the
    target signal resets (goes to 0 or flips), to avoid immediate whipsaw re-entry.
    """
    df = df.reset_index(drop=True)
    open_ = df["Open"].astype(float).values
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    close = df["Close"].astype(float).values
    atr_series = atr(df, cfg.atr_period).values
    dates = pd.to_datetime(df["Datetime"]).dt.date.astype(str).tolist()
    n = len(df)

    trades: List[SimTrade] = []
    pos = 0
    entry_idx = 0
    entry_price = 0.0
    trail_stop = 0.0
    latch_dir = 0  # direction we were stopped out of; suppresses same-dir re-entry

    def _close_trade(exit_idx: int, exit_price: float, reason: str):
        nonlocal pos
        gross = pos * (exit_price - entry_price) / entry_price
        holding = max(1, exit_idx - entry_idx)
        cost = (cfg.commission + cfg.slippage) * 2.0
        if pos == -1:
            cost += cfg.borrow_annual * (holding / cfg.trading_days_per_year)
        trades.append(SimTrade(
            symbol=symbol, decision_date=dates[entry_idx], direction=pos,
            entry_date=dates[entry_idx], entry_price=round(entry_price, 4),
            exit_date=dates[exit_idx], exit_price=round(exit_price, 4),
            exit_reason=reason, confidence=1.0, holding_days=holding,
            gross_pnl_pct=round(gross, 5), cost_pct=round(cost, 5),
            pnl_pct=round(gross - cost, 5), note=f"{exit_mode}",
        ))
        pos = 0

    for i in range(max(1, warmup), n):
        # ── 1) Manage open position with bar i's range (trailing stop) ──
        if pos != 0 and exit_mode == "trailing":
            if pos == 1 and low[i] <= trail_stop:
                px = trail_stop if open_[i] >= trail_stop else open_[i]
                _close_trade(i, px, "trail_stop")
                latch_dir = 1
            elif pos == -1 and high[i] >= trail_stop:
                px = trail_stop if open_[i] <= trail_stop else open_[i]
                _close_trade(i, px, "trail_stop")
                latch_dir = -1

        # ── 2) Act on the target decided at bar i-1, executed at bar i open ──
        tgt = int(target[i - 1])
        if not cfg.allow_short and tgt < 0:
            tgt = 0
        # clear the stop latch once the signal resets/flips
        if latch_dir != 0 and tgt != latch_dir:
            latch_dir = 0
        # suppress re-entry into the same direction we were just stopped out of
        if latch_dir != 0 and tgt == latch_dir:
            tgt = 0

        if tgt != pos:
            if pos != 0:
                _close_trade(i, open_[i], "regime")
            if tgt != 0:
                pos = tgt
                entry_idx = i
                entry_price = open_[i]
                a = atr_series[i] * atr_mult
                trail_stop = entry_price - a if pos == 1 else entry_price + a

        # ── 3) Ratchet the trailing stop using bar i close ──
        if pos != 0 and exit_mode == "trailing":
            a = atr_series[i] * atr_mult
            if pos == 1:
                trail_stop = max(trail_stop, close[i] - a)
            else:
                trail_stop = min(trail_stop, close[i] + a)

    # Close any residual position at the final close
    if pos != 0:
        _close_trade(n - 1, close[n - 1], "final")

    return trades


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy registry  (per-symbol strategies)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StratSpec:
    name: str
    gen: Callable[[pd.DataFrame, bool], np.ndarray]   # (df, allow_short) -> target
    warmup: int


PER_SYMBOL_STRATEGIES: List[StratSpec] = [
    StratSpec("donchian_20_10", lambda d, s: donchian_target(d, 20, 10, s), 21),
    StratSpec("donchian_55_20", lambda d, s: donchian_target(d, 55, 20, s), 56),
    StratSpec("ma_cross_20_100", lambda d, s: ma_cross_target(d, 20, 100, s), 100),
    StratSpec("ma_cross_50_200", lambda d, s: ma_cross_target(d, 50, 200, s), 200),
    StratSpec("tsmom_60", lambda d, s: tsmom_target(d, 60, s), 60),
    StratSpec("tsmom_120", lambda d, s: tsmom_target(d, 120, s), 120),
]

EXIT_MODES = ["regime", "trailing"]
DIRECTION_MODES = ["long_flat", "long_short"]


# ─────────────────────────────────────────────────────────────────────────────
#  Dual momentum (portfolio-level rotation)
# ─────────────────────────────────────────────────────────────────────────────

def dual_momentum(histories: dict[str, pd.DataFrame], lookback: int,
                  cfg: SimConfig, start_date: str | None) -> dict:
    """
    Monthly: hold the single symbol with the highest trailing `lookback`-day return,
    but only if that return is positive (absolute momentum filter); else go to cash.
    Returns metrics on the resulting portfolio equity curve.
    """
    # Align all symbols on a common date index
    closes = pd.DataFrame({
        sym: df.set_index("Datetime")["Close"].astype(float)
        for sym, df in histories.items()
    }).dropna()
    if start_date:
        closes = closes[closes.index >= pd.Timestamp(start_date) - pd.Timedelta(days=lookback * 2)]

    # Month-end rebalance dates
    month_ends = closes.resample("ME").last().index
    month_ends = [d for d in month_ends if d in closes.index]

    trades: List[SimTrade] = []
    daily_ret = closes.pct_change().fillna(0.0)
    held: Optional[str] = None
    held_since: Optional[pd.Timestamp] = None
    entry_px = 0.0

    for k, d in enumerate(month_ends):
        if start_date and str(d.date()) < start_date:
            continue
        window = closes.loc[:d]
        if len(window) <= lookback:
            continue
        mom = window.iloc[-1] / window.iloc[-lookback - 1] - 1.0
        best = mom.idxmax()
        target = best if mom[best] > 0 else None

        if target != held:
            # close held position at this date's close
            if held is not None:
                exit_px = float(closes.loc[d, held])
                gross = (exit_px - entry_px) / entry_px
                cost = (cfg.commission + cfg.slippage) * 2.0
                trades.append(SimTrade(
                    symbol=held, decision_date=str(held_since.date()), direction=1,
                    entry_date=str(held_since.date()), entry_price=round(entry_px, 4),
                    exit_date=str(d.date()), exit_price=round(exit_px, 4),
                    exit_reason="rotate", confidence=1.0,
                    holding_days=max(1, (d - held_since).days),
                    gross_pnl_pct=round(gross, 5), cost_pct=round(cost, 5),
                    pnl_pct=round(gross - cost, 5), note=f"dualmom_{lookback}",
                ))
            held = target
            if held is not None:
                entry_px = float(closes.loc[d, held])
                held_since = d

    if held is not None:
        last_d = closes.index[-1]
        exit_px = float(closes.loc[last_d, held])
        gross = (exit_px - entry_px) / entry_px
        cost = (cfg.commission + cfg.slippage) * 2.0
        trades.append(SimTrade(
            symbol=held, decision_date=str(held_since.date()), direction=1,
            entry_date=str(held_since.date()), entry_price=round(entry_px, 4),
            exit_date=str(last_d.date()), exit_price=round(exit_px, 4),
            exit_reason="final", confidence=1.0,
            holding_days=max(1, (last_d - held_since).days),
            gross_pnl_pct=round(gross, 5), cost_pct=round(cost, 5),
            pnl_pct=round(gross - cost, 5), note=f"dualmom_{lookback}",
        ))

    m = compute_metrics(trades, cfg)
    # Equal-weight buy&hold of the 3 names as the reference
    bh = float((closes.iloc[-1] / closes.iloc[0] - 1.0).mean())
    m["buy_hold_return"] = bh
    m["excess_return"] = m["total_return"] - bh
    return m


# ─────────────────────────────────────────────────────────────────────────────
#  Emit backtest_results-style directory + charts (reuses visualize.render)
# ─────────────────────────────────────────────────────────────────────────────

def emit_visual_dir(out_root: Path, cfg_name: str, symbols: List[str],
                    histories: dict[str, pd.DataFrame],
                    trades_by_sym: dict[str, List[SimTrade]],
                    cfg: SimConfig, start: str | None, title: str) -> Path:
    """
    Write per-config output files in the exact schema produced by backtest.py
    (summary.csv, all_trades.json, all_signals.json, {sym}_history.csv,
    {sym}_trades.csv) then call visualize.render() to produce the PNG charts:
      • <sym>_detail.png      price + entry/exit markers + per-trade P&L + equity
      • equity_curves.png     per-symbol agent vs B&H grid
      • portfolio_equity.png  equal-weighted portfolio equity
      • returns_bar.png       total-return bar chart
    """
    cdir = out_root / "configs" / cfg_name
    cdir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []
    all_trades: dict[str, list] = {}
    all_signals: dict[str, list] = {}

    for sym in symbols:
        df = histories[sym]
        trades = trades_by_sym.get(sym, [])
        m = compute_metrics(trades, cfg)
        first_idx = (
            int((df["Datetime"] >= pd.Timestamp(start)).idxmax()) if start else 0
        )
        bh = buy_hold_return(df, first_idx)

        summary_rows.append({
            "symbol":             sym,
            "n_trades":           m["n_trades"],
            "n_long":             m["n_long"],
            "n_short":            m["n_short"],
            "agent_total_return": m["total_return"],
            "buy_hold_return":    bh,
            "excess_return":      m["total_return"] - bh,
            "sharpe_annual":      m["sharpe_annual"],
            "win_rate":           m["win_rate"],
            "max_drawdown":       m["max_drawdown"],
        })

        trades_to_frame(trades).to_csv(cdir / f"{sym}_trades.csv", index=False)
        all_trades[sym] = [asdict(t) for t in trades]
        all_signals[sym] = [
            {"date": t.entry_date, "direction": t.direction, "note": t.note}
            for t in trades
        ]
        df.to_csv(cdir / f"{sym}_history.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(cdir / "summary.csv", index=False)
    (cdir / "all_trades.json").write_text(json.dumps(all_trades, indent=2, default=str))
    (cdir / "all_signals.json").write_text(json.dumps(all_signals, indent=2, default=str))

    try:
        from visualize import render as _render
        _render(cdir, title=title)
    except Exception as e:  # pragma: no cover
        print(f"  [WARN] chart render failed for {cfg_name}: {e}")

    return cdir


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_matrix(symbols: List[str], start: str | None, end: str | None,
               out_dir: Path, emit_charts: bool = True) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = SimConfig(
        commission=0.0002, slippage=0.0005, borrow_annual=0.01,
        trend_filter=False, confidence_gate=0.0, use_atr_stops=False,
        periods_per_year=252.0,
    )

    histories: dict[str, pd.DataFrame] = {}
    print(f"Fetching {len(symbols)} symbols ...")
    for sym in symbols:
        histories[sym] = fetch_history(sym, start, end)
        print(f"  {sym:<6} {len(histories[sym])} bars "
              f"({histories[sym]['Datetime'].iloc[0].date()} -> "
              f"{histories[sym]['Datetime'].iloc[-1].date()})")

    rows: List[dict] = []
    config_trades: dict[str, dict[str, List[SimTrade]]] = {}
    config_cfg: dict[str, SimConfig] = {}
    for spec in PER_SYMBOL_STRATEGIES:
        for direction in DIRECTION_MODES:
            allow_short = direction == "long_short"
            for exit_mode in EXIT_MODES:
                cfg_name = f"{spec.name}__{direction}__{exit_mode}"
                config_trades[cfg_name] = {}
                for sym in symbols:
                    df = histories[sym]
                    cfg = SimConfig(**{**base_cfg.__dict__, "allow_short": allow_short})
                    config_cfg[cfg_name] = cfg
                    # restrict decisions to >= start_date
                    target = spec.gen(df, allow_short)
                    if start:
                        before = df["Datetime"] < pd.Timestamp(start)
                        target[before.values] = 0.0
                    trades = simulate_target(
                        sym, df, target, cfg,
                        exit_mode=exit_mode, warmup=spec.warmup,
                    )
                    config_trades[cfg_name][sym] = trades
                    m = compute_metrics(trades, cfg)
                    # buy & hold from first in-range bar
                    first_idx = int((df["Datetime"] >= pd.Timestamp(start)).idxmax()) if start else 0
                    bh = buy_hold_return(df, first_idx)
                    rows.append({
                        "strategy": spec.name,
                        "direction": direction,
                        "exit": exit_mode,
                        "symbol": sym,
                        "n_trades": m["n_trades"],
                        "total_return": m["total_return"],
                        "buy_hold": bh,
                        "excess_return": m["total_return"] - bh,
                        "sharpe": m["sharpe_annual"],
                        "win_rate": m["win_rate"],
                        "avg_pnl_pct": m["avg_pnl_pct"],
                        "max_dd": m["max_drawdown"],
                        "n_long": m["n_long"],
                        "n_short": m["n_short"],
                    })

    # Dual momentum (portfolio level)
    for lb in (90, 180):
        m = dual_momentum(histories, lb, base_cfg, start)
        rows.append({
            "strategy": f"dual_momentum_{lb}",
            "direction": "rotation",
            "exit": "monthly",
            "symbol": "PORTFOLIO",
            "n_trades": m["n_trades"],
            "total_return": m["total_return"],
            "buy_hold": m["buy_hold_return"],
            "excess_return": m["excess_return"],
            "sharpe": m["sharpe_annual"],
            "win_rate": m["win_rate"],
            "avg_pnl_pct": m["avg_pnl_pct"],
            "max_dd": m["max_drawdown"],
            "n_long": m["n_long"],
            "n_short": m["n_short"],
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "summary.csv", index=False)

    # Aggregate per (strategy, direction, exit) across symbols
    agg = (
        df_out[df_out["symbol"] != "PORTFOLIO"]
        .groupby(["strategy", "direction", "exit"], as_index=False)
        .agg(
            avg_return=("total_return", "mean"),
            avg_excess=("excess_return", "mean"),
            avg_sharpe=("sharpe", "mean"),
            avg_win=("win_rate", "mean"),
            avg_trades=("n_trades", "mean"),
            avg_max_dd=("max_dd", "mean"),
        )
        .sort_values("avg_excess", ascending=False)
    )
    agg.to_csv(out_dir / "aggregate.csv", index=False)

    # ── Emit per-config directories + charts (backtest_results format) ──────────
    if emit_charts:
        print(f"\nRendering per-config charts into {out_dir / 'configs'} ...")
        for cfg_name, trades_by_sym in config_trades.items():
            title = f"Strategy Lab — {cfg_name.replace('__', '  ·  ')}  |  2015→today  |  daily"
            cdir = emit_visual_dir(
                out_dir, cfg_name, symbols, histories, trades_by_sym,
                config_cfg[cfg_name], start, title,
            )
            print(f"  [{cfg_name}] -> {cdir}")

    return df_out, agg


def _fmt_pct(x: float) -> str:
    return f"{x*100:+.1f}%"


def main():
    p = argparse.ArgumentParser(description="Classic quant strategy lab")
    p.add_argument("--symbols", nargs="+", default=SYMBOLS)
    p.add_argument("--start", default=START_DATE)
    p.add_argument("--end", default=END_DATE)
    p.add_argument("--out", default="strategy_results")
    args = p.parse_args()

    df_out, agg = run_matrix(args.symbols, args.start, args.end, Path(args.out))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)

    print("\n" + "=" * 78)
    print("AGGREGATE — per strategy/direction/exit, averaged across symbols")
    print("=" * 78)
    show = agg.copy()
    for c in ("avg_return", "avg_excess", "avg_max_dd"):
        show[c] = show[c].map(_fmt_pct)
    show["avg_win"] = (agg["avg_win"] * 100).map(lambda v: f"{v:.0f}%")
    show["avg_sharpe"] = agg["avg_sharpe"].map(lambda v: f"{v:.2f}")
    show["avg_trades"] = agg["avg_trades"].map(lambda v: f"{v:.0f}")
    print(show.to_string(index=False))

    print("\n" + "=" * 78)
    print("DUAL MOMENTUM (portfolio rotation)")
    print("=" * 78)
    dm = df_out[df_out["symbol"] == "PORTFOLIO"]
    for _, r in dm.iterrows():
        print(f"  {r['strategy']:<18} return {_fmt_pct(r['total_return'])}  "
              f"vs B&H {_fmt_pct(r['buy_hold'])}  "
              f"excess {_fmt_pct(r['excess_return'])}  "
              f"sharpe {r['sharpe']:.2f}  trades {r['n_trades']}")

    print(f"\nFull per-symbol detail written to {args.out}/summary.csv")
    print(f"Aggregate written to {args.out}/aggregate.csv")


if __name__ == "__main__":
    main()
