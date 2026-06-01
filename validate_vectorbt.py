"""
Validate decision-agent / backtest signals with VectorBT (issue #4).

Consumes the ``all_signals.json`` produced by ``backtest.py`` (or any
compatible signals file) plus the per-symbol price history, converts the
dated directional signals into a forward-filled position series, and evaluates
it two ways:

  * **vectorbt** backend — the requested validation engine. Builds a
    ``vbt.Portfolio.from_signals`` (long/short) with fees + slippage and prints
    vectorbt's own stats. Used automatically when vectorbt is importable.
  * **engine** fallback — uses the project's deterministic ``backtest_engine``
    so the script always runs even where vectorbt (numba) can't be installed.

The signal→position conversion is backend-independent and unit-tested.

Usage
-----
    # After running:  python backtest.py --symbols NVDA AAPL --out backtest_results
    python validate_vectorbt.py --signals backtest_results/all_signals.json \
        --history-dir backtest_results --symbol NVDA --fees 0.0007

    # All symbols in the signals file
    python validate_vectorbt.py --signals backtest_results/all_signals.json \
        --history-dir backtest_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_signals(path: str | Path) -> dict:
    """
    Load a signals file. Supports:
      * backtest.py's all_signals.json:  {symbol: [{decision_idx, direction, ...}]}
      * master_portfolio --output JSON:  [{ticker, decision, ...}] (per-ticker latest)
    Returns {symbol: [signal_dict, ...]}.
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        return data
    if isinstance(data, list):  # master_portfolio results — one signal per ticker
        out: dict = {}
        word2dir = {"BUY": 1, "LONG": 1, "SELL": -1, "SHORT": -1, "HOLD": 0}
        for row in data:
            sym = row.get("ticker") or row.get("symbol")
            if not sym:
                continue
            out.setdefault(sym, []).append({
                "decision_idx": row.get("decision_idx", 0),
                "direction": word2dir.get(str(row.get("decision", "HOLD")).upper(), 0),
                "confidence": row.get("confidence", 1.0),
                "risk_reward_ratio": row.get("risk_reward_ratio", 2.0),
                "date": row.get("date"),
            })
        return out
    raise ValueError(f"Unrecognised signals file format: {type(data)}")


def load_history(history_dir: str | Path, symbol: str) -> pd.DataFrame:
    csv = Path(history_dir) / f"{symbol}_history.csv"
    if not csv.exists():
        raise FileNotFoundError(f"History not found: {csv}")
    df = pd.read_csv(csv)
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Signal → position conversion (backend-independent, testable)
# ─────────────────────────────────────────────────────────────────────────────

def signals_to_position(df: pd.DataFrame, signals: List[dict]) -> pd.Series:
    """
    Build a per-bar target position series in {-1, 0, +1}.

    Each signal sets the position from the bar *after* its decision_idx (entry on
    next open) and holds until the next signal's entry bar. Index aligns to df.
    """
    pos = pd.Series(0, index=df.index, dtype=int)
    sig = sorted(signals, key=lambda s: int(s["decision_idx"]))
    for k, s in enumerate(sig):
        start = int(s["decision_idx"]) + 1
        end = (int(sig[k + 1]["decision_idx"]) + 1) if k + 1 < len(sig) else len(df)
        start = min(max(start, 0), len(df))
        end = min(max(end, start), len(df))
        pos.iloc[start:end] = int(s["direction"])
    return pos


def position_to_signal_arrays(pos: pd.Series):
    """Convert a {-1,0,1} position series into vectorbt's 4 boolean signal arrays."""
    prev = pos.shift(1).fillna(0)
    long_entries = (pos == 1) & (prev != 1)
    long_exits = (pos != 1) & (prev == 1)
    short_entries = (pos == -1) & (prev != -1)
    short_exits = (pos != -1) & (prev == -1)
    return long_entries, long_exits, short_entries, short_exits


# ─────────────────────────────────────────────────────────────────────────────
#  Backends
# ─────────────────────────────────────────────────────────────────────────────

def vectorbt_available() -> bool:
    try:
        import vectorbt  # noqa: F401
        return True
    except Exception:
        return False


def validate_with_vectorbt(df: pd.DataFrame, signals: List[dict],
                           fees: float = 0.0007, slippage: float = 0.0005) -> dict:
    """Run vectorbt Portfolio.from_signals on the converted position series."""
    import vectorbt as vbt

    close = df["Close"].astype(float)
    if "Datetime" in df.columns:
        close.index = pd.to_datetime(df["Datetime"])
    pos = signals_to_position(df, signals)
    pos.index = close.index
    le, lx, se, sx = position_to_signal_arrays(pos)

    pf = vbt.Portfolio.from_signals(
        close, entries=le, exits=lx, short_entries=se, short_exits=sx,
        fees=fees, slippage=slippage, freq="1D",
    )
    bh = vbt.Portfolio.from_holding(close, freq="1D")
    return {
        "backend": "vectorbt",
        "total_return": float(pf.total_return()),
        "sharpe_ratio": float(pf.sharpe_ratio()),
        "max_drawdown": float(pf.max_drawdown()),
        "win_rate": float(pf.trades.win_rate()) if pf.trades.count() > 0 else 0.0,
        "n_trades": int(pf.trades.count()),
        "buy_hold_return": float(bh.total_return()),
    }


def validate_with_engine(df: pd.DataFrame, signals: List[dict],
                         fees: float = 0.0007, slippage: float = 0.0005) -> dict:
    """Fallback validation using the project's deterministic engine."""
    from backtest_engine import Signal, SimConfig, simulate, buy_hold_return

    cfg = SimConfig(commission=fees / 2, slippage=slippage,
                    trend_filter=False, use_atr_stops=False, periods_per_year=252)
    sigs = [Signal(decision_idx=int(s["decision_idx"]), direction=int(s["direction"]),
                   confidence=float(s.get("confidence", 1.0)),
                   risk_reward_ratio=float(s.get("risk_reward_ratio", 2.0)))
            for s in signals]
    m = simulate("VALIDATE", df, sigs, cfg)["metrics"]
    return {
        "backend": "engine",
        "total_return": m["total_return"],
        "sharpe_ratio": m["sharpe_annual"],
        "max_drawdown": m["max_drawdown"],
        "win_rate": m["win_rate"],
        "n_trades": m["n_trades"],
        "buy_hold_return": buy_hold_return(df, 0),
    }


def validate(df: pd.DataFrame, signals: List[dict], fees: float = 0.0007,
             slippage: float = 0.0005, backend: str = "auto") -> dict:
    use_vbt = backend == "vectorbt" or (backend == "auto" and vectorbt_available())
    if use_vbt:
        try:
            return validate_with_vectorbt(df, signals, fees, slippage)
        except Exception as e:  # fall back gracefully
            print(f"⚠️  vectorbt backend failed ({type(e).__name__}: {e}); using engine fallback.")
    return validate_with_engine(df, signals, fees, slippage)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _print_row(symbol: str, r: dict):
    print(f"{symbol:<8} [{r['backend']:^9}] "
          f"ret {r['total_return']:+.2%} | B&H {r['buy_hold_return']:+.2%} | "
          f"excess {r['total_return'] - r['buy_hold_return']:+.2%} | "
          f"Sharpe {r['sharpe_ratio']:+.2f} | MDD {r['max_drawdown']:.2%} | "
          f"win {r['win_rate']:.0%} | n {r['n_trades']}")


def main():
    p = argparse.ArgumentParser(description="Validate agent signals with VectorBT (issue #4).")
    p.add_argument("--signals", default="backtest_results/all_signals.json",
                   help="Signals JSON (backtest.py all_signals.json or master_portfolio --output)")
    p.add_argument("--history-dir", default="backtest_results",
                   help="Directory containing {SYMBOL}_history.csv files")
    p.add_argument("--symbol", default=None, help="Validate one symbol (default: all in file)")
    p.add_argument("--fees", type=float, default=0.0007, help="Round-trip fees fraction")
    p.add_argument("--slippage", type=float, default=0.0005, help="Slippage fraction")
    p.add_argument("--backend", choices=["auto", "vectorbt", "engine"], default="auto")
    p.add_argument("--output", default=None, help="Write results JSON to this path")
    args = p.parse_args()

    sig_map = load_signals(args.signals)
    symbols = [args.symbol] if args.symbol else list(sig_map.keys())
    if not vectorbt_available() and args.backend != "engine":
        print("ℹ️  vectorbt not installed — using the deterministic engine fallback "
              "(pip install vectorbt to use the vectorbt backend).")

    results = {}
    print(f"{'='*100}\n  Signal validation  |  fees={args.fees}  slippage={args.slippage}\n{'='*100}")
    for sym in symbols:
        try:
            df = load_history(args.history_dir, sym)
        except FileNotFoundError as e:
            print(f"{sym:<8} SKIP — {e}")
            continue
        r = validate(df, sig_map[sym], fees=args.fees, slippage=args.slippage, backend=args.backend)
        results[sym] = r
        _print_row(sym, r)

    if args.output and results:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.output}")
    return results


if __name__ == "__main__":
    main()
