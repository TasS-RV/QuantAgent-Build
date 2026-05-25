"""Visualize backtest results: per-stock equity curves + portfolio summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def _equity_curve_from_trades(trades_df: pd.DataFrame) -> pd.Series:
    if trades_df.empty:
        return pd.Series(dtype=float)
    trades_df = trades_df.sort_values("decision_date").reset_index(drop=True)
    eq = (1 + trades_df["pnl_pct"]).cumprod()
    eq.index = pd.to_datetime(trades_df["exit_date"])
    return eq


def _buy_hold_curve(history_df: pd.DataFrame, ref_dates: pd.DatetimeIndex) -> pd.Series:
    h = history_df.copy()
    h["Datetime"] = pd.to_datetime(h["Datetime"])
    h = h.set_index("Datetime").sort_index()
    close = h["Close"]
    bh = close / close.iloc[0]
    return bh.reindex(close.index.union(ref_dates)).ffill().reindex(ref_dates)


def render(results_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    summary = pd.read_csv(results_dir / "summary.csv")
    all_trades = json.loads((results_dir / "all_trades.json").read_text())

    symbols = sorted(all_trades.keys())
    n = len(symbols)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    # --- Per-stock equity curves ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows))
    axes = np.atleast_2d(axes).flatten()
    for ax, sym in zip(axes, symbols):
        trades_df = pd.DataFrame(all_trades[sym])
        hist = pd.read_csv(results_dir / f"{sym}_history.csv")
        agent_eq = _equity_curve_from_trades(trades_df)
        if agent_eq.empty:
            ax.set_title(f"{sym}: no trades")
            continue
        bh_eq = _buy_hold_curve(hist, agent_eq.index)

        ax.plot(agent_eq.index, agent_eq.values, label="Agent", linewidth=1.6, color="#1f6feb")
        ax.plot(bh_eq.index, bh_eq.values, label="Buy & Hold", linewidth=1.4, color="#888", linestyle="--")
        ax.axhline(1.0, color="#000", linewidth=0.5, alpha=0.3)
        row = summary[summary["symbol"] == sym].iloc[0]
        ax.set_title(
            f"{sym}  agent={row['agent_total_return']*100:.1f}%  "
            f"B&H={row['buy_hold_return']*100:.1f}%  "
            f"win={row['win_rate']*100:.0f}%  "
            f"sharpe={row['sharpe_annual']:.2f}"
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.25)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("QuantAgent vs Buy & Hold — Top 10 USA, 5y, weekly rebalance", fontsize=14)
    fig.tight_layout()
    fig.savefig(results_dir / "equity_curves.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- Aggregate portfolio (equal-weighted) ---
    all_dates = set()
    eqs = {}
    for sym in symbols:
        trades_df = pd.DataFrame(all_trades[sym])
        eq = _equity_curve_from_trades(trades_df)
        if eq.empty:
            continue
        eqs[sym] = eq
        all_dates.update(eq.index)
    if eqs:
        idx = pd.DatetimeIndex(sorted(all_dates))
        aligned = pd.DataFrame({sym: eq.reindex(idx).ffill().fillna(1.0) for sym, eq in eqs.items()})
        portfolio = aligned.mean(axis=1)

        # Buy & hold portfolio
        bh_aligned = {}
        for sym in symbols:
            hist = pd.read_csv(results_dir / f"{sym}_history.csv")
            bh_aligned[sym] = _buy_hold_curve(hist, idx).ffill().fillna(1.0)
        bh_portfolio = pd.DataFrame(bh_aligned).mean(axis=1)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(portfolio.index, portfolio.values, label="Agent portfolio (EW)", linewidth=2.2, color="#1f6feb")
        ax.plot(bh_portfolio.index, bh_portfolio.values, label="Buy & Hold portfolio (EW)",
                linewidth=2.0, color="#888", linestyle="--")
        ax.axhline(1.0, color="#000", linewidth=0.5, alpha=0.3)
        ax.set_title(
            f"Equal-Weighted Portfolio: Agent={portfolio.iloc[-1]-1:+.1%}  "
            f"B&H={bh_portfolio.iloc[-1]-1:+.1%}"
        )
        ax.set_ylabel("Equity (start=1.0)")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(results_dir / "portfolio_equity.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    # --- Bar chart: agent vs B&H total return per symbol ---
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(summary))
    w = 0.4
    ax.bar(x - w / 2, summary["agent_total_return"] * 100, w, label="Agent", color="#1f6feb")
    ax.bar(x + w / 2, summary["buy_hold_return"] * 100, w, label="Buy & Hold", color="#888")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["symbol"], rotation=0)
    ax.set_ylabel("Total Return (%)")
    ax.set_title("Total Return: Agent vs Buy & Hold")
    ax.axhline(0, color="#000", linewidth=0.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / "returns_bar.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- Console summary ---
    print("\n=== Summary ===")
    s = summary.copy()
    pct_cols = ["agent_total_return", "buy_hold_return", "excess_return", "win_rate", "max_drawdown"]
    for c in pct_cols:
        s[c] = (s[c] * 100).round(2)
    s["sharpe_annual"] = s["sharpe_annual"].round(2)
    print(s.to_string(index=False))
    print(f"\nWrote: {results_dir / 'equity_curves.png'}")
    print(f"Wrote: {results_dir / 'portfolio_equity.png'}")
    print(f"Wrote: {results_dir / 'returns_bar.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="backtest_results")
    args = p.parse_args()
    render(args.results)


if __name__ == "__main__":
    main()
