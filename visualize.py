"""Visualize backtest results: per-stock detail charts + portfolio summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# ── Colour palette (dark-finance theme) ──────────────────────────────────────
_C = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "border":  "#30363d",
    "text":    "#c9d1d9",
    "muted":   "#8b949e",
    "green":   "#3fb950",
    "red":     "#f85149",
    "blue":    "#58a6ff",
    "orange":  "#d29922",
}


def _apply_dark(ax: plt.Axes) -> None:
    ax.set_facecolor(_C["panel"])
    ax.tick_params(colors=_C["text"], labelsize=8)
    ax.yaxis.label.set_color(_C["text"])
    ax.xaxis.label.set_color(_C["text"])
    ax.title.set_color(_C["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(_C["border"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=_C["border"], linewidth=0.5, alpha=0.5)


# ── Shared helpers ────────────────────────────────────────────────────────────

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
    close = h["Close"].astype(float)
    bh = close / close.iloc[0]
    return bh.reindex(close.index.union(ref_dates)).ffill().reindex(ref_dates)


# ── Per-symbol detail chart ───────────────────────────────────────────────────

def render_ticker_detail(
    sym: str,
    trades_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    signals: list[dict],
    out_dir: Path,
) -> Path:
    """
    3-panel chart for one symbol:
      Panel 1  Price line + entry/exit markers + holding-period shading
      Panel 2  Per-trade P&L bar chart (green = win, red = loss)
      Panel 3  Cumulative equity curve (agent vs buy & hold)

    Saved as  <out_dir>/<sym>_detail.png
    """
    fig = plt.figure(figsize=(16, 12), facecolor=_C["bg"])
    gs  = fig.add_gridspec(3, 1, height_ratios=[5, 3, 2.5], hspace=0.45)
    ax_price = fig.add_subplot(gs[0])
    ax_bars  = fig.add_subplot(gs[1])
    ax_eq    = fig.add_subplot(gs[2])

    for ax in (ax_price, ax_bars, ax_eq):
        _apply_dark(ax)

    # ── Prepare history ───────────────────────────────────────────────────────
    hist_df = hist_df.copy()
    hist_df["Datetime"] = pd.to_datetime(hist_df["Datetime"])
    hist_df = hist_df.sort_values("Datetime").reset_index(drop=True)
    dates_dt = hist_df["Datetime"]
    close    = hist_df["Close"].astype(float)

    # ── PANEL 1: Price + trade markers ───────────────────────────────────────
    ax_price.plot(dates_dt, close, color=_C["blue"], linewidth=1.2,
                  label="Close price", zorder=2)

    legend_handles = [
        Line2D([0], [0], color=_C["blue"], linewidth=1.5, label="Close"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=_C["green"],
               markersize=9, linestyle="None", label="Long entry"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=_C["red"],
               markersize=9, linestyle="None", label="Short entry"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_C["green"],
               markersize=7, linestyle="None", label="Exit (profit)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_C["red"],
               markersize=7, linestyle="None", label="Exit (loss)"),
    ]

    if not trades_df.empty:
        t_df = trades_df.copy()
        t_df["entry_date"] = pd.to_datetime(t_df["entry_date"])
        t_df["exit_date"]  = pd.to_datetime(t_df["exit_date"])

        for _, t in t_df.iterrows():
            is_long = int(t["direction"]) == 1
            is_win  = float(t["pnl_pct"]) > 0
            ec      = _C["green"] if is_long  else _C["red"]
            xc      = _C["green"] if is_win   else _C["red"]
            mk      = "^"         if is_long  else "v"

            # Shade holding period
            ax_price.axvspan(t["entry_date"], t["exit_date"],
                             alpha=0.07, color=xc, zorder=1)

            # Entry marker
            ax_price.scatter(t["entry_date"], t["entry_price"],
                             marker=mk, s=100, color=ec, zorder=5,
                             edgecolors=_C["bg"], linewidths=0.8)

            # Exit marker
            ax_price.scatter(t["exit_date"], t["exit_price"],
                             marker="o", s=70, color=xc, zorder=5,
                             edgecolors=_C["bg"], linewidths=0.8)

            # P&L annotation near exit
            sign_offset = 10 if is_win else -14
            ax_price.annotate(
                f"{float(t['pnl_pct'])*100:+.1f}%",
                xy=(t["exit_date"], t["exit_price"]),
                xytext=(4, sign_offset), textcoords="offset points",
                fontsize=7, color=xc, ha="left",
            )

    ax_price.legend(handles=legend_handles, loc="upper left",
                    fontsize=8, facecolor=_C["panel"],
                    labelcolor=_C["text"], framealpha=0.7)
    ax_price.set_title(f"{sym}  —  Price  ·  Entry / Exit Markers", fontsize=11)
    ax_price.set_ylabel("Price")
    ax_price.tick_params(axis="x", rotation=30)

    # ── PANEL 2: Per-trade P&L bars ───────────────────────────────────────────
    if not trades_df.empty:
        t_df = t_df.sort_values("decision_date").reset_index(drop=True)
        pnl_pcts  = t_df["pnl_pct"].values * 100
        bar_x     = np.arange(len(pnl_pcts))
        bar_cols  = [_C["green"] if p > 0 else _C["red"] for p in pnl_pcts]
        reasons   = t_df["exit_reason"].values
        dec_dates = pd.to_datetime(t_df["decision_date"])
        directions = t_df["direction"].values

        bars = ax_bars.bar(bar_x, pnl_pcts, color=bar_cols,
                           width=0.65, zorder=3, edgecolor=_C["border"], linewidth=0.4)

        # P&L label on each bar
        for bar, p in zip(bars, pnl_pcts):
            y_off = 0.15 if p >= 0 else -0.25
            va    = "bottom" if p >= 0 else "top"
            ax_bars.text(bar.get_x() + bar.get_width() / 2,
                         p + y_off, f"{p:+.1f}%",
                         ha="center", va=va, fontsize=7, color=_C["text"])

        # Direction + reason labels on x-axis
        xlabels = []
        for d, r, di in zip(dec_dates, reasons, directions):
            dir_str = "L" if di == 1 else "S"
            xlabels.append(f"{d.strftime('%b %d')}\n{dir_str}·{r}")
        ax_bars.set_xticks(bar_x)
        ax_bars.set_xticklabels(xlabels, fontsize=7, rotation=0, ha="center")

        ax_bars.axhline(0, color=_C["muted"], linewidth=0.8)
        ax_bars.set_ylabel("P&L per trade (%)")
        wins  = int((t_df["pnl_pct"] > 0).sum())
        total = len(t_df)
        net   = t_df["pnl_pct"].sum() * 100
        ax_bars.set_title(
            f"Per-Trade P&L   ·   {total} trades  ·  "
            f"{wins} wins ({wins/total:.0%})  ·  net {net:+.1f}%  "
            f"   (L=Long  S=Short  ·  exit: stop/target/horizon)",
            fontsize=9,
        )
    else:
        ax_bars.text(0.5, 0.5, "No trades executed", ha="center", va="center",
                     transform=ax_bars.transAxes, color=_C["muted"], fontsize=12)

    # ── PANEL 3: Cumulative equity ────────────────────────────────────────────
    if not trades_df.empty:
        t_df_sorted = t_df.sort_values("exit_date").reset_index(drop=True)
        eq      = (1 + t_df_sorted["pnl_pct"].values).cumprod()
        eq_x    = np.arange(len(eq))
        eq_dates = pd.to_datetime(t_df_sorted["exit_date"])

        ax_eq.plot(eq_x, eq, color=_C["blue"], linewidth=2.0,
                   label=f"Agent  ({eq[-1]-1:+.1%})", zorder=3)
        ax_eq.fill_between(eq_x, 1.0, eq,
                           where=eq >= 1.0, alpha=0.15, color=_C["green"])
        ax_eq.fill_between(eq_x, 1.0, eq,
                           where=eq  < 1.0, alpha=0.15, color=_C["red"])
        ax_eq.axhline(1.0, color=_C["muted"], linewidth=0.8, linestyle="--")

        # Buy & hold reference line (normalised to same start date)
        bh_start_dt = eq_dates.iloc[0]
        hist_after  = hist_df[hist_df["Datetime"] >= bh_start_dt].reset_index(drop=True)
        if len(hist_after) > 0:
            bh_vals = (hist_after["Close"].astype(float)
                       / float(hist_after["Close"].iloc[0])).values
            # sample B&H at same number of points as trades
            sample_idx = np.linspace(0, len(bh_vals) - 1, len(eq), dtype=int)
            bh_sampled = bh_vals[sample_idx]
            ax_eq.plot(eq_x, bh_sampled, color=_C["muted"], linewidth=1.4,
                       linestyle="--",
                       label=f"B&H  ({bh_sampled[-1]-1:+.1%})", zorder=2)

        ax_eq.set_xticks(eq_x[::max(1, len(eq_x)//10)])
        ax_eq.set_xticklabels(
            [eq_dates.iloc[i].strftime("%b %d") for i in eq_x[::max(1, len(eq_x)//10)]],
            fontsize=7, rotation=30, ha="right",
        )
        ax_eq.set_ylabel("Equity  (start = 1.0)")
        ax_eq.set_title("Cumulative P&L  ·  Agent vs Buy & Hold", fontsize=9)
        ax_eq.legend(fontsize=8, facecolor=_C["panel"],
                     labelcolor=_C["text"], framealpha=0.7)
    else:
        ax_eq.text(0.5, 0.5, "No trades executed", ha="center", va="center",
                   transform=ax_eq.transAxes, color=_C["muted"], fontsize=12)

    out_path = out_dir / f"{sym}_detail.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=_C["bg"])
    plt.close(fig)
    return out_path


# ── Summary-level charts (existing, enhanced) ─────────────────────────────────

def render(results_dir: str | Path, title: str | None = None) -> None:
    """
    Render all charts from a completed backtest run.

    Generates per-symbol detail charts plus the existing summary charts.
    Output files are written into results_dir.
    """
    results_dir = Path(results_dir)
    summary     = pd.read_csv(results_dir / "summary.csv")
    all_trades  = json.loads((results_dir / "all_trades.json").read_text())
    signals_path = results_dir / "all_signals.json"
    all_signals  = json.loads(signals_path.read_text()) if signals_path.exists() else {}

    symbols = sorted(all_trades.keys())

    # ── Per-symbol detail charts ───────────────────────────────────────────────
    for sym in symbols:
        trades_df = pd.DataFrame(all_trades[sym])
        hist_path = results_dir / f"{sym}_history.csv"
        if not hist_path.exists():
            continue
        hist_df   = pd.read_csv(hist_path)
        sigs      = all_signals.get(sym, [])
        out_path  = render_ticker_detail(sym, trades_df, hist_df, sigs, results_dir)
        print(f"  Chart: {out_path}")

    # ── Per-stock equity curves (overview grid) ───────────────────────────────
    n     = len(symbols)
    ncols = 2
    nrows = max(1, (n + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows),
                             facecolor=_C["bg"])
    axes = np.atleast_2d(axes).flatten()

    for ax, sym in zip(axes, symbols):
        _apply_dark(ax)
        trades_df = pd.DataFrame(all_trades[sym])
        hist_df   = pd.read_csv(results_dir / f"{sym}_history.csv")
        agent_eq  = _equity_curve_from_trades(trades_df)
        if agent_eq.empty:
            ax.text(0.5, 0.5, f"{sym}: no trades", ha="center", va="center",
                    transform=ax.transAxes, color=_C["muted"])
            continue
        bh_eq = _buy_hold_curve(hist_df, agent_eq.index)
        ax.plot(agent_eq.index, agent_eq.values,
                label="Agent", linewidth=1.6, color=_C["blue"])
        ax.plot(bh_eq.index, bh_eq.values,
                label="B&H", linewidth=1.4, color=_C["muted"], linestyle="--")
        ax.axhline(1.0, color=_C["border"], linewidth=0.5)
        row = summary[summary["symbol"] == sym].iloc[0]
        ax.set_title(
            f"{sym}  agent={row['agent_total_return']*100:.1f}%  "
            f"B&H={row['buy_hold_return']*100:.1f}%  "
            f"win={row['win_rate']*100:.0f}%  "
            f"sharpe={row['sharpe_annual']:.2f}",
            fontsize=9,
        )
        ax.legend(loc="upper left", fontsize=8,
                  facecolor=_C["panel"], labelcolor=_C["text"])

    for ax in axes[n:]:
        ax.set_visible(False)

    run_title = title or "QuantAgent — backtest summary"
    fig.suptitle(run_title, fontsize=13, color=_C["text"])
    fig.tight_layout()
    out = results_dir / "equity_curves.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=_C["bg"])
    plt.close(fig)
    print(f"  Chart: {out}")

    # ── Equal-weighted portfolio equity ───────────────────────────────────────
    eqs = {}
    for sym in symbols:
        trades_df = pd.DataFrame(all_trades[sym])
        eq = _equity_curve_from_trades(trades_df)
        if not eq.empty:
            eqs[sym] = eq

    if eqs:
        all_dates = pd.DatetimeIndex(sorted({d for eq in eqs.values() for d in eq.index}))
        aligned = pd.DataFrame(
            {sym: eq.reindex(all_dates).ffill().fillna(1.0) for sym, eq in eqs.items()}
        )
        portfolio = aligned.mean(axis=1)

        bh_aligned = {}
        for sym in symbols:
            hist_df = pd.read_csv(results_dir / f"{sym}_history.csv")
            bh_aligned[sym] = _buy_hold_curve(hist_df, all_dates).ffill().fillna(1.0)
        bh_portfolio = pd.DataFrame(bh_aligned).mean(axis=1)

        fig, ax = plt.subplots(figsize=(12, 5), facecolor=_C["bg"])
        _apply_dark(ax)
        ax.plot(portfolio.index, portfolio.values,
                label=f"Agent EW  ({portfolio.iloc[-1]-1:+.1%})",
                linewidth=2.2, color=_C["blue"])
        ax.plot(bh_portfolio.index, bh_portfolio.values,
                label=f"B&H EW  ({bh_portfolio.iloc[-1]-1:+.1%})",
                linewidth=2.0, color=_C["muted"], linestyle="--")
        ax.fill_between(portfolio.index, 1.0, portfolio.values,
                        where=portfolio.values >= 1.0, alpha=0.12, color=_C["green"])
        ax.fill_between(portfolio.index, 1.0, portfolio.values,
                        where=portfolio.values  < 1.0, alpha=0.12, color=_C["red"])
        ax.axhline(1.0, color=_C["border"], linewidth=0.6, linestyle="--")
        ax.set_title("Equal-Weighted Portfolio Equity", fontsize=12, color=_C["text"])
        ax.set_ylabel("Equity  (start = 1.0)")
        ax.legend(loc="upper left", facecolor=_C["panel"], labelcolor=_C["text"])
        fig.tight_layout()
        out = results_dir / "portfolio_equity.png"
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=_C["bg"])
        plt.close(fig)
        print(f"  Chart: {out}")

    # ── Total return bar chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(summary) * 1.4), 5),
                           facecolor=_C["bg"])
    _apply_dark(ax)
    x = np.arange(len(summary))
    w = 0.38
    agent_rets = summary["agent_total_return"] * 100
    bh_rets    = summary["buy_hold_return"]    * 100

    ax.bar(x - w / 2, agent_rets, w,
           label="Agent",
           color=[_C["green"] if v >= 0 else _C["red"] for v in agent_rets],
           edgecolor=_C["border"], linewidth=0.5)
    ax.bar(x + w / 2, bh_rets, w,
           label="Buy & Hold", color=_C["muted"],
           edgecolor=_C["border"], linewidth=0.5, alpha=0.75)

    # Value labels on bars
    for xi, (av, bv) in enumerate(zip(agent_rets, bh_rets)):
        ax.text(xi - w / 2, av + (0.5 if av >= 0 else -1.5),
                f"{av:+.1f}%", ha="center", fontsize=8,
                color=_C["green"] if av >= 0 else _C["red"])
        ax.text(xi + w / 2, bv + (0.5 if bv >= 0 else -1.5),
                f"{bv:+.1f}%", ha="center", fontsize=8, color=_C["text"])

    ax.set_xticks(x)
    ax.set_xticklabels(summary["symbol"], rotation=0)
    ax.set_ylabel("Total Return (%)")
    ax.set_title("Total Return: Agent vs Buy & Hold", fontsize=12, color=_C["text"])
    ax.axhline(0, color=_C["muted"], linewidth=0.7)
    ax.legend(facecolor=_C["panel"], labelcolor=_C["text"])
    fig.tight_layout()
    out = results_dir / "returns_bar.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=_C["bg"])
    plt.close(fig)
    print(f"  Chart: {out}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n=== Summary ===")
    s = summary.copy()
    pct_cols = ["agent_total_return", "buy_hold_return", "excess_return",
                "win_rate", "max_drawdown"]
    for c in pct_cols:
        if c in s.columns:
            s[c] = (s[c] * 100).round(2)
    if "sharpe_annual" in s.columns:
        s["sharpe_annual"] = s["sharpe_annual"].round(2)
    print(s.to_string(index=False))


def main():
    p = argparse.ArgumentParser(
        description="Render charts from a backtest results directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results", default="backtest_results",
                   help="Directory produced by backtest.py")
    p.add_argument("--title", default=None,
                   help="Optional title for the overview chart")
    args = p.parse_args()
    render(args.results, title=args.title)


if __name__ == "__main__":
    main()
