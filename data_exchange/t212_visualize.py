"""
Trading 212 Portfolio Dashboard
=================================
Standalone dark-theme visualisation of your open T212 positions.
Produces a single PNG with:
  • Allocation donut (by market value)
  • Return-vs-entry bar chart (per position, % + currency P&L)
  • Position detail table (qty / avg / current / change / value / P&L)
  • Summary strip (total value, unrealized P&L, cash, # positions)

Usage
-----
    python data_exchange/t212_visualize.py           # live T212 account
    python data_exchange/t212_visualize.py --mock    # mock data (no key needed)
    python data_exchange/t212_visualize.py --demo    # T212 practice account
    python data_exchange/t212_visualize.py --out portfolio.png
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Ensure siblings (trading212_client) and repo root are importable
_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parent
for _p in (str(_DIR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from trading212_client import Position


# ─────────────────────────────────────────────────────────────────────────────
#  Dark theme palette
# ─────────────────────────────────────────────────────────────────────────────

_C = {
    "bg":     "#0d1117",
    "panel":  "#161b22",
    "border": "#30363d",
    "green":  "#3fb950",
    "red":    "#f85149",
    "blue":   "#58a6ff",
    "orange": "#e3b341",
    "purple": "#bc8cff",
    "text":   "#c9d1d9",
    "muted":  "#8b949e",
    "white":  "#f0f6fc",
}

_SLICE_PALETTE = [
    "#58a6ff", "#3fb950", "#e3b341", "#bc8cff",
    "#f85149", "#79c0ff", "#56d364", "#ffa657",
    "#d2a8ff", "#ff7b72",
]


def _apply_dark(ax, spines: bool = True):
    ax.set_facecolor(_C["panel"])
    ax.tick_params(colors=_C["muted"], labelsize=8.5)
    if spines:
        for spine in ax.spines.values():
            spine.set_color(_C["border"])


# ─────────────────────────────────────────────────────────────────────────────
#  Dashboard renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_portfolio(
    positions: List[Position],
    cash: Optional[dict] = None,
    out_path: str | Path = "t212_portfolio.png",
    title: str = "Trading 212 Portfolio",
) -> Path:
    """
    Render the portfolio dashboard to `out_path` and return the resolved path.
    """
    if not positions:
        print("[WARN] No positions to visualise.")
        return Path(out_path)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Derived per-position values ───────────────────────────────────────────
    symbols    = [p.yf_symbol for p in positions]
    values     = [p.market_value for p in positions]
    ppls       = [p.ppl for p in positions]
    pct_chgs   = [p.pnl_pct for p in positions]

    total_value = sum(values)
    total_ppl   = sum(ppls)
    cash_free   = float(cash.get("free", 0)) if cash else None
    n_pos       = len(positions)

    ts          = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M UTC")
    ppl_color   = _C["green"] if total_ppl >= 0 else _C["red"]
    ppl_sign    = "+" if total_ppl >= 0 else ""

    # ── Figure + grid ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9), facecolor=_C["bg"])

    # 2 rows × 2 cols; bottom row spans full width for the table
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[2.8, 1.0],
        hspace=0.40,
        wspace=0.28,
        left=0.04, right=0.97,
        top=0.83, bottom=0.06,
    )
    ax_pie = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_tbl = fig.add_subplot(gs[1, :])

    # ── Header strip ──────────────────────────────────────────────────────────
    fig.text(
        0.04, 0.945, title,
        ha="left", va="center",
        color=_C["white"], fontsize=16, fontweight="bold",
    )
    fig.text(
        0.04, 0.905, ts,
        ha="left", va="center",
        color=_C["muted"], fontsize=9,
    )

    # Key metrics (right side of header)
    metrics = [
        ("Total Value",    f"{total_value:,.2f}"),
        ("Unrealized P&L", f"{ppl_sign}{total_ppl:,.2f}"),
        ("Positions",      str(n_pos)),
    ]
    if cash_free is not None:
        metrics.append(("Cash Available", f"{cash_free:,.2f}"))

    x_start = 0.45
    x_step  = (0.97 - x_start) / max(len(metrics), 1)
    for i, (label, val) in enumerate(metrics):
        x     = x_start + (i + 0.5) * x_step
        color = ppl_color if "P&L" in label else _C["blue"]
        fig.text(x, 0.945, val,   ha="center", va="center",
                 color=color, fontsize=14, fontweight="bold")
        fig.text(x, 0.905, label, ha="center", va="center",
                 color=_C["muted"], fontsize=8.5)

    # Thin horizontal rule under the header
    fig.add_artist(
        plt.Line2D(
            [0.04, 0.97], [0.885, 0.885],
            transform=fig.transFigure,
            color=_C["border"], linewidth=0.8,
        )
    )

    # ── Allocation donut ──────────────────────────────────────────────────────
    ax_pie.set_facecolor(_C["panel"])
    slice_colors = [_SLICE_PALETTE[i % len(_SLICE_PALETTE)] for i in range(n_pos)]

    wedges, _, autotexts = ax_pie.pie(
        values,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        colors=slice_colors,
        wedgeprops={"linewidth": 1.5, "edgecolor": _C["bg"], "width": 0.50},
        pctdistance=0.76,
    )
    for at in autotexts:
        at.set_color(_C["bg"])
        at.set_fontsize(8.5)
        at.set_fontweight("bold")

    # Legend sitting in the hollow of the donut
    ax_pie.legend(
        wedges, [f"{s}  {v:,.0f}" for s, v in zip(symbols, values)],
        loc="center",
        frameon=False,
        labelcolor=_C["text"],
        fontsize=8.5,
        handlelength=0.9,
        handletextpad=0.4,
    )
    ax_pie.set_title(
        "Portfolio Allocation", color=_C["muted"], fontsize=10, pad=8,
    )

    # ── Return vs entry bar chart ─────────────────────────────────────────────
    _apply_dark(ax_bar)
    y_pos  = list(range(n_pos))
    colors = [_C["green"] if p >= 0 else _C["red"] for p in pct_chgs]
    bars   = ax_bar.barh(y_pos, pct_chgs, color=colors, height=0.55, zorder=2)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(symbols, color=_C["text"], fontsize=9.5)
    ax_bar.set_xlabel("Unrealized Return (%)", color=_C["muted"], fontsize=8)
    ax_bar.axvline(0, color=_C["border"], linewidth=1.2, zorder=3)
    ax_bar.grid(axis="x", color=_C["border"], linewidth=0.5, zorder=1, alpha=0.5)
    ax_bar.invert_yaxis()
    ax_bar.set_title("Return vs Entry Price", color=_C["muted"], fontsize=10, pad=8)

    # Extend right axis first so label offset is based on final range
    xlim = ax_bar.get_xlim()
    ax_bar.set_xlim(xlim[0], max(xlim[1], 2) * 1.6)

    # All labels pinned just right of the zero line — never overlaps y-ticks
    x_offset = (ax_bar.get_xlim()[1] - ax_bar.get_xlim()[0]) * 0.015
    for bar, pct, ppl in zip(bars, pct_chgs, ppls):
        sign  = "+" if pct >= 0 else ""
        psign = "+" if ppl >= 0 else ""
        label = f"{sign}{pct:.1f}%  ({psign}{ppl:,.0f})"
        ax_bar.text(
            x_offset, bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left",
            color=_C["text"], fontsize=8,
        )

    # ── Position detail table ─────────────────────────────────────────────────
    ax_tbl.set_facecolor(_C["panel"])
    ax_tbl.axis("off")

    hdrs   = ["Symbol", "Qty", "Avg Price", "Current", "Change %",
              "Mkt Value", "Unrealized P&L", ""]
    col_xs = [0.01, 0.10, 0.20, 0.32, 0.44, 0.56, 0.70, 0.87]

    # Column headers
    for h, x in zip(hdrs, col_xs):
        ax_tbl.text(
            x, 0.90, h, transform=ax_tbl.transAxes,
            color=_C["muted"], fontsize=8, fontweight="bold", va="top",
        )

    # Divider line
    ax_tbl.axhline(
        0.80, color=_C["border"], linewidth=0.8,
        xmin=0.0, xmax=1.0,
    )

    row_y   = 0.65
    # Compress row spacing so up to 8 positions fit; minimum 0.09 per row
    row_gap = max(0.09, 0.62 / max(n_pos, 1))

    for i, pos in enumerate(positions):
        y = row_y - i * row_gap
        if y < 0.0:
            ax_tbl.text(
                0.5, y + row_gap * 0.5,
                f"(+ {n_pos - i} more positions)",
                transform=ax_tbl.transAxes,
                color=_C["muted"], fontsize=7.5, ha="center", va="top",
            )
            break

        pct      = pct_chgs[i]
        pct_col  = _C["green"] if pct >= 0 else _C["red"]
        ppl_col  = _C["green"] if pos.ppl >= 0 else _C["red"]
        sign_p   = "+" if pct >= 0 else ""
        sign_ppl = "+" if pos.ppl >= 0 else ""
        etf_lbl  = " ▲ ETF" if pos.is_etf else ""

        row = [
            (pos.yf_symbol,                   _C["white"]),
            (f"{pos.quantity:.4g}",            _C["text"]),
            (f"{pos.average_price:.2f}",       _C["text"]),
            (f"{pos.current_price:.2f}",       _C["text"]),
            (f"{sign_p}{pct:.2f}%",            pct_col),
            (f"{pos.market_value:,.2f}",       _C["text"]),
            (f"{sign_ppl}{pos.ppl:,.2f}",      ppl_col),
            (etf_lbl,                          _C["orange"]),
        ]
        for (txt, color), x in zip(row, col_xs):
            ax_tbl.text(
                x, y, txt, transform=ax_tbl.transAxes,
                color=color, fontsize=9, va="top",
            )

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_C["bg"])
    plt.close(fig)
    print(f"Dashboard saved -> {out_path.resolve()}")
    return out_path.resolve()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Trading 212 portfolio dashboard (dark-theme PNG).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mock", action="store_true",
        help="Use mock portfolio data (no API key required)",
    )
    p.add_argument(
        "--demo", action="store_true",
        help="Connect to the T212 *practice* account instead of live",
    )
    p.add_argument(
        "--api-key", default=None, metavar="KEY",
        help="T212 API key (overrides auto-load from trading212_key.txt)",
    )
    p.add_argument(
        "--out", default="t212_portfolio.png", metavar="PATH",
        help="Output PNG path",
    )
    p.add_argument(
        "--no-cash", action="store_true",
        help="Skip the cash-balance API call (saves one request)",
    )
    args = p.parse_args()

    from trading212_client import Trading212Client, mock_portfolio

    if args.mock:
        positions = mock_portfolio()
        cash      = None
        print(f"Using mock portfolio  ({len(positions)} positions).")
    else:
        client    = Trading212Client(api_key=args.api_key, demo=args.demo)
        account   = "practice" if args.demo else "live"
        print(f"Fetching {account} portfolio from Trading 212 ...")
        positions = client.get_portfolio()
        cash      = None
        if not args.no_cash:
            try:
                cash = client.get_cash()
            except Exception as e:
                print(f"[WARN] Could not fetch cash balance: {e}")
        print(f"  {len(positions)} open position(s) found.")

    if not positions:
        print("No open positions — nothing to visualise.")
        return

    dash_title = "Trading 212 Portfolio"
    if args.demo:
        dash_title += "  (Practice Account)"
    if args.mock:
        dash_title += "  [MOCK DATA]"

    render_portfolio(positions, cash=cash, out_path=args.out, title=dash_title)


if __name__ == "__main__":
    main()
