"""
visualise_agents.py
───────────────────
Interactive runners for the three perception agents.  Each function fetches
live market data and either prints quantitative metrics, renders an interactive
HTML chart (indicator), or displays a chart image (pattern / trend).

These are development / debugging tools, NOT unit tests.

Quick start
───────────
    # Indicator signals + HTML chart
    python visualise_agents.py indicator --symbol XOM --period 6mo --window 90

    # Pattern agent (LLM vision, requires API key)
    python visualise_agents.py pattern --symbol NVDA --period 1mo

    # Trend agent (pure-math)
    python visualise_agents.py trend --symbol GOOG --window 10 --offset 5

    # Run all three sequentially
    python visualise_agents.py all --symbol AAPL
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]


# ─────────────────────────────────────────────────────────────────────────────
#  Shared data fetcher
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_kline(
    symbol: str,
    timeframe: str,
    period: str,
    window: int,
    offset: int,
    need_volume: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """Download OHLCV from yfinance and return (kline_dict, full_df)."""
    print(f"Fetching {period} of {timeframe} data for {symbol}...")

    df = yf.download(tickers=symbol, period=period, interval=timeframe, progress=False)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {symbol!r}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = (
            df.columns.get_level_values(0)
            if "Close" in df.columns.get_level_values(0)
            else df.columns.get_level_values(1)
        )

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Datetime"})

    cols = REQUIRED_COLUMNS if need_volume else REQUIRED_COLUMNS[:-1]
    for c in cols:
        if c not in df.columns:
            if c == "Volume":
                df["Volume"] = 1.0
            else:
                raise ValueError(f"Missing column: {c}")
    df = df[cols]

    if offset > 0:
        df_slice = df.iloc[-(window + offset):-offset].reset_index(drop=True)
    else:
        df_slice = df.tail(window).reset_index(drop=True)

    if len(df_slice) < window:
        print(f"  [WARN] Only {len(df_slice)} candles available (requested {window}).")

    kline: dict = {}
    for col in cols:
        if col == "Datetime":
            kline[col] = df_slice[col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        else:
            kline[col] = df_slice[col].tolist()

    return kline, df


# ─────────────────────────────────────────────────────────────────────────────
#  1. Indicator agent runner
# ─────────────────────────────────────────────────────────────────────────────

def run_indicator(
    symbol: str = "XOM",
    timeframe: str = "1d",
    period: str = "6mo",
    window: int = 90,
    offset: int = 0,
    visualize: bool = True,
    save_path: str = "indicator_chart.html",
    show_plot: bool = True,
) -> dict | None:
    """
    Compute TA-Lib indicators and optionally render an interactive Plotly chart.
    Returns the raw metrics dict.
    """
    from indicator_agent import quantify_indicators_from_kline, visualize_indicator_signals

    try:
        kline, full_df = _fetch_kline(symbol, timeframe, period, window, offset)
        metrics = quantify_indicators_from_kline(kline)

        print("\n" + "=" * 60)
        print(f"=== INDICATOR METRICS — {symbol} ===")
        print("=" * 60)
        print(json.dumps(metrics, indent=4))
        print("=" * 60)

        if visualize:
            visualize_indicator_signals(
                full_df, symbol,
                save_path  = save_path,
                show_plot  = show_plot,
            )
        return metrics

    except Exception as exc:
        print(f"[ERROR] indicator runner: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  2. Pattern agent runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pattern(
    symbol: str = "NVDA",
    timeframe: str = "1d",
    period: str = "1mo",
    window: int = 45,
) -> None:
    """
    Run the full LangGraph pipeline and print the pattern report alongside the
    TA-Lib micro-math verification. Requires an LLM API key (defaults to Gemini).
    """
    import talib
    from trading_graph import TradingGraph
    from default_config import DEFAULT_CONFIG
    import static_util

    kline, df = _fetch_kline(symbol, timeframe, period, window, offset=0, need_volume=False)

    print("Generating chart images...")
    p_image = static_util.generate_kline_image(kline)
    t_image = static_util.generate_trend_image(kline)

    initial_state = {
        "kline_data":     kline,
        "analysis_results": None,
        "messages":       [],
        "time_frame":     timeframe,
        "stock_name":     symbol,
        "pattern_image":  p_image.get("pattern_image"),
        "trend_image":    t_image.get("trend_image"),
    }

    my_config = DEFAULT_CONFIG.copy()
    my_config.update({
        "agent_llm_provider": "google",
        "graph_llm_provider": "google",
        "agent_llm_model":    "gemini-2.5-flash",
        "graph_llm_model":    "gemini-2.5-flash",
    })

    print("Initialising TradingGraph (Gemini)...")
    engine = TradingGraph(config=my_config)

    try:
        final_state = engine.graph.invoke(initial_state)
        qualitative_report = final_state.get("pattern_report", "")

        # TA-Lib micro verification
        o = df["Open"].values
        h = df["High"].values
        l = df["Low"].values
        c = df["Close"].values

        micro_name, micro_dir = "None", 0
        engulfing = talib.CDLENGULFING(o, h, l, c)[-1]
        hammer    = talib.CDLHAMMER(o, h, l, c)[-1]
        doji      = talib.CDLDOJI(o, h, l, c)[-1]
        if engulfing > 0:
            micro_name, micro_dir = "Bullish Engulfing", 1
        elif engulfing < 0:
            micro_name, micro_dir = "Bearish Engulfing", -1
        elif hammer != 0:
            micro_name, micro_dir = "Hammer", 1
        elif doji != 0:
            micro_name, micro_dir = "Doji (Indecision)", 0

        print("\n" + "=" * 60)
        print(f"=== 1. LLM PATTERN REPORT — {symbol} ===")
        print("=" * 60)
        print(qualitative_report)

        print("\n" + "=" * 60)
        print(f"=== 2. TA-LIB MICRO VERIFICATION — {symbol} ===")
        print("=" * 60)
        print(json.dumps(
            {"micro_pattern": micro_name, "micro_direction": micro_dir}, indent=4
        ))
        print("=" * 60)

    except Exception as exc:
        print(f"[ERROR] pattern runner: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  3. Trend agent runner
# ─────────────────────────────────────────────────────────────────────────────

def run_trend(
    symbol: str = "GOOG",
    timeframe: str = "1d",
    period: str = "3mo",
    window: int = 45,
    offset: int = 0,
) -> dict | None:
    """
    Compute the linear-regression trend channel and print the metrics.
    Returns the raw metrics dict.
    """
    from trend_agent import quantify_trend_from_kline

    try:
        kline, _ = _fetch_kline(
            symbol, timeframe, period, window, offset, need_volume=False
        )
        metrics = quantify_trend_from_kline(kline)

        print("\n" + "=" * 60)
        print(f"=== TREND METRICS — {symbol} ===")
        print("=" * 60)
        print(json.dumps(metrics, indent=4))
        print("=" * 60)

        return metrics

    except Exception as exc:
        print(f"[ERROR] trend runner: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise / inspect individual perception agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "agent",
        choices=["indicator", "pattern", "trend", "all"],
        help="Which agent to run",
    )
    p.add_argument("--symbol",    default="XOM",  help="Ticker symbol")
    p.add_argument("--timeframe", default="1d",   help="yfinance interval")
    p.add_argument("--period",    default="6mo",  help="yfinance period")
    p.add_argument("--window",    type=int, default=90, help="Candle window size")
    p.add_argument("--offset",    type=int, default=0,  help="Candles to shift back from today")
    p.add_argument("--no-plot",   action="store_true",  help="Skip opening browser/plot window")
    p.add_argument("--chart-path", default="indicator_chart.html",
                   help="Output path for indicator HTML chart")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    show = not args.no_plot

    if args.agent in ("indicator", "all"):
        run_indicator(
            symbol    = args.symbol,
            timeframe = args.timeframe,
            period    = args.period,
            window    = args.window,
            offset    = args.offset,
            visualize = True,
            save_path = args.chart_path,
            show_plot = show,
        )

    if args.agent in ("pattern", "all"):
        run_pattern(
            symbol    = args.symbol,
            timeframe = args.timeframe,
            period    = args.period,
            window    = min(args.window, 45),
        )

    if args.agent in ("trend", "all"):
        run_trend(
            symbol    = args.symbol,
            timeframe = args.timeframe,
            period    = args.period,
            window    = args.window,
            offset    = args.offset,
        )
