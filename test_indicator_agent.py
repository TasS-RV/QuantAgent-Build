import json

import pandas as pd
import yfinance as yf

from indicator_agent import quantify_indicators_from_kline, visualize_indicator_signals

REQUIRED_COLUMNS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]


def _fetch_kline_data(symbol, timeframe, period, window, offset):
    print(f"Fetching {period} of {timeframe} data for {symbol}...")

    df = yf.download(tickers=symbol, period=period, interval=timeframe, progress=False)
    if df.empty:
        raise ValueError("No data fetched.")

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Datetime"})

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        if "Volume" in missing_cols:
            df["Volume"] = 1.0
        else:
            raise ValueError(f"Failed to parse data. Missing: {missing_cols}")

    df = df[REQUIRED_COLUMNS]

    if offset > 0:
        df_slice = df.iloc[-(window + offset) : -offset].reset_index(drop=True)
    else:
        df_slice = df.tail(window).reset_index(drop=True)

    if len(df_slice) < window:
        print(f"Warning: Only {len(df_slice)} candles available for this slice (requested {window}).")

    kline_data = {}
    for col in REQUIRED_COLUMNS:
        if col == "Datetime":
            kline_data[col] = df_slice[col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        else:
            kline_data[col] = df_slice[col].tolist()

    return kline_data, df


def run_indicator_agent(
    symbol="NVDA",
    timeframe="1d",
    period="6mo",
    window=45,
    offset=0,
    *,
    visualize=False,
    viz_kwargs=None,
):
    """
    Standalone runner for indicator quantification outside the LangChain pipeline.
    Fetches market data and delegates all indicator math to indicator_agent.
    """
    try:
        kline_data, full_df = _fetch_kline_data(symbol, timeframe, period, window, offset)
        metrics = quantify_indicators_from_kline(kline_data)

        print("\n" + "=" * 60)
        print(f"=== QUANTITATIVE INDICATOR METRICS ({symbol}) ===")
        print("=" * 60)
        print(json.dumps(metrics, indent=4))
        print("=" * 60 + "\n")

        if visualize:
            visualize_indicator_signals(full_df, symbol, **(viz_kwargs or {}))

        return metrics
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        return None


if __name__ == "__main__":
    # --- Visualization toggles ---
    VISUALIZE = True
    SHOW_RSI = True
    SHOW_STOCH = True
    SHOW_WILLR = True
    SHOW_MACD = True
    SHOW_ROC = True
    SHOW_FINAL = True
    SAVE_CHART = True
    SHOW_PLOT_WINDOW = True
    CHART_PATH = "indicator_chart.html"

    run_indicator_agent(
        symbol="XOM",
        timeframe="1d",
        period="6mo",
        window=90,
        offset=0,
        visualize=VISUALIZE,
        viz_kwargs={
            "show_rsi": SHOW_RSI,
            "show_stoch": SHOW_STOCH,
            "show_willr": SHOW_WILLR,
            "show_macd": SHOW_MACD,
            "show_roc": SHOW_ROC,
            "show_final": SHOW_FINAL,
            "save_path": CHART_PATH if SAVE_CHART else None,
            "show_plot": SHOW_PLOT_WINDOW,
        },
    )
