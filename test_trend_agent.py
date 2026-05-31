import json

import pandas as pd
import yfinance as yf

from trend_agent import quantify_trend_from_kline

REQUIRED_COLUMNS = ["Datetime", "Open", "High", "Low", "Close"]


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

    return kline_data


def run_trend_agent(symbol="NVDA", timeframe="1d", period="3mo", window=45, offset=0):
    """
    Standalone runner for trend quantification outside the LangChain pipeline.
    Fetches market data and delegates all trend math to trend_agent.
    """
    try:
        kline_data = _fetch_kline_data(symbol, timeframe, period, window, offset)
        metrics = quantify_trend_from_kline(kline_data)

        print("\n" + "=" * 60)
        print(f"=== MATHEMATICAL TREND METRICS ({symbol}) ===")
        print("=" * 60)
        print(json.dumps(metrics, indent=4))
        print("=" * 60 + "\n")

        return metrics
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        return None


if __name__ == "__main__":
    # window = Size of the trend channel (default 45)
    # offset = How many candles backward to shift the end date (0 = today)

    # Example 1: Run on the current most recent 45 days
    # run_trend_agent(symbol="BTC-USD", timeframe="1d", period="3mo", window=45, offset=0)

    # Example 2: Backtest a 45-day trend from exactly 14 days ago
    run_trend_agent(symbol="GOOG", timeframe="1d", period="1mo", window=10, offset=5)
