"""Single-symbol demo of the new OpenAI Agents SDK pipeline.

Usage:
    python run_single.py --symbol NVDA --period 6mo
"""

from __future__ import annotations

import argparse
import json
import sys

# yfinance needs sqlite3; shim if missing.
try:
    import sqlite3  # noqa: F401
except ImportError:
    import pysqlite3

    sys.modules["sqlite3"] = pysqlite3

import pandas as pd
import yfinance as yf

from quant_agents import run_pipeline


def fetch(symbol: str, period: str, interval: str) -> dict:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit(f"No data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Datetime"})
    df = df[["Datetime", "Open", "High", "Low", "Close"]].tail(60)
    return {
        "Datetime": pd.to_datetime(df["Datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "Open": df["Open"].astype(float).tolist(),
        "High": df["High"].astype(float).tolist(),
        "Low": df["Low"].astype(float).tolist(),
        "Close": df["Close"].astype(float).tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="NVDA")
    p.add_argument("--period", default="6mo")
    p.add_argument("--interval", default="1d")
    args = p.parse_args()

    kline = fetch(args.symbol, args.period, args.interval)
    result = run_pipeline(args.symbol, kline, timeframe=args.interval)

    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
