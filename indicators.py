"""Technical indicators powered by pandas-ta (pure Python, no C deps)."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def _ensure_df(kline_data: dict | pd.DataFrame) -> pd.DataFrame:
    if isinstance(kline_data, pd.DataFrame):
        return kline_data
    return pd.DataFrame(kline_data)


def _tail_round(series: pd.Series, n: int = 28) -> list[float]:
    return series.fillna(0).round(4).tail(n).tolist()


def compute_rsi(kline_data, period: int = 14, tail: int = 28) -> dict:
    df = _ensure_df(kline_data)
    rsi = ta.rsi(df["Close"], length=period)
    return {"rsi": _tail_round(rsi, tail)}


def compute_macd(
    kline_data,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
    tail: int = 28,
) -> dict:
    df = _ensure_df(kline_data)
    macd_df = ta.macd(df["Close"], fast=fastperiod, slow=slowperiod, signal=signalperiod)
    macd_col = f"MACD_{fastperiod}_{slowperiod}_{signalperiod}"
    hist_col = f"MACDh_{fastperiod}_{slowperiod}_{signalperiod}"
    sig_col = f"MACDs_{fastperiod}_{slowperiod}_{signalperiod}"
    return {
        "macd": _tail_round(macd_df[macd_col], tail),
        "macd_signal": _tail_round(macd_df[sig_col], tail),
        "macd_hist": _tail_round(macd_df[hist_col], tail),
    }


def compute_stoch(
    kline_data,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
    tail: int = 28,
) -> dict:
    df = _ensure_df(kline_data)
    stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=k_period, d=d_period, smooth_k=smooth_k)
    k_col = f"STOCHk_{k_period}_{d_period}_{smooth_k}"
    d_col = f"STOCHd_{k_period}_{d_period}_{smooth_k}"
    return {
        "stoch_k": _tail_round(stoch[k_col], tail),
        "stoch_d": _tail_round(stoch[d_col], tail),
    }


def compute_roc(kline_data, period: int = 10, tail: int = 28) -> dict:
    df = _ensure_df(kline_data)
    roc = ta.roc(df["Close"], length=period)
    return {"roc": _tail_round(roc, tail)}


def compute_willr(kline_data, period: int = 14, tail: int = 28) -> dict:
    df = _ensure_df(kline_data)
    willr = ta.willr(df["High"], df["Low"], df["Close"], length=period)
    return {"willr": _tail_round(willr, tail)}


def detect_candle_patterns(kline_data) -> dict:
    """Lightweight pure-Python detection for engulfing, hammer, doji on the last candle."""
    df = _ensure_df(kline_data).tail(2).reset_index(drop=True)
    if len(df) < 2:
        return {"pattern": "None", "direction": 0}

    prev = df.iloc[-2]
    cur = df.iloc[-1]

    body = abs(cur["Close"] - cur["Open"])
    prev_body = abs(prev["Close"] - prev["Open"])
    rng = cur["High"] - cur["Low"]
    if rng == 0:
        return {"pattern": "None", "direction": 0}

    upper_wick = cur["High"] - max(cur["Close"], cur["Open"])
    lower_wick = min(cur["Close"], cur["Open"]) - cur["Low"]

    # Doji: tiny body
    if body / rng < 0.1:
        return {"pattern": "Doji", "direction": 0}

    # Bullish engulfing
    if (
        prev["Close"] < prev["Open"]
        and cur["Close"] > cur["Open"]
        and cur["Open"] <= prev["Close"]
        and cur["Close"] >= prev["Open"]
        and body > prev_body
    ):
        return {"pattern": "Bullish Engulfing", "direction": 1}

    # Bearish engulfing
    if (
        prev["Close"] > prev["Open"]
        and cur["Close"] < cur["Open"]
        and cur["Open"] >= prev["Close"]
        and cur["Close"] <= prev["Open"]
        and body > prev_body
    ):
        return {"pattern": "Bearish Engulfing", "direction": -1}

    # Hammer: small body near top, long lower wick
    if lower_wick >= 2 * body and upper_wick <= body and body / rng < 0.4:
        return {"pattern": "Hammer", "direction": 1}

    # Shooting star: small body near bottom, long upper wick
    if upper_wick >= 2 * body and lower_wick <= body and body / rng < 0.4:
        return {"pattern": "Shooting Star", "direction": -1}

    return {"pattern": "None", "direction": 0}


def all_indicators(kline_data) -> dict:
    """Compute the full indicator pack used by the indicator agent."""
    out = {}
    out.update(compute_rsi(kline_data))
    out.update(compute_macd(kline_data))
    out.update(compute_stoch(kline_data))
    out.update(compute_roc(kline_data))
    out.update(compute_willr(kline_data))
    out.update({"candle_pattern": detect_candle_patterns(kline_data)})
    return out
