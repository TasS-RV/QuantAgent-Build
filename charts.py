"""Chart image generation: K-line and trend-annotated candlestick charts."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

import color_style as color

matplotlib.use("Agg")


def _check_trend_line(support: bool, pivot: int, slope: float, y: pd.Series) -> float:
    intercept = -slope * pivot + y.iloc[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
    diffs = line_vals - y
    if support and diffs.max() > 1e-5:
        return -1.0
    if not support and diffs.min() < -1e-5:
        return -1.0
    return float((diffs**2).sum())


def _optimize_slope(support: bool, pivot: int, init_slope: float, y: pd.Series) -> tuple[float, float]:
    slope_unit = (y.max() - y.min()) / len(y)
    opt_step, min_step, curr_step = 1.0, 0.0001, 1.0
    best_slope = init_slope
    best_err = _check_trend_line(support, pivot, init_slope, y)
    if best_err < 0:
        return (init_slope, -init_slope * pivot + y.iloc[pivot])

    derivative = None
    get_derivative = True
    while curr_step > min_step:
        if get_derivative:
            test_err = _check_trend_line(support, pivot, best_slope + slope_unit * min_step, y)
            derivative = test_err - best_err
            if test_err < 0:
                test_err = _check_trend_line(support, pivot, best_slope - slope_unit * min_step, y)
                derivative = best_err - test_err
            if test_err < 0:
                break
            get_derivative = False

        test_slope = best_slope - slope_unit * curr_step if derivative > 0 else best_slope + slope_unit * curr_step
        test_err = _check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            curr_step *= 0.5
        else:
            best_err = test_err
            best_slope = test_slope
            get_derivative = True

    return (best_slope, -best_slope * pivot + y.iloc[pivot])


def _fit_trendlines_close(close: pd.Series) -> tuple[tuple[float, float], tuple[float, float]]:
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    line = coefs[0] * x + coefs[1]
    upper_pivot = int((close - line).argmax())
    lower_pivot = int((close - line).argmin())
    return (
        _optimize_slope(True, lower_pivot, coefs[0], close),
        _optimize_slope(False, upper_pivot, coefs[0], close),
    )


def _kline_df(kline_data) -> pd.DataFrame:
    df = pd.DataFrame(kline_data) if not isinstance(kline_data, pd.DataFrame) else kline_data.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")
    return df[["Open", "High", "Low", "Close"]]


def generate_kline_image(kline_data, tail: int = 40, save_path: str | None = None) -> dict:
    df = _kline_df(kline_data).tail(tail)
    fig, axlist = mpf.plot(
        df,
        type="candle",
        style=color.my_color_style,
        figsize=(12, 6),
        returnfig=True,
        block=False,
    )
    axlist[0].set_ylabel("Price")
    axlist[0].set_xlabel("Datetime")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return {
        "pattern_image": base64.b64encode(buf.read()).decode("utf-8"),
        "pattern_image_description": "Candlestick K-line chart.",
    }


def generate_trend_image(kline_data, tail: int = 50, save_path: str | None = None) -> dict:
    candles = _kline_df(kline_data).tail(tail).copy()
    support_coefs, resist_coefs = _fit_trendlines_close(candles["Close"])

    x = np.arange(len(candles))
    support_line = support_coefs[0] * x + support_coefs[1]
    resist_line = resist_coefs[0] * x + resist_coefs[1]

    apds = [
        mpf.make_addplot(support_line, color="blue", width=1.2),
        mpf.make_addplot(resist_line, color="red", width=1.2),
    ]
    fig, axlist = mpf.plot(
        candles,
        type="candle",
        style=color.my_color_style,
        addplot=apds,
        returnfig=True,
        figsize=(12, 6),
        block=False,
    )
    axlist[0].set_ylabel("Price")
    axlist[0].set_xlabel("Datetime")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return {
        "trend_image": base64.b64encode(buf.read()).decode("utf-8"),
        "trend_image_description": "Trend-annotated candlestick chart (blue=support, red=resistance).",
        "support_slope": float(support_coefs[0]),
        "resist_slope": float(resist_coefs[0]),
    }
