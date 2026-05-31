import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import talib
import yfinance as yf
from plotly.subplots import make_subplots

REQUIRED_COLUMNS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]

# Trend-following normalization: high momentum / positive ROC / rising oscillators → bullish (+1)
COMPONENT_LABELS = {
    "rsi_signal": "RSI (14)",
    "stoch_signal": "Stochastic %K",
    "willr_signal": "Williams %R",
    "macd_signal": "MACD Hist",
    "roc_signal": "ROC (10)",
    "final_signal": "Final Aggregate",
}


def _compute_indicator_series(df):
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values

    rsi = talib.RSI(c, timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(
        c, fastperiod=12, slowperiod=26, signalperiod=9
    )
    roc = talib.ROC(c, timeperiod=10)
    slowk, slowd = talib.STOCH(
        h,
        l,
        c,
        fastk_period=14,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    willr = talib.WILLR(h, l, c, timeperiod=14)

    return {
        "dates": df["Datetime"],
        "close": c,
        "rsi": rsi,
        "macd": macd,
        "macdsignal": macdsignal,
        "macdhist": macdhist,
        "roc": roc,
        "slowk": slowk,
        "slowd": slowd,
        "willr": willr,
    }


def _normalize_indicator_signals(series):
    """
    Map each indicator to [-1, 1] using trend-following logic:
    - Oscillators: higher reading = stronger bullish momentum
    - MACD hist / ROC: positive = bullish, scaled by recent volatility
    """
    rsi = series["rsi"]
    slowk = series["slowk"]
    willr = series["willr"]
    macdhist = series["macdhist"]
    roc = series["roc"]

    rsi_sig = np.clip((rsi - 50.0) / 30.0, -1.0, 1.0)
    stoch_sig = np.clip((slowk - 50.0) / 30.0, -1.0, 1.0)
    willr_sig = np.clip((willr + 50.0) / 30.0, -1.0, 1.0)

    macd_std = pd.Series(macdhist).rolling(window=30, min_periods=1).std().values
    roc_std = pd.Series(roc).rolling(window=30, min_periods=1).std().values

    macd_sig = np.tanh(macdhist / (macd_std + 1e-9))
    roc_sig = np.tanh(roc / (roc_std + 1e-9))

    stacked = np.vstack([rsi_sig, stoch_sig, willr_sig, macd_sig, roc_sig])
    final_sig = np.array([
        np.nanmean(stacked[:, i]) if np.any(~np.isnan(stacked[:, i])) else np.nan
        for i in range(stacked.shape[1])
    ])

    return {
        "rsi_signal": rsi_sig,
        "stoch_signal": stoch_sig,
        "willr_signal": willr_sig,
        "macd_signal": macd_sig,
        "roc_signal": roc_sig,
        "final_signal": final_sig,
    }


def _last_valid(values):
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    return float(valid[-1]) if valid.size else np.nan


def quantify_indicators(df):
    """Single-snapshot indicator quantification for the Decision Agent."""
    if df.empty or len(df) < 35:
        return {"error": "Not enough data to compute indicators. Need at least 35 candles."}

    series = _compute_indicator_series(df)
    signals = _normalize_indicator_signals(series)

    curr_rsi = _last_valid(series["rsi"])
    curr_macdhist = _last_valid(series["macdhist"])
    curr_roc = _last_valid(series["roc"])
    curr_stoch = _last_valid(series["slowk"])
    curr_willr = _last_valid(series["willr"])

    return {
        "raw_metrics": {
            "RSI_14": round(curr_rsi, 2) if not np.isnan(curr_rsi) else 0.0,
            "MACD_Hist": round(curr_macdhist, 4) if not np.isnan(curr_macdhist) else 0.0,
            "Stochastic_K": round(curr_stoch, 2) if not np.isnan(curr_stoch) else 0.0,
            "Williams_R": round(curr_willr, 2) if not np.isnan(curr_willr) else 0.0,
            "ROC_10": round(curr_roc, 2) if not np.isnan(curr_roc) else 0.0,
        },
        "component_signals": {
            "rsi_signal": round(_last_valid(signals["rsi_signal"]), 3),
            "stoch_signal": round(_last_valid(signals["stoch_signal"]), 3),
            "willr_signal": round(_last_valid(signals["willr_signal"]), 3),
            "macd_signal": round(_last_valid(signals["macd_signal"]), 3),
            "roc_signal": round(_last_valid(signals["roc_signal"]), 3),
        },
        "final_indicator_signal": round(_last_valid(signals["final_signal"]), 3),
    }


def visualize_indicator_signals(
    df,
    symbol,
    *,
    show_rsi=True,
    show_stoch=True,
    show_willr=True,
    show_macd=True,
    show_roc=True,
    show_final=True,
    save_path=None,
    show_plot=True,
):
    """Interactive chart: price, each normalized component, and final aggregate."""
    print("Generating interactive Plotly visualization...")

    series = _compute_indicator_series(df)
    signals = _normalize_indicator_signals(series)
    dates = series["dates"]

    component_colors = {
        "rsi_signal": "#9b59b6",
        "stoch_signal": "#3498db",
        "willr_signal": "#e67e22",
        "macd_signal": "#1abc9c",
        "roc_signal": "#e74c3c",
        "final_signal": "#f1c40f",
    }

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.62, 0.38],
        subplot_titles=(f"{symbol} Price", "Normalized Component Signals [-1 to 1]"),
    )

    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=symbol,
        ),
        row=1,
        col=1,
    )

    toggles = {
        "rsi_signal": show_rsi,
        "stoch_signal": show_stoch,
        "willr_signal": show_willr,
        "macd_signal": show_macd,
        "roc_signal": show_roc,
        "final_signal": show_final,
    }

    for key, enabled in toggles.items():
        if not enabled:
            continue
        line_width = 3 if key == "final_signal" else 1.5
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=signals[key],
                mode="lines",
                name=COMPONENT_LABELS[key],
                line=dict(color=component_colors[key], width=line_width),
            ),
            row=2,
            col=1,
        )

    for level, color, dash, label in [
        (0.0, "gray", "dash", "Neutral (0)"),
        (0.5, "green", "dot", "Bullish (+0.5)"),
        (-0.5, "red", "dot", "Bearish (-0.5)"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=[level] * len(dates),
                mode="lines",
                name=label,
                line=dict(color=color, dash=dash, width=1),
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=f"{symbol} — Indicator Signals (Trend-Following Normalization)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=820,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Signal", range=[-1.05, 1.05], row=2, col=1)

    if save_path:
        fig.write_html(save_path)
        print(f"Saved interactive chart to {save_path}")

    if show_plot:
        fig.show()
    else:
        fig.write_html(save_path or "indicator_chart.html")

    return fig


def _fetch_df(symbol, timeframe, period):
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

    return df[REQUIRED_COLUMNS]


def test_quantitative_indicator_agent(
    symbol="NVDA",
    timeframe="1d",
    period="6mo",
    window=45,
    offset=0,
    *,
    show_plot=False,
    viz_kwargs=None,
):
    df = _fetch_df(symbol, timeframe, period)

    if show_plot:
        visualize_indicator_signals(df, symbol, **(viz_kwargs or {}))

    if offset > 0:
        df_slice = df.iloc[-(window + offset) : -offset].reset_index(drop=True)
    else:
        df_slice = df.tail(window).reset_index(drop=True)

    print(f"Running quantitative indicator math on {len(df_slice)} candles...")
    quantitative_metrics = quantify_indicators(df_slice)

    print("\n" + "=" * 60)
    print(f"=== QUANTITATIVE INDICATOR METRICS ({symbol}) ===")
    print("=" * 60)
    print(json.dumps(quantitative_metrics, indent=4))
    print("=" * 60 + "\n")

    return quantitative_metrics, df


if __name__ == "__main__":
    # --- Visualization toggles ---
    SHOW_PLOT = True
    SHOW_RSI = True
    SHOW_STOCH = True
    SHOW_WILLR = True
    SHOW_MACD = True
    SHOW_ROC = True
    SHOW_FINAL = True
    SAVE_CHART = True
    CHART_PATH = "indicator_chart.html"

    test_quantitative_indicator_agent(
        symbol="GOOG",
        timeframe="1d",
        period="6mo",
        window=70,
        offset=0,
        show_plot=SHOW_PLOT,
        viz_kwargs={
            "show_rsi": SHOW_RSI,
            "show_stoch": SHOW_STOCH,
            "show_willr": SHOW_WILLR,
            "show_macd": SHOW_MACD,
            "show_roc": SHOW_ROC,
            "show_final": SHOW_FINAL,
            "save_path": CHART_PATH if SAVE_CHART else None,
            "show_plot": SHOW_PLOT,
        },
    )
