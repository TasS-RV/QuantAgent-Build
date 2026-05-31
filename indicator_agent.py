"""
Agent for technical indicator analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to compute and interpret indicators like MACD, RSI, ROC, Stochastic, and Williams %R.
Combines LLM qualitative analysis with rigid TA-Lib trend-following mathematics.
"""

import copy
import json

import numpy as np
import pandas as pd
import talib
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

COMPONENT_LABELS = {
    "rsi_signal": "RSI (14)",
    "stoch_signal": "Stochastic %K",
    "willr_signal": "Williams %R",
    "macd_signal": "MACD Hist",
    "roc_signal": "ROC (10)",
    "final_signal": "Final Aggregate",
}


def _kline_to_dataframe(kline_data):
    """Convert kline_data (dict-of-lists or list-of-dicts) to a OHLCV DataFrame."""
    if isinstance(kline_data, dict) and "Close" in kline_data:
        df = pd.DataFrame(kline_data)
    else:
        df = pd.DataFrame(kline_data)

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Volume" not in df.columns:
        df["Volume"] = 1.0
    return df


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


def quantify_indicators_from_kline(kline_data):
    """Run indicator quantification on kline_data (dict-of-lists or list-of-dicts)."""
    return quantify_indicators(_kline_to_dataframe(kline_data))


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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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
    elif save_path:
        pass
    else:
        fig.write_html("indicator_chart.html")

    return fig


def create_indicator_agent(llm, toolkit):
    """
    Create an indicator analysis agent node for HFT. The agent uses LLM and indicator tools to analyze OHLCV data.
    """

    def indicator_agent_node(state):
        tools = [
            toolkit.compute_macd,
            toolkit.compute_rsi,
            toolkit.compute_roc,
            toolkit.compute_stoch,
            toolkit.compute_willr,
        ]
        time_frame = state["time_frame"]

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a high-frequency trading (HFT) analyst assistant operating under time-sensitive conditions. "
                    "You must analyze technical indicators to support fast-paced trading execution.\n\n"
                    "You have access to tools: compute_rsi, compute_macd, compute_roc, compute_stoch, and compute_willr. "
                    "Use them by providing appropriate arguments like `kline_data` and the respective periods.\n\n"
                    f"⚠️ The OHLC data provided is from a {time_frame} intervals, reflecting recent market behavior. "
                    "You must interpret this data quickly and accurately.\n\n"
                    "Here is the OHLC data:\n{kline_data}.\n\n"
                    "Call necessary tools, and analyze the results.\n",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        ).partial(kline_data=json.dumps(state["kline_data"], indent=2))

        chain = prompt | llm.bind_tools(tools)
        messages = state.get("messages", [])
        if not messages:
            messages = [HumanMessage(content="Begin indicator analysis.")]

        ai_response = chain.invoke({"messages": messages})
        messages.append(ai_response)

        if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
            for call in ai_response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                tool_fn = next(t for t in tools if t.name == tool_name)
                tool_result = tool_fn.invoke(tool_args)
                messages.append(
                    ToolMessage(
                        tool_call_id=call["id"], content=json.dumps(tool_result)
                    )
                )

        max_iterations = 5
        iteration = 0
        final_response = None

        while iteration < max_iterations:
            iteration += 1
            final_response = chain.invoke({"messages": messages})
            messages.append(final_response)

            if not hasattr(final_response, "tool_calls") or not final_response.tool_calls:
                break

            for call in final_response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                tool_fn = next(t for t in tools if t.name == tool_name)
                tool_result = tool_fn.invoke(tool_args)
                messages.append(
                    ToolMessage(
                        tool_call_id=call["id"], content=json.dumps(tool_result)
                    )
                )

        if final_response:
            report_content = final_response.content
            if not report_content or (isinstance(report_content, str) and not report_content.strip()):
                for msg in reversed(messages):
                    if (
                        hasattr(msg, "content")
                        and msg.content
                        and isinstance(msg.content, str)
                        and msg.content.strip()
                        and not hasattr(msg, "tool_calls")
                    ):
                        report_content = msg.content
                        break
        else:
            report_content = "Indicator analysis completed, but no detailed report was generated."

        kline_data = state.get("kline_data", {})
        math_metrics = {"error": "Quantitative evaluation failed"}

        try:
            math_metrics = quantify_indicators_from_kline(kline_data)
        except Exception as e:
            print(f"Mathematical indicator evaluation failed: {e}")

        combined_report = json.dumps(
            {
                "llm_analysis": report_content,
                "quantitative_metrics": math_metrics,
            },
            indent=4,
        )

        return {
            "messages": messages,
            "indicator_report": combined_report,
        }

    return indicator_agent_node
