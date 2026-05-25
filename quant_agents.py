"""Multi-agent trading pipeline built on the OpenAI Agents SDK (2025/2026)."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from agents import Agent, ModelSettings, Runner
from pydantic import BaseModel, Field

import charts
import indicators
from default_config import DEFAULT_CONFIG


# Ensure OPENAI_API_KEY is on the env so the Agents SDK picks it up.
if not os.environ.get("OPENAI_API_KEY") and DEFAULT_CONFIG.get("openai_api_key"):
    os.environ["OPENAI_API_KEY"] = DEFAULT_CONFIG["openai_api_key"]


# ---------- Structured outputs ----------

class IndicatorReport(BaseModel):
    momentum_bias: Literal["bullish", "bearish", "neutral"]
    overbought_oversold: Literal["overbought", "oversold", "neutral"]
    macd_signal: Literal["bullish_cross", "bearish_cross", "trend_up", "trend_down", "flat"]
    summary: str = Field(..., description="2-3 sentence interpretation")


class PatternReport(BaseModel):
    macro_pattern_name: str
    direction: Literal[-1, 0, 1]
    confidence_score: float = Field(ge=0.0, le=1.0)
    justification: str


class TrendReport(BaseModel):
    prediction: Literal["upward", "downward", "sideways"]
    support_break: bool
    resistance_break: bool
    justification: str


class TradeDecision(BaseModel):
    decision: Literal["LONG", "SHORT"]
    confidence: float = Field(ge=0.0, le=1.0)
    risk_reward_ratio: float = Field(ge=1.0, le=3.0)
    justification: str


# ---------- Agent factories ----------

def _model_settings(temperature: float) -> ModelSettings:
    return ModelSettings(temperature=temperature)


def make_indicator_agent(model: str, temperature: float = 0.1) -> Agent:
    return Agent(
        name="indicator_analyst",
        model=model,
        model_settings=_model_settings(temperature),
        instructions=(
            "You are a quantitative HFT indicator analyst. "
            "You are given a JSON payload of recent technical indicator values (RSI, MACD, Stochastic, "
            "ROC, Williams %R) over the last ~28 bars. "
            "Interpret the momentum, oscillator state, MACD crossover state, and overall bias. "
            "Be decisive — do not hedge."
        ),
        output_type=IndicatorReport,
    )


def make_pattern_agent(model: str, temperature: float = 0.1) -> Agent:
    return Agent(
        name="pattern_analyst",
        model=model,
        model_settings=_model_settings(temperature),
        instructions=(
            "You are a quantitative vision analyst. You receive a candlestick (K-line) chart image. "
            "Identify the single most prominent macro pattern from: Inverse H&S, Double Bottom/Top, "
            "Rounded Bottom/Top, Falling/Rising Wedge, Ascending/Descending Triangle, "
            "Bullish/Bearish Flag, Rectangle, Symmetrical Triangle, or 'None'. "
            "Set direction = 1 for bullish, -1 for bearish, 0 for neutral/none. "
            "Confidence reflects how clear the structure is."
        ),
        output_type=PatternReport,
    )


def make_trend_agent(model: str, temperature: float = 0.1) -> Agent:
    return Agent(
        name="trend_analyst",
        model=model,
        model_settings=_model_settings(temperature),
        instructions=(
            "You are a trend analysis vision agent. The chart shows recent candles with a blue support "
            "trendline and red resistance trendline. Analyze how price interacts with these lines and "
            "predict the short-term trend direction (upward/downward/sideways). "
            "Flag whether price has clearly broken support or resistance."
        ),
        output_type=TrendReport,
    )


def make_decision_agent(model: str, temperature: float = 0.0) -> Agent:
    return Agent(
        name="trade_decider",
        model=model,
        model_settings=_model_settings(temperature),
        instructions=(
            "You are a high-frequency quantitative trader. Based on the three reports below, issue an "
            "IMMEDIATE LONG or SHORT order. HOLD is NOT permitted. "
            "Weight signals as follows: prioritize alignment across all three reports; require "
            "confirmation for pattern signals; use trendline slope when reports disagree. "
            "Suggest a risk-reward ratio between 1.2 and 1.8 based on signal strength."
        ),
        output_type=TradeDecision,
    )


# ---------- Pipeline ----------

@dataclass
class PipelineResult:
    symbol: str
    timeframe: str
    indicator_report: IndicatorReport
    pattern_report: PatternReport
    trend_report: TrendReport
    decision: TradeDecision

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "indicator_report": self.indicator_report.model_dump(),
            "pattern_report": self.pattern_report.model_dump(),
            "trend_report": self.trend_report.model_dump(),
            "decision": self.decision.model_dump(),
        }


def _image_input(prompt: str, image_b64: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"},
            ],
        }
    ]


async def _run_indicator(agent: Agent, indicator_data: dict, symbol: str, timeframe: str) -> IndicatorReport:
    payload = {k: v for k, v in indicator_data.items()}
    prompt = (
        f"Symbol: {symbol} | Timeframe: {timeframe}\n"
        f"Indicator payload:\n{json.dumps(payload, indent=2)}"
    )
    result = await Runner.run(agent, input=prompt)
    return result.final_output


async def _run_pattern(agent: Agent, image_b64: str, symbol: str, timeframe: str) -> PatternReport:
    prompt = (
        f"Identify the macro pattern in this {timeframe} candlestick chart of {symbol}. "
        "Return strict JSON in the required schema."
    )
    result = await Runner.run(agent, input=_image_input(prompt, image_b64))
    return result.final_output


async def _run_trend(agent: Agent, image_b64: str, symbol: str, timeframe: str) -> TrendReport:
    prompt = (
        f"Predict the short-term trend for this {timeframe} {symbol} chart. "
        "Blue line = support, red line = resistance. Return strict JSON."
    )
    result = await Runner.run(agent, input=_image_input(prompt, image_b64))
    return result.final_output


async def _run_decision(
    agent: Agent,
    indicator: IndicatorReport,
    pattern: PatternReport,
    trend: TrendReport,
    symbol: str,
    timeframe: str,
) -> TradeDecision:
    prompt = (
        f"Symbol: {symbol} | Timeframe: {timeframe}\n\n"
        f"Indicator report:\n{indicator.model_dump_json(indent=2)}\n\n"
        f"Pattern report:\n{pattern.model_dump_json(indent=2)}\n\n"
        f"Trend report:\n{trend.model_dump_json(indent=2)}"
    )
    result = await Runner.run(agent, input=prompt)
    return result.final_output


async def run_pipeline_async(
    symbol: str,
    kline_data: dict,
    timeframe: str = "1d",
    config: dict | None = None,
) -> PipelineResult:
    cfg = config or DEFAULT_CONFIG
    agent_model = cfg["agent_llm_model"]
    vision_model = cfg["vision_llm_model"]
    decision_model = cfg["decision_llm_model"]
    temperature = cfg["temperature"]

    indicator_data = indicators.all_indicators(kline_data)
    kline_b64 = charts.generate_kline_image(kline_data)["pattern_image"]
    trend_b64 = charts.generate_trend_image(kline_data)["trend_image"]

    indicator_agent = make_indicator_agent(agent_model, temperature)
    pattern_agent = make_pattern_agent(vision_model, temperature)
    trend_agent = make_trend_agent(vision_model, temperature)
    decision_agent = make_decision_agent(decision_model, 0.0)

    # Run the three analysts in parallel
    indicator_report, pattern_report, trend_report = await asyncio.gather(
        _run_indicator(indicator_agent, indicator_data, symbol, timeframe),
        _run_pattern(pattern_agent, kline_b64, symbol, timeframe),
        _run_trend(trend_agent, trend_b64, symbol, timeframe),
    )

    decision = await _run_decision(
        decision_agent, indicator_report, pattern_report, trend_report, symbol, timeframe
    )

    return PipelineResult(
        symbol=symbol,
        timeframe=timeframe,
        indicator_report=indicator_report,
        pattern_report=pattern_report,
        trend_report=trend_report,
        decision=decision,
    )


def run_pipeline(
    symbol: str,
    kline_data: dict,
    timeframe: str = "1d",
    config: dict | None = None,
) -> PipelineResult:
    """Synchronous wrapper around run_pipeline_async."""
    return asyncio.run(run_pipeline_async(symbol, kline_data, timeframe, config))
