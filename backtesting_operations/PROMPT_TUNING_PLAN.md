# LLM Agent Prompt Tuning Plan
**2026-06-06**

This plan covers iterative tuning of the three LLM perception agents
(Indicator, Pattern, Trend) when running in **LLM mode** (`NO_LLM_MODE = False`).
The goal is to find prompt formulations that outperform the pure-quant baseline
established by the 12 backtest configs.

---

## 1. Why tune prompts at all?

The quant pipeline (`NO_LLM_MODE = True`) is deterministic but limited to
hard-coded mathematical rules. The LLM agents can interpret context — macro regime,
sector-specific patterns, qualitative momentum — that the quant formulas cannot.
However, a poorly-written prompt produces inconsistent, overconfident, or biased
signals. Prompt tuning finds the formulations that translate the LLM's reasoning
into reliably directional signals.

---

## 2. Where the prompts live

| Agent | File | Key function | What the prompt controls |
|---|---|---|---|
| Indicator | `indicator_agent.py` | `create_indicator_agent()` | How the LLM interprets RSI/MACD/Stoch values and weighs them into a direction |
| Trend | `trend_agent.py` | `create_trend_agent()` | How the LLM reads the regression channel, support/resistance, and slope |
| Pattern | `pattern_agent.py` | `create_pattern_agent()` | Vision prompt that names the macro chart formation and assigns direction + confidence |

All three write to the same state keys as the quant nodes, so the downstream
decision node (`decision_agent_quant.py`) is unchanged.

Temperature is set globally in `default_config.py`:
```python
"agent_llm_temperature":  0.1,   # agent perception nodes
"graph_llm_temperature":  0.1,   # decision maker
```

---

## 3. Temperature strategy

### Exploration phase  (temperature = 0.8 – 1.0)

Run each prompt variant at **high temperature** across a short date window
(e.g. 3 months, one symbol). High temperature forces the model to explore
a wider distribution of reasoning paths.

**Purpose:** stress-test the prompt. If a prompt produces consistent
directional signals even with high randomness, it has found a genuine
analytical framework — not just pattern-matching the data.

**How to set:**
```python
# default_config.py — exploration phase
"agent_llm_temperature":  0.9,
```

### Validation phase  (temperature = 0.0 – 0.1)

Once a prompt variant looks promising from exploration, lock in with
**low temperature** and run the full backtest (2015 → today, all 3 symbols).
Low temperature makes signals reproducible — the same kline window will always
produce the same decision, enabling fair comparison across configs.

**How to set:**
```python
# default_config.py — production / validation
"agent_llm_temperature":  0.0,
```

### Why not just use low temperature throughout?

At low temperature the model converges quickly and you may mistake a locally
consistent but wrong prior for a good prompt. High-temperature exploration
reveals the prompt's failure modes faster and with fewer backtest runs.

---

## 4. Per-agent tuning dimensions

### 4a — Indicator Agent

**Current behaviour:** receives OHLCV + pre-computed indicator values, produces
a directional summary and signal strength.

**Dimensions to vary:**

| Dimension | Option A | Option B | Option C |
|---|---|---|---|
| Output format | Single direction + confidence | Bullish/bearish breakdown per indicator | Weighted vote across indicators |
| RSI framing | "RSI > 70 = overbought" | "RSI relative to its 30-bar avg" | "RSI divergence from price" |
| Conflict handling | "If indicators disagree, return HOLD" | "Weight majority vote" | "Defer to momentum (MACD)" |
| Timeframe context | No context given | Explicitly told it's daily bars | Told the sector (tech / energy / healthcare) |
| Confidence scale | 0–1 linear | High / Medium / Low buckets | Only signal if 3+ indicators agree |

### 4b — Trend Agent

**Current behaviour:** receives regression channel metrics, produces trend
direction, support/resistance levels, and signal strength.

**Dimensions to vary:**

| Dimension | Option A | Option B | Option C |
|---|---|---|---|
| Channel interpretation | "Position in channel = signal" | "Channel slope is primary signal" | "Both position and slope required to agree" |
| Support/resistance usage | Report only | "Only buy near support" | "Price must touch band, not just approach" |
| Regime detection | None | "If slope > threshold, trending; else ranging" | Explicitly ask model to classify regime |
| Signal dampening | None | "Reduce signal by 50% if in ranging market" | "Set signal = 0 if channel width < threshold" |

### 4c — Pattern Agent

**Current behaviour:** receives a rendered candlestick chart image, identifies
a macro structural pattern, assigns direction and confidence.

**Dimensions to vary:**

| Dimension | Option A | Option B | Option C |
|---|---|---|---|
| Pattern scope | All 8 macro formations | Only reversal patterns | Only continuation patterns |
| Confidence calibration | Open-ended 0–1 | "Only score > 0.6 counts as a signal" | Force binary: confirmed / not confirmed |
| Context injection | None | Inject current price level and ATR | Inject sector name and recent news context |
| Multi-timeframe framing | Single chart | "Imagine this is the weekly view" | "Assume this is an hourly chart for a swing trader" |
| Failure mode instruction | None | "If pattern is ambiguous, return direction=0" | "If fewer than 20 candles visible, return None" |

---

## 5. The tuning loop

```
For each agent (indicator → trend → pattern):

  1. Read the current prompt in create_<agent>_agent()
  2. Define 2–3 prompt variants based on the dimensions above
  3. Set temperature = 0.9 in default_config.py
  4. Run a SHORT backtest (one symbol, 6 months) for each variant:
       python backtest.py --llm --provider google \
           --symbols GOOGL \
           --start 2024-01-01 --end 2024-06-30 \
           --out backtest_results/prompt_tuning/<agent>_v<N>
  5. Record: n_trades, win_rate, excess_return, signal consistency
  6. Select the best-performing variant
  7. Set temperature = 0.0
  8. Run FULL backtest (all 3 symbols, 2015 → today):
       python backtest.py --llm --provider google \
           --run-config 1_baseline \
           --out backtest_results/prompt_tuning/<agent>_v<N>_full
  9. Compare against quant baseline (run_1 from RESULTS_LOG.md)
  10. If LLM mode beats quant on ≥ 2/3 symbols → promote to production prompt
      If not → try next variant or iterate on the winning prompt further
```

**Tune agents independently** — fix indicator and trend prompts first (they are
called first in the graph), then tune pattern on top of stable upstream signals.

---

## 6. Prompt variant template

Use this format when writing a new prompt variant for any agent.
Include it as a comment above the prompt string in the source file.

```python
# PROMPT VERSION: v2
# CHANGED FROM v1: Added sector context injection, removed open-ended confidence.
# EXPLORATION RESULT: 58% win rate on GOOGL 6-month window at temp=0.9
# VALIDATION RESULT: TBD
SYSTEM_PROMPT = """
...
"""
```

---

## 7. Results recording

Append each tuning run to `backtesting_operations/PROMPT_TUNING_LOG.md`:

```
| Agent     | Version | Temp | Symbol | Period         | Win%  | Excess return | Beats quant? |
|-----------|---------|------|--------|----------------|-------|---------------|--------------|
| indicator | v1      | 0.9  | GOOGL  | 2024-01 to -06 |       |               |              |
| indicator | v2      | 0.9  | GOOGL  | 2024-01 to -06 |       |               |              |
| indicator | v2      | 0.0  | ALL    | 2015 → today   |       |               |              |
```

---

## 8. Practical notes on API cost

- Exploration runs: 1 symbol × 6 months × daily cadence ≈ **125 LLM calls**
  per agent per variant. At ~$0.001/call for Gemini Flash → ~$0.12 per variant.
- Full validation: 3 symbols × ~2,700 bars ≈ **8,100 calls** total.
  Run full validation only for the 1–2 best prompt variants per agent.
- Use `--start 2024-01-01 --end 2024-06-30` for cheap exploration.
- Use `gemini-2.5-flash-lite` (already default in `default_config.py`) — it
  is the cheapest model that still follows structured output instructions.
- Disable chart generation during tuning runs: `--no-charts`

Suggested token budget: **exploration = 5–10 variants × $0.12 = ~$1.00 per agent**.
Full validation per agent: ~$8. Total tuning budget ≈ **$30–40** across all three.
