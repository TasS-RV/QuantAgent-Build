# Quantitative Decision Agent & Master Portfolio Script

## Overview

This change strips the qualitative LLM response from the final decision step of the QuantAgent pipeline and replaces it with a deterministic mathematical function. The perception agents (Indicator, Pattern, Trend) are unchanged — they still run their LLM analysis — but their **quantitative outputs** are now the sole input to the decision layer.

This makes the final trade signal:
- **Reproducible** — same data always produces the same signal
- **Backtestable** — no probabilistic LLM variance
- **Fast** — no extra API call in the decision step

---

## Architecture Before vs After

### Before

```
Indicator Agent (LLM + TA-Lib)
        ↓
Pattern Agent (LLM vision)
        ↓
Trend Agent (LLM + linear regression)
        ↓
Decision Maker (LLM) ← synthesises all three reports qualitatively → "LONG" or "SHORT"
```

### After

```
Indicator Agent (LLM + TA-Lib)   → quantitative_metrics.final_indicator_signal ∈ [-1, 1]
        ↓
Pattern Agent (LLM vision)        → direction × confidence_score                ∈ [-1, 1]
        ↓
Trend Agent (LLM + lin. reg.)     → quantitative_metrics.normalized_signal      ∈ [-1, 1]
        ↓
Decision Maker (pure math)        → weighted sum → BUY / HOLD / SELL / SHORT
                                    + price target, stop-loss, R:R ratio
```

---

## Signal Wiring

Each perception agent already computed a normalised scalar alongside its LLM narrative. The new decision agent reads only those scalars:

| Agent | Field consumed | Range |
|-------|---------------|-------|
| Indicator | `quantitative_metrics.final_indicator_signal` | −1 to +1 |
| Trend | `quantitative_metrics.normalized_signal` | −1 to +1 |
| Pattern | `direction × confidence_score` (parsed from LLM JSON output) | −1 to +1 |

### Weighted Combination

```
combined_signal = 0.40 × indicator_signal
                + 0.40 × trend_signal
                + 0.20 × pattern_signal
```

All weights are configurable in `DECISION_CONFIG` inside `master_portfolio.py`.

### Decision Thresholds

| combined_signal | Decision |
|----------------|----------|
| ≥ +0.15 | **BUY** |
| −0.15 to +0.15 | **HOLD** |
| ≤ −0.15 | **SELL** |
| ≤ −0.35 (and `allow_short=True`) | **SHORT** |

Thresholds are configurable.

### Price Levels

- **Stop-loss**: `current_price ± ATR × atr_multiplier` (default ×2.0)
- **Target**: nearest support/resistance from the trend agent's linear-regression channel, capped by `risk_reward_target × SL_distance` (default 2.0×)
- **Protective override**: if an `entry_price` is supplied and unrealised loss exceeds −5 %, a `HOLD` signal is automatically promoted to `SELL`

---

## Files Changed / Added

### New: `decision_agent_quant.py`

Pure-math decision node. No LLM instantiated or called.

Key public API:

```python
# Used directly (e.g. in tests or backtesting loops)
decision: TradeDecision = make_trade_decision(state, weights=..., thresholds=..., ...)

# Used inside LangGraph
node = create_quant_decision_node(allow_short=True, atr_multiplier_sl=2.0)
graph.add_node("Decision Maker", node)
```

`TradeDecision` dataclass fields:

```python
ticker, decision, combined_signal, signal_strength,
current_price, entry_price, unrealized_pnl_pct,
target_price, stop_loss, risk_reward_ratio, atr,
signal_breakdown, decision_rationale
```

### Modified: `agent_state.py`

Added `entry_price: Optional[float]` to `IndicatorAgentState`.  
This carries the user's average cost basis through the LangGraph state so the decision node can compute unrealised P&L and apply the protective SELL override.

### Modified: `graph_setup.py`

Added two new graph compilation methods to `SetGraph`:

**`set_graph_quant(**kwargs)`**  
Same Indicator → Pattern → Trend perception pipeline as before, but the final `Decision Maker` node is `create_quant_decision_node` (no LLM).

**`set_graph_full_quant(**kwargs)`**  
Fully deterministic — zero LLM calls across the entire pipeline:
- Indicator: `quantify_indicators_from_kline()` (TA-Lib, existing function)
- Trend: `quantify_trend_from_kline()` (linear regression, existing function)
- Pattern: TA-Lib CDL candlestick aggregation (see `quant_nodes.py`)
- Decision: `create_quant_decision_node` (same as above)

The original `set_graph()` (LLM decision maker) is untouched.

### New: `quant_nodes.py`

Three drop-in LangGraph nodes for the fully-deterministic pipeline:

- `quant_indicator_node` — wraps `quantify_indicators_from_kline()`
- `quant_trend_node` — wraps `quantify_trend_from_kline()`
- `quant_pattern_node` — aggregates 15 TA-Lib `CDL*` functions (Engulfing, Hammer, MorningStar, ThreeWhiteSoldiers, Doji, etc.) over the last 5 candles into a single `direction × confidence` signal

### Modified: `pattern_agent.py`

Fixed a pre-existing bug: the JSON schema embedded in `pattern_text` contained `{` / `}` characters that LangChain's `ChatPromptTemplate` was misinterpreting as template variable placeholders. Replaced `ChatPromptTemplate` + `MessagesPlaceholder` with direct `SystemMessage` / `HumanMessage` construction, bypassing the template engine entirely.

### Modified: `indicator_agent.py`

Fixed the same class of bug: `chain.invoke(messages)` was passing a raw list where `ChatPromptTemplate` expected a dict. Changed to `chain.invoke({"messages": messages})`.

### New: `master_portfolio.py`

Portfolio-level runner script. Fetches OHLCV data from Yahoo Finance, runs the full LangGraph pipeline for each ticker, and prints a consolidated decision table.

---

## Master Portfolio Script

### Configuration (edit at the top of the file)

```python
PORTFOLIO = {
    "AAPL":    {"entry_price": 189.50, "lookback_days": 120, "timeframe": "1d"},
    "TSLA":    {"entry_price":   None, "lookback_days":  90, "timeframe": "1d"},
    "NVDA":    {"entry_price": 870.00, "lookback_days":  60, "timeframe": "1d"},
    "BTC-USD": {"entry_price":   None, "lookback_days":  90, "timeframe": "1d"},
}
```

| Field | Description |
|-------|-------------|
| `entry_price` | Average cost of your current position. `None` = flat (no position). |
| `lookback_days` | Calendar days of history to fetch from Yahoo Finance. |
| `timeframe` | yfinance interval string: `1d`, `1h`, `15m`, `5m`, `1wk`, etc. |

```python
DECISION_CONFIG = {
    "weights":            {"indicator": 0.40, "trend": 0.40, "pattern": 0.20},
    "thresholds":         {"buy": 0.15, "sell": -0.15, "short": -0.35},
    "atr_multiplier_sl":  2.0,
    "risk_reward_target": 2.0,
    "allow_short":        True,
}
```

### Usage

```bash
# Default — LLM perception agents, quant decision node
python master_portfolio.py

# Specify provider and API key
python master_portfolio.py --provider anthropic --api-key sk-ant-...

# Long-only portfolio, tighter stops
python master_portfolio.py --no-short --atr-mult 1.5

# Save results to JSON for downstream processing / backtesting
python master_portfolio.py --output results.json

# Print per-ticker signal breakdown
python master_portfolio.py --breakdown
```

### Output

```
========================================================================
  QuantAgent Portfolio  —  2026-05-31 18:30
  Provider: openai  |  Tickers: 4
========================================================================

 Ticker  Decision  Signal  Strength   Price    Entry    PnL%    Target   StopLoss  R:R   ATR
   AAPL  BUY       +0.312  Moderate  188.45   189.50   -0.6%   194.20   183.10    1:2.0  2.68
   TSLA  HOLD      +0.041  Weak      245.30      —        —     251.40   238.90    1:1.5  4.12
   NVDA  SHORT    -0.421   Strong    812.40   870.00   -6.6%   793.20   824.10    1:2.1  6.30
BTC-USD  BUY       +0.187  Moderate  67840.0     —        —    71200.0  64300.0   1:2.0  1420
```

### yfinance Data Limits by Timeframe

| Interval | Max lookback |
|----------|-------------|
| `1m` | 7 days |
| `5m` / `15m` / `30m` | 60 days |
| `1h` | ~730 days |
| `1d` / `1wk` | 10 years |

---

## Pipeline Comparison

| Mode | Command / method | LLM calls | Deterministic | API quota |
|------|-----------------|-----------|--------------|-----------|
| Original (LLM decision) | `set_graph()` | 4+ per ticker | No | High |
| Quant decision only | `set_graph_quant()` | 3 per ticker (perception only) | Decision only | Medium |
| Fully deterministic | `set_graph_full_quant()` | 0 | Yes | None |

The master portfolio script uses **`set_graph_quant()`** by default (quant decision, LLM perception).  
To run zero-LLM, instantiate `TradingGraph` and call `graph_setup.set_graph_full_quant()` directly.
