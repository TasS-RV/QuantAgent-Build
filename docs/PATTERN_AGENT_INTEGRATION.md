# Pattern Agent Integration
**2026-06-06**

---

## Overview

The pattern agent has two operating modes depending on whether `NO_LLM_MODE` is
enabled in `master_portfolio.py`. Both modes write to the same `pattern_report`
state key, so the downstream decision node is unaffected by which mode is active.

---

## Mode 1 — TA-Lib CDL (NO_LLM_MODE = True)

**File:** `quant_pipeline/quant_nodes.py` → `quant_pattern_node()`

The quantitative pipeline uses TA-Lib's candlestick recognition functions to
detect micro-reversal patterns on a bar-by-bar basis. This requires no API key
and runs in microseconds.

### How it works

1. The last `lookback=5` candles are scanned for each of the 14 CDL functions.
2. Each function returns `+100` (bullish), `-100` (bearish), or `0` (no signal).
3. Scores are summed across all triggered patterns into a `total_score`.
4. Direction and confidence are derived:

```
direction  = sign(total_score)
confidence = min(|total_score| / (14 × 100) × 3, 1.0)
```

The `× 3` multiplier spreads the confidence range so a single fired pattern
maps to roughly 0.21 confidence rather than 0.07.

### Active patterns

| Pattern | Type |
|---|---|
| Engulfing | Reversal |
| Hammer / InvertedHammer | Reversal |
| MorningStar / EveningStar | Reversal |
| ThreeWhiteSoldiers / ThreeBlackCrows | Continuation |
| ShootingStar | Reversal |
| DarkCloudCover / Piercing | Reversal |
| Harami | Reversal |
| Marubozu | Momentum |
| Doji | Indecision |
| AbandonedBaby | Reversal |

### Output schema

```json
{
  "macro_pattern_name": "Engulfing + 1 more",
  "direction": 1,
  "confidence_score": 0.43,
  "justification": "TA-Lib CDL: Engulfing, Hammer. Aggregate score=200."
}
```

---

## Mode 2 — LLM Vision Agent (NO_LLM_MODE = False)

**File:** `pattern_agent.py` → `create_pattern_agent()`

The LLM receives a rendered candlestick chart image and identifies macro structural
formations spanning 20–80 candles. Requires a valid API key for the chosen provider.

### Detected formations

Head & Shoulders, Inverse Head & Shoulders, Double Bottom / Top,
Rounded Bottom / Top, Falling / Rising Wedge,
Ascending / Descending Triangle, Bullish / Bearish Flag,
Rectangle, Symmetrical Triangle.

### Output schema (same keys)

```json
{
  "macro_pattern_name": "Inverse Head and Shoulders",
  "direction": 1,
  "confidence_score": 0.8,
  "justification": "Clear neckline break with right shoulder confirmation."
}
```

---

## Signal chain position

```
Indicator Agent  (TA-Lib: RSI / MACD / Stoch / WillR / ROC)   weight: 0.40
       ↓
Pattern Agent    (TA-Lib CDL  —or—  LLM vision)                weight: 0.20
       ↓
Trend Agent      (scipy linear-regression channel)              weight: 0.40
       ↓
Decision Maker   (weighted combination → BUY / HOLD / SELL / SHORT)
```

---

## Toggle

`master_portfolio.py` exposes two top-level variables:

| Variable | Effect |
|---|---|
| `NO_LLM_MODE = True` | Use TA-Lib CDL pattern node (Mode 1) |
| `NO_LLM_MODE = False` | Use LLM vision pattern agent (Mode 2) |
| `USE_CDL_PATTERNS = False` | Disable pattern node entirely in NO_LLM_MODE; pattern weight redistributed proportionally to indicator + trend |

`USE_CDL_PATTERNS` has no effect when `NO_LLM_MODE = False`.

---

## Key distinction

| | TA-Lib CDL (Mode 1) | LLM Vision (Mode 2) |
|---|---|---|
| Candle window | 1 – 5 bars | 20 – 80 bars |
| Pattern type | Micro-reversal (single candle sequences) | Macro structural formations |
| Speed | < 1 ms | 2 – 10 s (API round-trip) |
| Requires API key | No | Yes |
| Deterministic | Yes | No |
| Best for | Daily / intraday reversal signals | Swing / position trade setups |
