# Algorithm Tuning Plan — Test_Algos Branch
**2026-06-06**

---

## Goal

Systematically tweak the three quantitative perception agents (Indicator, Trend, Pattern)
and the Decision layer to find configurations that improve backtest performance.
Every parameter listed below is a real, currently hard-coded value in the source — this
document maps each one to its file, line context, and tuning options.

---

## The Signal Chain

```
kline data (OHLCV)
       │
       ├─ Indicator Agent ──► indicator_signal  ∈ [-1, 1]   weight: 0.40
       ├─ Trend Agent     ──► trend_signal      ∈ [-1, 1]   weight: 0.40
       └─ Pattern Agent   ──► pattern_signal    ∈ [-1, 1]   weight: 0.20
                                      │
                              combined = Σ w·signal
                                      │
                              Decision Node
                              ┌─ ≥ +0.15  →  BUY
                              ├─ ≤ -0.15  →  SELL
                              ├─ ≤ -0.35  →  SHORT
                              └─ else     →  HOLD
```

All three signals are independently computed on the same kline window, then
combined in `quant_pipeline/decision_agent_quant.py`.

---

## 1. Indicator Agent  (`indicator_agent.py`)

### 1a — TA-Lib periods (hard-coded in `_compute_indicator_series`)

| Indicator | Parameter | Current | Tuning range | Notes |
|---|---|---|---|---|
| RSI | `timeperiod` | **14** | 7 – 21 | Shorter = noisier, more signals |
| MACD fast | `fastperiod` | **12** | 8 – 16 | |
| MACD slow | `slowperiod` | **26** | 20 – 30 | Must be > fast |
| MACD signal | `signalperiod` | **9** | 5 – 12 | |
| Stoch %K | `fastk_period` | **14** | 5 – 21 | |
| Stoch smooth | `slowk_period / slowd_period` | **3 / 3** | 2 – 5 | Heavier = smoother |
| Williams %R | `timeperiod` | **14** | 7 – 21 | Mirrors RSI sensitivity |
| ROC | `timeperiod` | **10** | 5 – 20 | Rate-of-change lookback |

### 1b — Normalization formula (hard-coded in `_normalize_indicator_signals`)

The three oscillators (RSI, Stoch, WillR) are normalized by:

```python
rsi_sig   = clip((rsi   - 50.0) / 30.0, -1, 1)   # center=50, scale=30
stoch_sig = clip((slowk - 50.0) / 30.0, -1, 1)
willr_sig = clip((willr + 50.0) / 30.0, -1, 1)
```

MACD and ROC use volatility-normalization:
```python
macd_sig = tanh(macdhist / rolling_30bar_std)
roc_sig  = tanh(roc      / rolling_30bar_std)
```

**Tuning knobs:**
- **Oscillator center** (currently `50`): shifting this changes the bullish/bearish bias.
  A center of `55` on RSI means "mildly bullish by default".
- **Oscillator scale** (currently `30`): controls saturation speed.
  `scale=20` → signals hit ±1 faster (more aggressive); `scale=40` → slower (more conservative).
- **MACD/ROC rolling window** (currently `30`): the standard-deviation window.
  Shorter (e.g. `14`) makes it react faster to regime changes.

### 1c — Component aggregation (hard-coded in `_normalize_indicator_signals`)

```python
final_sig = nanmean([rsi_sig, stoch_sig, willr_sig, macd_sig, roc_sig])
```

Currently an **equal-weight average**. Two upgrade paths:

| Approach | How | Where to change |
|---|---|---|
| Weighted average | Replace `nanmean` with `np.average(..., weights=[w1,w2,w3,w4,w5])` | `_normalize_indicator_signals` |
| Drop a component | Remove one indicator from the stacked array | `_normalize_indicator_signals` |
| Add a component | Add EMA-cross, ADX, Bollinger position, CCI, etc. via TA-Lib | `_compute_indicator_series` + `_normalize_indicator_signals` |

Candidate additions: `talib.ADX` (trend strength), `talib.CCI` (commodity channel), EMA-9 vs EMA-21 cross.

---

## 2. Trend Agent  (`trend_agent.py`)

### 2a — Channel model (hard-coded in `quantify_trend_strength`)

```python
std_dev       = std(y - regression_line)
upper_channel = regression_line + 2 * std_dev    # ← multiplier = 2
lower_channel = regression_line - 2 * std_dev
```

**Channel width multiplier** (currently `2`):
- `1.5` → tighter bands, price breaks out more often → more signals, more noise
- `2.5` → wider bands, fewer breakouts → fewer but cleaner signals

### 2b — Signal direction from channel position

```python
position = (price - lower) / (upper - lower)     # 0 = at support, 1 = at resistance
normalized_signal = 1.0 - (position * 2.0)       # 1 = at support (bullish), -1 = at resistance (bearish)
```

This is **mean-reversion logic**: price at support → BUY signal.
To flip to **trend-continuation** logic: price above midline → bullish.

```python
# Mean-reversion (current):
normalized_signal = 1.0 - (position * 2.0)

# Trend-continuation alternative:
normalized_signal = (position - 0.5) * 2.0
```

This is a significant behavioural switch — worth testing as a separate config.

### 2c — Slope amplification (hard-coded)

```python
if slope > 0 and signal > 0:
    signal = min(signal * 1.2, 1.0)   # ← boost factor = 1.2
elif slope < 0 and signal < 0:
    signal = max(signal * 1.2, -1.0)
```

**Amplification factor** (currently `1.2`): when price is near support AND trend is rising,
the signal is boosted.
- `1.0` = no amplification (disable entirely)
- `1.5` = stronger slope confirmation
- Could extend to penalize when slope and position disagree

### 2d — Lookback window

The trend agent uses whatever candles are passed in from `master_portfolio.py` /
`backtest.py` (via `lookback_days`). The regression is fitted on the **entire window**.

| Window (days) | Character |
|---|---|
| 20 – 30 | Short-term momentum |
| 45 – 60 | Medium swing |
| 90 – 120 | Macro trend — slow, less noise |

Shorter windows make the regression line react faster; longer ones are more stable.
This is already configurable in `demo_portfolio.json` (`lookback_days` per ticker).

---

## 3. Pattern Agent  (`quant_pipeline/quant_nodes.py`)

### 3a — Pattern lookback window

```python
def quantify_candlestick_patterns(kline_data, lookback: int = 5):
```

**`lookback`** (currently `5`): how many of the most recent candles each TA-Lib
CDL function is allowed to fire on.
- `1` = only the last candle (very strict)
- `3` = last 3 candles (standard)
- `5` = default — relaxed detection
- `10` = allows older signals to count (increases pattern noise)

### 3b — Active pattern list

Currently 14 patterns in `_CDL_PATTERNS`. Each can be individually enabled/disabled:

| Pattern | Type | Typical edge |
|---|---|---|
| Engulfing | Reversal | Strong — high accuracy |
| Hammer / InvertedHammer | Reversal | Bottom detection |
| MorningStar / EveningStar | Reversal | Multi-candle, higher reliability |
| ThreeWhiteSoldiers / ThreeBlackCrows | Continuation | Strong trend confirmation |
| ShootingStar | Reversal | Top detection |
| DarkCloudCover / Piercing | Reversal | Moderate |
| Harami | Reversal | Weak on its own |
| Marubozu | Momentum | Strong body = commitment |
| Doji | Indecision | Neutral — often noise |
| AbandonedBaby | Reversal | Rare, high reliability |

Quick experiment: run a config with only the high-reliability patterns
(Engulfing, MorningStar/EveningStar, ThreeWhite/BlackSoldiers, AbandonedBaby).

### 3c — Confidence scaling

```python
confidence = min(abs(total_score) / _MAX_SCORE * 3, 1.0)   # ← multiplier = 3
```

`_MAX_SCORE = 14 × 100 = 1400`. The `× 3` spreads the confidence range so a single
pattern (score = 100/1400 ≈ 0.07) maps to confidence ≈ 0.21 rather than 0.07.

**Multiplier** (currently `3`):
- `1` = conservative confidence (most signals < 0.3)
- `3` = default
- `5` = aggressive (one pattern → high confidence)

---

## 4. Decision Layer  (`quant_pipeline/decision_agent_quant.py`)

### 4a — Agent weights

```python
DEFAULT_WEIGHTS = {"indicator": 0.40, "trend": 0.40, "pattern": 0.20}
```

Must sum to 1.0. Pass overrides to `backtest.py` via `DECISION_CONFIG["weights"]`
or `master_portfolio.py`.

| Config name | Indicator | Trend | Pattern | Use case |
|---|---|---|---|---|
| Current default | 0.40 | 0.40 | 0.20 | Balanced |
| Trend-heavy | 0.20 | 0.60 | 0.20 | Trending markets |
| Indicator-heavy | 0.60 | 0.30 | 0.10 | Mean-reversion / HFT |
| Pattern-heavy | 0.20 | 0.30 | 0.50 | Event-driven reversals |
| Trend + indicator, no pattern | 0.50 | 0.50 | 0.00 | Quant-only, no CDL |

### 4b — Decision thresholds

```python
DEFAULT_THRESHOLDS = {"buy": 0.15, "sell": -0.15, "short": -0.35}
```

| Setting | buy | sell | short | Effect |
|---|---|---|---|---|
| Tight (aggressive) | 0.05 | -0.05 | -0.20 | More trades, more noise |
| Current | 0.15 | -0.15 | -0.35 | Balanced |
| Wide (conservative) | 0.25 | -0.25 | -0.50 | Fewer, higher-conviction trades |
| Asymmetric (bullish bias) | 0.10 | -0.20 | -0.40 | Easier to BUY, harder to SELL/SHORT |

### 4c — Stop-loss sizing

```python
sl_distance = atr * atr_multiplier_sl    # default: 2.0
```

| Multiplier | Stop width | Trade-off |
|---|---|---|
| 1.0 | Tight | More stops hit, but better R:R when it works |
| 2.0 | Default | ATR-standard |
| 3.0 | Wide | Fewer whipsaw exits; larger losses when wrong |

ATR period is also hard-coded at `14` in `_compute_atr()` — can be shortened (e.g. `7`)
for more reactive stops on shorter timeframes.

### 4d — Risk:reward target

```python
risk_reward_target = 2.0   # target = sl_distance × 2
```

Targets are also capped at the nearest S/R level from the trend agent, so setting
this very high (e.g. 4.0) will still be constrained by where resistance/support is.

---

## 5. How to Run Experiments

### Single run (inspect one config)
```bash
python master_portfolio.py --breakdown
```

### Backtest a config change
Edit `DECISION_CONFIG` in `master_portfolio.py` or `backtest.py`, then:
```bash
python backtest.py --symbol AAPL --start 2023-01-01
python backtest.py --universe SP500_TOP20 --start 2022-01-01
```

### Compare two configs back-to-back
```bash
# Config A (default)
python backtest.py --symbol AAPL --output results/default.json

# Config B (trend-heavy weights)
# Edit DECISION_CONFIG weights in backtest.py, then:
python backtest.py --symbol AAPL --output results/trend_heavy.json
```

---

## 6. Suggested Experiment Sequence

Work from macro to micro — change one thing at a time.

| # | What to test | Change | Hypothesis |
|---|---|---|---|
| 1 | Baseline | No change | Record default metrics to compare against |
| 2 | Trend weights | Indicator 0.20 / Trend 0.60 / Pattern 0.20 | Trending assets should improve |
| 3 | Mean-reversion vs continuation | Flip trend signal formula (§2b) | Test which regime the target assets are in |
| 4 | Wider thresholds | buy=0.25, sell=-0.25 | Fewer trades, check if win rate improves |
| 5 | Shorter RSI / tighter indicators | RSI period 7, ROC period 5 | Faster response — check if it over-trades |
| 6 | Tighter channel | Channel multiplier 1.5 | More trend breakouts detected |
| 7 | High-conviction patterns only | Remove Doji, Harami from `_CDL_PATTERNS` | Noise reduction in pattern signal |
| 8 | Pattern lookback 1 | `lookback=1` | Only last candle patterns — strictest filter |
| 9 | Indicator-only | weights 0.60 / 0.40 / 0.00 | Eliminate pattern noise entirely |
| 10 | LLM pattern layer | `--llm --provider google` | Compare vision-based pattern vs TA-Lib |

---

## 7. Where Each Parameter Lives

| Parameter | File | Function / line |
|---|---|---|
| RSI / MACD / ROC / Stoch / WillR periods | `indicator_agent.py` | `_compute_indicator_series` |
| Oscillator normalization center + scale | `indicator_agent.py` | `_normalize_indicator_signals` |
| MACD/ROC rolling std window | `indicator_agent.py` | `_normalize_indicator_signals` |
| Component aggregation weights | `indicator_agent.py` | `_normalize_indicator_signals` |
| Trend channel width multiplier | `trend_agent.py` | `quantify_trend_strength` |
| Slope amplification factor | `trend_agent.py` | `quantify_trend_strength` |
| Mean-reversion vs continuation formula | `trend_agent.py` | `quantify_trend_strength` |
| Pattern lookback window | `quant_pipeline/quant_nodes.py` | `quantify_candlestick_patterns` |
| Active CDL pattern list | `quant_pipeline/quant_nodes.py` | `_CDL_PATTERNS` |
| Confidence scaling multiplier | `quant_pipeline/quant_nodes.py` | `quantify_candlestick_patterns` |
| Agent weights | `quant_pipeline/decision_agent_quant.py` | `DEFAULT_WEIGHTS` |
| Decision thresholds | `quant_pipeline/decision_agent_quant.py` | `DEFAULT_THRESHOLDS` |
| ATR multiplier (stop-loss width) | `quant_pipeline/decision_agent_quant.py` | `make_trade_decision` |
| ATR period | `quant_pipeline/decision_agent_quant.py` | `_compute_atr` |
| Risk:reward target | `quant_pipeline/decision_agent_quant.py` | `make_trade_decision` |
| Per-ticker lookback window | `demo_portfolio.json` | `lookback_days` field |
