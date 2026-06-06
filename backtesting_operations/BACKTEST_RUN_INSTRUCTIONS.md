# Backtest Run Instructions — QuantAgent Tuning Experiments
**2026-06-06**

Each section below is a self-contained prompt block.
Paste the MASTER CONTEXT once at the start of a session, then paste the
relevant CONFIG PROMPT to execute that experiment.
No prior conversation history is needed — every prompt is fully self-contained.

---

## HOW TO USE

1. Open a new Claude Code / Cursor / GPT-4o session with repo access.
2. Paste **MASTER CONTEXT** first (one time per session).
3. Paste one **CONFIG PROMPT** block — the LLM will make edits, run the
   command, report metrics, then revert any file changes.
4. Save the printed metrics into `backtesting_operations/RESULTS_LOG.md`.
5. Repeat from step 3 for the next config.

Work in order: Group A first (no edits), then B and C one at a time.

---

---

# ═══════════════════════════════════════════════════════════════
# MASTER CONTEXT  —  paste this once at the start of every session
# ═══════════════════════════════════════════════════════════════

```
You are operating inside the QuantAgent-Build repository.
All paths below are relative to the repo root.

REPO LAYOUT (relevant files only)
──────────────────────────────────
backtest.py                          ← main entry point
backtest_engine.py                   ← simulation engine (do not edit)
indicator_agent.py                   ← TA-Lib indicator computation
trend_agent.py                       ← linear-regression channel
quant_pipeline/
    quant_nodes.py                   ← TA-Lib CDL pattern node
    decision_agent_quant.py          ← weighted signal combiner
backtesting_operations/RESULTS_LOG.md  ← where to record results

CURRENT HARD-CODED VALUES (do not change unless the config says so)
─────────────────────────────────────────────────────────────────────
indicator_agent.py → _compute_indicator_series:
    talib.RSI(c, timeperiod=14)
    talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    talib.ROC(c, timeperiod=10)
    talib.STOCH(..., fastk_period=14, slowk_period=3, slowd_period=3, ...)
    talib.WILLR(h, l, c, timeperiod=14)

indicator_agent.py → _normalize_indicator_signals:
    rsi_sig   = np.clip((rsi   - 50.0) / 30.0, -1.0, 1.0)   # scale = 30.0
    stoch_sig = np.clip((slowk - 50.0) / 30.0, -1.0, 1.0)
    willr_sig = np.clip((willr + 50.0) / 30.0, -1.0, 1.0)
    rolling window for MACD/ROC std: window=30

trend_agent.py → quantify_trend_strength:
    upper_channel = regression_line + (2 * std_dev)         # multiplier = 2
    lower_channel = regression_line - (2 * std_dev)
    normalized_signal = 1.0 - (position * 2.0)              # mean-reversion
    normalized_signal = min(normalized_signal * 1.2, 1.0)   # slope boost = 1.2

quant_pipeline/quant_nodes.py → quantify_candlestick_patterns:
    def quantify_candlestick_patterns(kline_data: dict, lookback: int = 5):
    _CDL_PATTERNS contains 14 entries including ("Doji", ...) and ("Harami", ...)

BACKTEST SETUP
──────────────
Symbols  : GOOGL, XOM, JNJ  (tech / energy / healthcare)
Start    : 2015-01-01
Cadence  : daily ("D")
Window   : 60 bars (overridden to 20 for config 10_swing_momentum)
Mode     : --no-llm (pure quant, zero LLM calls)
Output   : backtest_results/run_<N>/  where N = config number

HOW TO RUN
──────────
Activate the project venv, then from the repo root:
    python backtest.py --run-config <NAME> --out backtest_results/run_<N> --no-llm

KEY METRICS TO RECORD FROM STDOUT / backtest_results/run_<N>/summary.csv
─────────────────────────────────────────────────────────────────────────
For each symbol (GOOGL, XOM, JNJ) record:
    agent_total_return      excess_return (vs buy-hold)
    sharpe_annual           win_rate
    max_drawdown            n_trades
Also note the final line: "Agent beats 200dma trend-follower on X/3 symbols."
```

---

---

# ═══════════════════════════════════════════════════════════════
# GROUP A PROMPT  —  configs 1 / 2 / 6 / 7 / 8 / 9
# NO SOURCE FILE EDITS REQUIRED — paste as a single block
# ═══════════════════════════════════════════════════════════════

```
TASK: Run the six pure-weight / no-edit experiment configs in sequence.
No source file changes are needed. Run each command, wait for it to finish,
then record the metrics before running the next one.

For each run, record:
  symbol | agent_return | excess_return | sharpe | win_rate | max_dd | n_trades

RUN 1 — Baseline (control)
    python backtest.py --run-config 1_baseline --out backtest_results/run_1 --no-llm

RUN 2 — Trend-heavy weights
    python backtest.py --run-config 2_trend_heavy --out backtest_results/run_2 --no-llm

RUN 6 — No pattern signal
    python backtest.py --run-config 6_indicator_only --out backtest_results/run_6 --no-llm

RUN 7 — Pure indicator only
    python backtest.py --run-config 7_pure_indicator --out backtest_results/run_7 --no-llm

RUN 8 — Pure trend only
    python backtest.py --run-config 8_pure_trend --out backtest_results/run_8 --no-llm

RUN 9 — Pure pattern only
    python backtest.py --run-config 9_pure_pattern --out backtest_results/run_9 --no-llm

After all 6 runs complete, read each backtest_results/run_N/summary.csv and
print all results in a single comparison table with columns:
    config | GOOGL_return | XOM_return | JNJ_return | avg_sharpe | avg_win_rate
```

---

---

# ═══════════════════════════════════════════════════════════════
# CONFIG 3 PROMPT  —  3_trend_continuation
# 1 source file edit required
# ═══════════════════════════════════════════════════════════════

```
TASK: Run experiment config 3_trend_continuation.

STEP 1 — Edit trend_agent.py
In function quantify_trend_strength, find and replace EXACTLY:
    OLD: normalized_signal = 1.0 - (position * 2.0)
    NEW: normalized_signal = (position - 0.5) * 2.0
(There is only one occurrence of this line. Do not change any other line.)

STEP 2 — Run the backtest
    python backtest.py --run-config 3_trend_continuation \
        --out backtest_results/run_3 --no-llm

STEP 3 — Record metrics
Read backtest_results/run_3/summary.csv and report all rows.

STEP 4 — REVERT the edit
Restore trend_agent.py to:
    normalized_signal = 1.0 - (position * 2.0)
Confirm the file is reverted before reporting done.
```

---

---

# ═══════════════════════════════════════════════════════════════
# CONFIG 4 PROMPT  —  4_aggressive
# 6 values to change in indicator_agent.py
# ═══════════════════════════════════════════════════════════════

```
TASK: Run experiment config 4_aggressive.

STEP 1 — Edit indicator_agent.py  (6 value changes, all in _compute_indicator_series
          or _normalize_indicator_signals — do not touch any other function)

  1a. talib.RSI(c, timeperiod=14)
      →  talib.RSI(c, timeperiod=7)

  1b. talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
      →  talib.MACD(c, fastperiod=8, slowperiod=20, signalperiod=6)

  1c. talib.ROC(c, timeperiod=10)
      →  talib.ROC(c, timeperiod=5)

  1d. In _normalize_indicator_signals, the oscillator normalization lines:
      OLD: rsi_sig   = np.clip((rsi   - 50.0) / 30.0, -1.0, 1.0)
           stoch_sig = np.clip((slowk - 50.0) / 30.0, -1.0, 1.0)
           willr_sig = np.clip((willr + 50.0) / 30.0, -1.0, 1.0)
      NEW: rsi_sig   = np.clip((rsi   - 50.0) / 20.0, -1.0, 1.0)
           stoch_sig = np.clip((slowk - 50.0) / 20.0, -1.0, 1.0)
           willr_sig = np.clip((willr + 50.0) / 20.0, -1.0, 1.0)

STEP 2 — Run the backtest
    python backtest.py --run-config 4_aggressive \
        --out backtest_results/run_4 --no-llm

STEP 3 — Record metrics from backtest_results/run_4/summary.csv

STEP 4 — REVERT all 6 changes in indicator_agent.py to their original values:
    timeperiod=14, fastperiod=12, slowperiod=26, signalperiod=9,
    timeperiod=10 (ROC), scale divisor back to 30.0 for all three clips.
Confirm revert before reporting done.
```

---

---

# ═══════════════════════════════════════════════════════════════
# CONFIG 5 PROMPT  —  5_conservative
# 2 files to edit
# ═══════════════════════════════════════════════════════════════

```
TASK: Run experiment config 5_conservative.

STEP 1a — Edit indicator_agent.py → _normalize_indicator_signals
    OLD: rsi_sig   = np.clip((rsi   - 50.0) / 30.0, -1.0, 1.0)
         stoch_sig = np.clip((slowk - 50.0) / 30.0, -1.0, 1.0)
         willr_sig = np.clip((willr + 50.0) / 30.0, -1.0, 1.0)
    NEW: rsi_sig   = np.clip((rsi   - 50.0) / 40.0, -1.0, 1.0)
         stoch_sig = np.clip((slowk - 50.0) / 40.0, -1.0, 1.0)
         willr_sig = np.clip((willr + 50.0) / 40.0, -1.0, 1.0)

STEP 1b — Edit trend_agent.py → quantify_trend_strength
    OLD: upper_channel = regression_line + (2 * std_dev)
         lower_channel = regression_line - (2 * std_dev)
    NEW: upper_channel = regression_line + (2.5 * std_dev)
         lower_channel = regression_line - (2.5 * std_dev)

STEP 2 — Run the backtest
    python backtest.py --run-config 5_conservative \
        --out backtest_results/run_5 --no-llm

STEP 3 — Record metrics from backtest_results/run_5/summary.csv

STEP 4 — REVERT both files:
    indicator_agent.py scale divisors back to 30.0
    trend_agent.py channel multipliers back to 2
Confirm both files are reverted before reporting done.
```

---

---

# ═══════════════════════════════════════════════════════════════
# CONFIG 10 PROMPT  —  10_swing_momentum
# 3 source file edits + auto window override (20 bars)
# ═══════════════════════════════════════════════════════════════

```
TASK: Run experiment config 10_swing_momentum.
NOTE: This config uses window_bars=20 automatically (set in EXPERIMENT_CONFIGS).

STEP 1a — Edit indicator_agent.py → _compute_indicator_series  (same as config 4)
    RSI:  timeperiod=14 → 7
    ROC:  timeperiod=10 → 5
    MACD: fastperiod=12, slowperiod=26, signalperiod=9
       →  fastperiod=8,  slowperiod=20, signalperiod=6

STEP 1b — Edit indicator_agent.py → _normalize_indicator_signals
    Scale divisors: 30.0 → 20.0  (same 3 lines as config 4)

STEP 1c — Edit trend_agent.py → quantify_trend_strength
    OLD: normalized_signal = 1.0 - (position * 2.0)
    NEW: normalized_signal = (position - 0.5) * 2.0

STEP 2 — Run the backtest (window_bars=20 is applied automatically)
    python backtest.py --run-config 10_swing_momentum \
        --out backtest_results/run_10 --no-llm

STEP 3 — Record metrics from backtest_results/run_10/summary.csv
Note: with window=20 and daily cadence from 2015, expect ~2,500 decision bars
per symbol — n_trades will be significantly higher than other configs.

STEP 4 — REVERT all three files to original values:
    indicator_agent.py: RSI 7→14, ROC 5→10, MACD 8/20/6→12/26/9, scale 20→30
    trend_agent.py:     normalized_signal = 1.0 - (position * 2.0)
Confirm all reverts before reporting done.
```

---

---

# ═══════════════════════════════════════════════════════════════
# CONFIG 11 PROMPT  —  11_sr_bands
# 2 edits in trend_agent.py
# ═══════════════════════════════════════════════════════════════

```
TASK: Run experiment config 11_sr_bands.

STEP 1a — Edit trend_agent.py → quantify_trend_strength
    OLD: upper_channel = regression_line + (2 * std_dev)
         lower_channel = regression_line - (2 * std_dev)
    NEW: upper_channel = regression_line + (1.5 * std_dev)
         lower_channel = regression_line - (1.5 * std_dev)

STEP 1b — Edit trend_agent.py → quantify_trend_strength  (slope amplification)
    OLD: normalized_signal = min(normalized_signal * 1.2, 1.0)
    NEW: normalized_signal = min(normalized_signal * 1.5, 1.0)
(There is a matching line for the negative case — update it too:)
    OLD: normalized_signal = max(normalized_signal * 1.2, -1.0)
    NEW: normalized_signal = max(normalized_signal * 1.5, -1.0)

STEP 2 — Run the backtest
    python backtest.py --run-config 11_sr_bands \
        --out backtest_results/run_11 --no-llm

STEP 3 — Record metrics from backtest_results/run_11/summary.csv

STEP 4 — REVERT trend_agent.py:
    channel multipliers back to 2
    slope amplification back to 1.2 (both the min and max lines)
Confirm revert before reporting done.
```

---

---

# ═══════════════════════════════════════════════════════════════
# CONFIG 12 PROMPT  —  12_reversal_hunter
# 2 edits in quant_pipeline/quant_nodes.py
# ═══════════════════════════════════════════════════════════════

```
TASK: Run experiment config 12_reversal_hunter.

STEP 1a — Edit quant_pipeline/quant_nodes.py → _CDL_PATTERNS list
Remove these two entries (the entire tuple lines including the trailing comma):
    ("Doji",   lambda o,h,l,c: talib.CDLDOJI(o,h,l,c)),
    ("Harami", lambda o,h,l,c: talib.CDLHARAMI(o,h,l,c)),
The remaining list should have 12 entries. Do not touch any other entry.

STEP 1b — Edit quant_pipeline/quant_nodes.py → quantify_candlestick_patterns
    OLD: def quantify_candlestick_patterns(kline_data: dict, lookback: int = 5):
    NEW: def quantify_candlestick_patterns(kline_data: dict, lookback: int = 1):

STEP 2 — Run the backtest
    python backtest.py --run-config 12_reversal_hunter \
        --out backtest_results/run_12 --no-llm

STEP 3 — Record metrics from backtest_results/run_12/summary.csv

STEP 4 — REVERT quant_pipeline/quant_nodes.py:
    Re-add ("Doji", ...) and ("Harami", ...) lines to _CDL_PATTERNS in their
    original positions (Doji was 13th, AbandonedBaby 14th).
    Restore lookback default to 5.
Confirm list has 14 entries and lookback=5 before reporting done.
```

---

---

## RESULTS RECORDING TEMPLATE

After each run, paste results into `backtesting_operations/RESULTS_LOG.md`
using this table format:

```
| Config                | GOOGL return | XOM return | JNJ return | Avg Sharpe | Avg Win% | Avg DD | Avg trades | Beats SMA |
|-----------------------|-------------|------------|------------|------------|----------|--------|------------|-----------|
| 1_baseline            |             |            |            |            |          |        |            |  X/3      |
| 2_trend_heavy         |             |            |            |            |          |        |            |  X/3      |
| 3_trend_continuation  |             |            |            |            |          |        |            |  X/3      |
| 4_aggressive          |             |            |            |            |          |        |            |  X/3      |
| 5_conservative        |             |            |            |            |          |        |            |  X/3      |
| 6_indicator_only      |             |            |            |            |          |        |            |  X/3      |
| 7_pure_indicator      |             |            |            |            |          |        |            |  X/3      |
| 8_pure_trend          |             |            |            |            |          |        |            |  X/3      |
| 9_pure_pattern        |             |            |            |            |          |        |            |  X/3      |
| 10_swing_momentum     |             |            |            |            |          |        |            |  X/3      |
| 11_sr_bands           |             |            |            |            |          |        |            |  X/3      |
| 12_reversal_hunter    |             |            |            |            |          |        |            |  X/3      |
```

**Interpretation guide**
- `excess_return > 0`  → agent beats buy-and-hold
- `sharpe_annual > 1`  → good risk-adjusted return
- `win_rate > 55%`     → edge exists
- `max_drawdown < 20%` → manageable risk
- `Beats SMA 3/3`      → beats dumb 200dma baseline on all symbols
