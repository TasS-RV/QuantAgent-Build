# Backtest Results Log
**Started: 2026-06-06**

Symbols: GOOGL / XOM / JNJ  |  Start: 2015-01-01  |  Cadence: daily  |  Window: 60 bars
Mode: pure-quant (NO_LLM_MODE = True)

---

## Quant Config Results

| Config | GOOGL return | XOM return | JNJ return | Avg Sharpe | Avg Win% | Avg trades | Beats SMA |
|---|---|---|---|---|---|---|---|
| 1_baseline           | −81.5% | −91.3% | −82.0% | — | 47.9% | 1065 | 0/3 |
| 2_trend_heavy        | −86.3% | −89.9% | −92.7% | — | 47.3% | 1115 | 0/3 |
| 3_trend_continuation | −88.3% | −79.6% | −82.4% | — | 52.1% | 1478 | 0/3 |
| 4_aggressive         | −96.7% | −94.1% | −93.1% | — | 48.8% | 1651 | 0/3 |
| 5_conservative       | −49.4% | −52.6% | −56.9% | — | 47.1% |  384 | 0/3 |
| 6_indicator_only     | −93.4% | −94.1% | −87.2% | — | 48.2% | 1256 | 0/3 |
| 7_pure_indicator     | −92.9% | −91.7% | −90.3% | — | 51.1% | 1675 | 0/3 |
| 8_pure_trend         | −90.3% | −92.6% | −91.0% | — | 47.6% | 1179 | 0/3 |
| 9_pure_pattern       | −75.4% | −75.4% | −83.9% | — | 50.0% | 1209 | 0/3 |
| 10_swing_momentum    | −86.3% | −85.7% | −66.1% | — | **56.9%** | 1193 | 0/3 |
| 11_sr_bands          | −93.4% | −88.7% | −91.1% | — | 48.9% | 1315 | 0/3 |
| 12_reversal_hunter   | −64.4% | −77.9% | −60.3% | — | 49.4% |  575 | 0/3 |

### Per-Symbol Detail

| Config | GOOGL trades | GOOGL return | GOOGL win% | XOM trades | XOM return | XOM win% | JNJ trades | JNJ return | JNJ win% |
|---|---|---|---|---|---|---|---|---|---|
| 1_baseline           | 1133 | −81.5% | 50.3% | 1015 | −91.3% | 45.8% | 1048 | −82.0% | 47.5% |
| 2_trend_heavy        | 1104 | −86.3% | 50.8% | 1128 | −89.9% | 46.9% | 1112 | −92.7% | 44.2% |
| 3_trend_continuation | 1489 | −88.3% | 52.5% | 1463 | −79.6% | 52.2% | 1483 | −82.4% | 51.6% |
| 4_aggressive         | 1718 | −96.7% | 50.2% | 1606 | −94.1% | 47.9% | 1629 | −93.1% | 48.4% |
| 5_conservative       |  416 | −49.4% | 48.3% |  361 | −52.6% | 47.6% |  375 | −56.9% | 45.3% |
| 6_indicator_only     | 1310 | −93.4% | 49.2% | 1190 | −94.1% | 47.0% | 1269 | −87.2% | 48.5% |
| 7_pure_indicator     | 1673 | −92.9% | 51.9% | 1663 | −91.7% | 50.8% | 1689 | −90.3% | 50.6% |
| 8_pure_trend         | 1152 | −90.3% | 50.3% | 1218 | −92.6% | 47.2% | 1168 | −91.0% | 45.2% |
| 9_pure_pattern       | 1329 | −75.4% | 52.7% | 1100 | −75.4% | 50.2% | 1198 | −83.9% | 47.1% |
| 10_swing_momentum    | 1207 | −86.3% | **57.7%** | 1203 | −85.7% | **56.3%** | 1168 | −66.1% | **56.6%** |
| 11_sr_bands          | 1312 | −93.4% | 50.5% | 1334 | −88.7% | 49.9% | 1298 | −91.1% | 46.2% |
| 12_reversal_hunter   |  588 | −64.4% | 51.5% |  585 | −77.9% | 46.7% |  551 | −60.3% | 50.1% |

---

## Notes & Observations

### Root Cause Analysis (2026-06-06)

**Two compounding problems identified:**

#### Problem 1 — Overtrading cost drag (dominant effect)
- Daily signal rate: ~35–40% of bars generate a trade entry
- Round-trip cost: 0.14% (0.02% commission + 0.05% slippage × 2 sides)
- Compounding over ~1,100 trades: `(1 − 0.0014)^1133 ≈ 0.20` = **~−80% loss from costs alone**
- Validated with zero-cost diagnostic: GOOGL improved from −81.5% → −9.7% (costs explained 91% of the loss)
- **Fix**: Raise signal thresholds or reduce cadence — config 5 (416 trades) is the proof: same signal quality, far less damage

#### Problem 2 — Mean-reversion bias vs long bull market
- `trend_agent.py` uses `1.0 − (position × 2.0)`: price at TOP of channel = bearish/sell
- GOOGL +1,303% over 2015–2026 → price lives near the top of every 60-bar channel
- Systematic SELL/SHORT bias on a 10-year bull market
- Config 3 (continuation formula) raised win rate to 52% but *increased* trades by 30% (more crossings), net negative
- **Fix**: Use continuation formula AND wide thresholds together

### Rankings

**Fewest losses (absolute return):**
1. Config 5 conservative: avg −52.9% (only 384 trades — wide thresholds choke overtrading)
2. Config 12 reversal hunter: avg −67.5% (588 trades — short lookback limits signal frequency)
3. Config 9 pure pattern: avg −78.2% (patterns fire less often than oscillators)

**Best signal quality (win rate):**
1. Config 10 swing_momentum: **56.9% avg** — fast indicators + continuation formula is genuinely directional
2. Config 3 trend_continuation: 52.1% — continuation formula works but kills trade count management
3. Config 7 pure_indicator / config 9 pure_pattern: ~51%

**Worst performers:**
- Config 4 aggressive: most trades (1,651 avg), worst return (−94.6% avg)
- Config 2 trend_heavy: worst per-symbol consistency (XOM/JNJ weakest)
- Config 11 sr_bands: no benefit from tighter channel — same trade frequency, worse returns than baseline

### Key Conclusions

1. **Trade frequency is the primary lever.** Every config with <600 trades outperformed every config with >1,000 trades, regardless of signal type.
2. **Config 10's win rate (56.9%) is real signal.** At ~1,200 trades and 0.14% cost, it still underperforms B&H, but the signal itself is correct more than 56% of the time — better than any other configuration.
3. **Pattern-only (config 9) beats indicator-only (config 7)** on return despite similar win rates — CDL patterns fire less frequently and avoid the oscillator-driven overtrading.
4. **The trend agent is the weakest component in isolation** (config 8 worst of the pure configs). The mean-reversion formula hurts more than it helps on secular bull markets.

---

## Proposed Next Experiments

### Hybrid Config — "5+10 Combined"
Merge config 5's conservative thresholds with config 10's fast indicators + continuation formula:
```
weights     : indicator=0.35, trend=0.50, pattern=0.15
thresholds  : buy=0.25, sell=-0.25, short=-0.50   (from config 5)
window_bars : 20
indicators  : RSI=7, ROC=5, MACD 8/20/6, scale=20  (from config 10)
trend       : continuation formula (position - 0.5) * 2.0  (from config 10)
```
**Hypothesis**: High win rate (57%+) × low trade count (target <500) = first potentially profitable config.

### Confidence Gate Diagnostic
Re-run config 10 with `--confidence-gate 0.4` to discard low-conviction signals.
**Hypothesis**: Drops trade count by ~40% with minimal win-rate penalty.

### Weekly Cadence Diagnostic
Re-run config 5 with `--cadence W` (weekly bars) to remove noise.
**Hypothesis**: Reduces trades from ~400 to ~80; costs no longer dominate.

---

## Prompt Tuning Results

| Agent | Version | Temp | Symbol | Period | Win% | Excess return | Beats quant? |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
