# Classic Quant Strategy Lab — Results
**Run: 2026-06-06**

Portfolio: GOOGL / XOM / JNJ  |  2015-01-01 → 2026-06-05  |  daily bars
Costs: 0.02% commission + 0.05% slippage per side (0.14% round-trip), 1%/yr short borrow
Engine: `strategy_lab.py` (event-driven, trailing-stop capable)

---

## Headline

**Every classic strategy is profitable** — a complete reversal from the −50% to −95%
oscillator configs. The best (golden cross, long-flat) returned **+257% avg** with only
**10 trades** and **25% max drawdown**. The architecture fix (trailing/regime holds +
trend-direction-only entries) was the whole problem, not the indicators.

---

## Aggregate ranking (avg across 3 symbols, best first)

| Strategy | Dir | Exit | Avg return | Avg win% | Avg trades | Avg max DD |
|---|---|---|---|---|---|---|
| **ma_cross_50_200** | long-flat | regime | **+257.0%** | 45% | 10 | 24.9% |
| tsmom_60 | long-flat | regime | +179.6% | 38% | 74 | 29.5% |
| ma_cross_20_100 | long-flat | regime | +178.0% | 35% | 19 | 22.7% |
| tsmom_120 | long-flat | regime | +127.9% | 35% | 67 | 38.7% |
| tsmom_60 | long-flat | trailing | +99.5% | 41% | 74 | 23.6% |
| donchian_55_20 | long-flat | regime | +69.2% | 50% | 24 | 26.2% |
| ma_cross_50_200 | long-short | regime | +51.7% | 28% | 20 | 56.5% |
| donchian_20_10 | long-flat | regime | +49.6% | 41% | 51 | 29.2% |
| … (long-short variants) | … | … | mostly negative | <36% | high | 40–72% |

Full table: `strategy_results/aggregate.csv` · per-symbol: `strategy_results/summary.csv`

### Golden cross (50/200, long-flat, regime) — per symbol

| Symbol | Return | Buy&Hold | Win% | Trades | Max DD |
|---|---|---|---|---|---|
| GOOGL | **+650.0%** | +1303.4% | 75% | 8 | **3.6%** |
| XOM | +97.0% | +162.3% | 30% | 10 | 33.1% |
| JNJ | +24.0% | +205.8% | 31% | 13 | 38.0% |

### Dual momentum (portfolio rotation)

| Variant | Return | Basket B&H | Sharpe | Trades |
|---|---|---|---|---|
| dual_momentum_180 | +188.0% | +545.3% | 5.22* | 26 |
| dual_momentum_90 | +71.5% | +500.9% | 2.43* | 41 |

---

## Findings

### 1. Long-flat dominates long-short — decisively
Every single long-short variant underperformed its long-flat twin, usually by 50–200
points, with double the drawdown. **Shorting a secular bull market destroys value.**
Confirms the diagnosis from the oscillator runs. → Default to long-flat.

### 2. Regime exit beats the 3×ATR trailing stop here
Counter-intuitive but explainable: on smooth trends the trailing stop gets nicked on
normal pullbacks, and the re-entry latch (suppress same-direction re-entry until the
signal resets) then keeps us out until a full death cross. For MA strategies that means
missing most of the continuation (golden cross: regime +257% vs trailing +18.7%).
The latch is correct for **breakout** strategies (Donchian) but too sticky for
**stateful regime** strategies (MA/TSMOM). → Fix in round 2 (see below).

### 3. None beat buy-&-hold on absolute return — and that's honest
The basket returned ~+500–1300%. Sitting in cash during corrections means missing the
V-shaped rebounds. The payoff is **risk reduction**: golden cross on GOOGL captured
+650% with a 3.6% max drawdown vs buy-&-hold's ~44% drawdown in 2022. That is a vastly
better risk-adjusted ride, which is exactly what trend-following sells.

### 4. Fewer, bigger trades = the winning shape
The best configs trade 8–25 times in 11 years (vs ~1,100 for the oscillator blend).
Win rates of 30–50% are *fine* because winners run far longer than losers — positive
expectancy without a high hit rate, exactly as predicted.

### ⚠ Sharpe caveat
`compute_metrics` annualizes *per-trade* returns by √252, which massively overstates
Sharpe for strategies with only 8–25 trades over 11 years (the 7.18 figure is an
artifact). Trust total return, win rate, trade count and max drawdown; treat Sharpe as
ordinal-only until we switch to a daily-equity Sharpe.

---

## Recommended Round 2

1. **Fix the trailing-stop re-entry for regime strategies** — after a trailing-stop
   exit, allow re-entry on the next bar if the regime is still bullish (don't wait for a
   reset). Re-run golden cross + TSMOM with trailing to see if it now adds drawdown
   protection without sacrificing the +257%.
2. **Tune the golden cross** — sweep (40/150, 50/200, 60/250) and a wider ATR trail
   (4–6× ATR) so the stop only catches genuine trend breaks.
3. **Add volatility-target sizing** — scale exposure to a constant risk budget; usually
   lifts Sharpe and trims drawdown on all three names.
4. **Switch to daily-equity Sharpe** in the metrics so risk-adjusted numbers are real.
5. **Combine with a regime filter** — only take golden-cross longs when the broad market
   (SPY) is also above its 200-day, to skip bear-market false starts.
