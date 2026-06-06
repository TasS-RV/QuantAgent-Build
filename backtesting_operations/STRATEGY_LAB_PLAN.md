# Classic Quant Strategy Lab — Test Plan
**Created: 2026-06-06**

Portfolio: GOOGL / XOM / JNJ  |  Start: 2015-01-01  |  Daily bars

---

## Why this exists

The 12 oscillator-blend configs all lost 50–95%. Root cause was **not** the signal —
it was the architecture:

1. **Fixed-horizon exit caps winners.** The old engine exits each trade at the next
   signal bar or a fixed ATR target (`target = stop × RR`). That forces the payoff
   ratio toward 1:1, so even a 56% win rate can't overcome costs.
2. **Symmetric mean-reversion shorts a bull market.** The trend agent treats
   "price near channel top" as bearish; on a +1,300% stock that's a permanent
   short bias.

The well-proven strategy families fix both by design: **let winners run via trailing
stops**, and **only take trades in the direction of the established trend**. Their win
rates are often 35–45%, but expectancy is positive because winners are 3–5× losers.

```
Expectancy = Win% × Avg_Win − Loss% × Avg_Loss
```

---

## What's being built

`strategy_lab.py` — a standalone, vectorized, event-driven backtester. It reuses the
existing cost model (`SimConfig`), trade record (`SimTrade`), metrics
(`compute_metrics`) and buy-&-hold baseline from `backtest_engine.py`, so every number
is directly comparable to the 12 prior configs. The new piece is the **position
simulator**, which supports:

- **Trailing-stop exits** (ATR-based) — lets winners run, the key fix
- **Regime exits** — exit only when the entry signal reverses
- **Long-flat vs long-short** direction modes

All strategies decide at bar *i* close, execute at bar *i+1* open (no look-ahead).

---

## Strategies under test

| # | Strategy | Family | Entry | Native exit |
|---|---|---|---|---|
| A | **Donchian breakout** | Trend / breakout (Turtles) | Close > N-day high | Close < M-day low |
| B | **MA crossover** | Trend | Fast MA > Slow MA | Fast MA < Slow MA |
| C | **Time-series momentum** | Trend / CTA | L-day return > 0 | L-day return < 0 |
| D | **Dual momentum** | Cross-sectional rotation | Hold strongest symbol if its abs. momentum > 0 | Rotate / go to cash monthly |

### Parameter variants

- **Donchian**: `(entry=20, exit=10)` fast Turtle, `(entry=55, exit=20)` slow Turtle
- **MA cross**: `(20/100)` medium, `(50/200)` classic golden/death cross
- **TSMOM**: lookback `60` (≈3mo), `120` (≈6mo)
- **Dual momentum**: lookback `90`, `180`; monthly rebalance

### Combination matrix (per-symbol strategies A–C)

```
{strategy variant} × {exit: regime | trailing} × {direction: long-flat | long-short}
```

Trailing stop = 3 × ATR(14) ratchet. Dual momentum (D) is portfolio-level, long-flat,
monthly — run separately.

---

## Cost & friction (held constant, identical to prior 12 configs)

- Commission 0.02% / side, slippage 0.05% / side → 0.14% round-trip
- Short borrow 1%/yr pro-rata
- ATR(14) for stop distance
- Buy-&-hold computed from first decision bar for honest excess-return comparison

---

## Success criteria

A strategy "works" if, **net of costs**, it:
1. Beats the 200-day SMA trend-follower baseline, AND
2. Has positive expectancy (avg trade PnL > 0) despite a possibly sub-50% win rate, AND
3. Produces a materially smaller trade count than the ~1,100/symbol overtrading regime.

Stretch goal: positive absolute return on at least one symbol while keeping max
drawdown below buy-&-hold.

---

## Execution order

1. Run full matrix (A–C × variants × exit × direction) + dual momentum across all 3 symbols.
2. Record into `STRATEGY_LAB_RESULTS.md`.
3. Identify the best family + exit mode.
4. Round 2: tune the winner's lookback / ATR multiple / add volatility-target sizing.
