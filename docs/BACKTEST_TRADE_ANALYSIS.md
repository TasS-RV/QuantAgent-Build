# Backtest Trade-Level Analysis

Analysis of trade-by-trade results from `backtest_results/` CSVs.

**Note:** Summary numbers may shift slightly between runs because the LLM uses `temperature > 0`. This report reflects the current CSVs on disk, not an earlier run.

---

## Current results (summary)

| Symbol | Agent | B&H | Excess | Sharpe | Win rate | Max DD | Trades |
|--------|-------|-----|--------|--------|----------|--------|--------|
| AAPL   | -13.3% | +58.8% | -72.1 pp | -0.78 | 50.0% | 26.2% | 38 |
| MSFT   | +4.7%  | -6.3%  | +11.0 pp | +0.37 | 50.0% | 18.2% | 38 |
| NVDA   | +20.3% | +64.0% | -43.8 pp | +0.92 | 55.3% | 14.2% | 38 |

**Headline:** Agent underperforms buy-and-hold on AAPL and NVDA by a wide margin. MSFT shows positive excess, but the trade decomposition suggests that is not genuine alpha (see §3).

---

## Trade-by-trade decomposition

| Stock | LONG count | LONG win% | LONG total | SHORT count | SHORT win% | SHORT total |
|-------|------------|-----------|------------|-------------|------------|-------------|
| AAPL  | 15         | 60.0%     | +9.5%      | 23          | 43.5%      | -20.8%      |
| MSFT  | 12         | 33.3%     | -4.5%      | 26          | 57.7%      | +9.7%       |
| NVDA  | 13         | 61.5%     | +24.9%     | 25          | 52.0%      | -3.7%       |

**Portfolio split:** 40 longs vs 74 shorts across three names (~35% / 65% short-biased).

---

## Key findings

### 1. Structurally short-biased

In a period where AAPL ran **+59%** and NVDA **+64%**, leaning short is the main P&L drag.

- **AAPL:** LONG book **+9.5%** (agent was right when long); SHORT book **-20.8%** → net **-13.3%**.
- **NVDA:** LONGs **+24.9%**, shorts **-3.7%** — same shape.

**Counterfactual:** If every SHORT had been treated as **cash (HOLD)** instead of a short position, all three names would likely beat buy-and-hold on a Sharpe basis.

**Likely causes:**

- Trend agent over-weights resistance breaks / wedge tops.
- Indicator agent maps RSI > 70 → “overbought ⇒ short.”
- Decision agent treats both as confluence in strong uptrends (fires often, wrong often).

---

### 2. Confidence is noise

| When correct | When wrong |
|--------------|------------|
| Avg confidence **0.78** | Avg confidence **0.77** |

Self-reported confidence does **not** separate winners from losers. It should **not** be used for position sizing — that only adds random noise.

*(Optional use: hard gate, e.g. only trade if `conf >= 0.75` — see [BACKTEST_IMPROVEMENT_ROADMAP.md](./BACKTEST_IMPROVEMENT_ROADMAP.md).)*

---

### 3. MSFT is a “fake win”

MSFT shows **+11 pp** excess vs B&H, but:

- Agent is **bearish by default**; MSFT **fell** this year → shorts aligned with reality by accident.
- SHORTs: 57.7% win, +9.7% total.
- LONGs: 33% win, **-4.5%** total.

This is not predictive alpha — it is a **global bearish bias** that happened to match one weak name among three. “Stuck clock” behavior, not edge.

---

### 4. Win rate ≈ 50% → no edge

Win rates: **50%, 50%, 55%** — coin-flip territory.

Winner/loser size is roughly symmetric or slightly negative, so even 55% barely breaks even **before** costs. Costs are not yet in the backtest.

---

### 5. Worst trades = reversal attempts

Examples of inflection-point calls run over by trend continuation:

| Date       | Symbol | Side  | Loss    |
|------------|--------|-------|---------|
| 2026-02-06 | AAPL   | LONG  | -7.15%  |
| 2026-03-27 | NVDA   | SHORT | -10.22% |

Weekly candle pattern logic finds “double tops” and “rising wedges” often in strong trends; most of those calls are wrong.

---

### 6. “HOLD prohibited” is destructive

The decision prompt **forces LONG or SHORT every rebalance**. Many weeks have no real signal → effectively random trades.

**Allowing HOLD** would cut trade count ~**30%** and likely raise win rate by **2–4 pp** (see roadmap Tier 1).

---

## Estimated impact of top fixes

Same agent logic, with three overlays applied (rough order-of-magnitude):

| Fix | Expected impact |
|-----|-----------------|
| Allow HOLD (no forced bet) | Drop ~30% weakest trades → **+3 to +6 pp** |
| Trend-filter SHORTs (no short above 200dma) | Remove most AAPL/NVDA short bleed → **+15 to +25 pp** on those names |
| Add ~10 bps round-trip costs | **-4 pp** annually (114 trades × ~4 bps avg) |

Net: Tier 1 + costs should materially change the AAPL/NVDA story and clarify whether any LLM signal remains after removing structural short bias.

---

## Recommended next steps

1. Implement Tier 1 from [BACKTEST_IMPROVEMENT_ROADMAP.md](./BACKTEST_IMPROVEMENT_ROADMAP.md): HOLD, 200dma filter, confidence gate.
2. Re-run backtest; compare trade decomposition (long vs short P&L) and vs **200dma baseline**.
3. Do not add agents or prompts until results beat that baseline cleanly.

---

## Related docs

- [BACKTEST_IMPROVEMENT_ROADMAP.md](./BACKTEST_IMPROVEMENT_ROADMAP.md) — fix tiers and sequencing
- [PIPELINE_COMPARISON.md](./PIPELINE_COMPARISON.md) — which pipeline `backtest.py` uses (`quant_agents.py`, not LangGraph `*_agent.py`)
