# Backtest Improvement Roadmap

Summary of recommended fixes to stop weak trades, add realistic friction, and validate whether the LLM pipeline adds value over simple baselines.

**Core thesis:** Three Tier 1 fixes alone should flip MSFT-style results toward cleaner AAPL/NVDA outcomes. Do not add complexity until the backtest beats a dumb 200-day SMA trend follower.

---

## Tier 1 — Stop the bleeding (this week)

Priority fixes with the highest expected impact.

### 1. Add HOLD

**Problem:** Forcing LONG or SHORT every week is the biggest unforced error.

**Change (`quant_agents.py`):**
- Extend `TradeDecision.decision` to `Literal["LONG", "SHORT", "HOLD"]`
- Remove *"HOLD is NOT permitted"* from the decision agent instructions

**Backtest:** Treat HOLD as cash for that period.

**Expected impact:** ~30% fewer weak trades; win rate +2–4 pp.

---

### 2. Trend filter — never fight the 200-day SMA

**Problem:** SHORT book lost heavily on AAPL (e.g. ~20.8%) from shorting into uptrends.

**Rule (apply in `backtest.py` after agent decision):**
```python
sma200 = df["Close"].iloc[max(0, end_idx-199):end_idx+1].mean()
price = df["Close"].iloc[end_idx]
if direction == -1 and price > sma200:
    direction = 0  # force HOLD
# Symmetrically: don't LONG below 200dma in clear downtrends
```

**Expected impact:** Saves most AAPL/NVDA-style losses from counter-trend shorts.

---

### 3. Confidence as a gate, not position size

**Problem:** Confidence does not correlate with accuracy (~0.78 when right vs ~0.77 when wrong).

**Rule:**
```python
if conf < 0.75:
    direction = 0  # only trade high-confidence calls
```

**Validation:** If Sharpe improves, confidence is at least monotonic. If not, remove confidence from the output schema.

---

## Tier 2 — Realistic costs and exits

### 4. Subtract trading costs

Current backtest assumes free trading. Add at PnL calculation:

| Parameter | Value | Notes |
|---|---|---|
| `COMMISSION` | 0.0002 | ~2 bps/side (retail) |
| `SLIPPAGE` | 0.0005 | ~5 bps/side, conservative |
| `borrow_daily` | 0.001 / 252 | Short borrow ~10 bps/yr |

Round-trip costs + borrow over hold period. Stops false-positive “profitable” results.

---

### 5. Use `risk_reward_ratio` for stops

**Problem:** Decision agent outputs `risk_reward_ratio`; backtest ignores it.

**Approach:** ATR-based stop (e.g. 1.5× ATR), target = `risk_reward_ratio × stop_distance`. Scan bars from entry to next rebalance; exit on first stop or target hit (use High/Low for intra-bar fills).

**Expected impact:** Caps tail losses (e.g. -10% NVDA SHORT → ~-2% stop-out).

---

## Tier 3 — Better signal (after Tier 1+2)

Only invest here once Tier 1+2 show a repeatable edge.

| # | Idea | Rationale |
|---|---|---|
| 6 | **Richer indicator context** | Add SMA50/200 position, ATR, sector/SPX strength, distance from 52w high as JSON scalars — cheap, dense signal |
| 7 | **Drop vision agents** | Pattern + trend vision calls are slow, expensive, weak for time series; replace with numerical features (slope, % above 50dma, max drawdown) |
| 8 | **Self-consistency voting** | 3 decision samples; 3/3 → full trade, 2/1 → half size, split → HOLD |
| 9 | **Regime pre-classifier** | Upstream agent: trending_up / trending_down / range / high_vol → route to different decision prompts |

**Start with #7** (drop vision, add numerical features) for A/B testing in Week 2.

---

## Tier 4 — Test discipline

### 10. Compare against trivial baselines

Do not benchmark only vs buy-and-hold. Required baselines:

1. Buy & hold *(existing)*
2. Always LONG every rebalance
3. **200dma trend follower** — LONG if price > 200dma, else cash
4. Random LONG/SHORT (50/50)

**Pass/fail:** If the agent does not beat #3, the LLM is destroying value.

---

### 11. Out-of-sample window

Hold out **2021–2022** as true OOS. Prompt tweaks driven by in-sample backtests = leakage. Large OOS divergence = overfit to recent regime.

---

### 12. Expand universe

Top-10 mega-caps (2025–26) favor momentum; B&H is hard to beat. Test:

- S&P mid-caps
- Crypto (BTC, ETH, SOL)
- Sector ETFs (XLF, XLE, XLK)

---

## Tier 5 — If still losing: pivot use case

LLMs are **weak** at chart pattern recognition vs dedicated quant models (HMMs, gradient boosting, logistic regression on indicators) at far lower cost.

LLMs are **strong** at:

- Earnings calls, 10-Qs, news flow
- Sentiment classification
- Cross-asset narrative reasoning
- Risk overlays on positions + events

**Better product shape:** LLM flags stocks with material catalysts from text; quant model decides direction. Use each tool for what it does best.

---

## Practical sequencing

| When | Action |
|---|---|
| **Day 1** | Tier 1: HOLD + trend filter + confidence gate → re-run backtest |
| **Day 2** | Tier 2: costs + stops → verify Day 1 survives friction |
| **Day 3** | Tier 4.10: vs 200dma baseline — if fail, pause LLM investment |
| **Week 2** | One Tier 3 A/B (prefer #7: drop vision) |

**Rule:** No new agents, indicators, or prompts until the backtest cleanly beats the 200dma baseline. Complexity on a non-edge = a more confident way to lose money.

---

## Files most affected

| File | Changes |
|---|---|
| `quant_agents.py` | HOLD in schema; decision prompt; optional Tier 3 context |
| `backtest.py` | Trend filter, confidence gate, costs, ATR stops, baselines |
| `visualize.py` | May need updates if output schema or trade logic changes |
