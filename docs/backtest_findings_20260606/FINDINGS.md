# LLM Backtest Findings — 2026-06-06

**Pipeline:** LangGraph quant-decision (Indicator + Pattern-vision + Trend-vision → weighted quant decision)
**Model:** `gemini-2.5-flash-lite`, temperature 0 · **Universe:** GOOGL, XOM, JNJ
**Window:** 60-bar rolling · **Cadence:** weekly (W-FRI) · **Period:** 2025-06-01 → 2026-06-06 (~1yr, 50 decisions/symbol)
**Overlays:** 200dma trend filter, ATR(1.5×) stops, costs (2bps comm + 5bps slip/side)
**Cost:** **$5.58** / $10 cap · 901 LLM calls · 4.38M tokens · run via 6 parallel workers (~6× speedup)

## Results
| Symbol | Agent | Buy&Hold | Excess | Sharpe | Win% | MaxDD | Trades | Beats 200dma? |
|--------|------:|---------:|-------:|-------:|-----:|------:|-------:|:-------------:|
| GOOGL  | +5.4% | +112.9% | **−107.5%** | 0.64 | 64% | 11.6% | 22 (22L/0S) | ❌ |
| XOM    | +4.1% | +48.4%  | **−44.3%**  | 0.73 | 67% | 4.2%  | 18 (17L/1S) | ❌ |
| JNJ    | −1.5% | +53.9%  | **−55.4%**  | −0.12| 52% | 10.5% | 21 (20L/1S) | ❌ |

Baselines (total return): always-long beat the agent on all 3; 200dma trend-follower beat it on all 3.
**Agent beats the dumb 200dma baseline on 0/3 symbols.**

## Key findings
1. **Severe under-capture of trend.** Win rates are fine (52–67%) but returns are tiny (+5%, +4%, −1.5%) while the names ran +49–113%. The agent is in cash most of the time and harvests small moves — the 200dma filter + ATR stops repeatedly cut it out of the big up-legs.
2. **Almost no shorts fired** (0–1 per name) — the trend filter correctly blocked counter-trend shorts in a bull regime, but the upside wasn't captured either.
3. **Doesn't clear the roadmap gate.** Per `BACKTEST_IMPROVEMENT_ROADMAP.md`, the rule is "don't add complexity until it beats a 200dma trend follower." It currently loses to that baseline on every name — so the next work is signal quality / exposure, not more features.
4. **Cost is non-trivial.** $5.58 for 150 weekly decisions ≈ $0.037/decision on flash-lite. At a daily cadence or larger universe this dominates P&L — the cost stop-loss + cost-in-P&L reporting matters.

## Suggested next steps
- **Raise exposure / let winners run** — the trailing-stop exit in `strategy_lab.py` is the right direction; the LLM pipeline's ATR stop is too tight relative to the trend.
- **Re-weight toward trend-continuation** (the trend signal is currently mean-reverting within the channel) and A/B via the `prompt_tuning.py` harness (momentum vs mean-reversion personas).
- **Gate on confidence + regime** before spending tokens; skip low-conviction weeks entirely (cheaper + fewer weak trades).

_All charts/CSVs in this folder. Generated deterministically from the saved run; reproducible with the parallel driver._
