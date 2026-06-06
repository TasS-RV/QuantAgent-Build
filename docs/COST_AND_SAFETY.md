# Cost Stop-Loss & Trading Safety

Two independent safety systems guard this project. Both fail safe (engaged by
default) and are verified by unit + end-to-end tests.

---

## 1. API cost stop-loss (`cost_guard.py`)

A hard ceiling on LLM spend that **aborts before** a call that would breach the
cap — a runaway backtest or scheduler loop can never silently burn credits.

### How it works
- `preflight(model)` runs *before* each LLM request; raises `CostLimitExceeded`
  if the kill switch is on, a cap is already hit, or the next call is estimated
  to breach the dollar cap. **Nothing is spent.**
- `record(in, out, model)` runs *after* with the real token usage; trips if a cap
  is breached.
- Provider-agnostic: a LangChain callback is auto-attached to every LLM in
  `trading_graph._create_llm`; the OpenAI Agents SDK path uses `_guarded_run`;
  the direct Gemini path uses `record_gemini_response`.

> No LLM API returns a dollar cost — they return token usage
> (OpenAI `usage`, Anthropic `usage`, Gemini `usage_metadata`). The guard prices
> tokens locally, optionally via the maintained offline [`tokencost`](https://github.com/AgentOps-AI/tokencost)
> map when installed, else a built-in table. This mirrors how litellm/langfuse work.

### Hard fail-safe defaults (apply even if unconfigured)
| Cap | Default | Env var |
|---|---|---|
| Dollars | `$5.00` | `QUANT_MAX_USD` |
| Tokens | `2,000,000` | `QUANT_MAX_TOKENS` |
| Calls | `400` | `QUANT_MAX_CALLS` |
| Kill switch | off | `QUANT_KILL_SWITCH=1` |

CLI: `backtest.py` / `master_portfolio.py` accept `--max-usd / --max-tokens /
--max-calls`. Every run prints the armed banner and a final cost report
(`cost_report.json`), including **cost drag on P&L** — so token spend is part of
the return, not a hidden tax.

```bash
python backtest.py --llm --provider google --max-usd 1.0   # cap this run at $1
QUANT_KILL_SWITCH=1 python master_portfolio.py --llm        # block all LLM calls
```

---

## 2. Trading safety rails (`safety.py`)

Decisions are LLM/quant-generated and **advisory**. Live order placement is
impossible unless *all three* hold:

1. `PAPER_TRADING` is off  (`QUANT_PAPER_TRADING=0`)
2. live trading is explicitly armed (`QUANT_LIVE_TRADING=1`)
3. the trading kill switch is off (`QUANT_TRADING_KILL_SWITCH` unset)

Any order path must call `require_live_trading()`, which raises
`LiveTradingBlocked` in the default configuration. `Trading212Client.place_order`
is gated by it (the client only *reads* portfolios today).

```python
from safety import require_live_trading, safe_to_live
if safe_to_live():           # False by default
    require_live_trading()   # raises unless fully armed
```
