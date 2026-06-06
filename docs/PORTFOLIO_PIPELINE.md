# QuantAgent — Portfolio Pipeline
**Last updated: 2026-06-06**

---

## Overview

`master_portfolio.py` runs a full analysis-and-alert cycle across a portfolio of tickers.  
Each run: fetch data → analyse → decide → enrich → (optionally) push to Telegram.

```
Portfolio Source
     │
     ▼
load_portfolio()
     │  USE_T212=True  →  Trading 212 API  (live or practice)
     │  USE_T212=False →  demo_portfolio.json  (static file)
     │
     ▼
For each ticker
     │
     ├─ fetch_kline_data()          yfinance OHLCV
     │
     ├─ LangGraph pipeline
     │       Indicator node   (RSI / MACD / Stoch / WillR / ROC)
     │       Trend node        (linear-regression channel)
     │       Pattern node      (candlestick vision — LLM mode only)
     │       Quant Decision    (weighted signal → BUY / HOLD / SELL / SHORT)
     │
     ├─ build_trade_card()          adds stop-limit, TP%, SL%, quantity, allocation
     │
     └─ collect → all_results[]
           │
           ├─ print_summary_table()
           ├─ (optional) JSON export   --output
           └─ (optional) Telegram      --telegram
                   format_portfolio_telegram()
                   → data_exchange/Tele_bot.py → sendMessage
```

---

## Portfolio source

| Config var | Default | CLI flag |
|---|---|---|
| `USE_T212` | `False` | `--t212` / `--no-t212` |
| `T212_DEMO` | `False` | `--t212-demo` |
| `PORTFOLIO_FILE` | `demo_portfolio.json` | — |

**demo_portfolio.json** — static file at repo root. Edit freely.  
Each entry: `entry_price`, `quantity`, `lookback_days`, `timeframe`, `allocation_pct`.

**Trading 212 live pull** — requires `T212_key` + `T212_secret` in `.env.keys`.  
`allocation_pct` is computed automatically from live market values.

---

## Decision engine

Signals are combined with configurable weights (edit `DECISION_CONFIG`):

| Source | Default weight |
|---|---|
| Indicator (RSI / MACD / Stoch / WillR / ROC) | 40 % |
| Trend (linear-regression channel) | 40 % |
| Pattern (LLM vision — only in `--llm` mode) | 20 % |

Combined signal ∈ [−1, +1]. Thresholds: `≥ +0.15` → BUY · `≤ −0.15` → SELL · `≤ −0.35` → SHORT.

Stop-loss = current price ± ATR × `atr_multiplier_sl` (default 2.0).  
Target = SL distance × `risk_reward_target` (default 2.0 → 1:2 R:R).

---

## Trade card fields

After the pipeline, `build_trade_card()` enriches each decision with:

| Field | Description |
|---|---|
| `stop_limit` | SL × 0.995 (BUY) or SL × 1.005 (SELL/SHORT) |
| `tp_pct` | % distance from current price to target |
| `sl_pct` | % distance from current price to stop-loss |
| `quantity` | shares held (from portfolio source) |
| `allocation_pct` | % of total portfolio value |
| `is_etf` | flagged from known ETF ticker list |

---

## LLM modes

| Mode | Flag | Pattern agent | API calls |
|---|---|---|---|
| Full quant (default) | `--no-llm` | disabled — pattern signal = 0 | none |
| LLM mode | `--llm` | enabled | 1 per ticker |

Provider is set with `--provider` (default `featherless`).  
Keys are loaded from `.env.keys` at repo root.

---

## Telegram

Set `TELEGRAM_ENABLED = True` in the config block, or pass `--telegram` at runtime.  
Keys: `telegram_keys.json` in repo root, or `.env.keys` (`telegram_bot_token` / `telegram_chat_id`).

Each Telegram message contains one block per ticker: action, signal score, price, allocation, PnL, target (+%), stop-loss (+%), and stop-limit.

---

## Quick-start examples

```bash
# Demo portfolio, no LLM, no Telegram
python master_portfolio.py

# Live T212 positions, no LLM, no Telegram
python master_portfolio.py --t212

# Demo portfolio + Telegram alert
python master_portfolio.py --telegram

# Live T212 + Gemini LLM + Telegram
python master_portfolio.py --t212 --llm --provider google --telegram

# Practice T212 account + save results to file
python master_portfolio.py --t212 --t212-demo --output results.json
```

---

## Key files

| File | Purpose |
|---|---|
| `master_portfolio.py` | Entry point — pipeline orchestrator |
| `demo_portfolio.json` | Static portfolio (tickers, entry prices, allocations) |
| `data_exchange/trading212_client.py` | T212 REST API client |
| `data_exchange/Tele_bot.py` | Telegram sender |
| `data_exchange/portfolio_scheduler.py` | Scheduled / recurring runs |
| `data_exchange/t212_visualize.py` | Standalone portfolio dashboard (dark-theme chart) |
| `.env.keys` | API keys — not committed to VCS |
| `env_keys.py` | Central key loader (`get_key(*names)`) |
