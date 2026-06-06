# Token Usage Log — Gemini Provider

Tracks output token consumption per run, keyed by date, model, and portfolio configuration.

---

## Run: 2026-06-05 — Gemini 3.1 Flash Lite

### Token Usage

| Model                     | Output Tokens |
|---------------------------|---------------|
| **Gemini 3.1 Flash Lite** | **101,730**   |

> Source: Google AI Studio usage dashboard, Jun 5 2026 spike visible in output tokens chart.

### Ticker Configuration

| Ticker | Lookback (days) | Timeframe | Entry Price | Candles Loaded |
|--------|-----------------|-----------|-------------|----------------|
| AAPL   | 120             | 1d        | 189.50      | 83             |

**Pipeline mode:** LLM agents (`--provider google --llm`)  
**Agents per ticker:** indicator agent × trend agent × pattern agent × quant decision node

### Notes

- 101.73K output tokens for a **single ticker** (AAPL, 83 candles, 1d) is the per-ticker baseline cost.
- Full 4-ticker portfolio (AAPL + TSLA + NVDA + BTC-USD) would be approximately ~400K output tokens.
- Each agent (indicator, trend, pattern) runs its own LLM inference loop; output token count scales with number of tickers and agent iterations.
- To reduce token usage: decrease `lookback_days`, reduce tickers, or switch to `--no-llm` mode (zero tokens).

---

*Add new entries below as runs are completed.*
