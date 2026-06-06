# QuantAgent — Production Readiness Improvements

A pragmatic, ordered list of what to fix/add before shipping this to prod. Grouped by severity. Items are sized roughly: S (hours), M (days), L (week+).

---

## 0. Critical — do before any external traffic

- **Rotate the leaked OpenAI key.** [S] The key currently in `.env` was posted in chat history. Revoke at https://platform.openai.com/api-keys, mint a fresh one, replace in `.env`. Add a pre-commit hook (`detect-secrets` or `gitleaks`) so this never re-happens.
- **No real-money trading from this codebase yet.** [S] Add a hard-coded `PAPER_TRADING=True` flag and a `safe_to_live()` check that refuses to place orders unless explicitly overridden. Decisions are LLM-generated; treat as advisory.
- **Look-ahead bias audit.** [M] The backtest uses `df.iloc[:end_idx+1]` and trades at `end_idx+1`'s open — verify there is no future-bar leakage anywhere (e.g. via adjusted-close vs unadjusted, or via yfinance re-stating splits). Cross-validate against a known broker-grade dataset (Polygon, Tiingo, Norgate) on at least 2 stocks for 1 year.

## 1. Backtest realism — current results are optimistic

- **Transaction costs & slippage.** [S] Today the harness uses raw open-to-open returns. Add: 5–10 bps commission per side + 5–15 bps slippage (worse for SHORTs and less liquid names). For top-10 mega-caps the impact is small but a weekly rebalance × 10 = ~520 trades/year, so 20 bps/round-trip = ~1% annual drag.
- **Short selling cost & borrow availability.** [S] Hard-borrow rates on names like TSLA/NVDA hit 50–300 bps annualized; add a daily borrow fee for the SHORT side.
- **Position sizing.** [M] Currently every trade is "all in." Switch to confidence-weighted sizing (`weight = confidence × max_position`), with portfolio-level risk cap (e.g. max gross 100%, max single name 20%).
- **Stop loss / take profit.** [M] The agent suggests a `risk_reward_ratio` but the backtest ignores it. Use it: enter at next open, place stop at `entry × (1 - ATR×k)`, target at `entry + risk_reward_ratio × stop_distance`. Intra-bar fills via High/Low.
- **No-decision / abstention.** [S] Today the agent is forced to pick LONG/SHORT (HOLD prohibited). Add HOLD as a third option and let it abstain when reports disagree — measure if it helps net-of-cost returns.
- **Walk-forward parameter sanity.** [M] Currently no parameters are fit, but indicator periods (RSI 14, MACD 12/26/9) are global. Confirm those are not implicit overfits — sweep across reasonable ranges on out-of-sample windows.
- **Universe survivorship bias.** [S] "Top 10 USA stocks (early 2026)" is survivor-biased — these are *today's* winners. For honest 5-year results, use the *historical* top-10 as of 2021 (or rolling top-10 by market cap each year).
- **Multiple-bars-ahead leakage in indicator window.** [S] Make sure the indicator agent only sees data through the decision bar's close (it does — but lock it down with a unit test).

## 2. Agent reliability & determinism

- **Output schema validation w/ retry.** [S] `output_type=` on Agents SDK already enforces JSON shape, but add a top-level retry-with-clarification when validation fails (currently any failure surfaces as a 0-direction "skipped" trade).
- **Temperature = 0 everywhere.** [S] Currently 0.1 for analysts; drop to 0 in prod for reproducibility. Keep the decision agent at 0.
- **Seed/replay determinism.** [M] Persist every input prompt + raw model response so any decision can be replayed. Use OpenAI Agents SDK's built-in tracing or pipe to your own SQLite/Postgres.
- **Agent self-consistency.** [M] For high-confidence prod use, sample the decision agent k=3 times and require agreement (or use majority vote). Cuts variance considerably.
- **Cost-tier routing.** [S] Use `gpt-4o-mini` for indicator (text-only, cheap), but consider `gpt-4o` for vision and decision where the marginal accuracy matters. Easy A/B.

## 3. Data layer

- **Replace yfinance.** [M] yfinance scrapes Yahoo's unofficial API — flaky, rate-limited, occasionally returns wrong adjusted prices. Pay for **Polygon** or **Tiingo** (~$50/mo) for production data. Also they expose intraday & corporate-actions cleanly.
- **Corporate actions.** [S] Splits and dividends — yfinance auto-adjusts but check that the adjustment is applied consistently to *all* OHLC columns (it sometimes only adjusts Close). Validate with a known stock split (NVDA 10:1 in June 2024).
- **Caching layer.** [S] Add a local parquet/duckdb cache keyed on (symbol, interval, period). Right now every backtest re-downloads. (And install `_sqlite3` in the deploy image so the `pysqlite3` shim isn't needed.)
- **Higher-frequency support.** [M] If this is "HFT-like" as the prompts claim, 1d bars are way too slow. Add 5m/15m/1h intervals with proper market-hours handling.

## 4. Engineering hygiene

- **Delete dead LangChain code.** [S] `trading_graph.py`, `graph_setup.py`, `graph_util.py`, `indicator_agent.py`, `pattern_agent.py`, `trend_agent.py`, `decision_agent.py`, `agent_state.py`, `Algo_state.py`, `API_client.py`, `static_util.py`, `Indicator_node.py`, `1-algo_core.py` — all replaced by the new pipeline. Remove to avoid drift.
- **`requirements.txt` pinning.** [S] Pin versions: `pandas-ta==0.4.71b0`, `openai-agents>=0.17`, `openai>=2.0`. Add `pysqlite3-binary` for environments missing _sqlite3.
- **Type checking.** [S] Run `mypy --strict` or `pyright`. The Pydantic models help; add type hints to the rest.
- **Tests.** [M] Add: (a) unit tests for `indicators.py` against reference values (you can compare to TA-Lib outputs as oracle for 1 fixed dataset); (b) golden-file test for the agents using a recorded VCR-style fixture; (c) integration test that runs 5 decisions end-to-end against a stub OpenAI server.
- **Pre-commit.** [S] `ruff`, `black`, `gitleaks`, `mypy`. Add to `.github/workflows/ci.yml`.
- **Logging.** [S] Replace `print()` with `logging`. Structured JSON logs for prod, plus per-decision artifacts (input prompt, image hash, raw output) written to S3/disk.

## 5. Cost & latency

- **Concurrency tuning.** [S] Current backtest runs 4 decision points concurrently per stock; bump to 8–16 once Tier 2+ on OpenAI. Stocks themselves run sequentially — parallelize across stocks too.
- **Batch API.** [M] For backtests, OpenAI's Batch API is ~50% cheaper and tolerates 24h latency. Refactor `backtest.py` to a 2-phase: (1) submit all prompts as a batch, (2) collect & evaluate.
- **Prompt caching.** [S] The system prompts for the three analyst agents are identical across all 2,600 calls. Use the OpenAI prompt-cache hint to get the discount.
- **Image size.** [S] Charts are rendered at 12×6 inches @ 120dpi → ~1440×720. Vision tokens scale with image size. Render at 7×4 @ 100 dpi (≈700×400) — still legible, halves vision token cost.

## 6. Observability & ops

- **Tracing.** [S] OpenAI Agents SDK has built-in tracing — enable + export to Langfuse, Helicone, or Phoenix. Lets you see token counts and latency per agent per decision.
- **Cost monitor.** [S] Per-run cost tally written to summary CSV; hard ceiling enforced (e.g. abort if spend > $100).
- **Health checks.** [S] Add a `/health` endpoint that runs a known-good decision against fixed data and verifies the output schema.
- **Alerting.** [S] PagerDuty/Slack alert on: API 5xx > 5%, decision latency p95 > 30s, daily cost > threshold.

## 7. Web UI (`web_interface.py`)

- **It's 46KB of mixed concerns.** [M] Split into proper Flask blueprints (or move to FastAPI + a React/Next.js frontend). Today it instantiates a `TradingGraph` per request, which is wasteful and won't survive any traffic.
- **AuthN/Z.** [M] No auth at all currently. Add OAuth (Google/GitHub) and per-user rate limits before any external exposure.
- **API key handling.** [S] Don't accept the OpenAI key via the web form. Always read from the server's env. Multi-tenant ⇒ each tenant should bring their own key via secure vault.

## 8. Live execution layer (the thing that's not in the repo yet)

- **Broker integration.** [L] Alpaca / IBKR / Tradier — paper account first, real later. Map agent decisions to orders with retries, idempotency keys, and reconciliation.
- **Order lifecycle.** [L] Pre-trade risk checks → submit → monitor → reconcile fills → update position state → log to audit trail. None of this exists in the codebase.
- **Kill switch.** [S] One env var that disables all order submission instantly.
- **Reconciliation.** [M] Daily reconciliation between expected positions and broker positions; alert on any mismatch.

## 9. Research / model improvements

- **Ensemble of model families.** [M] Run the same decision through `gpt-4o-mini`, `claude-sonnet-4-6`, `gemini-2.5-flash` and majority-vote. LLM ensembles consistently beat single models on noisy tasks.
- **RAG for market context.** [M] Index recent earnings call transcripts, 10-Qs, and news for each stock; retrieve relevant chunks per decision. The agents currently see *only* price/indicator data.
- **Regime detection.** [M] A meta-agent that classifies the current market as trending/mean-reverting/high-vol and adjusts the weight given to pattern vs. trend agents.
- **Reinforcement-learning fine-tune.** [L] Once you have N=5k+ decision-outcome pairs, do RLHF or DPO on a small open-source model — could replicate the multi-agent decision in a single fast model call.

---

## TL;DR

If you want to ship in a quarter:

1. Week 1: Rotate key, delete dead code, add tests, transaction costs, paper-trading flag.
2. Week 2: Polygon data, structured logging, tracing, batch API for backtests.
3. Week 3: Stop/target execution, position sizing, regime gate.
4. Week 4: Broker integration on paper account, kill switch, monitoring.
5. Then run paper for 4–8 weeks before any real capital.
