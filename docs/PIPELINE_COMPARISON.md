# QuantAgent Pipeline Comparison

This repo contains **two separate implementations** of the same multi-agent trading workflow. They share similar roles (indicator → pattern → trend → decision) but use different code paths.

## Quick reference

| | Original pipeline | New pipeline (backtest) |
|---|---|---|
| **Entry point** | `TradingGraph` in `trading_graph.py` | `run_pipeline()` / `run_pipeline_async()` in `quant_agents.py` |
| **Framework** | LangChain + LangGraph | OpenAI Agents SDK |
| **Used by** | `web_interface.py`, `test_trend_agent.py`, README examples | `backtest.py`, `run_single.py` |
| **Agent modules** | `indicator_agent.py`, `pattern_agent.py`, `trend_agent.py`, `decision_agent.py` | Inline factories in `quant_agents.py` (`make_*_agent`) |
| **Orchestration** | `graph_setup.py` (state graph) | `asyncio.gather` + sequential decision step |

## Original pipeline (LangGraph)

```
trading_graph.py
  └── graph_setup.py
        ├── indicator_agent.py   → create_indicator_agent()
        ├── pattern_agent.py     → create_pattern_agent()
        ├── trend_agent.py       → create_trend_agent()
        └── decision_agent.py    → create_final_trade_decider()
```

- LangGraph state machine wires agents together.
- Uses `TechnicalTools`, `agent_state.py`, and LangChain LLM clients.
- This is the path documented in the main README.

## New pipeline (OpenAI Agents SDK)

```
backtest.py / run_single.py
  └── quant_agents.py
        ├── make_indicator_agent()
        ├── make_pattern_agent()
        ├── make_trend_agent()
        └── make_decision_agent()
```

- Agents are defined inline; does **not** import the original `*_agent.py` files.
- Three analysts run in parallel; decision agent runs after.
- Shared utilities only: `indicators.py`, `charts.py`.

## What this means in practice

- **`backtest.py` results** reflect the **new** `quant_agents.py` pipeline, not the original LangGraph agents.
- **`web_interface.py`** still uses the **original** LangGraph pipeline.
- Behavior may differ between the two (prompts, models, tool usage, parsing).

## Note

`IMPROVEMENTS.md` lists the LangChain/LangGraph stack as legacy code intended for removal once the new pipeline is fully adopted.
