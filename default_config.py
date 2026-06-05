import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass


# Merged configuration for both pipelines that now live in this repo:
#   * LangGraph + Gemini pipeline   -> trading_graph.py, indicator/trend/pattern agents,
#                                      master_portfolio.py, decision_agent_quant.py
#   * OpenAI Agents SDK backtest    -> quant_agents.py, backtest.py, run_single.py
#
# NOTE on `agent_llm_model`: the LangGraph pipeline treats it as a Gemini model and routes
# via `agent_llm_provider`. The OpenAI Agents SDK backtest reads the SAME key as an OpenAI
# model name. They cannot share one default, so the fork's primary (Gemini) wins by default.
# To run the pandas-ta / OpenAI Agents SDK backtest, set `llm_provider="openai"` and override
# `AGENT_LLM_MODEL` (e.g. `gpt-4o-mini`) via env var or a config dict. See README → Backtesting.
DEFAULT_CONFIG = {
<<<<<<< HEAD
    "agent_llm_model": "gpt-4o-mini",
    "graph_llm_model": "gpt-4o",
    "agent_llm_provider": "openai",  # "openai", "anthropic", "qwen", "minimax", or "minimax_cn"
    "graph_llm_provider": "openai",  # "openai", "anthropic", "qwen", "minimax", or "minimax_cn"
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,
    "api_key": "sk-",  # OpenAI API key
=======
    # --- Primary agent/graph models (LangGraph + Gemini pipeline) ---
    "agent_llm_model": os.environ.get("AGENT_LLM_MODEL", "gemini-2.5-flash-lite"),
    "graph_llm_model": os.environ.get("GRAPH_LLM_MODEL", "gemini-2.5-flash"),
    "agent_llm_provider": "google",
    "graph_llm_provider": "google",
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,

    # --- OpenAI Agents SDK backtest stack (quant_agents.py / backtest.py) ---
    "llm_provider": os.environ.get("LLM_PROVIDER", "openai"),
    "vision_llm_model": os.environ.get("VISION_LLM_MODEL", "gpt-4o-mini"),
    "decision_llm_model": os.environ.get("DECISION_LLM_MODEL", "gpt-4o-mini"),
    "temperature": 0.1,

    # --- Provider API keys ---
    "api_key": "sk-",  # OpenAI API key
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
    "anthropic_api_key": "sk-",  # Anthropic API key (optional, can also use ANTHROPIC_API_KEY env var)
    "qwen_api_key": "sk-",  # Qwen API key (optional, can also use DASHSCOPE_API_KEY env var)
    "minimax_api_key": "",  # MiniMax API key (optional, can also use MINIMAX_API_KEY env var)
    "minimax_cn_api_key": "",  # MiniMax CN API key (optional, can also use MINIMAX_CN_API_KEY or MINIMAX_API_KEY env var)
<<<<<<< HEAD
=======
    "google_api_key": "",  # Google Gemini API key (optional; auto-loaded from ../Gemini_API.txt)
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
}
