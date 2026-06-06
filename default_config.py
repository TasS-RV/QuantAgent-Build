import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass


# Configuration for the LangGraph pipeline:
#   trading_graph.py, indicator/trend/pattern agents,
#   master_portfolio.py, quant_pipeline/
#
# API keys are loaded from .env.keys at repo root (via env_keys.py).
# Values here are fallbacks / defaults only.
DEFAULT_CONFIG = {
    # --- LangGraph agent/graph models ---
    "agent_llm_model":        os.environ.get("AGENT_LLM_MODEL", "gemini-2.5-flash-lite"),
    "graph_llm_model":        os.environ.get("GRAPH_LLM_MODEL", "gemini-2.5-flash"),
    "agent_llm_provider":     "google",
    "graph_llm_provider":     "google",
    "agent_llm_temperature":  0.1,
    "graph_llm_temperature":  0.1,
    "temperature":            0.1,

    # --- Provider API keys (prefer .env.keys over hardcoding here) ---
    "api_key":             os.environ.get("OPENAI_API_KEY", ""),
    "openai_api_key":      os.environ.get("OPENAI_API_KEY", ""),
    "anthropic_api_key":   os.environ.get("ANTHROPIC_API_KEY", ""),
    "qwen_api_key":        os.environ.get("DASHSCOPE_API_KEY", ""),
    "minimax_api_key":     os.environ.get("MINIMAX_API_KEY", ""),
    "minimax_cn_api_key":  os.environ.get("MINIMAX_CN_API_KEY", os.environ.get("MINIMAX_API_KEY", "")),
    "google_api_key":      os.environ.get("GOOGLE_API_KEY", ""),
    "featherless_api_key": os.environ.get("FEATHERLESS_API_KEY", ""),
}
