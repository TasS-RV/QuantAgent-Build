import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass


DEFAULT_CONFIG = {
    "llm_provider": "openai",
    "agent_llm_model": os.environ.get("AGENT_LLM_MODEL", "gpt-4o-mini"),
    "vision_llm_model": os.environ.get("VISION_LLM_MODEL", "gpt-4o-mini"),
    "decision_llm_model": os.environ.get("DECISION_LLM_MODEL", "gpt-4o-mini"),
    "temperature": 0.1,
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
}
