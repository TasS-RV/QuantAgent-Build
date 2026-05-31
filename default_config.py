DEFAULT_CONFIG = {
    "agent_llm_model": "gemini-2.5-flash-lite",
    "graph_llm_model": "gemini-2.5-flash",
    "agent_llm_provider": "google",
    "graph_llm_provider": "google",
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,
    "api_key": "sk-",  # OpenAI API key
    "anthropic_api_key": "sk-",  # Anthropic API key (optional, can also use ANTHROPIC_API_KEY env var)
    "qwen_api_key": "sk-",  # Qwen API key (optional, can also use DASHSCOPE_API_KEY env var)
    "minimax_api_key": "",  # MiniMax API key (optional, can also use MINIMAX_API_KEY env var)
    "minimax_cn_api_key": "",  # MiniMax CN API key (optional, can also use MINIMAX_CN_API_KEY or MINIMAX_API_KEY env var)
    "google_api_key": "",  # Google Gemini API key (optional; auto-loaded from ../Gemini_API.txt)
}
