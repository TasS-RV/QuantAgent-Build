"""
TradingGraph: Orchestrates the multi-agent trading system using LangChain and LangGraph.
Initializes LLMs, toolkits, and agent nodes for indicator, pattern, and trend analysis.
"""

import os
<<<<<<< HEAD
=======
from pathlib import Path
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
from typing import Dict

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_qwq import ChatQwen
<<<<<<< HEAD
=======
from langchain_google_genai import ChatGoogleGenerativeAI
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
from langgraph.prebuilt import ToolNode

from default_config import DEFAULT_CONFIG
from graph_setup import SetGraph
from graph_util import TechnicalTools


<<<<<<< HEAD
SUPPORTED_PROVIDERS = ("openai", "anthropic", "qwen", "minimax", "minimax_cn")
=======
SUPPORTED_PROVIDERS = ("openai", "anthropic", "qwen", "minimax", "minimax_cn", "google")
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
MINIMAX_PROVIDER_CONFIG = {
    "minimax": {
        "label": "MiniMax",
        "config_key": "minimax_api_key",
        "env_keys": ("MINIMAX_API_KEY",),
        "base_url": "https://api.minimax.io/v1",
        "console_url": "https://platform.minimaxi.com/",
    },
    "minimax_cn": {
        "label": "MiniMax CN",
        "config_key": "minimax_cn_api_key",
        "env_keys": ("MINIMAX_CN_API_KEY", "MINIMAX_API_KEY"),
        "base_url": "https://api.minimaxi.com/v1",
        "console_url": "https://platform.minimaxi.com/",
    },
}


class TradingGraph:
    """
    Main orchestrator for the multi-agent trading system.
    Sets up LLMs, toolkits, and agent nodes for indicator, pattern, and trend analysis.
    """

    def __init__(self, config=None):
        # --- Configuration and LLMs ---
        self.config = config if config is not None else DEFAULT_CONFIG.copy()
<<<<<<< HEAD

        # Initialize LLMs with provider support
        self.agent_llm = self._create_llm(
            provider=self.config.get("agent_llm_provider", "openai"),
            model=self.config.get("agent_llm_model", "gpt-4o-mini"),
            temperature=self.config.get("agent_llm_temperature", 0.1),
        )
        self.graph_llm = self._create_llm(
            provider=self.config.get("graph_llm_provider", "openai"),
            model=self.config.get("graph_llm_model", "gpt-4o"),
            temperature=self.config.get("graph_llm_temperature", 0.1),
        )
=======
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
        self.toolkit = TechnicalTools()
        self.agent_llm = None
        self.graph_llm = None
        self.graph_setup = None
        self.graph = None

<<<<<<< HEAD
        # --- Create tool nodes for each agent ---
        # self.tool_nodes = self._set_tool_nodes()
=======
        # Build the LLMs + graph. If no API key is configured yet (common on a
        # fresh start before the user enters one in the web UI), defer instead of
        # crashing — refresh_llms() will build them once a key is available.
        try:
            self._build_llms_and_graph()
        except ValueError as e:
            print(
                f"⚠️  TradingGraph: LLMs not initialised yet ({e}). "
                "Set an API key (env var, config, or the web UI) and the graph "
                "will be built automatically when the provider/key is updated."
            )
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c

    def _build_llms_and_graph(self):
        """(Re)create the LLMs and the LangGraph graph from the current config.

        Raises ValueError if the selected provider has no API key — callers that
        want to tolerate that (e.g. __init__) must catch it.
        """
        self.agent_llm = self._create_llm(
            provider=self.config.get("agent_llm_provider", "openai"),
            model=self.config.get("agent_llm_model", "gpt-4o-mini"),
            temperature=self.config.get("agent_llm_temperature", 0.1),
        )
        self.graph_llm = self._create_llm(
            provider=self.config.get("graph_llm_provider", "openai"),
            model=self.config.get("graph_llm_model", "gpt-4o"),
            temperature=self.config.get("graph_llm_temperature", 0.1),
        )
        self.graph_setup = SetGraph(
            self.agent_llm,
            self.graph_llm,
            self.toolkit,
<<<<<<< HEAD
            # self.tool_nodes,
        )

        # --- The main LangGraph graph object ---
=======
        )
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
        self.graph = self.graph_setup.set_graph()

    def _get_api_key(self, provider: str = "openai") -> str:
        """
        Get API key with proper validation and error handling.
        
        Args:
            provider: The provider name ("openai", "anthropic", "qwen", "minimax", or "minimax_cn")
        
        Returns:
            str: The API key for the specified provider
            
        Raises:
            ValueError: If API key is missing or invalid
        """
        if provider == "openai":
            # First check if API key is provided in config
            api_key = self.config.get("api_key")
            
            # If not in config, check environment variable
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY")
            
            # Validate the API key
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Please set it using one of these methods:\n"
                    "1. Set environment variable: export OPENAI_API_KEY='your-key-here'\n"
                    "2. Update the config with: config['api_key'] = 'your-key-here'\n"
                    "3. Use the web interface to update the API key"
                )
            
            if api_key == "your-openai-api-key-here" or api_key == "":
                raise ValueError(
                    "Please replace the placeholder API key with your actual OpenAI API key. "
                    "You can get one from: https://platform.openai.com/api-keys"
                )
        elif provider == "anthropic":
            # First check if API key is provided in config
            api_key = self.config.get("anthropic_api_key")
            
            # If not in config, check environment variable
            if not api_key:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            
            # Validate the API key
            if not api_key:
                raise ValueError(
                    "Anthropic API key not found. Please set it using one of these methods:\n"
                    "1. Set environment variable: export ANTHROPIC_API_KEY='your-key-here'\n"
                    "2. Update the config with: config['anthropic_api_key'] = 'your-key-here'\n"
                )
            
            if api_key == "":
                raise ValueError(
                    "Please provide your actual Anthropic API key. "
                    "You can get one from: https://console.anthropic.com/"
                )
        elif provider == "qwen":
            # First check if API key is provided in config
            api_key = self.config.get("qwen_api_key")

            # If not in config, check environment variable
            if not api_key:
                api_key = os.environ.get("DASHSCOPE_API_KEY")

            # Validate the API key
            if not api_key:
                raise ValueError(
                    "Qwen API key not found. Please set it using one of these methods:\n"
                    "1. Set environment variable: export DASHSCOPE_API_KEY='your-key-here'\n"
                    "2. Update the config with: config['qwen_api_key'] = 'your-key-here'\n"
                )

            if api_key == "":
                raise ValueError(
                    "Please provide your actual Qwen API key. "
                    "You can get one from: https://dashscope.console.aliyun.com/"
                )
<<<<<<< HEAD
=======
        elif provider == "google":
            api_key = self.config.get("google_api_key")

            if not api_key:
                api_key = os.environ.get("GOOGLE_API_KEY")

            if not api_key:
                key_path = Path(__file__).resolve().parent.parent / "Gemini_API.txt"
                if key_path.is_file():
                    api_key = key_path.read_text(encoding="utf-8").strip()

            if not api_key:
                raise ValueError(
                    "Google API key not found. Please set it using one of these methods:\n"
                    "1. Save your key to: "
                    f"{Path(__file__).resolve().parent.parent / 'Gemini_API.txt'}\n"
                    "2. Set environment variable: export GOOGLE_API_KEY='your-key-here'\n"
                    "3. Update the config with: config['google_api_key'] = 'your-key-here'"
                )


>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
        elif provider in MINIMAX_PROVIDER_CONFIG:
            provider_config = MINIMAX_PROVIDER_CONFIG[provider]
            api_key = self.config.get(provider_config["config_key"])

            if not api_key:
                for env_key in provider_config["env_keys"]:
                    api_key = os.environ.get(env_key)
                    if api_key:
                        break

            if not api_key:
                env_exports = "\n".join(
                    f"{idx}. Set environment variable: export {env_key}='your-key-here'"
                    for idx, env_key in enumerate(provider_config["env_keys"], start=1)
                )
                raise ValueError(
                    f"{provider_config['label']} API key not found. Please set it using one of these methods:\n"
                    f"{env_exports}\n"
                    f"{len(provider_config['env_keys']) + 1}. Update the config with: "
                    f"config['{provider_config['config_key']}'] = 'your-key-here'\n"
                    f"{len(provider_config['env_keys']) + 2}. Use the web interface to update the API key"
                )

            if api_key == "":
                raise ValueError(
                    f"Please provide your actual {provider_config['label']} API key. "
                    f"You can get one from: {provider_config['console_url']}"
                )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Must be one of {', '.join(SUPPORTED_PROVIDERS)}")
        
        return api_key

    def _create_llm(
        self, provider: str, model: str, temperature: float
    ) -> BaseChatModel:
        """
        Create an LLM instance based on the provider.
<<<<<<< HEAD

        Args:
            provider: The provider name ("openai", "anthropic", "qwen", "minimax", or "minimax_cn")
            model: The model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022", "qwen-vl-max-latest", "MiniMax-M2.7")
            temperature: The temperature setting for the model

        Returns:
            BaseChatModel: An instance of the appropriate LLM class
        """
=======
        """
        # Force lowercase to prevent "Google" vs "google" capitalization crashes
        provider = provider.lower() 
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
        api_key = self._get_api_key(provider)

        if provider == "openai":
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
            )
        elif provider == "anthropic":
            # ChatAnthropic handles SystemMessage extraction automatically
<<<<<<< HEAD
            # It extracts SystemMessage from the message list and passes it as 'system' parameter
            # The messages array should contain at least one non-SystemMessage
=======
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                api_key=api_key,
            )
        elif provider == "qwen":
            return ChatQwen(
                model=model,
                temperature=temperature,
                api_key=api_key,
                max_retries=4,
            )
        elif provider in MINIMAX_PROVIDER_CONFIG:
<<<<<<< HEAD
            # MiniMax uses OpenAI-compatible APIs; CN and global differ by base URL.
            # Temperature must be in (0.0, 1.0] for MiniMax.
=======
            # MiniMax uses OpenAI-compatible APIs
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
            clamped_temp = max(0.01, min(temperature, 1.0))
            return ChatOpenAI(
                model=model,
                temperature=clamped_temp,
                api_key=api_key,
                openai_api_base=MINIMAX_PROVIDER_CONFIG[provider]["base_url"],
            )
<<<<<<< HEAD
        else:
            raise ValueError(f"Unsupported provider: {provider}. Must be one of {', '.join(SUPPORTED_PROVIDERS)}")

=======
        elif provider == "google":
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=api_key,
                max_retries=4,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Must be one of {', '.join(SUPPORTED_PROVIDERS)}")
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
    # def _set_tool_nodes(self) -> Dict[str, ToolNode]:
    #     """
    #     Define tool nodes for each agent type (indicator, pattern, trend).
    #     """
    #     return {
    #         "indicator": ToolNode(
    #             [
    #                 self.toolkit.compute_macd,
    #                 self.toolkit.compute_roc,
    #                 self.toolkit.compute_rsi,
    #                 self.toolkit.compute_stoch,
    #                 self.toolkit.compute_willr,
    #             ]
    #         ),
    #         "pattern": ToolNode(
    #             [
    #                 self.toolkit.generate_kline_image,
    #             ]
    #         ),
    #         "trend": ToolNode([self.toolkit.generate_trend_image]),
    #     }

    def refresh_llms(self):
        """
        Refresh the LLM objects with the current API key from environment.
        This is called when the API key is updated. Raises ValueError if the
        selected provider still has no API key (the web UI uses this to prompt).
        """
<<<<<<< HEAD
        # Recreate LLM objects with current config values
        self.agent_llm = self._create_llm(
            provider=self.config.get("agent_llm_provider", "openai"),
            model=self.config.get("agent_llm_model", "gpt-4o-mini"),
            temperature=self.config.get("agent_llm_temperature", 0.1),
        )
        self.graph_llm = self._create_llm(
            provider=self.config.get("graph_llm_provider", "openai"),
            model=self.config.get("graph_llm_model", "gpt-4o"),
            temperature=self.config.get("graph_llm_temperature", 0.1),
        )

        # Recreate the graph setup with new LLMs
        self.graph_setup = SetGraph(
            self.agent_llm,
            self.graph_llm,
            self.toolkit,
            # self.tool_nodes,
        )

        # Recreate the main graph
        self.graph = self.graph_setup.set_graph()
=======
        self._build_llms_and_graph()

    def ensure_initialized(self):
        """
        Build the LLMs/graph if construction was deferred (no API key at start).
        Raises a descriptive ValueError if a key is still missing — call this
        before using ``self.graph`` so consumers get a clear message instead of
        an AttributeError on a None graph.
        """
        if self.graph is None or self.graph_setup is None:
            self._build_llms_and_graph()
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c

    def update_api_key(self, api_key: str, provider: str = "openai"):
        """
        Update the API key in the config and refresh LLMs.
        This method is called by the web interface when API key is updated.
        
        Args:
            api_key (str): The new API key
            provider (str): The provider name, defaults to "openai"
        """
        if provider == "openai":
            # Update the config with the new API key
            self.config["api_key"] = api_key
            
            # Also update the environment variable for consistency
            os.environ["OPENAI_API_KEY"] = api_key
        elif provider == "anthropic":
            # Update the config with the new API key
            self.config["anthropic_api_key"] = api_key
            
            # Also update the environment variable for consistency
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif provider == "qwen":
            # Update the config with the new API key
            self.config["qwen_api_key"] = api_key

            # Also update the environment variable for consistency
            os.environ["DASHSCOPE_API_KEY"] = api_key
        elif provider == "minimax":
            # Update the config with the new API key
            self.config["minimax_api_key"] = api_key

            # Also update the environment variable for consistency
            os.environ["MINIMAX_API_KEY"] = api_key
        elif provider == "minimax_cn":
            # Update the config with the new API key
            self.config["minimax_cn_api_key"] = api_key

            # Keep CN credentials separate from the global MiniMax key.
            os.environ["MINIMAX_CN_API_KEY"] = api_key
        else:
            raise ValueError(f"Unsupported provider: {provider}. Must be one of {', '.join(SUPPORTED_PROVIDERS)}")
        
        # Refresh the LLMs with the new API key
        self.refresh_llms()
