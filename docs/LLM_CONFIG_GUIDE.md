# LLM Provider & API Key Configuration Guide

There are three layers where you can change the LLM provider or API key.
They are listed from **quickest** (no file edits needed) to **most permanent** (changes the hardcoded defaults).

---

## Layer 1 — CLI flag (one-off runs, no file edits)

Pass `--provider` and `--api-key` directly when running the master portfolio script.

```bash
python master_portfolio.py --provider openai    --api-key sk-...
python master_portfolio.py --provider anthropic --api-key sk-ant-...
python master_portfolio.py --provider qwen      --api-key sk-...
```

Valid `--provider` values: `openai`, `anthropic`, `qwen`, `minimax`, `minimax_cn`, `google`

---

## Layer 2 — Environment variables (persistent per shell session or `.env`)

Set the relevant variable before running. The system checks these automatically if no key is found in config.

| Provider | Environment variable |
|----------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Qwen (Alibaba) | `DASHSCOPE_API_KEY` |
| MiniMax | `MINIMAX_API_KEY` |
| MiniMax CN | `MINIMAX_CN_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` |

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python master_portfolio.py --provider anthropic
```

**Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=sk-ant-...
python master_portfolio.py --provider anthropic
```

**Google Gemini shortcut** — the system also reads a plain-text key file one folder above the repo:
```
..\Gemini_API.txt   ← paste your key here, nothing else in the file
```
If that file exists and is non-empty, it is loaded automatically when `--provider google` is used.

---

## Layer 3 — Hardcoded defaults (permanent change across all runs)

Edit these two files when you want to change the default provider or models permanently.

---

### File A: `master_portfolio.py` — lines 85–89

Controls what the **portfolio script** uses when no `--provider` flag is given.

```python
# master_portfolio.py  ←  lines 85–89

DEFAULT_LLM_PROVIDER = "google"          # ← change to: "openai" | "anthropic" | "qwen" | "minimax" | "google"

GEMINI_KEY_FILE = Path(__file__).resolve().parent.parent / "Gemini_API.txt"   # Google only
DEFAULT_GEMINI_AGENT_MODEL = "gemini-2.0-flash-lite"   # ← model for indicator / pattern agents
DEFAULT_GEMINI_GRAPH_MODEL = "gemini-2.0-flash-lite"   # ← model for trend / vision analysis
```

**Agent LLM vs Graph LLM:**

| Variable | Used by |
|----------|---------|
| `DEFAULT_GEMINI_AGENT_MODEL` | `agent_llm` — Indicator Agent tool calls, Pattern Agent tool calls |
| `DEFAULT_GEMINI_GRAPH_MODEL` | `graph_llm` — Pattern vision analysis, Trend vision analysis |

For non-Google providers the model names are inherited from `default_config.py` (see below).

---

### File B: `default_config.py` — lines 1–14

The **system-wide default config** used by `TradingGraph` and the web interface whenever no override is passed.

```python
# default_config.py  ←  lines 1–14

DEFAULT_CONFIG = {
    # ── Active provider ────────────────────────────────────────────────────────
    "agent_llm_provider": "google",          # ← provider for indicator / pattern agents
    "graph_llm_provider": "google",          # ← provider for trend / vision / decision agents

    # ── Model names ────────────────────────────────────────────────────────────
    "agent_llm_model": "gemini-2.5-flash-lite",   # ← model ID for agent_llm_provider
    "graph_llm_model": "gemini-2.5-flash",         # ← model ID for graph_llm_provider

    # ── Temperatures ───────────────────────────────────────────────────────────
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,

    # ── API keys ───────────────────────────────────────────────────────────────
    "api_key":           "sk-",   # OpenAI
    "anthropic_api_key": "sk-",   # Anthropic / Claude
    "qwen_api_key":      "sk-",   # Qwen (Alibaba DashScope)
    "minimax_api_key":   "",       # MiniMax (international)
    "minimax_cn_api_key":"",       # MiniMax (China endpoint)
    "google_api_key":    "",       # Google Gemini (or use Gemini_API.txt / env var)
}
```

---

## Supported Providers & Recommended Models

| Provider | `provider` value | Agent model (lighter) | Graph model (smarter / vision) | Key source |
|----------|-----------------|----------------------|-------------------------------|------------|
| **OpenAI** | `openai` | `gpt-4o-mini` | `gpt-4o` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| **Anthropic** | `anthropic` | `claude-haiku-4-5-20251001` | `claude-sonnet-4-6` | [console.anthropic.com](https://console.anthropic.com/) |
| **Google Gemini** | `google` | `gemini-2.0-flash-lite` | `gemini-2.0-flash` | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| **Qwen (Alibaba)** | `qwen` | `qwen3-max` | `qwen3-vl-plus` | [dashscope.aliyun.com](https://dashscope.console.aliyun.com/) |
| **MiniMax** | `minimax` | `MiniMax-M2.7` | `MiniMax-M2.7` | [platform.minimaxi.com](https://platform.minimaxi.com/) |
| **MiniMax CN** | `minimax_cn` | `MiniMax-M2.7` | `MiniMax-M2.7` | [platform.minimaxi.com](https://platform.minimaxi.com/) |

> **Note:** Pattern and Trend agents use `graph_llm` for vision analysis (image understanding). Make sure the model you pick for `graph_llm_model` supports multimodal / image input.

---

## Example: Switching to Anthropic

### Option A — CLI (one run)
```bash
python master_portfolio.py --provider anthropic --api-key sk-ant-...
```

### Option B — Permanent (edit both files)

**`default_config.py`:**
```python
"agent_llm_provider": "anthropic",
"graph_llm_provider": "anthropic",
"agent_llm_model":    "claude-haiku-4-5-20251001",
"graph_llm_model":    "claude-sonnet-4-6",
"anthropic_api_key":  "sk-ant-...",
```

**`master_portfolio.py` line 85:**
```python
DEFAULT_LLM_PROVIDER = "anthropic"
```

---

## Example: Switching to OpenAI

### Option A — CLI (one run)
```bash
python master_portfolio.py --provider openai --api-key sk-...
```

### Option B — Permanent (edit both files)

**`default_config.py`:**
```python
"agent_llm_provider": "openai",
"graph_llm_provider": "openai",
"agent_llm_model":    "gpt-4o-mini",
"graph_llm_model":    "gpt-4o",
"api_key":            "sk-...",
```

**`master_portfolio.py` line 85:**
```python
DEFAULT_LLM_PROVIDER = "openai"
```

---

## How the Config Flows at Runtime

```
master_portfolio.py
  └─ build_llm_config(provider, api_key)          # master_portfolio.py lines 112–138
        └─ TradingGraph(config=llm_config)         # trading_graph.py line 47
              └─ _get_api_key(provider)            # trading_graph.py line 78
                    checks: config dict → env var → Gemini_API.txt (Google only)
              └─ _create_llm(provider, model, ...)  # trading_graph.py line 209
                    returns: ChatOpenAI / ChatAnthropic / ChatGoogleGenerativeAI / ...
```

Key resolution order (highest priority first):
1. `--api-key` CLI flag → passed into `llm_config` dict
2. `default_config.py` key fields (if non-empty)
3. Environment variable for the provider (see table above)
4. `Gemini_API.txt` file (Google only)
