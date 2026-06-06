"""
API cost stop-loss - a hard ceiling on LLM token/dollar spend.

The safety mechanism for the project: it counts LLM calls and token spend across
every provider and aborts BEFORE a call that would breach the configured cap, so
a runaway backtest or scheduler loop can never silently burn API credits.

* One process-wide CostGuard (get_guard()), configured from env or configure().
  Hard defaults apply even if nothing is set, so forgetting to configure fails safe.
* preflight(model): call right before an LLM request. Raises CostLimitExceeded if
  the kill switch is on, a cap is already reached, or the next call is estimated
  to breach the dollar cap. Nothing is spent.
* record(input_tokens, output_tokens, model): call after a request with the real
  usage. Updates totals and raises if a cap was breached.
* Provider-agnostic: make_langchain_callback() wires preflight+record on every
  LangChain invoke; the OpenAI Agents SDK and Gemini paths call preflight/record.

Env vars (optional):
    QUANT_MAX_USD      hard dollar ceiling          (default 5.00)
    QUANT_MAX_TOKENS   hard total-token ceiling     (default 2,000,000)
    QUANT_MAX_CALLS    hard LLM-call-count ceiling  (default 400)
    QUANT_KILL_SWITCH  "1"/"true" -> block ALL calls (default off)
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Optional


class CostLimitExceeded(RuntimeError):
    """Raised when an LLM call would breach (or has breached) a configured cap."""


# USD per 1,000,000 tokens (input, output). Prefix match -> dated ids resolve.
DEFAULT_PRICING: dict = {
    "gpt-4o-mini":           (0.15, 0.60),
    "gpt-4o":                (2.50, 10.00),
    "gpt-4.1-mini":          (0.40, 1.60),
    "gpt-4.1":               (2.00, 8.00),
    "o4-mini":               (1.10, 4.40),
    "gemini-3":              (0.15, 0.60),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash":      (0.30, 2.50),
    "gemini-2.5-pro":        (1.25, 10.00),
    "gemini-1.5-flash":      (0.075, 0.30),
    "claude-haiku":          (0.80, 4.00),
    "claude-sonnet":         (3.00, 15.00),
    "claude-opus":           (5.00, 25.00),
    "qwen":                  (0.40, 1.20),
    "minimax":               (0.30, 1.20),
    "meta-llama":            (0.20, 0.60),
}
_DEFAULT_PRICE = (1.00, 3.00)


def _price_for(model: Optional[str], pricing: dict):
    if not model:
        return _DEFAULT_PRICE
    m = model.lower()
    best = None
    for key, price in pricing.items():
        if m.startswith(key) and (best is None or len(key) > len(best[0])):
            best = (key, price)
    return best[1] if best else _DEFAULT_PRICE


# Optional integration with the `tokencost` library (AgentOps-AI): a maintained,
# offline price map for 400+ models. Used when installed; we silently fall back
# to the built-in table for unknown models or if the package is absent.
_TOKENCOST_OK: Optional[bool] = None


def _tokencost_estimate(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    global _TOKENCOST_OK
    if _TOKENCOST_OK is False:
        return None
    try:
        from tokencost import calculate_cost_by_tokens
    except Exception:
        _TOKENCOST_OK = False
        return None
    try:
        cin = float(calculate_cost_by_tokens(int(input_tokens), model, "input"))
        cout = float(calculate_cost_by_tokens(int(output_tokens), model, "output"))
        _TOKENCOST_OK = True
        return cin + cout
    except Exception:
        # Unknown model in tokencost's map — let caller use the built-in table.
        return None


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v not in (None, "") else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    try:
        return int(v) if v not in (None, "") else default
    except ValueError:
        return default


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class CostGuard:
    max_usd: Optional[float] = 5.00
    max_tokens: Optional[int] = 2_000_000
    max_calls: Optional[int] = 400
    kill_switch: bool = False
    pricing: dict = field(default_factory=lambda: dict(DEFAULT_PRICING))

    use_tokencost: bool = True   # prefer the maintained `tokencost` price map if installed

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def estimate_cost(self, model, input_tokens, output_tokens) -> float:
        # No LLM API returns a dollar cost — every tracker (tokencost, litellm,
        # langfuse) multiplies returned token usage by a price table, which is
        # what we do here. Prefer the maintained `tokencost` map (400+ models,
        # offline) when installed and the model is recognised; else fall back to
        # the built-in pricing table.
        if self.use_tokencost and model:
            tc = _tokencost_estimate(model, input_tokens, output_tokens)
            if tc is not None:
                return tc
        cin, cout = _price_for(model, self.pricing)
        return (input_tokens / 1e6) * cin + (output_tokens / 1e6) * cout

    def preflight(self, model=None, est_input_tokens=1500, est_output_tokens=500) -> None:
        """Raise CostLimitExceeded if this call must not proceed. No spend."""
        with self._lock:
            if self.kill_switch:
                raise CostLimitExceeded("Kill switch active (QUANT_KILL_SWITCH) - LLM calls blocked.")
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise CostLimitExceeded(
                    f"Call cap reached: {self.calls}/{self.max_calls} LLM calls. "
                    f"Raise QUANT_MAX_CALLS to continue.")
            if self.max_tokens is not None and self.total_tokens >= self.max_tokens:
                raise CostLimitExceeded(
                    f"Token cap reached: {self.total_tokens:,}/{self.max_tokens:,}. "
                    f"Raise QUANT_MAX_TOKENS to continue.")
            if self.max_usd is not None:
                projected = self.cost_usd + self.estimate_cost(model, est_input_tokens, est_output_tokens)
                if projected > self.max_usd:
                    raise CostLimitExceeded(
                        f"Dollar cap would be breached: ${self.cost_usd:.4f} spent, next "
                        f"~${projected - self.cost_usd:.4f} -> ${projected:.4f} > cap "
                        f"${self.max_usd:.2f}. Raise QUANT_MAX_USD to continue.")

    def record(self, input_tokens=0, output_tokens=0, model=None) -> float:
        """Add actual usage to running totals; raise if a cap was breached."""
        with self._lock:
            cost = self.estimate_cost(model, input_tokens, output_tokens)
            self.calls += 1
            self.input_tokens += int(input_tokens)
            self.output_tokens += int(output_tokens)
            self.cost_usd += cost
            breach = None
            if self.max_usd is not None and self.cost_usd > self.max_usd:
                breach = f"dollar cap ${self.max_usd:.2f} exceeded (spent ${self.cost_usd:.4f})"
            elif self.max_tokens is not None and self.total_tokens > self.max_tokens:
                breach = f"token cap {self.max_tokens:,} exceeded ({self.total_tokens:,} used)"
            elif self.max_calls is not None and self.calls > self.max_calls:
                breach = f"call cap {self.max_calls} exceeded ({self.calls} calls)"
        if breach:
            raise CostLimitExceeded(f"STOP-LOSS TRIPPED - {breach}. Halting LLM usage.")
        return cost

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def remaining_usd(self) -> float:
        return float("inf") if self.max_usd is None else max(0.0, self.max_usd - self.cost_usd)

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "max_usd": self.max_usd,
            "max_tokens": self.max_tokens,
            "max_calls": self.max_calls,
            "remaining_usd": round(self.remaining_usd(), 6),
        }

    def reset(self) -> None:
        with self._lock:
            self.calls = self.input_tokens = self.output_tokens = 0
            self.cost_usd = 0.0

    def banner(self) -> str:
        return (f"Cost stop-loss armed - cap ${self.max_usd} / {self.max_tokens:,} tok / "
                f"{self.max_calls} calls" + ("  [KILL SWITCH ON]" if self.kill_switch else ""))


_guard: Optional[CostGuard] = None
_guard_lock = threading.Lock()


def get_guard() -> CostGuard:
    """Return the process-wide guard, building it from env on first use."""
    global _guard
    if _guard is None:
        with _guard_lock:
            if _guard is None:
                _guard = CostGuard(
                    max_usd=_env_float("QUANT_MAX_USD", 5.00),
                    max_tokens=_env_int("QUANT_MAX_TOKENS", 2_000_000),
                    max_calls=_env_int("QUANT_MAX_CALLS", 400),
                    kill_switch=_env_bool("QUANT_KILL_SWITCH"),
                )
    return _guard


def configure(max_usd=None, max_tokens=None, max_calls=None, kill_switch=None, pricing=None) -> CostGuard:
    """Set caps programmatically (overrides env). Returns the guard."""
    g = get_guard()
    if max_usd is not None:
        g.max_usd = max_usd
    if max_tokens is not None:
        g.max_tokens = max_tokens
    if max_calls is not None:
        g.max_calls = max_calls
    if kill_switch is not None:
        g.kill_switch = kill_switch
    if pricing is not None:
        g.pricing.update(pricing)
    return g


def reset_guard() -> None:
    """Drop the singleton (mainly for tests)."""
    global _guard
    _guard = None


def make_langchain_callback(guard: Optional[CostGuard] = None):
    """LangChain BaseCallbackHandler enforcing the guard on every LLM call."""
    from langchain_core.callbacks import BaseCallbackHandler
    g = guard or get_guard()

    class _CostGuardHandler(BaseCallbackHandler):
        raise_error = True

        def on_llm_start(self, serialized, prompts, **kwargs):
            model = (serialized or {}).get("kwargs", {}).get("model") or (serialized or {}).get("name")
            est_in = sum(len(p) for p in (prompts or [])) // 4 or 1000
            g.preflight(model=model, est_input_tokens=est_in)

        def on_chat_model_start(self, serialized, messages, **kwargs):
            model = (serialized or {}).get("kwargs", {}).get("model") or (serialized or {}).get("name")
            est_in = sum(len(str(m)) for batch in (messages or []) for m in batch) // 4 or 1000
            g.preflight(model=model, est_input_tokens=est_in)

        def on_llm_end(self, response, **kwargs):
            in_tok, out_tok, model = _extract_langchain_usage(response)
            g.record(in_tok, out_tok, model)

    return _CostGuardHandler()


def _extract_langchain_usage(response):
    """Best-effort token + model extraction from a LangChain LLMResult."""
    in_tok = out_tok = 0
    llm_output = getattr(response, "llm_output", None) or {}
    model = llm_output.get("model_name") or llm_output.get("model")
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    in_tok = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
    out_tok = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
    if not (in_tok or out_tok):
        try:
            for gen_list in getattr(response, "generations", []) or []:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    um = getattr(msg, "usage_metadata", None) if msg else None
                    if um:
                        in_tok += um.get("input_tokens", 0)
                        out_tok += um.get("output_tokens", 0)
        except Exception:
            pass
    return int(in_tok or 0), int(out_tok or 0), model


def record_agents_sdk_result(result, model=None, guard: Optional[CostGuard] = None) -> float:
    """Record token usage from an OpenAI Agents SDK RunResult. Returns cost."""
    g = guard or get_guard()
    in_tok = out_tok = 0
    try:
        usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
        if usage is not None:
            in_tok = getattr(usage, "input_tokens", 0) or 0
            out_tok = getattr(usage, "output_tokens", 0) or 0
    except Exception:
        pass
    if not (in_tok or out_tok):
        try:
            for r in getattr(result, "raw_responses", []) or []:
                u = getattr(r, "usage", None)
                if u:
                    in_tok += getattr(u, "input_tokens", 0) or 0
                    out_tok += getattr(u, "output_tokens", 0) or 0
        except Exception:
            pass
    return g.record(in_tok, out_tok, model)


def record_gemini_response(response, model=None, guard: Optional[CostGuard] = None) -> float:
    """
    Record usage from a google-genai response (direct Gemini SDK path).
    Uses response.usage_metadata: prompt_token_count (input) and
    candidates_token_count + thoughts_token_count (billed output). Returns cost.
    """
    g = guard or get_guard()
    um = getattr(response, "usage_metadata", None)
    in_tok = out_tok = 0
    if um is not None:
        in_tok = getattr(um, "prompt_token_count", 0) or 0
        out_tok = (getattr(um, "candidates_token_count", 0) or 0) + \
                  (getattr(um, "thoughts_token_count", 0) or 0)
    return g.record(in_tok, out_tok, model)
