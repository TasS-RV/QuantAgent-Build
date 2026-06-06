"""
Prompt variant catalog + active-override registry (issue #8, Objective 2).

This is the artifact the prompt-tuning loop "plays with": a small, named set of
system-prompt variants for each perception agent (indicator / pattern / trend),
plus a process-wide registry the tuner sets before each backtest run.

Agents call ``get_prompt(agent_key, default_text)`` for their primary system
prompt. With no override set, the agent's existing default text is used verbatim
(zero behaviour change), so this is safe to wire in even when not tuning.

Keep variants few and meaningfully different — the search space is otherwise
infinite. Temperature should be ~0 during tuning so runs are comparable
(set via the LLM config, see backtesting_operations/PROMPT_TUNING_PLAN.md).
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ── Variant catalog ──────────────────────────────────────────────────────────
# Each agent_key maps variant_name -> system prompt text. "baseline" mirrors the
# agent's shipped prompt; the others encode different trader "psychologies".

PROMPT_VARIANTS: Dict[str, Dict[str, str]] = {
    "indicator": {
        "baseline": (
            "You are a high-frequency trading (HFT) analyst assistant operating under "
            "time-sensitive conditions. Interpret the momentum and oscillator state from "
            "the provided indicators and give a decisive bias."
        ),
        "momentum_aggressive": (
            "You are an aggressive momentum trader. Treat strong RSI/MACD/ROC alignment as "
            "a signal to lean firmly into the prevailing direction; do not fade strength. "
            "Only flag exhaustion when oscillators are at genuine extremes."
        ),
        "mean_reversion": (
            "You are a disciplined mean-reversion analyst. Treat overbought (RSI>70) and "
            "oversold (RSI<30) readings as fade opportunities against the recent move, and "
            "be skeptical of chasing extended momentum."
        ),
        "conservative": (
            "You are a risk-averse indicator analyst. Require multiple indicators to agree "
            "before issuing a directional bias; default to neutral when signals conflict."
        ),
    },
    "pattern": {
        "baseline": (
            "You are a trading pattern recognition assistant. Identify the single most "
            "prominent chart pattern and report its bullish/bearish implication."
        ),
        "breakout_focused": (
            "You are a breakout specialist. Prioritise continuation patterns (flags, "
            "ascending/descending triangles, range breakouts) and treat clean breakouts as "
            "high-conviction directional signals."
        ),
        "reversal_focused": (
            "You are a reversal specialist. Prioritise exhaustion/reversal structures "
            "(double tops/bottoms, head-and-shoulders, wedges) and be cautious about "
            "trend-continuation calls late in a move."
        ),
        "strict": (
            "You are a conservative pattern analyst. Only report a pattern when it is "
            "textbook-clear; otherwise return 'None' with low confidence."
        ),
    },
    "trend": {
        "baseline": (
            "You are a K-line trend pattern recognition assistant operating in a "
            "high-frequency trading context. Describe the prevailing trend, channel and "
            "any support/resistance breaks."
        ),
        "trend_following": (
            "You are a trend-following analyst. Favour the direction of the regression "
            "channel slope; treat pullbacks within an up-channel as buys and within a "
            "down-channel as sells."
        ),
        "range_aware": (
            "You are a range/channel analyst. When price sits mid-channel with a flat "
            "slope, call it sideways and avoid directional conviction; only commit near "
            "channel extremes or on a confirmed break."
        ),
    },
}

# ── Active-override registry ──────────────────────────────────────────────────
_ACTIVE: Dict[str, str] = {}   # agent_key -> prompt text currently in force


def set_active(agent_key: str, variant_name: str) -> str:
    """Activate a named variant for an agent. Returns the prompt text."""
    variants = PROMPT_VARIANTS.get(agent_key, {})
    if variant_name not in variants:
        raise KeyError(f"Unknown variant {variant_name!r} for agent {agent_key!r}. "
                       f"Available: {sorted(variants)}")
    _ACTIVE[agent_key] = variants[variant_name]
    return _ACTIVE[agent_key]


def set_active_combo(combo: Dict[str, str]) -> None:
    """Activate a {agent_key: variant_name} combination at once."""
    for k, v in combo.items():
        set_active(k, v)


def clear_active(agent_key: Optional[str] = None) -> None:
    if agent_key is None:
        _ACTIVE.clear()
    else:
        _ACTIVE.pop(agent_key, None)


def get_prompt(agent_key: str, default: str) -> str:
    """Return the active override for agent_key, else the supplied default."""
    return _ACTIVE.get(agent_key, default)


def variant_names(agent_key: str) -> List[str]:
    return sorted(PROMPT_VARIANTS.get(agent_key, {}))


def all_combos(agent_keys: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Cartesian product of variants across the given agents (default: all)."""
    import itertools
    keys = agent_keys or list(PROMPT_VARIANTS)
    name_lists = [variant_names(k) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*name_lists)]
