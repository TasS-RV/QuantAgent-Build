"""
Trading safety rails (IMPROVEMENTS_todo section 0 / 8).

The pipeline's decisions are LLM/quant-generated and must be treated as
ADVISORY. This module makes live order placement impossible unless three
independent conditions are all met, so no code path can accidentally trade real
money:

    1. PAPER_TRADING is False
    2. QUANT_LIVE_TRADING=1 is explicitly set in the environment
    3. the trading kill switch (QUANT_TRADING_KILL_SWITCH) is OFF

Any function that would submit a real order must call ``require_live_trading()``
first; in the default configuration it raises ``LiveTradingBlocked``.

Env vars
    QUANT_PAPER_TRADING        "0"/"false" to leave paper mode (default ON)
    QUANT_LIVE_TRADING         "1"/"true"  to arm live trading  (default OFF)
    QUANT_TRADING_KILL_SWITCH  "1"/"true"  to block ALL orders instantly
"""

from __future__ import annotations

import os


class LiveTradingBlocked(RuntimeError):
    """Raised when a live order is attempted while the safety rails are engaged."""


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# Default-ON paper trading. Importers may flip PAPER_TRADING at runtime, but the
# env gates below still have to agree before any live order is permitted.
PAPER_TRADING: bool = _env_bool("QUANT_PAPER_TRADING", True)


def paper_trading() -> bool:
    """True unless paper mode has been explicitly disabled (module + env)."""
    return PAPER_TRADING and _env_bool("QUANT_PAPER_TRADING", True)


def trading_kill_switch() -> bool:
    return _env_bool("QUANT_TRADING_KILL_SWITCH", False)


def live_trading_armed() -> bool:
    """All three conditions must hold for live trading to be permitted."""
    return (not paper_trading()) and _env_bool("QUANT_LIVE_TRADING", False) \
        and not trading_kill_switch()


def safe_to_live() -> bool:
    """Non-raising check — True only if live order placement is permitted."""
    return live_trading_armed()


def require_live_trading(action: str = "place order") -> None:
    """
    Gate that MUST precede any real order submission. Raises LiveTradingBlocked
    unless paper mode is off, QUANT_LIVE_TRADING=1, and the kill switch is off.
    """
    if trading_kill_switch():
        raise LiveTradingBlocked(
            f"Refusing to {action}: trading kill switch is ON (QUANT_TRADING_KILL_SWITCH).")
    if paper_trading():
        raise LiveTradingBlocked(
            f"Refusing to {action}: PAPER_TRADING is ON. Decisions are advisory. "
            f"To go live, set QUANT_PAPER_TRADING=0 AND QUANT_LIVE_TRADING=1 (and "
            f"understand the risk).")
    if not _env_bool("QUANT_LIVE_TRADING", False):
        raise LiveTradingBlocked(
            f"Refusing to {action}: live trading not armed. Set QUANT_LIVE_TRADING=1 "
            f"to explicitly enable real-money orders.")


def status() -> dict:
    return {
        "paper_trading": paper_trading(),
        "live_trading_armed": live_trading_armed(),
        "trading_kill_switch": trading_kill_switch(),
    }


def banner() -> str:
    if live_trading_armed():
        return "[!!!] LIVE TRADING ARMED — real orders may be placed."
    if trading_kill_switch():
        return "Trading kill switch ON — all orders blocked."
    return "PAPER TRADING (advisory only) — no real orders will be placed."
