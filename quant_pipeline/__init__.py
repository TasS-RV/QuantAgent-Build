"""
quant_pipeline — pure-math LangGraph nodes and decision logic (zero LLM calls).

Modules
───────
  decision_agent_quant  weighted signal combiner → BUY / HOLD / SELL / SHORT
  quant_nodes           LangGraph node wrappers (indicator / trend / pattern)
  quant_signal          standalone one-shot entry point (no LangGraph needed)
"""

# Ensure the repo root is on sys.path so every module in this package can
# import root-level modules (indicator_agent, trend_agent, agent_state …)
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Convenience re-exports so callers can do:
#   from quant_pipeline import create_quant_decision_node
#   from quant_pipeline import quant_indicator_node, quant_trend_node, quant_pattern_node
#   from quant_pipeline import compute_quant_decision
from quant_pipeline.decision_agent_quant import (  # noqa: E402
    create_quant_decision_node,
    make_trade_decision,
    DEFAULT_WEIGHTS,
    DEFAULT_THRESHOLDS,
    TradeDecision,
)
from quant_pipeline.quant_nodes import (  # noqa: E402
    quant_indicator_node,
    quant_trend_node,
    quant_pattern_node,
    quantify_candlestick_patterns,
)
from quant_pipeline.quant_signal import compute_quant_decision  # noqa: E402

__all__ = [
    "create_quant_decision_node",
    "make_trade_decision",
    "DEFAULT_WEIGHTS",
    "DEFAULT_THRESHOLDS",
    "TradeDecision",
    "quant_indicator_node",
    "quant_trend_node",
    "quant_pattern_node",
    "quantify_candlestick_patterns",
    "compute_quant_decision",
]
