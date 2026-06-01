"""Unit tests for the deterministic quant decision agent (decision_agent_quant.py).

Pure math — no LLM. Skipped automatically if numpy/pandas are unavailable.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401
    from decision_agent_quant import make_trade_decision
    HAVE_DEPS = True
except Exception:  # pragma: no cover
    HAVE_DEPS = False


def _kline():
    import numpy as np
    close = np.linspace(100, 120, 40)
    return {"Open": (close - 0.2).tolist(), "High": (close + 1).tolist(),
            "Low": (close - 1).tolist(), "Close": close.tolist()}


@unittest.skipUnless(HAVE_DEPS, "numpy/pandas not installed")
class TestQuantDecision(unittest.TestCase):
    def _state(self, ind, trend, pat, **extra):
        s = {"stock_name": "T", "kline_data": _kline(),
             "indicator_report": json.dumps({"quantitative_metrics": {"final_indicator_signal": ind}}),
             "trend_report": json.dumps({"quantitative_metrics": trend}),
             "pattern_report": json.dumps(pat)}
        s.update(extra)
        return s

    def test_strong_bullish_is_buy(self):
        d = make_trade_decision(self._state(
            0.6, {"normalized_signal": 0.5, "current_price": 120.0,
                  "support_level": 110.0, "resistance_level": 122.0},
            {"macro_pattern_name": "Double Bottom", "direction": 1, "confidence_score": 0.7}))
        self.assertEqual(d.decision, "BUY")

    def test_strong_bearish_is_short(self):
        d = make_trade_decision(self._state(
            -0.8, {"normalized_signal": -0.7, "current_price": 100.0,
                   "support_level": 90.0, "resistance_level": 110.0},
            {"macro_pattern_name": "Rising Wedge", "direction": -1, "confidence_score": 0.6}))
        self.assertEqual(d.decision, "SHORT")

    def test_neutral_is_hold(self):
        d = make_trade_decision(self._state(
            0.0, {"normalized_signal": 0.0, "current_price": 100.0,
                  "support_level": 97.0, "resistance_level": 103.0},
            {"macro_pattern_name": "None", "direction": 0, "confidence_score": 0.0}))
        self.assertEqual(d.decision, "HOLD")

    def test_protective_override_on_deep_loss(self):
        # Near-neutral signal would be HOLD, but a >5% unrealised loss promotes to SELL.
        d = make_trade_decision(self._state(
            -0.05, {"normalized_signal": -0.05, "current_price": 100.0,
                    "support_level": 95.0, "resistance_level": 105.0},
            {"macro_pattern_name": "None", "direction": 0, "confidence_score": 0.0},
            entry_price=110.0))
        self.assertEqual(d.decision, "SELL")
        self.assertLess(d.unrealized_pnl_pct, -5.0)

    def test_short_disabled_when_allow_short_false(self):
        d = make_trade_decision(self._state(
            -0.8, {"normalized_signal": -0.7, "current_price": 100.0,
                   "support_level": 90.0, "resistance_level": 110.0},
            {"macro_pattern_name": "Rising Wedge", "direction": -1, "confidence_score": 0.6}),
            allow_short=False)
        self.assertEqual(d.decision, "SELL")  # falls through to SELL, never SHORT


if __name__ == "__main__":
    unittest.main()
