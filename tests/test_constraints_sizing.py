"""Tests for hard-coded constraints (constraints.py), confidence position sizing
(backtest_engine), and the quant_signal market_context hook. No API calls."""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import constraints as C

try:
    import numpy as np
    import pandas as pd
    from backtest_engine import Signal, SimConfig, simulate
    from quant_pipeline.quant_signal import compute_quant_decision
    HAVE_DEPS = True
except Exception:  # pragma: no cover
    HAVE_DEPS = False


class TestConstraints(unittest.TestCase):
    def test_vix_gate_blocks_buy_in_high_vol(self):
        d, c = C.apply_constraints(1, 0.8, {"vix": 35}, rules=["vix_gate"])
        self.assertEqual(d, 0)
        # below threshold -> unchanged
        d2, _ = C.apply_constraints(1, 0.8, {"vix": 20}, rules=["vix_gate"])
        self.assertEqual(d2, 1)
        # shorts not gated by VIX
        d3, _ = C.apply_constraints(-1, 0.8, {"vix": 40}, rules=["vix_gate"])
        self.assertEqual(d3, -1)

    def test_breakout_bias_forces_long(self):
        d, c = C.apply_constraints(0, 0.1, {"is_bullish_breakout": True}, rules=["breakout_bias"])
        self.assertEqual(d, 1)
        self.assertGreater(c, 0.1)
        d2, _ = C.apply_constraints(0, 0.1, {"is_bearish_breakdown": True}, rules=["breakout_bias"])
        self.assertEqual(d2, -1)

    def test_trend_regime_filter(self):
        d, _ = C.apply_constraints(-1, 0.5, {"price": 110, "sma200": 100},
                                   rules=["trend_regime_filter"])
        self.assertEqual(d, 0)  # don't short above 200dma

    def test_no_context_passthrough(self):
        d, c = C.apply_constraints(1, 0.7, None)
        self.assertEqual((d, round(c, 3)), (1, 0.7))

    def test_detect_breakout(self):
        closes = [10] * 25 + [20]      # last bar breaks out above prior high
        kl = {"Close": closes, "High": closes, "Low": closes}
        self.assertTrue(C.detect_breakout(kl)["is_bullish_breakout"])
        closes2 = [10] * 25 + [2]
        kl2 = {"Close": closes2, "High": closes2, "Low": closes2}
        self.assertTrue(C.detect_breakout(kl2)["is_bearish_breakdown"])


@unittest.skipUnless(HAVE_DEPS, "numpy/pandas not installed")
class TestPositionSizing(unittest.TestCase):
    def setUp(self):
        n = 120
        close = np.linspace(100, 140, n)
        self.df = pd.DataFrame({
            "Datetime": pd.date_range("2024-01-01", periods=n, freq="D"),
            "Open": close - 0.1, "High": close + 1, "Low": close - 1, "Close": close,
        })

    def test_confidence_sizing_scales_pnl(self):
        sig = [Signal(20, 1, confidence=0.5, risk_reward_ratio=2.0)]
        full = simulate("T", self.df, sig,
                        SimConfig(trend_filter=False, use_atr_stops=False, position_sizing="full"))
        conf = simulate("T", self.df, sig,
                        SimConfig(trend_filter=False, use_atr_stops=False,
                                  position_sizing="confidence"))
        ft, ct = full["trades"][0], conf["trades"][0]
        self.assertEqual(ft.position_size, 1.0)
        self.assertEqual(ct.position_size, 0.5)
        self.assertAlmostEqual(ct.pnl_pct, ft.pnl_pct * 0.5, places=6)


@unittest.skipUnless(HAVE_DEPS, "numpy/pandas not installed")
class TestQuantSignalContext(unittest.TestCase):
    def test_market_context_applies_constraints(self):
        warnings.simplefilter("ignore")
        close = np.linspace(100, 130, 80)   # uptrend -> base decision tends BUY/HOLD
        kl = {"Open": (close - 0.2).tolist(), "High": (close + 1).tolist(),
              "Low": (close - 1).tolist(), "Close": close.tolist()}
        # Force a strong bullish signal, then gate it with high VIX.
        base = compute_quant_decision("T", kl, weights={"indicator": 0, "trend": 1, "pattern": 0})
        gated = compute_quant_decision("T", kl, weights={"indicator": 0, "trend": 1, "pattern": 0},
                                       market_context={"vix": 40})
        if base.decision == "BUY":
            self.assertEqual(gated.decision, "HOLD")
            self.assertIn("constraint-adjusted", gated.decision_rationale)


if __name__ == "__main__":
    unittest.main()
