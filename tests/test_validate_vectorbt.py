"""Tests for the VectorBT validation signal conversion + engine fallback (issue #4)."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
    import pandas as pd
    from validate_vectorbt import (
        signals_to_position, position_to_signal_arrays,
        validate_with_engine, load_signals,
    )
    HAVE_DEPS = True
except Exception:  # pragma: no cover
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy/pandas not installed")
class TestValidateVectorbt(unittest.TestCase):
    def setUp(self):
        n = 60
        close = np.linspace(100, 160, n)
        self.df = pd.DataFrame({
            "Datetime": pd.date_range("2024-01-01", periods=n, freq="D"),
            "Open": close - 0.1, "High": close + 1, "Low": close - 1, "Close": close,
        })
        self.signals = [
            {"decision_idx": 5, "direction": 1, "confidence": 0.9, "risk_reward_ratio": 2.0},
            {"decision_idx": 25, "direction": 0, "confidence": 0.5},
            {"decision_idx": 40, "direction": -1, "confidence": 0.8},
        ]

    def test_position_series(self):
        pos = signals_to_position(self.df, self.signals)
        self.assertEqual(pos.iloc[6], 1)
        self.assertEqual(pos.iloc[25], 1)
        self.assertEqual(pos.iloc[26], 0)
        self.assertEqual(pos.iloc[41], -1)
        self.assertEqual(pos.iloc[-1], -1)

    def test_signal_arrays(self):
        pos = signals_to_position(self.df, self.signals)
        le, lx, se, sx = position_to_signal_arrays(pos)
        self.assertEqual(int(le.sum()), 1)
        self.assertEqual(int(se.sum()), 1)
        self.assertTrue(le.iloc[6])
        self.assertTrue(lx.iloc[26])
        self.assertTrue(se.iloc[41])

    def test_engine_fallback(self):
        r = validate_with_engine(self.df, self.signals)
        self.assertEqual(r["backend"], "engine")
        self.assertGreaterEqual(r["n_trades"], 1)
        self.assertIn("buy_hold_return", r)

    def test_load_master_portfolio_format(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump([{"ticker": "NVDA", "decision": "SHORT", "confidence": 0.7}], f)
            path = f.name
        try:
            m = load_signals(path)
        finally:
            os.unlink(path)
        self.assertEqual(m["NVDA"][0]["direction"], -1)


if __name__ == "__main__":
    unittest.main()
