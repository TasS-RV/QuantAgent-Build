"""Tests for the classic strategy lab (strategy_lab.py) + its reuse of
backtest_engine.SimTrade after the position_size reconciliation. No API/network."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
    import pandas as pd
    from backtest_engine import SimConfig, SimTrade
    import strategy_lab as SL
    HAVE_DEPS = True
except Exception:  # pragma: no cover
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy/pandas not installed")
class TestStrategyLab(unittest.TestCase):
    def setUp(self):
        n = 400
        # trending series with a pullback so breakout/momentum strategies fire
        close = np.concatenate([np.linspace(50, 150, 300), np.linspace(150, 120, 100)])
        self.df = pd.DataFrame({
            "Datetime": pd.date_range("2022-01-01", periods=n, freq="D"),
            "Open": close - 0.1, "High": close + 1.0, "Low": close - 1.0, "Close": close,
        })

    def test_simtrade_default_position_size(self):
        # The reconciliation: SimTrade is constructible without position_size.
        t = SimTrade(symbol="X", decision_date="d", direction=1, entry_date="e",
                     entry_price=1.0, exit_date="x", exit_price=1.1, exit_reason="regime",
                     confidence=1.0, holding_days=3, gross_pnl_pct=0.1, cost_pct=0.0,
                     pnl_pct=0.1)
        self.assertEqual(t.position_size, 1.0)

    def test_donchian_target_and_simulate(self):
        target = SL.donchian_target(self.df, entry_n=20, exit_n=10, allow_short=False)
        self.assertEqual(len(target), len(self.df))
        trades = SL.simulate_target("X", self.df, target, SimConfig(trend_filter=False),
                                    exit_mode="trailing")
        self.assertIsInstance(trades, list)
        if trades:
            self.assertIsInstance(trades[0], SimTrade)
            self.assertIn(trades[0].direction, (-1, 1))

    def test_tsmom_target_long_flat(self):
        target = SL.tsmom_target(self.df, lookback=60, allow_short=False)
        self.assertTrue(set(np.unique(target[~np.isnan(target)])).issubset({0.0, 1.0}))

    def test_ma_cross_runs(self):
        target = SL.ma_cross_target(self.df, fast=20, slow=100, allow_short=True)
        trades = SL.simulate_target("X", self.df, target, SimConfig(trend_filter=False),
                                    exit_mode="regime")
        self.assertIsInstance(trades, list)


if __name__ == "__main__":
    unittest.main()
