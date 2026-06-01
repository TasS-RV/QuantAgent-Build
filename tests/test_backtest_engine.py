"""Unit tests for the deterministic backtest engine (backtest_engine.py).

Pure math — no LLM, no network. Skipped automatically if numpy/pandas are
unavailable in the environment.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
    import pandas as pd
    from backtest_engine import (
        Signal, SimConfig, simulate, apply_overlays, sma, atr,
        run_with_baselines, buy_hold_return,
    )
    HAVE_DEPS = True
except Exception:  # pragma: no cover
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy/pandas not installed")
class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        n = 300
        close = np.linspace(50, 150, n) + np.sin(np.arange(n) / 5) * 2
        self.df = pd.DataFrame({
            "Datetime": pd.date_range("2024-01-01", periods=n, freq="D"),
            "Open": close - 0.1, "High": close + 1.0, "Low": close - 1.0, "Close": close,
        })
        self.cfg = SimConfig(periods_per_year=52)

    def test_hold_produces_no_trade(self):
        r = simulate("T", self.df, [Signal(210, 0, 1.0)], self.cfg)
        self.assertEqual(r["metrics"]["n_trades"], 0)

    def test_trend_filter_blocks_short_in_uptrend(self):
        close, sma_s = self.df["Close"], sma(self.df["Close"], self.cfg.sma_window)
        self.assertEqual(apply_overlays(-1, 1.0, 250, close, sma_s, self.cfg), 0)
        self.assertEqual(apply_overlays(1, 1.0, 250, close, sma_s, self.cfg), 1)

    def test_confidence_gate(self):
        cfg = SimConfig(confidence_gate=0.75, trend_filter=False)
        close, sma_s = self.df["Close"], sma(self.df["Close"], cfg.sma_window)
        self.assertEqual(apply_overlays(1, 0.5, 250, close, sma_s, cfg), 0)
        self.assertEqual(apply_overlays(1, 0.9, 250, close, sma_s, cfg), 1)

    def test_costs_reduce_pnl(self):
        nocost = SimConfig(commission=0, slippage=0, trend_filter=False, use_atr_stops=False)
        cost = SimConfig(commission=0.001, slippage=0.001, trend_filter=False, use_atr_stops=False)
        sig = [Signal(100, 1, 1.0), Signal(120, 1, 1.0)]
        a = simulate("T", self.df, sig, nocost)["trades"][0]
        b = simulate("T", self.df, sig, cost)["trades"][0]
        self.assertLess(b.pnl_pct, a.pnl_pct)
        self.assertAlmostEqual(b.cost_pct, 0.004, places=9)  # 2*(comm+slip)

    def test_atr_stop_exit_on_crash(self):
        crash = np.concatenate([np.full(50, 100.0), np.linspace(100, 60, 50)])
        cdf = pd.DataFrame({
            "Datetime": pd.date_range("2024-01-01", periods=100, freq="D"),
            "Open": crash, "High": crash + 0.5, "Low": crash - 0.5, "Close": crash,
        })
        cfg = SimConfig(trend_filter=False, use_atr_stops=True, atr_mult=1.5)
        t = simulate("C", cdf, [Signal(49, 1, 1.0, 2.0)], cfg)["trades"][0]
        self.assertEqual(t.exit_reason, "stop")
        self.assertLess(t.pnl_pct, 0)

    def test_baselines_present(self):
        reb = list(range(200, len(self.df) - 1, 5))
        agent = [Signal(i, 1, 0.9, 2.0) for i in reb]
        bl = run_with_baselines("T", self.df, agent, reb, self.cfg)
        for k in ("agent", "always_long", "sma_trend_follower", "random", "buy_hold_return"):
            self.assertIn(k, bl)

    def test_buy_hold_return(self):
        r = buy_hold_return(self.df, 0)
        self.assertGreater(r, 0)  # synthetic series rises 50 -> ~150

    def test_trades_frame_satisfies_visualize_contract(self):
        # visualize.py builds equity from decision_date/exit_date/pnl_pct columns
        # of all_trades.json — guard that SimTrade keeps providing them.
        from backtest_engine import trades_to_frame
        reb = list(range(200, len(self.df) - 1, 10))
        cfg = SimConfig(trend_filter=False, periods_per_year=52)
        trades = simulate("T", self.df, [Signal(i, 1, 0.9, 2.0) for i in reb], cfg)["trades"]
        tf = trades_to_frame(trades)
        for col in ("decision_date", "exit_date", "pnl_pct"):
            self.assertIn(col, tf.columns)


if __name__ == "__main__":
    unittest.main()
