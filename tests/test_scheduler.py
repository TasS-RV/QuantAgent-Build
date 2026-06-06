"""Tests for Trading212 client, quant_signal, and the portfolio scheduler (issue #3).

No network: data fetch and Telegram send are injected. Skipped if numpy/pandas
(+ requests, for the T212 client import) are unavailable.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
    import pandas as pd  # noqa: F401
    import requests  # noqa: F401  (trading212_client imports it at module load)
    from data_exchange.trading212_client import t212_to_yfinance, mock_portfolio
    from quant_signal import compute_quant_decision
    import data_exchange.portfolio_scheduler as ps
    HAVE_DEPS = True
except Exception:  # pragma: no cover
    HAVE_DEPS = False


def _kline(n=80, start=100, stop=130):
    import numpy as np
    close = np.linspace(start, stop, n)
    return {
        "Datetime": [f"2024-01-{(i % 28) + 1:02d} 00:00:00" for i in range(n)],
        "Open": (close - 0.2).tolist(), "High": (close + 1).tolist(),
        "Low": (close - 1).tolist(), "Close": close.tolist(),
    }


@unittest.skipUnless(HAVE_DEPS, "deps not installed")
class TestTrading212(unittest.TestCase):
    def test_ticker_mapping(self):
        self.assertEqual(t212_to_yfinance("AAPL_US_EQ"), "AAPL")
        self.assertEqual(t212_to_yfinance("TSLA_US_EQ"), "TSLA")
        self.assertEqual(t212_to_yfinance("VUSA_l_EQ"), "VUSA.L")

    def test_mock_portfolio_flags_etf(self):
        positions = mock_portfolio()
        self.assertTrue(any(p.is_etf for p in positions))
        aapl = next(p for p in positions if p.yf_symbol == "AAPL")
        self.assertFalse(aapl.is_etf)


@unittest.skipUnless(HAVE_DEPS, "deps not installed")
class TestQuantSignal(unittest.TestCase):
    def test_decision_runs_trend_only(self):
        warnings.simplefilter("ignore")
        dec = compute_quant_decision("AAPL", _kline(), entry_price=110.0)
        self.assertIn(dec.decision, {"BUY", "SELL", "SHORT", "HOLD"})
        self.assertAlmostEqual(dec.unrealized_pnl_pct, (130 - 110) / 110 * 100, places=1)


@unittest.skipUnless(HAVE_DEPS, "deps not installed")
class TestScheduler(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.kline = _kline()

    def _fetch(self, symbol, lookback_days=120, interval="1d"):
        return self.kline

    def test_run_once_builds_message_with_etf_caveat(self):
        captured = {}
        holdings = [ps.Holding("AAPL", 110.0, 10, False),
                    ps.Holding("VUSA.L", 78.0, 20, True)]
        results = ps.run_once(holdings, fetch_fn=self._fetch,
                              send_fn=lambda m: captured.setdefault("msg", m) or True)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["error"] is None for r in results))
        self.assertIn("AlgoEdge Portfolio Update", captured["msg"])
        self.assertIn("ETF: surface price only", captured["msg"])

    def test_fetch_error_is_captured(self):
        def bad(*a, **k):
            raise RuntimeError("no data")
        res = ps.analyse_holdings([ps.Holding("ZZZ", None)], fetch_fn=bad)
        self.assertEqual(res[0]["decision"], "ERROR")
        self.assertIsNotNone(res[0]["error"])


if __name__ == "__main__":
    unittest.main()
