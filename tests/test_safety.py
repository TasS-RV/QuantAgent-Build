"""Tests for the trading safety rails (safety.py). Pure stdlib, no network."""

import importlib
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSafety(unittest.TestCase):
    def setUp(self):
        for k in ("QUANT_PAPER_TRADING", "QUANT_LIVE_TRADING", "QUANT_TRADING_KILL_SWITCH"):
            os.environ.pop(k, None)
        import safety
        importlib.reload(safety)
        self.safety = safety

    def tearDown(self):
        for k in ("QUANT_PAPER_TRADING", "QUANT_LIVE_TRADING", "QUANT_TRADING_KILL_SWITCH"):
            os.environ.pop(k, None)

    def test_default_is_paper_and_blocked(self):
        self.assertTrue(self.safety.paper_trading())
        self.assertFalse(self.safety.safe_to_live())
        with self.assertRaises(self.safety.LiveTradingBlocked):
            self.safety.require_live_trading()

    def test_paper_off_alone_still_blocked(self):
        # Leaving paper mode is NOT enough; live must be explicitly armed.
        os.environ["QUANT_PAPER_TRADING"] = "0"
        importlib.reload(self.safety)
        self.assertFalse(self.safety.safe_to_live())
        with self.assertRaises(self.safety.LiveTradingBlocked):
            self.safety.require_live_trading()

    def test_fully_armed_allows_live(self):
        os.environ["QUANT_PAPER_TRADING"] = "0"
        os.environ["QUANT_LIVE_TRADING"] = "1"
        importlib.reload(self.safety)
        self.assertTrue(self.safety.safe_to_live())
        self.safety.require_live_trading()  # must not raise

    def test_kill_switch_overrides_everything(self):
        os.environ["QUANT_PAPER_TRADING"] = "0"
        os.environ["QUANT_LIVE_TRADING"] = "1"
        os.environ["QUANT_TRADING_KILL_SWITCH"] = "1"
        importlib.reload(self.safety)
        self.assertFalse(self.safety.safe_to_live())
        with self.assertRaises(self.safety.LiveTradingBlocked):
            self.safety.require_live_trading()

    def test_t212_place_order_blocked_by_default(self):
        try:
            import requests  # noqa: F401
            from data_exchange.trading212_client import Trading212Client
        except Exception:
            self.skipTest("requests not installed")
        client = Trading212Client(api_key="x")
        with self.assertRaises(self.safety.LiveTradingBlocked):
            client.place_order("AAPL", 1)


if __name__ == "__main__":
    unittest.main()
