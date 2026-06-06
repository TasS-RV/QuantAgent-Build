"""Tests for the API cost stop-loss (cost_guard.py). Pure stdlib, no API calls."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cost_guard as cg


class TestCostGuard(unittest.TestCase):
    def setUp(self):
        for k in ("QUANT_MAX_USD", "QUANT_MAX_TOKENS", "QUANT_MAX_CALLS", "QUANT_KILL_SWITCH"):
            os.environ.pop(k, None)
        cg.reset_guard()

    def tearDown(self):
        cg.reset_guard()

    def test_call_cap_preflight(self):
        g = cg.configure(max_usd=None, max_tokens=None, max_calls=3)
        for _ in range(3):
            g.preflight("gpt-4o-mini")
            g.record(100, 50, "gpt-4o-mini")
        with self.assertRaises(cg.CostLimitExceeded):
            g.preflight("gpt-4o-mini")

    def test_dollar_preflight_blocks_before_spend(self):
        g = cg.configure(max_usd=0.001, max_tokens=None, max_calls=None)
        with self.assertRaises(cg.CostLimitExceeded):
            g.preflight("gpt-4o", est_input_tokens=100_000, est_output_tokens=50_000)
        self.assertEqual(g.calls, 0)        # nothing spent
        self.assertEqual(g.cost_usd, 0.0)

    def test_record_trips_stop_loss(self):
        g = cg.configure(max_usd=0.01, max_tokens=None, max_calls=None)
        with self.assertRaises(cg.CostLimitExceeded):
            g.record(1_000_000, 1_000_000, "gpt-4o")   # ~$12.50 >> $0.01
        self.assertGreater(g.cost_usd, 0.01)

    def test_token_cap(self):
        g = cg.configure(max_usd=None, max_tokens=1000, max_calls=None)
        g.preflight("gpt-4o-mini")
        with self.assertRaises(cg.CostLimitExceeded):
            g.record(800, 300, "gpt-4o-mini")         # 1100 > 1000 -> trips on record
        # already over; next preflight must also refuse
        with self.assertRaises(cg.CostLimitExceeded):
            g.preflight("gpt-4o-mini")

    def test_kill_switch(self):
        g = cg.configure(kill_switch=True)
        with self.assertRaises(cg.CostLimitExceeded):
            g.preflight("gpt-4o-mini")

    def test_pricing_prefix_match(self):
        g = cg.get_guard()
        self.assertAlmostEqual(g.estimate_cost("gpt-4o-mini", 1_000_000, 0), 0.15, places=6)
        self.assertAlmostEqual(g.estimate_cost("gpt-4o", 1_000_000, 0), 2.50, places=6)
        self.assertAlmostEqual(g.estimate_cost("gemini-3.1-flash-lite", 1_000_000, 0), 0.15, places=6)
        self.assertAlmostEqual(g.estimate_cost("claude-haiku-4-5-20251001", 1_000_000, 0), 0.80, places=6)
        # unknown model -> conservative fallback
        self.assertAlmostEqual(g.estimate_cost("totally-unknown", 1_000_000, 0), 1.00, places=6)

    def test_env_defaults(self):
        os.environ["QUANT_MAX_USD"] = "2.5"
        os.environ["QUANT_MAX_CALLS"] = "7"
        cg.reset_guard()
        g = cg.get_guard()
        self.assertEqual(g.max_usd, 2.5)
        self.assertEqual(g.max_calls, 7)

    def test_summary_and_remaining(self):
        g = cg.configure(max_usd=1.0, max_tokens=None, max_calls=None)
        g.record(1_000_000, 0, "gpt-4o-mini")          # $0.15
        s = g.summary()
        self.assertEqual(s["calls"], 1)
        self.assertAlmostEqual(s["cost_usd"], 0.15, places=4)
        self.assertAlmostEqual(g.remaining_usd(), 0.85, places=4)

    def test_default_caps_fail_safe(self):
        # With no config at all, hard defaults still apply (never unbounded).
        g = cg.get_guard()
        self.assertIsNotNone(g.max_usd)
        self.assertIsNotNone(g.max_calls)


if __name__ == "__main__":
    unittest.main()
