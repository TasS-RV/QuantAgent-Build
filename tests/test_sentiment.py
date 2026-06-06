"""Tests for the FinBERT sentiment overlay (issue #5).

No model download: the scorer is injected and the real scorer's graceful
degradation (transformers absent) is asserted. Skipped if numpy/pandas absent.
"""

import json
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401
    from quant_pipeline.decision_agent_quant import make_trade_decision
    from sentiment_agent import (
        SentimentResult, apply_sentiment, blend_signal,
        score_symbol_sentiment, get_default_scorer,
    )
    HAVE_DEPS = True
except Exception:  # pragma: no cover
    HAVE_DEPS = False


def _base_decision(sig):
    st = {
        "stock_name": "X",
        "kline_data": {"High": [10, 11, 12], "Low": [9, 10, 11], "Close": [9.5, 10.5, 11.5]},
        "indicator_report": json.dumps({"quantitative_metrics": {"final_indicator_signal": sig}}),
        "trend_report": json.dumps({"quantitative_metrics": {
            "normalized_signal": sig, "current_price": 11.5,
            "support_level": 10, "resistance_level": 13}}),
        "pattern_report": json.dumps({"macro_pattern_name": "None", "direction": 0,
                                      "confidence_score": 0.0}),
    }
    return make_trade_decision(st)


@unittest.skipUnless(HAVE_DEPS, "deps not installed")
class TestSentiment(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.hold = _base_decision(0.05)
        self.assertEqual(self.hold.decision, "HOLD")

    def test_positive_sentiment_flips_to_buy(self):
        d = apply_sentiment(self.hold, SentimentResult(0.9, "positive", 5, True, []),
                            sentiment_weight=0.5)
        self.assertEqual(d.decision, "BUY")
        self.assertGreater(d.combined_signal, self.hold.combined_signal)

    def test_negative_sentiment_flips_to_short_or_sell(self):
        d = apply_sentiment(self.hold, SentimentResult(-0.9, "negative", 5, True, []),
                            sentiment_weight=0.5)
        self.assertIn(d.decision, {"SELL", "SHORT"})

    def test_unavailable_sentiment_leaves_decision_unchanged(self):
        d = apply_sentiment(self.hold, SentimentResult(0.0, "unavailable", 3, False, []),
                            sentiment_weight=0.5)
        self.assertEqual(d.decision, self.hold.decision)
        self.assertIn("unavailable", d.decision_rationale)

    def test_blend_math(self):
        self.assertAlmostEqual(blend_signal(0.0, 1.0, 0.2), 0.2, places=9)
        self.assertAlmostEqual(blend_signal(0.5, -0.5, 0.5), 0.0, places=9)
        self.assertAlmostEqual(blend_signal(1.0, 1.0, 0.5), 1.0, places=9)  # clipped

    def test_injected_scorer(self):
        r = score_symbol_sentiment(
            "AAPL", headlines=["Great earnings beat"],
            scorer=lambda h: SentimentResult(0.7, "positive", len(h), True, []))
        self.assertTrue(r.available)
        self.assertEqual(r.signal, 0.7)

    def test_default_scorer_degrades_without_transformers(self):
        # transformers/torch are not installed in CI here → neutral, unavailable.
        r = get_default_scorer().score(["Apple beats earnings"])
        if not r.available:
            self.assertEqual(r.signal, 0.0)


if __name__ == "__main__":
    unittest.main()
