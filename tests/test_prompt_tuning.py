"""Tests for the prompt-tuning harness (prompt_library + prompt_tuning).
Uses a fake runner — ZERO LLM calls."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import prompt_library as PL
import prompt_tuning as PT
import cost_guard as cg


class TestPromptLibrary(unittest.TestCase):
    def tearDown(self):
        PL.clear_active()

    def test_catalog_has_baselines(self):
        for agent in ("indicator", "pattern", "trend"):
            self.assertIn("baseline", PL.variant_names(agent))

    def test_get_prompt_default_when_unset(self):
        PL.clear_active()
        self.assertEqual(PL.get_prompt("indicator", "DEFAULT"), "DEFAULT")

    def test_set_active_overrides(self):
        PL.set_active("trend", "trend_following")
        self.assertNotEqual(PL.get_prompt("trend", "DEFAULT"), "DEFAULT")
        self.assertIn("trend-following", PL.get_prompt("trend", "").lower())

    def test_unknown_variant_raises(self):
        with self.assertRaises(KeyError):
            PL.set_active("trend", "nope")

    def test_all_combos_cartesian(self):
        combos = PL.all_combos(["trend", "pattern"])
        self.assertEqual(len(combos), len(PL.variant_names("trend")) * len(PL.variant_names("pattern")))
        self.assertTrue(all("trend" in c and "pattern" in c for c in combos))


class TestRunTuning(unittest.TestCase):
    def setUp(self):
        cg.reset_guard()

    def tearDown(self):
        cg.reset_guard()
        PL.clear_active()

    def test_run_tuning_records_and_sets_active(self):
        seen = []

        def fake_run(combo):
            # the active prompt must reflect the combo while run_fn executes
            seen.append(PL.get_prompt("trend", "DEFAULT"))
            return {"mean_agent_return": 0.1, "mean_sharpe": 1.0, "mean_win_rate": 0.5}

        combos = PL.all_combos(["trend"])
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "RESULTS_LOG.md"
            results = PT.run_tuning(combos, fake_run, max_usd=None, results_log=log)
            self.assertEqual(len(results), len(combos))
            self.assertTrue(log.exists())
            self.assertIn("Prompt-tuning run", log.read_text())
        # active prompt was set during runs and cleared after
        self.assertEqual(len(seen), len(combos))
        self.assertEqual(PL.get_prompt("trend", "DEFAULT"), "DEFAULT")

    def test_cost_cap_halts_sweep(self):
        def expensive_run(combo):
            # simulate a real call that records heavy usage and trips the cap
            cg.get_guard().record(2_000_000, 2_000_000, "gpt-4o")
            return {"mean_sharpe": 0.0}

        combos = PL.all_combos(["trend"])     # several combos
        results = PT.run_tuning(combos, expensive_run, max_usd=0.01, results_log=None)
        # tripped on the first run -> stops well before enumerating all combos
        self.assertLess(len(results), len(combos))


if __name__ == "__main__":
    unittest.main()
