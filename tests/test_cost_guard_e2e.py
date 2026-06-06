"""End-to-end cost-guard test against a real LangChain invoke (FakeListChatModel).
No paid API. Skipped if langchain-core isn't installed."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cost_guard as cg

try:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    HAVE_LC = True
except Exception:  # pragma: no cover
    HAVE_LC = False


@unittest.skipUnless(HAVE_LC, "langchain-core not installed")
class TestCostGuardE2E(unittest.TestCase):
    def setUp(self):
        for k in ("QUANT_MAX_USD", "QUANT_MAX_TOKENS", "QUANT_MAX_CALLS", "QUANT_KILL_SWITCH"):
            os.environ.pop(k, None)
        cg.reset_guard()

    def tearDown(self):
        cg.reset_guard()

    def test_callback_records_invoke(self):
        g = cg.configure(max_usd=None, max_tokens=None, max_calls=10)
        llm = FakeListChatModel(responses=["ok"], callbacks=[cg.make_langchain_callback(g)])
        llm.invoke("hello")
        self.assertEqual(g.calls, 1)

    def test_kill_switch_blocks_invoke(self):
        g = cg.configure(kill_switch=True)
        llm = FakeListChatModel(responses=["never"], callbacks=[cg.make_langchain_callback(g)])
        with self.assertRaises(cg.CostLimitExceeded):
            llm.invoke("hello")
        self.assertEqual(g.calls, 0)   # blocked before the model ran

    def test_call_cap_halts_stream(self):
        g = cg.configure(max_usd=None, max_tokens=None, max_calls=2)
        llm = FakeListChatModel(responses=["a", "b", "c", "d"],
                                callbacks=[cg.make_langchain_callback(g)])
        ok = 0
        for _ in range(5):
            try:
                llm.invoke("x"); ok += 1
            except cg.CostLimitExceeded:
                break
        self.assertEqual(ok, 2)


if __name__ == "__main__":
    unittest.main()
