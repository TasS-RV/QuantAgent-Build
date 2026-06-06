"""
Prompt-tuning harness (issue #8, Objective 2 — "let the LLM play").

Enumerates combinations of agent prompt variants (prompt_library.py), runs a
short backtest for each under the API cost stop-loss, and records the metrics so
the best-performing prompt "psychology" can be chosen.

SAFETY / COST
─────────────
Every run goes through cost_guard, so the loop halts the instant the dollar /
token / call cap is hit — it can never silently burn credits. Set a tight cap
(e.g. --max-usd 1.0) and a low temperature (deterministic, comparable runs).
The default runner makes real LLM calls; pass your own run_fn (or use the tests'
fake) to exercise the harness with zero spend.

Usage
─────
    # dry-run the plan (no LLM calls) — lists the combos that WOULD run
    python prompt_tuning.py --dry-run

    # real tuning under a $1 cap on 1 symbol, daily, short window
    python prompt_tuning.py --symbols NVDA --max-usd 1.0 --provider google
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import prompt_library as PL

RESULTS_LOG = Path(__file__).resolve().parent / "backtesting_operations" / "RESULTS_LOG.md"


def default_backtest_runner(combo: Dict[str, str], *, symbols: List[str], provider: str,
                            start_date: Optional[str], cadence: str, window_bars: int) -> dict:
    """Run one LLM backtest with the active prompt combo. Returns summary metrics.

    Makes real LLM calls (gated by the cost guard). Lazy-imported so the harness
    and its tests don't require the full LLM stack.
    """
    import asyncio
    from backtest import backtest_universe, build_llm_config  # noqa: F401
    from backtest_engine import SimConfig

    out = asyncio.run(backtest_universe(
        symbols=symbols, start_date=start_date, end_date=None, period="6mo",
        cadence=cadence, window_bars=window_bars, concurrency=2,
        out_dir=f"backtest_results/tune_{'_'.join(combo.values())}",
        cfg=SimConfig(), use_llm=True, llm_provider=provider,
    ))
    summ = out["summary"]
    return {
        "mean_agent_return": float(summ["agent_total_return"].mean()) if len(summ) else 0.0,
        "mean_sharpe": float(summ["sharpe_annual"].mean()) if len(summ) else 0.0,
        "mean_win_rate": float(summ["win_rate"].mean()) if len(summ) else 0.0,
        "cost_usd": out.get("cost", {}).get("cost_usd", 0.0),
    }


def run_tuning(
    combos: List[Dict[str, str]],
    run_fn: Callable[[Dict[str, str]], dict],
    max_usd: Optional[float] = 1.0,
    max_calls: Optional[int] = None,
    results_log: Optional[Path] = RESULTS_LOG,
) -> List[dict]:
    """
    Run each prompt combo through run_fn under the cost stop-loss. Stops early if
    the cap trips. Returns a list of {combo, metrics} and appends a markdown
    table to results_log.
    """
    from cost_guard import configure, get_guard, CostLimitExceeded
    configure(max_usd=max_usd, max_calls=max_calls)

    results: List[dict] = []
    for combo in combos:
        PL.set_active_combo(combo)
        guard = get_guard()
        try:
            metrics = run_fn(combo)
        except CostLimitExceeded as e:
            print(f"[cost stop-loss] halting prompt tuning: {e}")
            break
        row = {"combo": combo, "metrics": metrics,
               "spend_usd": round(guard.cost_usd, 4)}
        results.append(row)
        print(f"  {combo}  ->  {metrics}")
    PL.clear_active()

    if results_log is not None and results:
        _append_results_log(results_log, results)
    return results


def _append_results_log(path: Path, results: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["", "## Prompt-tuning run", "",
             "| indicator | pattern | trend | return | sharpe | win | $spend |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        c, m = r["combo"], r["metrics"]
        lines.append(
            f"| {c.get('indicator','-')} | {c.get('pattern','-')} | {c.get('trend','-')} "
            f"| {m.get('mean_agent_return',0):.3f} | {m.get('mean_sharpe',0):.2f} "
            f"| {m.get('mean_win_rate',0):.2f} | {r.get('spend_usd',0)} |")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description="Prompt-tuning harness (issue #8 Obj2)")
    p.add_argument("--symbols", nargs="+", default=["NVDA"])
    p.add_argument("--provider", default="google")
    p.add_argument("--start", default=None)
    p.add_argument("--cadence", default="W-FRI")
    p.add_argument("--window-bars", type=int, default=60)
    p.add_argument("--max-usd", type=float, default=1.0, help="Cost stop-loss for the whole sweep")
    p.add_argument("--max-calls", type=int, default=None)
    p.add_argument("--agents", nargs="+", default=None,
                   help="Subset of agents to tune (default: indicator pattern trend)")
    p.add_argument("--dry-run", action="store_true", help="List combos; make no LLM calls")
    args = p.parse_args()

    combos = PL.all_combos(args.agents)
    print(f"Prompt-tuning: {len(combos)} combinations across {args.agents or list(PL.PROMPT_VARIANTS)}")
    if args.dry_run:
        for c in combos:
            print("  ", c)
        print(f"\n(dry run — no LLM calls. Real run would be capped at ${args.max_usd}.)")
        return

    def runner(combo):
        return default_backtest_runner(
            combo, symbols=args.symbols, provider=args.provider, start_date=args.start,
            cadence=args.cadence, window_bars=args.window_bars)

    results = run_tuning(combos, runner, max_usd=args.max_usd, max_calls=args.max_calls)
    print(f"\nCompleted {len(results)}/{len(combos)} combos. Results appended to {RESULTS_LOG}")
    best = max(results, key=lambda r: r["metrics"].get("mean_sharpe", 0), default=None)
    if best:
        print(f"Best by Sharpe: {best['combo']} -> {best['metrics']}")


if __name__ == "__main__":
    main()
