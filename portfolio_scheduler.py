"""
Scheduled portfolio analysis → Telegram (issue #3, steps 2 & 3).

Every `interval` seconds (default 1 hour) this:
  1. pulls the current holdings (Trading 212 portfolio, a mock, or a watchlist),
  2. fetches recent OHLC for each holding from Yahoo Finance,
  3. runs the deterministic quant decision (quant_signal.compute_quant_decision —
     no LLM, so it's free and repeatable), passing the position's average price
     as the entry so the protective-SELL override can fire, and
  4. pushes a consolidated BUY/SELL/HOLD message to Telegram.

Design note: data fetch and message send are injected so the core
(`analyse_holdings` / `run_once`) is unit-testable without yfinance or Telegram.

Usage
-----
    # one pass over the live Trading 212 portfolio, send to Telegram
    python portfolio_scheduler.py --source t212 --once

    # hourly loop over a fixed watchlist, print only (no Telegram)
    python portfolio_scheduler.py --source watchlist --symbols AAPL NVDA BTC-USD \
        --interval 3600 --no-telegram

    # dry run with the mock portfolio (no keys needed)
    python portfolio_scheduler.py --source mock --once --no-telegram
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class Holding:
    symbol: str                     # yfinance symbol
    entry_price: Optional[float]    # average cost, or None if just a watchlist item
    quantity: float = 0.0
    is_etf: bool = False


# ─────────────────────────────────────────────────────────────────────────────
#  Holdings sources
# ─────────────────────────────────────────────────────────────────────────────

def holdings_from_t212(use_mock: bool = False) -> List[Holding]:
    from trading212_client import Trading212Client, mock_portfolio
    positions = mock_portfolio() if use_mock else Trading212Client().get_portfolio()
    return [Holding(symbol=p.yf_symbol, entry_price=p.average_price,
                    quantity=p.quantity, is_etf=p.is_etf) for p in positions]


def holdings_from_watchlist(symbols: List[str]) -> List[Holding]:
    return [Holding(symbol=s, entry_price=None) for s in symbols]


# ─────────────────────────────────────────────────────────────────────────────
#  Data fetch (default: yfinance) — injectable
# ─────────────────────────────────────────────────────────────────────────────

def fetch_kline_yf(symbol: str, lookback_days: int = 120, interval: str = "1d") -> dict:
    import yfinance as yf
    period = f"{max(lookback_days, 60)}d"
    raw = yf.download(symbol, period=period, interval=interval,
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"No data for {symbol}")
    import pandas as pd
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    dt_col = "Datetime" if "Datetime" in raw.columns else raw.columns[0]
    return {
        "Datetime": pd.to_datetime(raw[dt_col]).dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "Open": raw["Open"].astype(float).tolist(),
        "High": raw["High"].astype(float).tolist(),
        "Low": raw["Low"].astype(float).tolist(),
        "Close": raw["Close"].astype(float).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Core analysis (testable)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_holdings(
    holdings: List[Holding],
    fetch_fn: Callable[[str, int, str], dict] = fetch_kline_yf,
    lookback_days: int = 120,
    interval: str = "1d",
    allow_short: bool = True,
    sentiment_weight: float = 0.0,
    sentiment_scorer: Optional[Callable] = None,
) -> List[dict]:
    """
    Run the deterministic decision for each holding. Returns a list of dicts.

    If sentiment_weight > 0, a FinBERT sentiment overlay (issue #5) is blended
    into each decision. Degrades gracefully (no change) if transformers/torch
    or news are unavailable. `sentiment_scorer` is injectable for tests.
    """
    from quant_signal import compute_quant_decision
    results = []
    for h in holdings:
        try:
            kline = fetch_fn(h.symbol, lookback_days, interval)
            dec = compute_quant_decision(h.symbol, kline, entry_price=h.entry_price,
                                         allow_short=allow_short)
            if sentiment_weight > 0:
                from sentiment_agent import apply_sentiment, score_symbol_sentiment
                sent = score_symbol_sentiment(h.symbol, scorer=sentiment_scorer)
                dec = apply_sentiment(dec, sent, sentiment_weight=sentiment_weight,
                                      allow_short=allow_short)
            results.append({
                "symbol": h.symbol, "is_etf": h.is_etf, "decision": dec.decision,
                "combined_signal": dec.combined_signal, "current_price": dec.current_price,
                "entry_price": dec.entry_price, "unrealized_pnl_pct": dec.unrealized_pnl_pct,
                "target_price": dec.target_price, "stop_loss": dec.stop_loss,
                "risk_reward_ratio": dec.risk_reward_ratio, "rationale": dec.decision_rationale,
                "error": None,
            })
        except Exception as e:
            results.append({"symbol": h.symbol, "is_etf": h.is_etf, "decision": "ERROR",
                            "error": f"{type(e).__name__}: {e}"})
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Telegram formatting + send (injectable)
# ─────────────────────────────────────────────────────────────────────────────

_ICON = {"BUY": "🟢", "SELL": "🔴", "SHORT": "🟠", "HOLD": "⚪", "ERROR": "⚠️"}


def format_telegram_message(results: List[dict]) -> str:
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"📊 *AlgoEdge Portfolio Update* — {ts}", ""]
    for r in results:
        if r["decision"] == "ERROR":
            lines.append(f"{_ICON['ERROR']} *{r['symbol']}* — {r['error']}")
            continue
        etf = " _(ETF: surface price only)_" if r.get("is_etf") else ""
        pnl = f" | PnL {r['unrealized_pnl_pct']:+.1f}%" if r.get("unrealized_pnl_pct") is not None else ""
        lines.append(
            f"{_ICON.get(r['decision'], '•')} *{r['symbol']}* — *{r['decision']}* "
            f"(sig {r['combined_signal']:+.2f}){etf}\n"
            f"   px {r['current_price']} → 🎯 {r['target_price']} | 🛡️ {r['stop_loss']} "
            f"| R:R {r['risk_reward_ratio']}{pnl}"
        )
    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    """Send a Markdown message using the bot credentials from Tele_bot."""
    import requests
    from Tele_bot import load_telegram_keys
    keys = load_telegram_keys()
    if not keys:
        print("No telegram_keys.json — skipping send.")
        return False
    url = f"https://api.telegram.org/bot{keys.get('bot_token')}/sendMessage"
    resp = requests.post(url, json={"chat_id": keys.get("chat_id"),
                                    "text": message, "parse_mode": "Markdown"},
                         timeout=15)
    if resp.status_code == 200:
        print("🚀 Portfolio update pushed to Telegram.")
        return True
    print(f"Telegram send failed: {resp.text}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_once(holdings: List[Holding], fetch_fn=fetch_kline_yf,
             send_fn: Optional[Callable[[str], bool]] = send_telegram,
             lookback_days: int = 120, interval: str = "1d",
             allow_short: bool = True, sentiment_weight: float = 0.0) -> List[dict]:
    results = analyse_holdings(holdings, fetch_fn, lookback_days, interval, allow_short,
                               sentiment_weight=sentiment_weight)
    msg = format_telegram_message(results)
    print(msg)
    if send_fn is not None:
        send_fn(msg)
    return results


def run_scheduler(get_holdings: Callable[[], List[Holding]], interval_sec: int = 3600,
                  send_fn: Optional[Callable[[str], bool]] = send_telegram, **kwargs):
    """Loop forever, re-pulling holdings each cycle (positions can change)."""
    print(f"⏱  Scheduler started — every {interval_sec}s. Ctrl-C to stop.")
    while True:
        try:
            run_once(get_holdings(), send_fn=send_fn, **kwargs)
        except Exception as e:
            print(f"Cycle error: {type(e).__name__}: {e}")
        time.sleep(interval_sec)


def main():
    p = argparse.ArgumentParser(description="Scheduled portfolio analysis → Telegram (issue #3).")
    p.add_argument("--source", choices=["t212", "mock", "watchlist"], default="mock")
    p.add_argument("--symbols", nargs="+", default=["AAPL", "NVDA", "MSFT"],
                   help="Watchlist symbols (when --source watchlist)")
    p.add_argument("--interval", type=int, default=3600, help="Seconds between cycles")
    p.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    p.add_argument("--no-telegram", action="store_true", help="Print only; don't send")
    p.add_argument("--no-short", action="store_true", help="Long-only recommendations")
    p.add_argument("--lookback-days", type=int, default=120)
    p.add_argument("--interval-bars", default="1d", help="yfinance bar interval (1d, 1h, ...)")
    p.add_argument("--sentiment-weight", type=float, default=0.0,
                   help="Blend FinBERT news sentiment into decisions (0 disables; e.g. 0.2)")
    args = p.parse_args()

    def get_holdings() -> List[Holding]:
        if args.source == "t212":
            return holdings_from_t212(use_mock=False)
        if args.source == "mock":
            return holdings_from_t212(use_mock=True)
        return holdings_from_watchlist(args.symbols)

    send_fn = None if args.no_telegram else send_telegram
    kwargs = dict(lookback_days=args.lookback_days, interval=args.interval_bars,
                  allow_short=not args.no_short, sentiment_weight=args.sentiment_weight)

    if args.once:
        run_once(get_holdings(), send_fn=send_fn, **kwargs)
    else:
        run_scheduler(get_holdings, interval_sec=args.interval, send_fn=send_fn, **kwargs)


if __name__ == "__main__":
    main()
