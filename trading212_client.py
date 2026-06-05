"""
Trading 212 API client — fetch the current portfolio (issue #3, step 1).

Reads open equity positions from the Trading 212 public API. The API key is
loaded (in priority order) from:
  1. the ``api_key`` argument
  2. the ``TRADING212_API_KEY`` environment variable
  3. ``../trading212_key.txt`` (one folder above the repo — same convention as
     the Gemini / Telegram secrets, keeping keys out of version control)

Use ``demo=True`` (or env ``TRADING212_DEMO=1``) to hit the practice account.

⚠️  ETF caveat (issue #3, step 2)
--------------------------------
Trading 212 returns an ETF (e.g. ``VUAG_l_EQ`` / S&P 500) as a *single*
position with one ticker and price. The decision pipeline therefore analyses
the ETF's **own** price series — its surface OHLC — not the 500 underlying
constituents. Treat ETF signals as a view on the fund's price action, not a
bottom-up aggregation of its holdings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests

LIVE_BASE = "https://live.trading212.com/api/v0"
DEMO_BASE = "https://demo.trading212.com/api/v0"


# ─────────────────────────────────────────────────────────────────────────────
#  Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    ticker: str                 # raw T212 ticker, e.g. "AAPL_US_EQ"
    quantity: float
    average_price: float
    current_price: float
    ppl: float                  # profit/loss in account currency
    yf_symbol: str              # best-effort yfinance symbol, e.g. "AAPL"
    is_etf: bool

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price


# Common ETF tickers (extend as needed) — used to flag the ETF caveat.
_KNOWN_ETF_HINTS = ("VUSA", "VUAG", "VWRL", "VWRP", "SPY", "VOO", "QQQ", "CSPX",
                    "EQQQ", "IWDA", "VUKE", "ISF", "VFEM", "AGGG")


def t212_to_yfinance(ticker: str) -> str:
    """
    Best-effort conversion of a Trading 212 ticker to a yfinance symbol.

    Examples:
      AAPL_US_EQ  -> AAPL
      TSLA_US_EQ  -> TSLA
      VUSA_l_EQ   -> VUSA.L   (London-listed)
      VUAGl_EQ    -> VUAG.L
    Falls back to the leading alphanumeric root if the pattern is unknown.
    """
    t = ticker.strip()
    root = t.split("_")[0]
    # US equities: AAPL_US_EQ -> AAPL
    if "_US_" in t or t.endswith("_US_EQ"):
        return root
    # London listings carry a lowercase 'l' segment
    if "_l_" in t or t.lower().endswith("l_eq") or t.endswith("_LON_EQ"):
        return f"{root}.L"
    return root


def _looks_like_etf(yf_symbol: str) -> bool:
    base = yf_symbol.split(".")[0].upper()
    return any(base.startswith(h) for h in _KNOWN_ETF_HINTS)


# ─────────────────────────────────────────────────────────────────────────────
#  Client
# ─────────────────────────────────────────────────────────────────────────────

def _load_key(api_key: Optional[str]) -> Optional[str]:
    if api_key:
        return api_key
    env = os.environ.get("TRADING212_API_KEY")
    if env:
        return env
    key_path = Path(os.getcwd()).resolve().parent / "trading212_key.txt"
    if key_path.exists():
        return key_path.read_text().strip()
    return None


class Trading212Client:
    def __init__(self, api_key: Optional[str] = None, demo: Optional[bool] = None,
                 timeout: int = 15):
        self.api_key = _load_key(api_key)
        if demo is None:
            demo = os.environ.get("TRADING212_DEMO", "") in ("1", "true", "True")
        self.base = DEMO_BASE if demo else LIVE_BASE
        self.timeout = timeout

    def _headers(self) -> dict:
        if not self.api_key:
            raise RuntimeError(
                "No Trading 212 API key. Set TRADING212_API_KEY, pass api_key=, "
                "or create ../trading212_key.txt. (Or use mock_portfolio() to test.)"
            )
        return {"Authorization": self.api_key}

    def get_portfolio(self) -> List[Position]:
        """GET /equity/portfolio → list of open Position objects."""
        resp = requests.get(f"{self.base}/equity/portfolio",
                            headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return [self._parse_position(p) for p in resp.json()]

    def get_cash(self) -> dict:
        """GET /equity/account/cash → raw cash dict (free/total/invested...)."""
        resp = requests.get(f"{self.base}/equity/account/cash",
                            headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_position(p: dict) -> Position:
        ticker = p.get("ticker", "")
        yf = t212_to_yfinance(ticker)
        return Position(
            ticker=ticker,
            quantity=float(p.get("quantity", 0.0)),
            average_price=float(p.get("averagePrice", 0.0)),
            current_price=float(p.get("currentPrice", 0.0)),
            ppl=float(p.get("ppl", 0.0)),
            yf_symbol=yf,
            is_etf=_looks_like_etf(yf),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Mock (for testing without a live key)
# ─────────────────────────────────────────────────────────────────────────────

def mock_portfolio() -> List[Position]:
    """A representative portfolio (incl. an ETF) for dry-run / tests."""
    raw = [
        {"ticker": "AAPL_US_EQ", "quantity": 10, "averagePrice": 180.0, "currentPrice": 195.0, "ppl": 150.0},
        {"ticker": "NVDA_US_EQ", "quantity": 5,  "averagePrice": 850.0, "currentPrice": 820.0, "ppl": -150.0},
        {"ticker": "VUSA_l_EQ",  "quantity": 20, "averagePrice": 78.0,  "currentPrice": 83.0,  "ppl": 100.0},
    ]
    return [Trading212Client._parse_position(p) for p in raw]
