# ── MOVED ──────────────────────────────────────────────────────────────────────
# This module now lives at  data_exchange/trading212_client.py
# This stub re-exports everything for backwards compatibility.
from data_exchange.trading212_client import *  # noqa: F401, F403
from data_exchange.trading212_client import (  # noqa: F401
    Position, Trading212Client, mock_portfolio, t212_to_yfinance,
    LIVE_BASE, DEMO_BASE,
)
