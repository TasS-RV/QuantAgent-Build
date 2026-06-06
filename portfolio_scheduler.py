# ── MOVED ──────────────────────────────────────────────────────────────────────
# This module now lives at  data_exchange/portfolio_scheduler.py
# This stub re-exports everything for backwards compatibility.
from data_exchange.portfolio_scheduler import *  # noqa: F401, F403
from data_exchange.portfolio_scheduler import (  # noqa: F401
    Holding, holdings_from_t212, holdings_from_watchlist,
    analyse_holdings, format_telegram_message, send_telegram,
    run_once, run_scheduler, main,
)

if __name__ == "__main__":
    main()
