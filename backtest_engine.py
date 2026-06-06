"""
Deterministic backtest engine — pure math, no LLM, no network.

This module separates **trade simulation** (deterministic) from **signal
generation** (the expensive, non-deterministic LLM step in ``backtest.py``).
Given a list of dated signals and an OHLC price history, it applies the risk
overlays from the Backtest Improvement Roadmap (issue #1) and produces trades +
summary metrics that are fully reproducible and unit-testable.

Overlays implemented (roadmap Tier 1 & 2):
  * HOLD            — direction 0 signals are treated as cash (no trade)
  * Trend filter    — never SHORT above the 200-day SMA, never LONG below it
  * Confidence gate — drop signals whose confidence < ``confidence_gate``
  * Trading costs   — commission + slippage per side, plus short borrow
  * ATR stops/target— exit on first stop or target hit (intra-bar via High/Low),
                      target distance = risk_reward_ratio × stop distance

Baselines (roadmap Tier 4): buy & hold, always-long, 200-day SMA trend
follower, and a seeded random long/short — so agent performance can be judged
against trivial strategies rather than buy & hold alone.

Depends only on numpy + pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Inputs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    """A single dated trade decision from the agent (or a baseline)."""
    decision_idx: int          # bar index at which the decision is made
    direction: int             # +1 long, -1 short, 0 hold/cash
    confidence: float = 1.0    # [0, 1]
    risk_reward_ratio: float = 2.0
    date: Optional[str] = None
    note: str = ""


@dataclass
class SimConfig:
    """Backtest friction + risk overlay configuration."""
    # Costs (fractions of notional, per side unless noted)
    commission: float = 0.0002          # ~2 bps/side
    slippage: float = 0.0005            # ~5 bps/side
    borrow_annual: float = 0.01         # ~1%/yr short borrow (charged pro-rata)
    # Overlays
    trend_filter: bool = True
    sma_window: int = 200
    confidence_gate: float = 0.0        # 0 disables; e.g. 0.75 to gate
    use_atr_stops: bool = True
    atr_period: int = 14
    atr_mult: float = 1.5
    allow_short: bool = True
    # Position sizing: "full" = all-in (size 1.0); "confidence" = size scales with
    # the signal's confidence, capped by max_position (IMPROVEMENTS_todo §1).
    position_sizing: str = "full"        # "full" | "confidence"
    max_position: float = 1.0            # cap on per-trade exposure fraction
    # Annualization for Sharpe (set from rebalance cadence)
    periods_per_year: float = 52.0
    trading_days_per_year: int = 252


@dataclass
class SimTrade:
    symbol: str
    decision_date: str
    direction: int
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    exit_reason: str           # "stop" | "target" | "horizon"
    confidence: float
    holding_days: int
    gross_pnl_pct: float       # signed return before costs (per unit notional)
    cost_pct: float            # total round-trip cost as a fraction
    pnl_pct: float             # net signed return on portfolio (size-weighted)
    position_size: float = 1.0  # exposure fraction (1.0=all-in); default keeps
                                # older constructors (e.g. strategy_lab) working
    note: str = ""


# ─────────────────────────────────────────────────────────────────────────────
#  Indicators (deterministic)
# ─────────────────────────────────────────────────────────────────────────────

def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=1).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-style ATR using a simple rolling mean of True Range."""
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Overlays
# ─────────────────────────────────────────────────────────────────────────────

def apply_overlays(direction: int, confidence: float, idx: int,
                   close: pd.Series, sma_series: pd.Series, cfg: SimConfig) -> int:
    """Return the post-overlay direction (may be forced to 0 = HOLD/cash)."""
    if direction == 0:
        return 0
    # Confidence gate
    if cfg.confidence_gate > 0 and confidence < cfg.confidence_gate:
        return 0
    # Disallow shorts entirely if configured
    if direction == -1 and not cfg.allow_short:
        return 0
    # Trend filter: never fight the long-term SMA
    if cfg.trend_filter:
        price = float(close.iloc[idx])
        ref = float(sma_series.iloc[idx])
        if direction == -1 and price > ref:
            return 0   # don't short an uptrend
        if direction == 1 and price < ref:
            return 0   # don't long a downtrend
    return direction


# ─────────────────────────────────────────────────────────────────────────────
#  Simulation
# ─────────────────────────────────────────────────────────────────────────────

def _exit_via_stops(df: pd.DataFrame, direction: int, entry_idx: int,
                    horizon_idx: int, entry: float, atr_at_entry: float,
                    rr: float, cfg: SimConfig):
    """
    Walk bars (entry_idx+1 .. horizon_idx) and return (exit_idx, exit_price,
    reason). Stop is checked before target on the same bar (conservative).
    Falls back to horizon open if neither is hit.
    """
    stop_dist = atr_at_entry * cfg.atr_mult
    if stop_dist <= 0:
        stop_dist = entry * 0.02
    target_dist = stop_dist * rr
    if direction == 1:
        stop = entry - stop_dist
        target = entry + target_dist
    else:
        stop = entry + stop_dist
        target = entry - target_dist

    for j in range(entry_idx, min(horizon_idx, len(df) - 1) + 1):
        hi = float(df["High"].iloc[j])
        lo = float(df["Low"].iloc[j])
        if direction == 1:
            if lo <= stop:
                return j, stop, "stop"
            if hi >= target:
                return j, target, "target"
        else:
            if hi >= stop:
                return j, stop, "stop"
            if lo <= target:
                return j, target, "target"
    # No stop/target hit — exit at horizon open
    return horizon_idx, float(df["Open"].iloc[horizon_idx]), "horizon"


def simulate(symbol: str, df: pd.DataFrame, signals: List[Signal],
             cfg: Optional[SimConfig] = None) -> dict:
    """
    Simulate a strategy described by ``signals`` over OHLC history ``df``.

    df must have columns Open/High/Low/Close and a Datetime column (or index
    convertible to dates). Returns {"trades": [...], "metrics": {...}}.
    """
    cfg = cfg or SimConfig()
    df = df.reset_index(drop=True)
    close = df["Close"].astype(float)
    sma_series = sma(close, cfg.sma_window)
    atr_series = atr(df, cfg.atr_period)
    dates = (
        pd.to_datetime(df["Datetime"]).dt.date.astype(str).tolist()
        if "Datetime" in df.columns else [str(i) for i in range(len(df))]
    )

    sig = sorted(signals, key=lambda s: s.decision_idx)
    trades: List[SimTrade] = []

    for k, s in enumerate(sig):
        direction = apply_overlays(s.direction, s.confidence, s.decision_idx,
                                   close, sma_series, cfg)
        if direction == 0:
            continue
        entry_idx = s.decision_idx + 1
        horizon_idx = (sig[k + 1].decision_idx + 1) if k + 1 < len(sig) else len(df) - 1
        if entry_idx >= len(df) or horizon_idx >= len(df) or horizon_idx <= entry_idx:
            continue

        entry = float(df["Open"].iloc[entry_idx])
        if cfg.use_atr_stops:
            exit_idx, exit_price, reason = _exit_via_stops(
                df, direction, entry_idx, horizon_idx, entry,
                float(atr_series.iloc[s.decision_idx]), s.risk_reward_ratio, cfg)
        else:
            exit_idx, exit_price, reason = horizon_idx, float(df["Open"].iloc[horizon_idx]), "horizon"

        gross = direction * (exit_price - entry) / entry
        holding_days = max(1, exit_idx - entry_idx)
        cost = (cfg.commission + cfg.slippage) * 2.0
        if direction == -1:
            cost += cfg.borrow_annual * (holding_days / cfg.trading_days_per_year)

        # Position sizing: scale exposure (and its cost) by confidence if enabled.
        if cfg.position_sizing == "confidence":
            size = max(0.0, min(s.confidence, 1.0)) * cfg.max_position
        else:
            size = cfg.max_position
        net = size * (gross - cost)

        trades.append(SimTrade(
            symbol=symbol,
            decision_date=dates[s.decision_idx],
            direction=direction,
            entry_date=dates[entry_idx],
            entry_price=round(entry, 4),
            exit_date=dates[exit_idx],
            exit_price=round(exit_price, 4),
            exit_reason=reason,
            confidence=round(s.confidence, 3),
            holding_days=holding_days,
            gross_pnl_pct=round(gross, 5),
            cost_pct=round(cost, 5),
            position_size=round(size, 4),
            pnl_pct=round(net, 5),
            note=s.note,
        ))

    return {"trades": trades, "metrics": compute_metrics(trades, cfg)}


def compute_metrics(trades: List[SimTrade], cfg: SimConfig) -> dict:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "total_return": 0.0, "sharpe_annual": 0.0,
                "win_rate": 0.0, "max_drawdown": 0.0, "avg_pnl_pct": 0.0,
                "n_long": 0, "n_short": 0, "exit_breakdown": {}}
    returns = pd.Series([t.pnl_pct for t in trades])
    equity = (1 + returns).cumprod()
    total_ret = float(equity.iloc[-1] - 1.0)
    sharpe = float(returns.mean() / returns.std() * (cfg.periods_per_year ** 0.5)) if returns.std() else 0.0
    win_rate = float((returns > 0).mean())
    mdd = float(((equity.cummax() - equity) / equity.cummax()).max())
    exit_breakdown: dict = {}
    for t in trades:
        exit_breakdown[t.exit_reason] = exit_breakdown.get(t.exit_reason, 0) + 1
    return {
        "n_trades": n,
        "total_return": total_ret,
        "sharpe_annual": sharpe,
        "win_rate": win_rate,
        "max_drawdown": mdd,
        "avg_pnl_pct": float(returns.mean()),
        "n_long": sum(1 for t in trades if t.direction == 1),
        "n_short": sum(1 for t in trades if t.direction == -1),
        "exit_breakdown": exit_breakdown,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Baselines (roadmap Tier 4)
# ─────────────────────────────────────────────────────────────────────────────

def buy_hold_return(df: pd.DataFrame, start_idx: int = 0) -> float:
    close = df["Close"].astype(float)
    return float(close.iloc[-1] / close.iloc[start_idx] - 1.0)


def _signals_from_directions(rebalance_idxs: List[int], directions: List[int],
                             rr: float = 2.0) -> List[Signal]:
    return [Signal(decision_idx=i, direction=d, confidence=1.0, risk_reward_ratio=rr)
            for i, d in zip(rebalance_idxs, directions)]


def baseline_signals(df: pd.DataFrame, rebalance_idxs: List[int], cfg: SimConfig,
                     seed: int = 42) -> dict:
    """Construct signal lists for each trivial baseline strategy."""
    close = df["Close"].astype(float)
    sma_series = sma(close, cfg.sma_window)
    rng = np.random.default_rng(seed)

    always_long = _signals_from_directions(rebalance_idxs, [1] * len(rebalance_idxs))
    sma_follow = _signals_from_directions(
        rebalance_idxs,
        [1 if float(close.iloc[i]) > float(sma_series.iloc[i]) else 0 for i in rebalance_idxs],
    )
    rand_dir = [int(d) for d in rng.choice([-1, 1], size=len(rebalance_idxs))]
    random_ls = _signals_from_directions(rebalance_idxs, rand_dir)
    return {"always_long": always_long, "sma_trend_follower": sma_follow, "random": random_ls}


def run_with_baselines(symbol: str, df: pd.DataFrame, agent_signals: List[Signal],
                       rebalance_idxs: List[int], cfg: Optional[SimConfig] = None) -> dict:
    """
    Run the agent strategy plus all baselines on the same history/costs.
    Baselines use a frictionless-trend-filter-off config so they are honest
    'dumb' references (no overlays), while the agent run uses ``cfg`` as given.
    Returns a dict of strategy_name -> metrics, plus buy_hold_return.
    """
    cfg = cfg or SimConfig()
    base_cfg = SimConfig(commission=cfg.commission, slippage=cfg.slippage,
                         borrow_annual=cfg.borrow_annual, trend_filter=False,
                         confidence_gate=0.0, use_atr_stops=False,
                         allow_short=True, periods_per_year=cfg.periods_per_year)

    out = {"symbol": symbol,
           "agent": simulate(symbol, df, agent_signals, cfg)["metrics"],
           "buy_hold_return": buy_hold_return(df, rebalance_idxs[0] if rebalance_idxs else 0)}
    for name, sigs in baseline_signals(df, rebalance_idxs, cfg).items():
        out[name] = simulate(symbol, df, sigs, base_cfg)["metrics"]
    return out


def trades_to_frame(trades: List[SimTrade]) -> pd.DataFrame:
    return pd.DataFrame([asdict(t) for t in trades])
