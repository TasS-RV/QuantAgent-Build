"""
FinBERT sentiment layer (issue #5).

Adds a financial-news sentiment signal (ProsusAI/finbert) that can nudge the
quant decision's combined signal. This is an *optional* overlay:

  * If `transformers` + `torch` (and the model) are available, headlines are
    scored and aggregated into a sentiment signal ∈ [-1, 1]
    (mean of P(positive) − P(negative)).
  * Otherwise the scorer degrades to a neutral 0.0 with `available=False`, so
    nothing downstream breaks.

The blending step (`apply_sentiment`) recombines the base decision's signal with
the sentiment signal and re-derives BUY / HOLD / SELL / SHORT from the same
thresholds — i.e. it *adjusts the weights of the decision output* as requested
in issue #5, without modifying the deterministic core.

Heavy deps are imported lazily and the news scorer is injectable, so the
blending logic and graceful-degradation path are fully unit-testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Optional

from decision_agent_quant import DEFAULT_THRESHOLDS, TradeDecision

MODEL_NAME = "ProsusAI/finbert"


@dataclass
class SentimentResult:
    signal: float            # [-1, 1]  (positive − negative)
    label: str               # "positive" | "negative" | "neutral" | "unavailable"
    n_headlines: int
    available: bool
    per_headline: List[dict] = None  # [{"text", "positive", "negative", "neutral"}]


def _label_from_signal(sig: float) -> str:
    return "positive" if sig > 0.15 else "negative" if sig < -0.15 else "neutral"


# ─────────────────────────────────────────────────────────────────────────────
#  FinBERT scorer (lazy, optional)
# ─────────────────────────────────────────────────────────────────────────────

class FinBERTSentiment:
    """Lazy wrapper around ProsusAI/finbert. Loads the model on first use."""

    def __init__(self):
        self._pipe = None
        self._tried = False

    def _ensure(self):
        if self._pipe is not None or self._tried:
            return
        self._tried = True
        try:
            from transformers import pipeline
            self._pipe = pipeline("text-classification", model=MODEL_NAME,
                                  top_k=None, truncation=True)
        except Exception:
            self._pipe = None

    def available(self) -> bool:
        self._ensure()
        return self._pipe is not None

    def score(self, headlines: List[str]) -> SentimentResult:
        headlines = [h for h in (headlines or []) if h and h.strip()]
        if not headlines:
            return SentimentResult(0.0, "neutral", 0, True, [])
        self._ensure()
        if self._pipe is None:
            return SentimentResult(0.0, "unavailable", len(headlines), False, [])

        per: List[dict] = []
        sigs = []
        for text, scores in zip(headlines, self._pipe(headlines)):
            probs = {s["label"].lower(): float(s["score"]) for s in scores}
            pos, neg = probs.get("positive", 0.0), probs.get("negative", 0.0)
            sigs.append(pos - neg)
            per.append({"text": text, "positive": pos, "negative": neg,
                        "neutral": probs.get("neutral", 0.0)})
        signal = float(sum(sigs) / len(sigs))
        return SentimentResult(round(signal, 4), _label_from_signal(signal),
                               len(headlines), True, per)


_default_scorer: Optional[FinBERTSentiment] = None


def get_default_scorer() -> FinBERTSentiment:
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = FinBERTSentiment()
    return _default_scorer


# ─────────────────────────────────────────────────────────────────────────────
#  Headlines source (optional, injectable)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_headlines_yf(symbol: str, limit: int = 10) -> List[str]:
    """Best-effort recent headlines for a symbol via yfinance. [] on failure."""
    try:
        import yfinance as yf
        news = yf.Ticker(symbol).news or []
    except Exception:
        return []
    out = []
    for item in news[:limit]:
        title = item.get("title") or item.get("content", {}).get("title")
        if title:
            out.append(title)
    return out


def score_symbol_sentiment(
    symbol: str,
    headlines: Optional[List[str]] = None,
    scorer: Optional[Callable[[List[str]], SentimentResult]] = None,
    fetch_fn: Callable[[str, int], List[str]] = fetch_headlines_yf,
) -> SentimentResult:
    """Score sentiment for a symbol. `scorer` and `fetch_fn` are injectable for tests."""
    if headlines is None:
        headlines = fetch_fn(symbol, 10)
    score = scorer if scorer is not None else get_default_scorer().score
    return score(headlines)


# ─────────────────────────────────────────────────────────────────────────────
#  Blending — adjust the decision with sentiment
# ─────────────────────────────────────────────────────────────────────────────

def blend_signal(base_signal: float, sentiment_signal: float,
                 sentiment_weight: float = 0.2) -> float:
    """Convex blend of the base combined signal and the sentiment signal."""
    sentiment_weight = max(0.0, min(1.0, sentiment_weight))
    blended = (1.0 - sentiment_weight) * base_signal + sentiment_weight * sentiment_signal
    return float(max(-1.0, min(1.0, blended)))


def _decision_from_signal(sig: float, thresholds: dict, allow_short: bool) -> str:
    if sig >= thresholds["buy"]:
        return "BUY"
    if sig <= thresholds["short"] and allow_short:
        return "SHORT"
    if sig <= thresholds["sell"]:
        return "SELL"
    return "HOLD"


def apply_sentiment(
    decision: TradeDecision,
    sentiment: SentimentResult,
    sentiment_weight: float = 0.2,
    thresholds: Optional[dict] = None,
    allow_short: bool = True,
) -> TradeDecision:
    """
    Return a new TradeDecision whose combined_signal and decision are adjusted by
    sentiment. Price levels are preserved from the base decision (sentiment moves
    direction/conviction, not the S/R-derived targets). If sentiment is
    unavailable the base decision is returned unchanged (signal still recorded).
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if not sentiment.available:
        note = " | sentiment=unavailable"
        return replace(decision, decision_rationale=decision.decision_rationale + note)

    new_signal = blend_signal(decision.combined_signal, sentiment.signal, sentiment_weight)
    new_decision = _decision_from_signal(new_signal, t, allow_short)
    abs_sig = abs(new_signal)
    strength = "Strong" if abs_sig >= 0.50 else "Moderate" if abs_sig >= 0.25 else "Weak"

    changed = "→ " + new_decision if new_decision != decision.decision else "unchanged"
    note = (f" | sentiment={sentiment.signal:+.2f} ({sentiment.label}, "
            f"n={sentiment.n_headlines}, w={sentiment_weight:.2f}) "
            f"| blended={new_signal:+.3f} [{changed}]")

    breakdown = dict(decision.signal_breakdown or {})
    breakdown["sentiment_signal"] = sentiment.signal
    breakdown["sentiment_weight"] = sentiment_weight

    return replace(
        decision,
        decision=new_decision,
        combined_signal=round(new_signal, 4),
        signal_strength=strength,
        signal_breakdown=breakdown,
        decision_rationale=decision.decision_rationale + note,
    )
