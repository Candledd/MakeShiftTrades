"""Multi-Timeframe (MTF) Confluence Analysis
==========================================

Cross-references 3m, 5m, and 15m timeframes with equal weighting to
determine the overall long/short consensus.  The 30m timeframe is included
as a supplementary/higher-timeframe bias layer — it carries only 0.4 weight
so it can add conviction to an already-strong signal without overriding
disagreement on the primary TFs.

Consensus rules
---------------
- Votes are weighted: 3m = 1.0 · 5m = 1.0 · 15m = 1.0 · 30m = 0.4
- LONG      : ≥60 % of total weight votes BUY
- SHORT     : ≥60 % of total weight votes SELL
- LEAN_LONG : BUY has plurality but < 60 %
- LEAN_SHORT: SELL has plurality but < 60 %
- NEUTRAL   : tied or no data

Entry / Exit selection
----------------------
- Entry  : lowest (most precise) TF that agrees with consensus and has
           an active FVG setup (3m → 5m → 15m → 30m)
- Stop   : taken from the same TF as entry
- Target : preferred from a higher TF (15m → 30m → 5m → 3m)

A ⚡ flag is set per-TF when price is *currently* at/inside the FVG zone
(strict signal fired, not just a pending setup).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

import pandas as pd

from charts.data import fetch_ohlcv
from charts.indicators.price_action import detect_market_structure
from src.strategy import SMCStrategy, TradeSignal

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Timeframe configuration
# ─────────────────────────────────────────────────────────────────────────────

_TIMEFRAMES: list[dict] = [
    {"interval": "3m",  "period": "5d",  "weight": 1.0},
    {"interval": "5m",  "period": "5d",  "weight": 1.0},
    {"interval": "15m", "period": "30d", "weight": 1.0},
    {"interval": "30m", "period": "30d", "weight": 0.4},   # supplementary HTF
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_trend(
    ms_events: list[dict],
) -> Optional[Literal["bullish", "bearish"]]:
    """Infer trend from the most recent market-structure event."""
    for ev in reversed(ms_events):
        t = ev.get("type", "")
        if "bull" in t:
            return "bullish"
        if "bear" in t:
            return "bearish"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Result data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TFResult:
    """Analysis output for a single timeframe."""
    interval:    str
    weight:      float
    trend:       Optional[Literal["bullish", "bearish"]]
    direction:   Optional[Literal["BUY", "SELL"]]
    entry:       Optional[float]
    stop_loss:   Optional[float]
    take_profit: Optional[float]
    risk_reward: Optional[float]
    confidence:  Optional[str]
    at_zone:     bool                         # True when strict signal fired
    reason:      Optional[str]
    ms_events:   list = field(default_factory=list)


@dataclass
class MTFResult:
    """Aggregated multi-timeframe consensus."""
    consensus:       str            # LONG | SHORT | LEAN_LONG | LEAN_SHORT | NEUTRAL
    consensus_score: float          # 0–100  (winning direction's weighted %)
    long_pct:        float          # weighted % voting BUY
    short_pct:       float          # weighted % voting SELL
    entry:           Optional[float]
    stop_loss:       Optional[float]
    take_profit:     Optional[float]
    risk_reward:     Optional[float]
    entry_tf:        Optional[str]  # which TF provided the entry level
    target_tf:       Optional[str]  # which TF provided the take-profit level
    timeframes:      dict           # per-TF detail dicts (serializable)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class MultiTimeframeAnalysis:
    """Run SMC analysis across 3m / 5m / 15m / 30m and produce a consensus.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. ``"NQ=F"``, ``"SPY"``.
    ms_term : str
        Market-structure detection term (``"short"`` / ``"intermediate"`` / ``"long"``).
    min_rr : float
        Minimum R/R ratio passed to the per-TF strategy (default 2.0).
    """

    def __init__(
        self,
        symbol: str,
        ms_term: str = "intermediate",
        min_rr: float = 2.0,
    ) -> None:
        self.symbol  = symbol
        self.ms_term = ms_term
        self.min_rr  = min_rr

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def analyze(self) -> MTFResult:
        """Fetch and analyze all timeframes, then build the consensus."""
        results: dict[str, TFResult] = {}

        for tf_cfg in _TIMEFRAMES:
            interval = tf_cfg["interval"]
            period   = tf_cfg["period"]
            weight   = tf_cfg["weight"]

            try:
                df = fetch_ohlcv(self.symbol, period=period, interval=interval)
                if df is None or len(df) < 30:
                    logger.warning("MTF %s %s: too few bars (%d), skipping.",
                                   self.symbol, interval, 0 if df is None else len(df))
                    continue

                strategy = SMCStrategy(
                    self.symbol,
                    interval=interval,
                    period=period,
                    ms_term=self.ms_term,
                    min_rr=self.min_rr,
                )

                # Strict signal: price IS at/inside the FVG zone right now
                strict_sig: Optional[TradeSignal] = strategy.analyze(df)
                # Fallback: best nearby FVG setup (price hasn't reached it yet)
                sig: Optional[TradeSignal] = strict_sig or strategy.find_setup(df)

                # Market structure for trend + recent events
                ms_events = detect_market_structure(df, term=self.ms_term)
                trend     = _extract_trend(ms_events)

                # Directional vote: signal direction first, then infer from trend
                direction: Optional[Literal["BUY", "SELL"]] = None
                if sig:
                    direction = sig.direction
                elif trend == "bullish":
                    direction = "BUY"
                elif trend == "bearish":
                    direction = "SELL"

                results[interval] = TFResult(
                    interval    = interval,
                    weight      = weight,
                    trend       = trend,
                    direction   = direction,
                    entry       = sig.entry       if sig else None,
                    stop_loss   = sig.stop_loss   if sig else None,
                    take_profit = sig.take_profit if sig else None,
                    risk_reward = sig.risk_reward if sig else None,
                    confidence  = sig.confidence  if sig else None,
                    at_zone     = strict_sig is not None,
                    reason      = sig.reason      if sig else None,
                    ms_events   = [
                        {
                            "label": ev.get("label"),
                            "type":  ev.get("type"),
                            "price": float(ev.get("price", 0)),
                        }
                        for ev in ms_events[-3:]
                    ],
                )

            except Exception:
                logger.warning(
                    "MTF: failed to analyze %s on %s", self.symbol, interval,
                    exc_info=True,
                )

        return self._build_consensus(results)

    # ------------------------------------------------------------------ #
    # Consensus builder                                                    #
    # ------------------------------------------------------------------ #

    def _build_consensus(self, results: dict[str, TFResult]) -> MTFResult:
        long_w  = 0.0
        short_w = 0.0
        total_w = 0.0

        for tf in results.values():
            w = tf.weight
            total_w += w
            if tf.direction == "BUY":
                long_w += w
            elif tf.direction == "SELL":
                short_w += w

        long_pct  = (long_w  / total_w * 100) if total_w else 0.0
        short_pct = (short_w / total_w * 100) if total_w else 0.0

        if long_pct >= 60.0:
            consensus = "LONG"
        elif short_pct >= 60.0:
            consensus = "SHORT"
        elif long_pct > short_pct:
            consensus = "LEAN_LONG"
        elif short_pct > long_pct:
            consensus = "LEAN_SHORT"
        else:
            consensus = "NEUTRAL"

        consensus_score = max(long_pct, short_pct)

        # Which direction are we building levels for?
        target_dir: Optional[Literal["BUY", "SELL"]] = (
            "BUY"  if "LONG"  in consensus else
            "SELL" if "SHORT" in consensus else
            None
        )

        # ── Entry: lowest TF that agrees with consensus and has a setup ──
        entry = stop_loss = None
        entry_tf: Optional[str] = None
        for interval in ["3m", "5m", "15m", "30m"]:
            tf = results.get(interval)
            if tf and tf.direction == target_dir and tf.entry is not None:
                entry     = tf.entry
                stop_loss = tf.stop_loss
                entry_tf  = interval
                break

        # ── Target: prefer higher TF (more room) ─────────────────────────
        take_profit: Optional[float] = None
        target_tf:   Optional[str]   = None
        for interval in ["15m", "30m", "5m", "3m"]:
            tf = results.get(interval)
            if tf and tf.direction == target_dir and tf.take_profit is not None:
                take_profit = tf.take_profit
                target_tf   = interval
                break

        # ── R/R from combined entry → stop → target ───────────────────────
        risk_reward: Optional[float] = None
        if entry is not None and stop_loss is not None and take_profit is not None:
            risk   = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            if risk > 0:
                risk_reward = round(reward / risk, 2)

        tf_summaries = {
            interval: {
                "trend":       tf.trend,
                "direction":   tf.direction,
                "entry":       tf.entry,
                "stop_loss":   tf.stop_loss,
                "take_profit": tf.take_profit,
                "risk_reward": tf.risk_reward,
                "confidence":  tf.confidence,
                "at_zone":     tf.at_zone,
                "reason":      tf.reason,
                "weight":      tf.weight,
                "ms_events":   tf.ms_events,
            }
            for interval, tf in results.items()
        }

        return MTFResult(
            consensus       = consensus,
            consensus_score = round(consensus_score, 1),
            long_pct        = round(long_pct,  1),
            short_pct       = round(short_pct, 1),
            entry           = entry,
            stop_loss       = stop_loss,
            take_profit     = take_profit,
            risk_reward     = risk_reward,
            entry_tf        = entry_tf,
            target_tf       = target_tf,
            timeframes      = tf_summaries,
        )
