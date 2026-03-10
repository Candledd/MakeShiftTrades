"""Multi-Timeframe (MTF) Confluence Analysis
==========================================

Cross-references 3m, 5m, 15m, and 30m timeframes with tiered weighting.
Higher timeframes are given more authority over the directional bias;
1m is included solely as an entry-precision layer and casts no vote.

Timeframe roles
---------------
- 1m  (weight 0.0) : precision entry trigger — FVG setup only, no vote
- 3m  (weight 0.5) : light fast-signal context
- 5m  (weight 1.0) : primary entry timeframe
- 15m (weight 2.0) : key setup confirmation — carries most weight
- 30m (weight 1.5) : broader trend bias / HTF context

Voting
------
Each TF votes on the **market-structure trend** (bullish → BUY, bearish → SELL).
Using FVG signal direction instead caused a systematic long-bias because
find_setup() resolves any active bullish FVG as a BUY regardless of context.

Consensus rules (total voting weight = 5.0)
-------------------------------------------
- LONG       : ≥60 % of total weight votes BUY  (≥3.0 pts)
- SHORT      : ≥60 % of total weight votes SELL
- LEAN_LONG  : BUY has plurality but < 60 %
- LEAN_SHORT : SELL has plurality but < 60 %
- NEUTRAL    : tied or no data

Entry / Exit selection
----------------------
- Entry  : most precise TF aligned with consensus that has an FVG setup
           tried in order: 1m → 3m → 5m → 15m → 30m
- Stop   : taken from the same TF as entry
- Target : preferred from a higher TF (15m → 30m → 5m → 3m)

A ⚡ flag is set per-TF when price is *currently* at/inside the FVG zone.
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
    {"interval": "1m",  "period": "1d",  "weight": 0.0},  # entry precision, no vote
    {"interval": "3m",  "period": "5d",  "weight": 0.5},
    {"interval": "5m",  "period": "5d",  "weight": 1.0},
    {"interval": "15m", "period": "1mo", "weight": 2.0},
]

# Ordered list used to filter which TFs are >= the active chart interval
_TF_ORDER: list[str] = ["1m", "3m", "5m", "15m"]


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
    """Run SMC analysis across timeframes at or above ``active_interval``.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. ``"NQ=F"``, ``"SPY"``.
    active_interval : str
        The chart interval currently selected by the user (e.g. ``"1m"``,
        ``"5m"``).  Only timeframes >= this value (in ``_TF_ORDER``) are
        fetched and analysed.
    ms_term : str
        Market-structure detection term (``"short"`` / ``"intermediate"`` / ``"long"``).
    min_rr : float
        Minimum R/R ratio passed to the per-TF strategy (default 2.0).
    """

    def __init__(
        self,
        symbol: str,
        active_interval: str = "1m",
        ms_term: str = "intermediate",
        min_rr: float = 2.0,
    ) -> None:
        self.symbol          = symbol
        self.active_interval = active_interval
        self.ms_term         = ms_term
        self.min_rr          = min_rr

        # Only keep TFs that are at or above the active interval
        try:
            cutoff = _TF_ORDER.index(active_interval)
        except ValueError:
            cutoff = 0  # unknown interval → include all
        self._active_tfs: list[dict] = [
            tf for tf in _TIMEFRAMES
            if tf["interval"] in _TF_ORDER[cutoff:]
        ]

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def analyze(self) -> MTFResult:
        """Fetch and analyze active timeframes, then build the consensus."""
        results: dict[str, TFResult] = {}

        for tf_cfg in self._active_tfs:
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

                # Vote on TREND from market structure only.
                # FVG signal direction is NOT used for voting — find_setup()
                # always resolves a bullish-trend FVG as BUY, which caused a
                # systematic long-bias across all timeframes.
                direction: Optional[Literal["BUY", "SELL"]] = None
                if trend == "bullish":
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

        # ── Entry: most precise TF aligned with consensus (lowest TF first) ──
        entry = stop_loss = None
        entry_tf: Optional[str] = None
        active_order = [tf["interval"] for tf in self._active_tfs]
        for interval in active_order:
            tf = results.get(interval)
            if tf and tf.direction == target_dir and tf.entry is not None:
                entry     = tf.entry
                stop_loss = tf.stop_loss
                entry_tf  = interval
                break

        # ── Target: prefer higher TF (more room) ─────────────────────────
        take_profit: Optional[float] = None
        target_tf:   Optional[str]   = None
        for interval in reversed(active_order):
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
