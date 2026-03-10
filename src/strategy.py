"""ICT Smart Money Concepts (SMC) Day Trading Strategy
=====================================================

Algorithm overview
------------------
This strategy is an implementation of the Inner Circle Trader (ICT) /
Smart Money Concepts methodology, one of the most widely studied and
community-validated open-source day-trading frameworks.

Entry logic (in order of required confluence):
  1. Trend  — determined by the most recent Break of Structure (BOS)
              or Change of Character (CHoCH) from market-structure analysis.
  2. FVG    — an active Fair Value Gap aligned with the trend is located.
              A signal fires when the current price is at or inside the
              FVG zone (the zone acts as a magnet / re-test area).
  3. Order Block (optional) — if the FVG overlaps or sits near a
              corresponding Order Block the signal is graded "strong".
  4. Engulfing (optional) — a liquidity-engulfing candle pattern at or
              near the FVG zone on the last 3 bars upgrades confidence.

Exit logic:
  - Take Profit: the nearest swing-high liquidity level (for longs) or
                 swing-low liquidity level (for shorts) found by the
                 pivot-based Liquidity Heatmap.
  - Stop Loss  : 0.5 × FVG-height beyond the FVG zone.
  - A minimum Risk/Reward ratio (default 2.0) is enforced; signals that
    do not meet it are discarded.

References / inspiration
------------------------
- ICT Mentorship (Michael J. Huddleston) — publicly available YouTube content
- LuxAlgo Pure Price Action — open-source Pine Script
- Nephew_Sam_ Liquidity Heatmap — open-source Pine Script
- TradingFinder FVG detection convention
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

import pandas as pd

from charts.data import fetch_ohlcv
from charts.indicators.engulfing import detect_engulfing
from charts.indicators.fvg import detect_fvg
from charts.indicators.liquidity import detect_liquidity_levels
from charts.indicators.price_action import (
    detect_market_structure,
    detect_order_blocks,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeSignal:
    """A fully validated trade signal ready to be acted on."""

    direction: Literal["BUY", "SELL"]
    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    confidence: Literal["standard", "strong"]       # strong = OB + engulfing confluence
    fvg_zone: tuple[float, float]                   # (bottom, top)
    trend: Literal["bullish", "bearish"]
    reason: str
    raw_data: dict = field(default_factory=dict, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _current_trend(structure_events: list[dict]) -> Optional[Literal["bullish", "bearish"]]:
    """Return the trend inferred from the most recent structure break.

    BOS_bull / CHoCH_bull  → bullish
    BOS_bear / CHoCH_bear  → bearish
    """
    for event in reversed(structure_events):
        t = event.get("type", "")
        if "bull" in t:
            return "bullish"
        if "bear" in t:
            return "bearish"
    return None


def _price_in_fvg(price: float, top: float, bottom: float,
                  approach_pct: float = 0.50) -> bool:
    """Return True if *price* is inside or approaching the FVG zone.

    approach_pct: if price is within this fraction of the zone height
    above/below the zone it is considered 'approaching'.
    """
    height = top - bottom
    tol = height * approach_pct
    return (bottom - tol) <= price <= (top + tol)


def _ob_near_fvg(order_blocks: list[dict],
                 fvg_bottom: float, fvg_top: float,
                 trend: Literal["bullish", "bearish"]) -> bool:
    """True if a matching Order Block overlaps or is adjacent to the FVG."""
    fvg_mid = (fvg_top + fvg_bottom) / 2
    for ob in order_blocks:
        if ob.get("type") != trend:
            continue
        ob_top    = ob["top"]
        ob_bottom = ob["bottom"]
        # Overlap check: intervals [ob_bottom, ob_top] and [fvg_bottom, fvg_top]
        if ob_bottom <= fvg_top and ob_top >= fvg_bottom:
            return True
        # Adjacent check: OB within one FVG-height of the zone mid
        if abs(ob_top - fvg_mid) <= (fvg_top - fvg_bottom):
            return True
    return False


def _engulfing_at_zone(df: pd.DataFrame,
                       trend: Literal["bullish", "bearish"],
                       lookback_bars: int = 3) -> bool:
    """Return True if a matching engulfing candle appeared in the last N bars."""
    engulf_df = detect_engulfing(df)
    if engulf_df.empty:
        return False
    cutoff = df.index[-lookback_bars]
    recent = engulf_df[(engulf_df["type"] == trend) & (engulf_df["date"] >= cutoff)]
    return not recent.empty


# ─────────────────────────────────────────────────────────────────────────────
# Public strategy class
# ─────────────────────────────────────────────────────────────────────────────

class SMCStrategy:
    """ICT / Smart Money Concepts day-trading strategy.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. "SPY", "NQ=F", "AAPL").
    interval : str
        Candle interval for signal generation (default "5m").
    period : str
        Lookback window for data fetch (default "5d").
    ms_term : str
        Market-structure detection window: "short", "intermediate", or "long".
    fvg_lookback : int
        Maximum number of bars back to consider an FVG active.
    approach_pct : float
        Fraction of FVG height used as approach tolerance (default 0.5).
    min_rr : float
        Minimum acceptable Risk/Reward ratio (default 2.0).
    require_ob : bool
        If True, require Order Block confluence to fire a signal.
    require_engulfing : bool
        If True, require an engulfing candle at the zone.
    """

    def __init__(
        self,
        symbol: str,
        interval: str = "5m",
        period: str = "5d",
        ms_term: str = "intermediate",
        fvg_lookback: int = 50,
        approach_pct: float = 0.50,
        min_rr: float = 2.0,
        require_ob: bool = False,
        require_engulfing: bool = False,
    ) -> None:
        self.symbol = symbol
        self.interval = interval
        self.period = period
        self.ms_term = ms_term
        self.fvg_lookback = fvg_lookback
        self.approach_pct = approach_pct
        self.min_rr = min_rr
        self.require_ob = require_ob
        self.require_engulfing = require_engulfing

    # ------------------------------------------------------------------ #
    # Data layer                                                           #
    # ------------------------------------------------------------------ #

    def fetch_data(self) -> pd.DataFrame:
        """Fetch OHLCV data for this symbol/interval/period."""
        return fetch_ohlcv(self.symbol, period=self.period, interval=self.interval)

    # ------------------------------------------------------------------ #
    # Core analysis                                                        #
    # ------------------------------------------------------------------ #

    def analyze(self, df: Optional[pd.DataFrame] = None) -> Optional[TradeSignal]:
        """Run the full SMC analysis pipeline.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Pre-fetched OHLCV DataFrame.  If None, data is fetched automatically.

        Returns
        -------
        TradeSignal or None
            A validated signal dict, or None if no trade setup is present.
        """
        if df is None:
            df = self.fetch_data()

        if len(df) < max(30, self.fvg_lookback):
            logger.warning("Not enough bars to run strategy (%d).", len(df))
            return None

        current_price = float(df["Close"].iloc[-1])

        # ── Step 1: Trend via Market Structure ──────────────────────────
        structure = detect_market_structure(df, term=self.ms_term)
        trend = _current_trend(structure)
        if trend is None:
            logger.debug("No confirmed market structure. Skipping.")
            return None

        # ── Step 2: Active FVGs aligned with trend ──────────────────────
        fvg_df = detect_fvg(df)
        if fvg_df.empty:
            logger.debug("No FVGs detected.")
            return None

        recent_cutoff = df.index[-self.fvg_lookback]
        aligned_fvgs = fvg_df[
            (fvg_df["active"] == True) &          # noqa: E712
            (fvg_df["type"] == trend) &
            (fvg_df["date"] >= recent_cutoff)
        ].sort_values("date", ascending=False)

        if aligned_fvgs.empty:
            logger.debug("No active %s FVGs in last %d bars.", trend, self.fvg_lookback)
            return None

        # ── Step 3: Find the first FVG where price is at/near zone ──────
        signal_fvg = None
        for _, fvg_row in aligned_fvgs.iterrows():
            if _price_in_fvg(current_price, fvg_row["top"], fvg_row["bottom"],
                              self.approach_pct):
                signal_fvg = fvg_row
                break

        if signal_fvg is None:
            logger.debug("Price not at/near any active %s FVG.", trend)
            return None

        fvg_top    = float(signal_fvg["top"])
        fvg_bottom = float(signal_fvg["bottom"])
        fvg_height = fvg_top - fvg_bottom

        # ── Step 4: Order Block confluence (optional or required) ────────
        order_blocks = detect_order_blocks(df, term=self.ms_term)
        has_ob = _ob_near_fvg(order_blocks, fvg_bottom, fvg_top, trend)
        if self.require_ob and not has_ob:
            logger.debug("OB confluence required but not found at FVG zone.")
            return None

        # ── Step 5: Engulfing confirmation (optional or required) ────────
        has_engulf = _engulfing_at_zone(df, trend)
        if self.require_engulfing and not has_engulf:
            logger.debug("Engulfing confirmation required but not found.")
            return None

        # ── Step 6: Entry / SL / TP calculation ─────────────────────────
        liquidity = detect_liquidity_levels(df)

        if trend == "bullish":
            entry      = fvg_bottom                        # retest from below
            stop_loss  = fvg_bottom - fvg_height * 0.5    # 0.5× height below zone
            # TP = nearest high-side liquidity above entry (highest strength first)
            candidates = sorted(
                [l for l in liquidity if l["dir"] == "high" and l["price"] > entry],
                key=lambda x: (x["price"] - entry),        # closest first
            )
        else:  # bearish
            entry      = fvg_top                           # retest from above
            stop_loss  = fvg_top + fvg_height * 0.5       # 0.5× height above zone
            candidates = sorted(
                [l for l in liquidity if l["dir"] == "low" and l["price"] < entry],
                key=lambda x: (entry - x["price"]),        # closest first
            )

        if not candidates:
            logger.debug("No liquidity target found for %s trade.", trend)
            return None

        take_profit = candidates[0]["price"]

        # ── Step 7: R/R gate ─────────────────────────────────────────────
        risk   = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        if risk == 0:
            return None
        rr = reward / risk
        if rr < self.min_rr:
            logger.debug("R/R %.2f below minimum %.2f. Skipping.", rr, self.min_rr)
            return None

        # ── Step 8: Confidence grading ───────────────────────────────────
        confidence: Literal["standard", "strong"] = "strong" if (has_ob and has_engulf) else "standard"

        direction: Literal["BUY", "SELL"] = "BUY" if trend == "bullish" else "SELL"
        reason = (
            f"{'Bullish' if trend == 'bullish' else 'Bearish'} FVG "
            f"[{fvg_bottom:.2f}–{fvg_top:.2f}] | "
            f"MS: {trend} | "
            f"OB: {'yes' if has_ob else 'no'} | "
            f"Engulf: {'yes' if has_engulf else 'no'} | "
            f"R/R: {rr:.2f}"
        )

        return TradeSignal(
            direction=direction,
            entry=round(entry, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            risk_reward=round(rr, 2),
            confidence=confidence,
            fvg_zone=(round(fvg_bottom, 4), round(fvg_top, 4)),
            trend=trend,
            reason=reason,
            raw_data={
                "current_price": current_price,
                "fvg_date":      str(signal_fvg["date"]),
                "has_ob":        has_ob,
                "has_engulf":    has_engulf,
                "ms_events":     len(structure),
            },
        )

    # ------------------------------------------------------------------ #
    # Nearest-setup finder (relaxed — price does NOT need to be at FVG)   #
    # ------------------------------------------------------------------ #

    def find_setup(self, df: Optional[pd.DataFrame] = None) -> Optional["TradeSignal"]:
        """Return the best active SMC setup even if price hasn't reached the zone.

        Unlike ``analyze()``, this does **not** require current price to be
        at/near the FVG.  It scans all active aligned FVGs and returns the
        first one that clears the R/R gate.  Use this to always populate the
        Entry / Stop / Target levels on the UI.
        """
        if df is None:
            df = self.fetch_data()

        if len(df) < max(30, self.fvg_lookback):
            return None

        current_price = float(df["Close"].iloc[-1])

        structure = detect_market_structure(df, term=self.ms_term)
        trend     = _current_trend(structure)
        if trend is None:
            return None

        fvg_df = detect_fvg(df)
        if fvg_df.empty:
            return None

        # find_setup scans ALL active aligned FVGs (no recency cutoff) so the
        # UI always has a pending level to display regardless of when the FVG
        # was formed.  Closest-to-current-price is tried first.
        aligned_fvgs = fvg_df[
            (fvg_df["active"] == True) &          # noqa: E712
            (fvg_df["type"]   == trend)
        ].copy()

        if aligned_fvgs.empty:
            return None

        aligned_fvgs["_dist"] = aligned_fvgs["bottom"].apply(
            lambda b: abs(current_price - b)
        )
        aligned_fvgs = aligned_fvgs.sort_values("_dist")

        order_blocks = detect_order_blocks(df, term=self.ms_term)
        liquidity    = detect_liquidity_levels(df)
        has_engulf   = _engulfing_at_zone(df, trend)

        for _, fvg_row in aligned_fvgs.iterrows():
            fvg_top    = float(fvg_row["top"])
            fvg_bottom = float(fvg_row["bottom"])
            fvg_height = fvg_top - fvg_bottom

            has_ob = _ob_near_fvg(order_blocks, fvg_bottom, fvg_top, trend)

            if trend == "bullish":
                entry     = fvg_bottom
                stop_loss = fvg_bottom - fvg_height * 0.5
                cands     = sorted(
                    [l for l in liquidity if l["dir"] == "high" and l["price"] > entry],
                    key=lambda x: (x["price"] - entry),
                )
            else:
                entry     = fvg_top
                stop_loss = fvg_top + fvg_height * 0.5
                cands     = sorted(
                    [l for l in liquidity if l["dir"] == "low" and l["price"] < entry],
                    key=lambda x: (entry - x["price"]),
                )

            if not cands:
                continue

            take_profit = cands[0]["price"]
            risk        = abs(entry - stop_loss)
            reward      = abs(take_profit - entry)
            if risk == 0:
                continue
            rr = reward / risk
            if rr < self.min_rr:
                continue

            confidence: Literal["standard", "strong"] = (
                "strong" if (has_ob and has_engulf) else "standard"
            )
            direction: Literal["BUY", "SELL"] = "BUY" if trend == "bullish" else "SELL"

            dist_pct = abs(current_price - entry) / max(entry, 1e-9) * 100
            reason = (
                f"Pending {'bullish' if trend == 'bullish' else 'bearish'} FVG "
                f"[{fvg_bottom:.2f}–{fvg_top:.2f}] "
                f"({dist_pct:.1f}% away) | "
                f"OB: {'yes' if has_ob else 'no'} | "
                f"Engulf: {'yes' if has_engulf else 'no'} | "
                f"R/R: {rr:.2f}"
            )

            return TradeSignal(
                direction=direction,
                entry=round(entry, 4),
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                risk_reward=round(rr, 2),
                confidence=confidence,
                fvg_zone=(round(fvg_bottom, 4), round(fvg_top, 4)),
                trend=trend,
                reason=reason,
                raw_data={
                    "current_price": current_price,
                    "fvg_date":      str(fvg_row["date"]),
                    "has_ob":        has_ob,
                    "has_engulf":    has_engulf,
                    "ms_events":     len(structure),
                    "price_at_zone": False,
                },
            )

        return None

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    def describe(self) -> str:
        lines = [
            "SMC Strategy Configuration",
            f"  Symbol     : {self.symbol}",
            f"  Interval   : {self.interval}",
            f"  Period     : {self.period}",
            f"  MS term    : {self.ms_term}",
            f"  FVG lookback: {self.fvg_lookback} bars",
            f"  Approach   : ±{self.approach_pct * 100:.0f}% of FVG height",
            f"  Min R/R    : {self.min_rr}:1",
            f"  Req OB     : {self.require_ob}",
            f"  Req Engulf : {self.require_engulfing}",
        ]
        return "\n".join(lines)
