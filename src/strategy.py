"""High-Probability SMC Scalping Strategy
=========================================

Strict ICT/SMC scalping engine using a sequential 6-step filter.
No ML dependency — pure rules-based SMC logic.

Pipeline
--------
  1. CONTEXT    — HTF trend + Equilibrium zone
  2. TIME       — Kill Zone active? (equities only)
  3. FUEL       — Recent liquidity sweep (PDH/PDL/swing point taken out)
  4. SHIFT      — CHoCH / BOS after the sweep
  5. ENTRY      — FVG from the structural shift + OB confluence
  6. RISK       — Valid SL/TP levels with R/R >= minimum

Entry / Exit design
-------------------
  - Entry   : retest of the FVG zone created by the structural shift
  - Stop    : beyond the swept swing point (not behind the FVG)
  - Target  : next opposing liquidity level or equilibrium
  - Min R/R : 1.5 (lower than the old 2.0 to account for wider SL)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from operator import itemgetter

import pandas as pd
import numpy as np

from charts.data import fetch_ohlcv
from charts.indicators.fvg import detect_fvg
from charts.indicators.levels import detect_equilibrium, detect_sessions
from charts.indicators.liquidity import detect_liquidity_levels
from charts.indicators.price_action import (
    detect_market_structure,
    detect_order_blocks,
    detect_swing_points,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Session / Kill-Zone configuration (ET)
# ─────────────────────────────────────────────────────────────────────────────

KILL_ZONES: list[dict] = [
    {"name": "ASIA",    "start": 19, "end": 23},   # 7–11 PM ET (prev day)
    {"name": "LONDON",  "start":  3, "end":  5},   # 3–5 AM ET
    {"name": "NY_AM",   "start":  7, "end": 10},   # 7–10 AM ET
    {"name": "NY_PM",   "start": 13, "end": 16},   # 1–4 PM ET
]

# Crypto tickers (yfinance format with dash) — bypass session checks
CRYPTO_TICKERS: set = {"BTC-USD", "ETH-USD", "BCH-USD", "LTC-USD", "UNI-USD", "LINK-USD"}

LUNCH_CHOP       = (11, 13)  # 11AM–1PM ET — skip
SWEEP_LOOKBACK   = 10        # bars back to check for liquidity sweep
MIN_FVG_SIZE_ATR = 0.1       # FVG must be at least 0.1× the current ATR
MAX_FVG_SIZE_ATR = 3.0       # FVG larger than this is unreliable (too wide)
MIN_CONFIDENCE   = 30.0      # minimum strategy confidence to even show the signal
MIN_RR_HARD      = 0.8       # absolute minimum risk/reward


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeSignal:
    """A fully validated SMC trade signal."""

    direction: Literal["BUY", "SELL"]
    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    confidence: float       # 0–100 based on confluence quality
    fvg_zone: tuple[float, float]
    trend: Literal["bullish", "bearish"]
    reason: str
    at_zone: bool           # True = price is AT the FVG right now
    raw_data: dict = field(default_factory=dict, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _current_trend(structure: list[dict]) -> Optional[Literal["bullish", "bearish"]]:
    """Return the trend from the most recent market-structure event."""
    for ev in reversed(structure):
        t = ev.get("type", "")
        if "bull" in t:
            return "bullish"
        if "bear" in t:
            return "bearish"
    return None


def _in_kill_zone() -> tuple[bool, str]:
    """Check whether the current ET hour is inside a kill zone.

    Returns
    -------
    (ok: bool, label: str)
      ``(True, "NY_AM")`` → valid session
      ``(False, "lunch_chop")`` → skip
    """
    try:
        # Eastern Time without pytz: UTC-4 (EDT) or UTC-5 (EST)
        now = datetime.now(timezone.utc)
        # EDT is UTC-4 (starts 2nd Sun Mar, ends 1st Sun Nov)
        _, week, dow = now.isocalendar()
        month = now.month
        # Simplified DST: EDT from March week 10 to November week 44
        is_edt = (month > 3 and month < 11) or (month == 3 and week >= 10) or (month == 11 and week < 44)
        offset = 4 if is_edt else 5
        et_hour = (now.hour - offset) % 24

        # Lunch chop — reject
        if LUNCH_CHOP[0] <= et_hour < LUNCH_CHOP[1]:
            return False, "lunch_chop"

        # Kill zones — accept
        for kz in KILL_ZONES:
            if kz["start"] <= et_hour < kz["end"]:
                return True, kz["name"]

        return False, "dead_zone"
    except Exception:
        return True, "unknown"


def _find_sweep_and_shift(
    df: pd.DataFrame,
    trend: Literal["bullish", "bearish"],
    lookback: int = SWEEP_LOOKBACK,
) -> Optional[dict[str, Any]]:
    """Look for a liquidity sweep followed by a structural shift in recent bars.

    For a **bullish** trend:
      - Price swept below a recent swing LOW (grabbed liquidity)
      - Then reversed and broke above the swing low's high (CHoCH)

    For a **bearish** trend:
      - Price swept above a recent swing HIGH
      - Then reversed and broke below the swing high's low

    Returns a dict with keys:
      sweep_bar : int       — index of the sweep candle
      sweep_px  : float     — price that was swept
      shift_bar : int       — index of the CHoCH candle
      fvg_top   : float     — top of the FVG left by the shift
      fvg_bottom: float     — bottom of the FVG
      succeeded : bool
    or None if no pattern found.
    """
    n = len(df)
    if n < lookback + 5:
        return None

    closes  = df["Close"].to_numpy()
    highs   = df["High"].to_numpy()
    lows    = df["Low"].to_numpy()
    opens   = df["Open"].to_numpy()
    idx_arr = np.arange(n)

    if trend == "bullish":
        # Find recent swing lows that were swept
        swings = detect_swing_points(df, term="short")
        recent_swings = [s for s in swings if s["type"] == "low"]
        if not recent_swings:
            return None

        # Check each swing low (most recent first)
        for sw in reversed(recent_swings[-4:]):  # check last 4 swings
            sw_idx = df.index.get_loc(sw["date"]) if sw["date"] in df.index else df.index.searchsorted(sw["date"])
            sw_px  = float(sw["price"])
            sw_high = float(sw.get("high", sw_px * 1.001))  # approximate

            # Look for a sweep in the next `lookback` bars
            window = slice(sw_idx + 1, min(sw_idx + 1 + lookback, n))
            if window.start >= n:
                continue
            sweep_candidates = idx_arr[window][lows[window] < sw_px]
            if len(sweep_candidates) == 0:
                continue
            sweep_idx = int(sweep_candidates[0])  # first sweep

            # After the sweep, look for a bullish CHoCH (close > sw_high)
            post_window = slice(sweep_idx + 1, min(sweep_idx + 1 + lookback, n))
            if post_window.start >= n:
                continue
            # A bullish CHoCH: a candle that closes above recent structure resistance
            # Simple proxy: close > the swing point's recent high and open < high (bullish candle)
            choches = idx_arr[post_window][
                (closes[post_window] > highs[sweep_idx]) &
                (closes[post_window] > opens[post_window])  # bullish candle
            ]
            if len(choches) == 0:
                # Try a looser condition — any bar closing above sw_high
                choches = idx_arr[post_window][closes[post_window] > sw_high]
            if len(choches) == 0:
                continue
            shift_idx = int(choches[0])

            # The shift candle(s) should have left an FVG
            # Bullish FVG: low[i] > high[i-2]  (candle i gap above candle i-2)
            fvg_top = None
            fvg_bottom = None
            for j in range(max(shift_idx - 1, 1), min(shift_idx + 3, n)):
                if j >= 2 and lows[j] > highs[j - 2]:
                    fvg_top = lows[j]
                    fvg_bottom = highs[j - 2]
                    break

            if fvg_top is not None:
                return {
                    "sweep_bar": int(sweep_idx),
                    "sweep_px": float(sw_px),
                    "shift_bar": int(shift_idx),
                    "fvg_top": float(fvg_top),
                    "fvg_bottom": float(fvg_bottom),
                    "succeeded": True,
                }

    else:  # bearish
        swings = detect_swing_points(df, term="short")
        recent_swings = [s for s in swings if s["type"] == "high"]
        if not recent_swings:
            return None

        for sw in reversed(recent_swings[-4:]):
            sw_idx = df.index.get_loc(sw["date"]) if sw["date"] in df.index else df.index.searchsorted(sw["date"])
            sw_px = float(sw["price"])
            sw_low = float(sw.get("low", sw_px * 0.999))

            window = slice(sw_idx + 1, min(sw_idx + 1 + lookback, n))
            if window.start >= n:
                continue
            sweep_candidates = idx_arr[window][highs[window] > sw_px]
            if len(sweep_candidates) == 0:
                continue
            sweep_idx = int(sweep_candidates[0])

            post_window = slice(sweep_idx + 1, min(sweep_idx + 1 + lookback, n))
            if post_window.start >= n:
                continue
            choches = idx_arr[post_window][
                (closes[post_window] < lows[sweep_idx]) &
                (closes[post_window] < opens[post_window])  # bearish candle
            ]
            if len(choches) == 0:
                choches = idx_arr[post_window][closes[post_window] < sw_low]
            if len(choches) == 0:
                continue
            shift_idx = int(choches[0])

            # Bearish FVG: high[i] < low[i-2]
            fvg_top = None
            fvg_bottom = None
            for j in range(max(shift_idx - 1, 1), min(shift_idx + 3, n)):
                if j >= 2 and highs[j] < lows[j - 2]:
                    fvg_top = lows[j - 2]
                    fvg_bottom = highs[j]
                    break

            if fvg_top is not None:
                return {
                    "sweep_bar": int(sweep_idx),
                    "sweep_px": float(sw_px),
                    "shift_bar": int(shift_idx),
                    "fvg_top": float(fvg_top),
                    "fvg_bottom": float(fvg_bottom),
                    "succeeded": True,
                }

    return None


def _price_in_fvg(price: float, top: float, bottom: float, tol_pct: float = 0.05) -> bool:
    """Return True if price is inside or within *tol_pct* of the FVG zone."""
    height = top - bottom
    if height <= 0:
        return False
    tol = height * tol_pct
    return (bottom - tol) <= price <= (top + tol)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Compute the current ATR value."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.tail(period).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Main strategy class
# ─────────────────────────────────────────────────────────────────────────────

class SMCStrategy:
    """High-probability ICT/SMC scalping engine.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. ``\"SPY\"``, ``\"NQ=F\"``, ``\"BTC-USD\"``).
    interval : str
        Candle interval for the entry timeframe (default ``\"1m\"``).
    period : str
        Lookback window for data fetch (default ``\"5d\"``).
    htf_interval : str
        Higher-timeframe for context (default ``\"5m\"``).
    ms_term : str
        Market structure term (default ``\"short\"`` for scalping).
    min_rr : float
        Minimum R/R (default 1.5).
    require_ob : bool
        Require Order Block confluence for signal (default True).
    """

    def __init__(
        self,
        symbol: str,
        interval: str = "1m",
        period: str = "5d",
        htf_interval: str = "5m",
        ms_term: str = "short",
        min_rr: float = 1.5,
        require_ob: bool = True,
    ) -> None:
        self.symbol = symbol.upper()
        self.interval = interval
        self.period = period
        self.htf_interval = htf_interval
        self.ms_term = ms_term
        self.min_rr = min_rr
        self.require_ob = require_ob

        self._is_crypto = self.symbol in CRYPTO_TICKERS

    # ------------------------------------------------------------------ #
    # Data fetching                                                       #
    # ------------------------------------------------------------------ #

    def fetch_data(self, interval: Optional[str] = None) -> pd.DataFrame:
        """Fetch OHLCV data for the given or default interval."""
        return fetch_ohlcv(
            self.symbol,
            period=self.period,
            interval=interval or self.interval,
        )

    # ------------------------------------------------------------------ #
    # Full analysis pipeline                                              #
    # ------------------------------------------------------------------ #

    def analyze(self, df: Optional[pd.DataFrame] = None) -> Optional[TradeSignal]:
        """Run the full 6-step SMC pipeline on the entry timeframe.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Pre-fetched entry-TF data. If None, fetches automatically.

        Returns
        -------
        TradeSignal or None
        """
        if df is None:
            df = self.fetch_data()

        if len(df) < 40:
            logger.debug("SMCStrategy: too few bars (%d).", len(df))
            return None

        # ── Step 1: HTF Context (trend + equilibrium) ─────────────────
        try:
            htf_df = self.fetch_data(interval=self.htf_interval)
            htf_structure = detect_market_structure(htf_df, term=self.ms_term)
            htf_trend = _current_trend(htf_structure)
        except Exception:
            htf_df = df
            htf_trend = _current_trend(detect_market_structure(df, term=self.ms_term))

        if htf_trend is None:
            logger.debug("No HTF trend — neutral market.")
            return None

        # Equilibrium zone (premium / discount)
        eq = detect_equilibrium(df)
        current_price = float(df["Close"].iloc[-1])
        in_discount  = eq and current_price < eq.get("eq", current_price)
        in_premium   = eq and current_price > eq.get("eq", current_price)

        # Location filter: only BUY in discount, only SELL in premium
        if htf_trend == "bullish" and not in_discount:
            logger.debug("Bullish trend but price not in discount — skipping.")
            # Still allow if equilibrium data is unavailable
            if eq is not None:
                return None
        elif htf_trend == "bearish" and not in_premium:
            if eq is not None:
                logger.debug("Bearish trend but price not in premium — skipping.")
                return None

        # ── Step 2: Time / Session filter (equities only) ─────────────
        if not self._is_crypto:
            ok, kz = _in_kill_zone()
            if not ok:
                logger.debug("Outside kill zone (%s) — skipping trade.", kz)
                return None

        # ── Step 3: Market structure + trend on entry TF ──────────────
        structure = detect_market_structure(df, term=self.ms_term)
        trend = _current_trend(structure)
        if trend is None:
            logger.debug("No market structure on entry TF.")
            return None

        # Trend alignment: entry TF must agree with HTF
        if trend != htf_trend:
            logger.debug("Entry TF trend (%s) ≠ HTF trend (%s).", trend, htf_trend)
            return None

        # ── Step 4: Sweep + CHoCH detection ───────────────────────────
        sweep_data = _find_sweep_and_shift(df, trend)
        if sweep_data is None:
            logger.debug("No sweep+CHoCH pattern in last %d bars.", SWEEP_LOOKBACK)
            return None

        fvg_top    = sweep_data["fvg_top"]
        fvg_bottom = sweep_data["fvg_bottom"]
        fvg_height = fvg_top - fvg_bottom

        # FVG size sanity: must be meaningful but not absurd
        atr_val = _atr(df["High"], df["Low"], df["Close"])
        if atr_val > 0:
            fvg_size_atr = fvg_height / atr_val
            if fvg_size_atr < MIN_FVG_SIZE_ATR:
                logger.debug("FVG too small (%.3f ATR) — skipping.", fvg_size_atr)
                return None
            if fvg_size_atr > MAX_FVG_SIZE_ATR:
                logger.debug("FVG too wide (%.3f ATR) — unreliable.", fvg_size_atr)
                return None

        # ── Step 5: OB confluence ─────────────────────────────────────
        obs = detect_order_blocks(df, term=self.ms_term)
        has_ob = False
        for ob in obs:
            if ob.get("type") != trend:
                continue
            ob_top = float(ob.get("top", 0))
            ob_bot = float(ob.get("bottom", 0))
            # Overlap with FVG
            if ob_bot <= fvg_top and ob_top >= fvg_bottom:
                has_ob = True
                break
            # Adjacent: within one FVG-height
            ob_mid = (ob_top + ob_bot) / 2
            fvg_mid = (fvg_top + fvg_bottom) / 2
            if abs(ob_mid - fvg_mid) <= fvg_height:
                has_ob = True
                break

        if self.require_ob and not has_ob:
            logger.debug("No OB near FVG — skipping.")
            return None

        # ── Step 6: Entry / SL / TP calculation ───────────────────────
        liquidity = detect_liquidity_levels(df)
        price_at_zone = _price_in_fvg(current_price, fvg_top, fvg_bottom)

        if trend == "bullish":
            entry = fvg_bottom if price_at_zone else current_price
            # SL beyond the swept swing low
            sl_candidates = [l for l in liquidity if l["dir"] == "low" and l["price"] < entry]
            stop_loss = sweep_data["sweep_px"] * 0.998 if sl_candidates else fvg_bottom - fvg_height * 0.5
            # TP at nearest high-level liquidity or 2x risk
            tp_candidates = sorted(
                [l for l in liquidity if l["dir"] == "high" and l["price"] > entry],
                key=lambda x: (x["price"] - entry),
            )
            take_profit = tp_candidates[0]["price"] if tp_candidates else entry + (entry - stop_loss) * 2
        else:
            entry = fvg_top if price_at_zone else current_price
            sl_candidates = [l for l in liquidity if l["dir"] == "high" and l["price"] > entry]
            stop_loss = sweep_data["sweep_px"] * 1.002 if sl_candidates else fvg_top + fvg_height * 0.5
            tp_candidates = sorted(
                [l for l in liquidity if l["dir"] == "low" and l["price"] < entry],
                key=lambda x: (entry - x["price"]),
            )
            take_profit = tp_candidates[0]["price"] if tp_candidates else entry - (stop_loss - entry) * 2

        # R/R gate
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        if risk < 0.001:
            return None
        rr = reward / risk
        if rr < self.min_rr:
            logger.debug("R/R %.2f below minimum %.2f.", rr, self.min_rr)
            return None

        # ── Confidence calculation (0–100) ────────────────────────────
        conf = 30.0  # base: valid setup
        if has_ob:
            conf += 20.0
        if price_at_zone:
            conf += 15.0
        if rr >= 2.0:
            conf += 10.0
        elif rr >= 2.5:
            conf += 15.0
        if eq is not None:
            conf += 10.0  # location context available
        if self._is_crypto:
            conf += 5.0  # 24/7 no session restriction
        conf = min(95.0, conf)  # never 100 from strategy alone

        sig_dir: Literal["BUY", "SELL"] = "BUY" if trend == "bullish" else "SELL"
        reason = (
            f"{'Bullish' if trend == 'bullish' else 'Bearish'} "
            f"FVG [{fvg_bottom:.2f}–{fvg_top:.2f}] | "
            f"Sweep @ {sweep_data['sweep_px']:.2f} | "
            f"OB: {'yes' if has_ob else 'no'} | "
            f"At zone: {'yes' if price_at_zone else 'no'} | "
            f"R/R: {rr:.2f}"
        )

        return TradeSignal(
            direction=sig_dir,
            entry=round(entry, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            risk_reward=round(rr, 2),
            confidence=round(conf, 1),
            fvg_zone=(round(fvg_bottom, 4), round(fvg_top, 4)),
            trend=trend,
            reason=reason,
            at_zone=price_at_zone,
            raw_data={
                "current_price": current_price,
                "has_ob":        has_ob,
                "has_engulf":    False,
                "ms_events":     len(structure),
                "sweep_px":      sweep_data["sweep_px"],
                "fvg_height":    fvg_height,
                "atr":           atr_val,
                "htf_trend":     htf_trend,
            },
        )

    # ------------------------------------------------------------------ #
    # Relaxed setup finder (UI display only — no sweep/OB requirement)   #
    # ------------------------------------------------------------------ #

    def find_setup(self, df: Optional[pd.DataFrame] = None) -> Optional[TradeSignal]:
        """Return a pending setup for UI display only.

        Unlike ``analyze()``, this method:
        - Does NOT require a sweep + CHoCH pattern
        - Does NOT require OB confluence
        - Does NOT enforce kill zones
        - Only looks for active FVGs aligned with HTF trend

        The returned signal should **never** be auto-executed.
        It populates the Entry / Stop / Target lines on the chart.
        """
        if df is None:
            df = self.fetch_data()
        if len(df) < 40:
            return None

        current_price = float(df["Close"].iloc[-1])

        try:
            htf_df = self.fetch_data(interval=self.htf_interval)
            htf_trend = _current_trend(detect_market_structure(htf_df, term=self.ms_term))
        except Exception:
            htf_trend = _current_trend(detect_market_structure(df, term=self.ms_term))

        if htf_trend is None:
            return None

        fvg_df = detect_fvg(df)
        if fvg_df.empty:
            return None

        aligned_fvgs = fvg_df[
            (fvg_df["active"] == True) &
            (fvg_df["type"] == htf_trend)
        ].copy()
        if aligned_fvgs.empty:
            return None

        # Closest to current price first
        aligned_fvgs["_dist"] = aligned_fvgs["bottom"].apply(
            lambda b: abs(current_price - b)
        )
        aligned_fvgs = aligned_fvgs.sort_values("_dist")

        liquidity = detect_liquidity_levels(df)

        for _, fvg_row in aligned_fvgs.iterrows():
            fvg_top    = float(fvg_row["top"])
            fvg_bottom = float(fvg_row["bottom"])
            fvg_height = fvg_top - fvg_bottom

            if htf_trend == "bullish":
                entry     = fvg_bottom
                stop_loss = fvg_bottom - fvg_height * 0.5
                tp_cands = sorted(
                    [l for l in liquidity if l["dir"] == "high" and l["price"] > entry],
                    key=lambda x: (x["price"] - entry),
                )
            else:
                entry     = fvg_top
                stop_loss = fvg_top + fvg_height * 0.5
                tp_cands = sorted(
                    [l for l in liquidity if l["dir"] == "low" and l["price"] < entry],
                    key=lambda x: (entry - x["price"]),
                )

            if not tp_cands:
                continue

            take_profit = tp_cands[0]["price"]
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            if risk == 0:
                continue
            rr = reward / risk
            if rr < 1.0:
                continue

            dist_pct = abs(current_price - entry) / max(entry, 1e-9) * 100
            sig_dir: Literal["BUY", "SELL"] = "BUY" if htf_trend == "bullish" else "SELL"

            return TradeSignal(
                direction=sig_dir,
                entry=round(entry, 4),
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                risk_reward=round(rr, 2),
                confidence=20.0,  # low — pending only
                fvg_zone=(round(fvg_bottom, 4), round(fvg_top, 4)),
                trend=htf_trend,
                reason=f"Pending FVG [{fvg_bottom:.2f}–{fvg_top:.2f}] ({dist_pct:.1f}% away)",
                at_zone=False,
                raw_data={"current_price": current_price, "pending": True},
            )

        return None

    # ------------------------------------------------------------------ #
    # Convenience                                                         #
    # ------------------------------------------------------------------ #

    def describe(self) -> str:
        return (
            f"SMC Scalper — {self.symbol} ({self.interval}, HTF={self.htf_interval})\n"
            f"  MS term: {self.ms_term} | Min R/R: {self.min_rr} | Require OB: {self.require_ob}"
        )
