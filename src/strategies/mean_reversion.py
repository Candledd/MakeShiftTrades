"""Equity Mean Reversion Strategy — Pure Z-Score & VWAP Deviation Model

Entries are triggered when price is both statistically overextended (Z-Score
exceeds configurable Bollinger Band sigma threshold) AND trading at a
meaningful premium/discount relative to VWAP (≥ 0.5%). This dual-condition
gate eliminates the arbitrary wick and volume rules previously used.

Indicators:
  - Bollinger Bands  (Z-Score computation — statistical overextension)
  - RSI              (secondary confirmation — exhaustion)
  - VWAP             (deviation filter + profit target)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.strategies import BaseStrategy, StrategySignal

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Ornstein-Uhlenbeck Half-Life estimation
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_ou_half_life(
    prices: np.ndarray,
    lookback: int = 80,
) -> Optional[float]:
    """Estimate the OU mean-reversion half-life via Kalman-filtered AR(1).

    Replaces the rolling ``numpy.polyfit`` regression with a 1D Kalman
    Filter to avoid the "ghosting" effect of fixed-window estimators.

    Formulates the AR(1) process :math:`S_t = \\alpha + \\beta S_{t-1} + \\varepsilon_t`
    into state-space form where the hidden state is :math:`[\\alpha,\\, \\beta]^T`.

    The discrete-time half-life is computed as:

        H = -\\ln(2) / \\ln(\\beta)

    where *β* is the AR(1) slope from the final Kalman state estimate.

    Parameters
    ----------
    prices : np.ndarray
        Log-price series (``np.log(close)``) with at least
        *lookback* + 2 observations.  The AR(1) state-space model
        :math:`S_t = \\alpha + \\beta S_{t-1}` operates directly on
        the (log-)price without pre-detrending.
    lookback : int
        How many price observations to regress over (default 80).

    Returns
    -------
    float or None
        Half-life in bars, or None if the estimate is not mathematically
        valid (β ≥ 1 — explosive; β ≤ 0 — not mean-reverting; or half-life
        outside the 5–50 bar sanity window).
    """
    if len(prices) < lookback + 2:
        return None

    # Trailing window (avoid copying the full array)
    seg = prices[-(lookback + 1):]

    # Guard against degenerate series (all identical / near-zero values)
    seg_var = np.var(seg)
    if seg_var < 1e-12:
        return None

    # ── Kalman filter setup ────────────────────────────────────────────
    # State vector: x = [α, β]^T   (S_t = α + β·S_{t-1} + ε_t)
    # Observation:         y_t = S_t
    # Measurement vector:  H_k = [1, S_{k-1}]

    # Process noise covariance (random walk prior — near-constant state)
    Q = np.eye(2) * 1e-4

    # Observation noise variance — a fraction of the data variance,
    # corresponding to the AR(1) innovation variance.
    R = 0.5 * seg_var + 1e-8

    # Initial state: start with zero coefficients
    x = np.zeros(2)
    # Initial covariance — high uncertainty
    P = np.eye(2) * 10.0

    I = np.eye(2)

    # ── Recursive Kalman update ────────────────────────────────────────
    for i in range(1, len(seg)):
        y = seg[i]                     # S_t
        H = np.array([1.0, seg[i - 1]])  # [1, S_{t-1}]

        # Predict (state transition is identity — random walk)
        P_pred = P + Q

        # Update
        innovation = y - H @ x
        innov_cov = H @ P_pred @ H + R
        K = P_pred @ H / innov_cov

        x = x + K * innovation
        P = (I - np.outer(K, H)) @ P_pred

    # ── Extract β from final state estimate ────────────────────────────
    beta = x[1]

    # Edge cases — non-stationary or non-reverting dynamics
    #   β >= 1  → explosive (unit root or worse)
    #   β <= -0.2 → allow slight oscillatory mean reversion, but reject extreme noise
    if beta >= 1.0:
        return None
    if beta <= -0.2:
        return None

    # Exact discrete-time half-life for AR(1) envelope: |β|^H = 0.5
    # Use abs(beta) so oscillatory mean reversion (beta < 0) calculates correctly
    half_life = -np.log(2.0) / np.log(abs(beta))

    # Bounds check — half-life must be meaningful for intraday mean reversion
    if half_life < 5.0 or half_life > 50.0:
        return None

    return half_life


# Minimum bars required for Kalman filter to converge
MR_MIN_BARS = getattr(config, 'MR_MIN_BARS', 100)  # Kalman filter requires 82 bars; use 100 as margin

class MeanReversionStrategy(BaseStrategy):
    """True intraday mean-reversion strategy for S&P 500 and NASDAQ indices.

    Works best in range-bound or low-volatility regimes.  Does **not**
    require higher-timeframe trend agreement — it is designed to fade
    short-term overextensions regardless of daily direction.

    Both **BUY** and **SELL** signals are emitted.
    """

    def __init__(self) -> None:
        super().__init__(
            name="mean_reversion",
            tickers=["SPY", "QQQ"],
            timeframe="15m",
            period="5d",
        )

    def analyze(self, df: pd.DataFrame, ticker: str) -> Optional[StrategySignal]:
        # -- Minimum bars ------------------------------------------------
        if len(df) < MR_MIN_BARS:
            logger.debug("%s: too few bars (%d)", self.name, len(df))
            return None

        close = df["Close"]

        # ── VPIN toxicity gate ------------------------------------------
        if len(df) >= config.VPIN_WINDOW:
            vpin = self.compute_vpin(df, config.VPIN_WINDOW)
            if vpin > config.VPIN_MR_BLOCK_THRESHOLD:
                logger.debug(
                    "%s %s: VPIN toxicity too high to fade (%.4f > %.2f).",
                    self.name, ticker, vpin, config.VPIN_MR_BLOCK_THRESHOLD,
                )
                return None

        atr_val = self.compute_atr(df)

        # -- Bollinger Bands (for Z-Score) --------------------------------
        sma20 = close.rolling(config.MR_BB_PERIOD).mean()
        std20 = close.rolling(config.MR_BB_PERIOD).std()

        # -- RSI ---------------------------------------------------------
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=config.MR_RSI_PERIOD - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=config.MR_RSI_PERIOD - 1, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # -- VWAP (daily-resetting intraday VWAP) -------------------------
        cum_pv = (df["Close"] * df["Volume"]).groupby(df.index.normalize()).cumsum()
        cum_vol = df["Volume"].groupby(df.index.normalize()).cumsum()
        vwap = cum_pv / cum_vol

        # -- Current values (last bar) -----------------------------------
        current_close = float(close.iloc[-1])
        current_sma20 = float(sma20.iloc[-1])
        current_rsi = float(rsi.iloc[-1])
        current_vwap = float(vwap.iloc[-1])
        current_std20 = float(std20.iloc[-1])

        if current_std20 <= 0 or current_vwap <= 0 or pd.isna(current_vwap):
            logger.debug("%s %s: invalid std20 or vwap, skipping", self.name, ticker)
            return None

        # -- Z-Score ------------------------------------------------------
        z_score = (current_close - current_sma20) / current_std20

        # -- VWAP Deviation ----------------------------------------------
        vwap_dev = (current_close - current_vwap) / current_vwap

        # -- Trigger Rules (Fast Checks First) ---------------------------
        direction: Optional[str] = None

        mr_vwap_dev_pct = getattr(config, 'MR_VWAP_DEV_PCT', 0.005)
        if z_score < -config.MR_BB_STD and vwap_dev < -mr_vwap_dev_pct and current_rsi < config.MR_RSI_OVERSOLD:
            direction = "BUY"
        elif z_score > config.MR_BB_STD and vwap_dev > mr_vwap_dev_pct and current_rsi > config.MR_RSI_OVERBOUGHT:
            direction = "SELL"

        if direction is None:
            return None

        # -- Institutional Filters: Kalman Regime & Volume Capitulation ---
        detrended = np.log(close).dropna().to_numpy()
        half_life = _estimate_ou_half_life(detrended)
        
        # 1. Statistical Regime Gate (OU Process)
        # If the Kalman filter AR(1) state shows the series is explosive (trending) 
        # rather than mean-reverting, we block the trade.
        if half_life is None:
            logger.debug("%s %s: Kalman filter indicates non-reverting regime, skipping.", self.name, ticker)
            return None

        # 2. Volume Capitulation Gate
        # Institutional sweeps cause volume spikes at extremes. Low-volume drift is a trap.
        avg_volume = float(df["Volume"].rolling(config.MR_BB_PERIOD).mean().iloc[-1])
        current_volume = float(df["Volume"].iloc[-1])
        vol_multiplier = config.MR_VOL_SPIKE_MULT if config.MR_VOL_SPIKE_MULT > 0 else 1.25
        
        if current_volume < (avg_volume * vol_multiplier):
            logger.debug("%s %s: Insufficient volume capitulation (Vol: %d < Avg: %d * %.2f)", self.name, ticker, current_volume, avg_volume, vol_multiplier)
            return None

        # -- Entry / Exit -----------------------------------------------
        entry = current_close
        order_type = "LIMIT" if config.USE_LIMIT_ORDERS_MR else "MARKET"

        # Target the SMA20 (Bollinger Band mean) — this is the statistical
        # center the Z-score measures deviation from, so it is the natural
        # mean-reversion target.  Previously we used min/max(VWAP, SMA20)
        # which picked the *closer* mean, producing tiny TP distances that
        # — combined with the stop clamp below — created stops tight enough
        # to get hit on noise, destroying expectancy.
        target_price = current_sma20

        # Stop Loss: use the raw ATR-based distance.  The old code clamped
        # stop_distance to tp_distance * 1.5, which created artificially
        # tight stops (often < 0.2% of price) that were hit on random noise
        # before the reversion could play out.  The MR_MIN_RR gate below
        # still filters trades whose R:R is unacceptable.
        stop_distance = config.MR_STOP_MULT * atr_val

        stop_loss = entry - stop_distance if direction == "BUY" else entry + stop_distance

        take_profit = target_price
        tp_distance = abs(target_price - entry)
        sl_distance = abs(entry - stop_loss)

        # Directional sanity check — TP must be on the correct side of entry
        if (direction == "BUY" and take_profit <= entry) or (direction == "SELL" and take_profit >= entry):
            return None

        # R/R quality gate (tp_distance is already computed above from target_price)
        if tp_distance < sl_distance * config.MR_MIN_RR:
            logger.debug(
                "%s %s: bad R/R (entry=%.2f, SL=%.2f, TP=%.2f, ratio=%.2f) < %.2f",
                self.name, ticker, entry, stop_loss, take_profit, (tp_distance / sl_distance if sl_distance > 0 else 0), config.MR_MIN_RR
            )
            return None

        # -- Confidence (0–90) ------------------------------------------
        confidence = min(90.0, abs(z_score) * 25.0)

        # -- Reason -----------------------------------------------------
        reason = (
            f"Mean reversion {direction} (Z-Score & VWAP): "
            f"Z={z_score:.2f}, VWAP Dev={vwap_dev*100:.2f}%"
        )

        # -- Trailing stop logic ----------------------------------------
        trail_logic = "vwap"

        # -- OU Half-Life Dynamic Time Stop -----------------------------
        # half_life is pre-calculated in the institutional filter above
        time_stop_bars = max(5, min(100, int(np.ceil(2.0 * half_life))))
        logger.debug(
            "%s %s: OU half-life=%.1f bars \u2192 time_stop=%d bars",
            self.name, ticker, half_life, time_stop_bars,
        )

        # -- Record signal ----------------------------------------------
        self.record_signal(ticker)

        return StrategySignal(
            ticker=ticker,
            direction=direction,
            entry=round(entry, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            confidence=round(confidence, 1),
            strategy_name=self.name,
            timeframe=self.timeframe,
            reason=reason,
            atr=round(atr_val, 4),
            timestamp=datetime.now(timezone.utc),
            order_type=order_type,
            time_stop_bars=time_stop_bars,
            trailing_stop_logic=trail_logic,
        )
