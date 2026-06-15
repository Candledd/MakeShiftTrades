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
    Q = np.eye(2) * 1e-6

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
    #   β <= 0  → not mean-reverting (white noise or oscillatory)
    if beta >= 1.0:
        return None
    if beta <= 0.0:
        return None

    # Exact discrete-time half-life for AR(1): β^H = 0.5
    half_life = -np.log(2.0) / np.log(beta)

    # Bounds check — half-life must be meaningful for intraday mean reversion
    if half_life < 5.0 or half_life > 50.0:
        return None

    return half_life


# Minimum bars required for Kalman filter to converge
MR_MIN_BARS = 100                   # Kalman filter requires 82 bars; use 100 as margin

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

        # -- Trigger Rules -----------------------------------------------
        direction: Optional[str] = None

        if z_score < -config.MR_BB_STD and vwap_dev < -0.005:
            direction = "BUY"
        elif z_score > config.MR_BB_STD and vwap_dev > 0.005:
            direction = "SELL"

        if direction is None:
            return None

        # -- Entry / Exit -----------------------------------------------
        entry = current_close
        order_type = "LIMIT" if config.USE_LIMIT_ORDERS_MR else "MARKET"

        # Target is VWAP
        target_price = current_vwap

        stop_distance = config.MR_STOP_MULT * atr_val
        stop_loss = entry - stop_distance if direction == "BUY" else entry + stop_distance

        take_profit = target_price

        # R/R gate: TP must meet the minimum R/R ratio to survive slippage
        tp_distance = abs(take_profit - entry)
        sl_distance = abs(entry - stop_loss)

        if (direction == "BUY" and take_profit <= entry) or (direction == "SELL" and take_profit >= entry):
            return None

        if tp_distance < sl_distance * config.MR_MIN_RR:
            logger.debug(
                "%s %s: bad R/R (entry=%.2f, SL=%.2f, TP=%.2f)",
                self.name, ticker, entry, stop_loss, take_profit,
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
        detrended = np.log(close).dropna().to_numpy()
        half_life = _estimate_ou_half_life(detrended)
        if half_life is not None:
            time_stop_bars = max(5, min(100, int(np.ceil(2.0 * half_life))))
            logger.debug(
                "%s %s: OU half-life=%.1f bars \u2192 time_stop=%d bars",
                self.name, ticker, half_life, time_stop_bars,
            )
        else:
            time_stop_bars = 6
            logger.debug(
                "%s %s: OU half-life N/A \u2192 time_stop=%d bars (fallback)",
                self.name, ticker, time_stop_bars,
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
