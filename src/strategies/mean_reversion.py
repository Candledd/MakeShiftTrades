"""Equity Mean Reversion Strategy — Playbook 1B

Catches overextensions in range-bound / low-volatility regimes.
Enters only after a **failed extension** (not on the first band touch),
reducing the risk of catching a falling knife.

Removed indicators (redundant):
  - Stochastic RSI   (duplicates info from RSI + BB distance)
  - Z-score          (duplicates info from BB position)
  - HTF trend        (mean reversion should NOT require trend agreement)

Remaining indicators:
  - Bollinger Bands  (primary edge — band touch)
  - RSI              (secondary confirmation — exhaustion)
  - VWAP             (profit target)
  - Volume           (climax-fade confirmation)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.strategies import BaseStrategy, StrategySignal

logger = logging.getLogger(__name__)

import config

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


# ── Named Constants (replacing magic numbers) ──────────────────────────────
# Volatility thresholds
MR_LOW_VOL_ATR_PCT = 0.003          # ATR% below which is considered low vol
MR_QUIET_LOW_VOL_ATR_PCT = 0.002    # Even lower ATR% for quiet playbook boost

# Band proximity factors (symmetric ~1% from the band)
MR_CLOSE_ABOVE_BAND_FACTOR = 0.99   # close > band * 0.99 (bounce / proximity)
MR_CLOSE_BELOW_BAND_FACTOR = 1.01   # close < band * 1.01 (rejection / proximity)

# Deep break thresholds for climax playbook
MR_DEEP_BREAK_LONG_FACTOR = 0.998   # prev_low < prev_lower * 0.998
MR_DEEP_BREAK_SHORT_FACTOR = 1.002  # prev_high > prev_upper * 1.002

# RSI thresholds for quiet playbook
MR_QUIET_RSI_MAX = 45.0             # RSI must be below this for quiet BUY
MR_QUIET_RSI_MIN = 55.0             # RSI must be above this for quiet SELL

# RSI extremes for climax confidence boost
MR_CLIMAX_RSI_EXTREME_LONG = 25.0   # RSI below this for climax BUY boost
MR_CLIMAX_RSI_EXTREME_SHORT = 75.0  # RSI above this for climax SELL boost

# VWAP deviation factors
MR_VWAP_DISCOUNT_FACTOR = 0.998     # close < vwap * 0.998 (quiet BUY)
MR_VWAP_PREMIUM_FACTOR = 1.002      # close > vwap * 1.002 (quiet SELL)
MR_VWAP_DEEP_DISCOUNT = 0.996       # close < vwap * 0.996 (confidence boost)
MR_VWAP_DEEP_PREMIUM = 1.004        # close > vwap * 1.004 (confidence boost)

# Volume / climax thresholds
MR_CLIMAX_VOL_RATIO = 2.0           # vol_ratio > 2.0 for climax boost

# Wick confirmation factor (fraction of candle body)
MR_WICK_CONFIRMATION_FRAC = 0.3     # wick >= 30% of candle for reversal confirmation

# Stop loss adjustment for quiet playbook
MR_QUIET_STOP_ADJ = 0.7             # stop_mult reduced by 30% for quiet playbook

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
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        atr_val = self.compute_atr(df)

        # -- Bollinger Bands ---------------------------------------------
        sma20 = close.rolling(config.MR_BB_PERIOD).mean()
        std20 = close.rolling(config.MR_BB_PERIOD).std()
        upper = sma20 + config.MR_BB_STD * std20
        lower = sma20 - config.MR_BB_STD * std20

        # -- RSI ---------------------------------------------------------
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=config.MR_RSI_PERIOD - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=config.MR_RSI_PERIOD - 1, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # -- VWAP (daily-resetting intraday VWAP) -------------------------
        # Group by trading day so the cumulative sum resets for each session
        cum_pv = (df["Close"] * df["Volume"]).groupby(df.index.normalize()).cumsum()
        cum_vol = df["Volume"].groupby(df.index.normalize()).cumsum()
        vwap = cum_pv / cum_vol

        # -- Current values (last bar) -----------------------------------
        current_close = float(close.iloc[-1])
        current_sma20 = float(sma20.iloc[-1])
        current_rsi = float(rsi.iloc[-1])
        current_upper = float(upper.iloc[-1])
        current_lower = float(lower.iloc[-1])
        current_vwap = float(vwap.iloc[-1])
        low_last = float(low.iloc[-1])
        high_last = float(high.iloc[-1])

        # -- Volume ratio ------------------------------------------------
        vol_mean = volume.rolling(20).mean().iloc[-1]
        if vol_mean <= 0:
            logger.debug("%s %s: volume mean <= 0, skipping", self.name, ticker)
            return None
        vol_ratio = float(volume.iloc[-1]) / float(vol_mean)

        # -- Determine Sub-Playbook --------------------------------------
        direction: Optional[str] = None
        playbook: Optional[str] = None
        
        target_price = current_sma20 # Default target
        
        # Volatility check for Quiet Playbook
        atr_pct = atr_val / current_close
        is_low_vol = atr_pct < MR_LOW_VOL_ATR_PCT
        
        for offset in range(2, min(len(df), 5)):
            prev_low = float(low.iloc[-offset])
            prev_high = float(high.iloc[-offset])
            prev_lower = float(lower.iloc[-offset])
            prev_upper = float(upper.iloc[-offset])
            
            # --- Playbook A: Climax Failed-Extension ---
            # Requires elevated volume, deep band break, and extreme RSI
            is_deep_break_long = prev_low < prev_lower * MR_DEEP_BREAK_LONG_FACTOR
            is_deep_break_short = prev_high > prev_upper * MR_DEEP_BREAK_SHORT_FACTOR
            
            if (
                is_deep_break_long
                and current_close > current_lower * MR_CLOSE_ABOVE_BAND_FACTOR
                and current_rsi < config.MR_RSI_OVERSOLD
                and vol_ratio > max(1.2, config.MR_VOL_SPIKE_MULT)
                and vol_ratio <= 10.0
            ):
                direction = "BUY"
                playbook = "A_CLIMAX"
                target_price = current_sma20
                break
                
            if (
                is_deep_break_short
                and current_close < current_upper * MR_CLOSE_BELOW_BAND_FACTOR
                and current_rsi > config.MR_RSI_OVERBOUGHT
                and vol_ratio > max(1.2, config.MR_VOL_SPIKE_MULT)
                and vol_ratio <= 10.0
            ):
                direction = "SELL"
                playbook = "A_CLIMAX"
                target_price = current_sma20
                break
                
            # --- Playbook B: Quiet VWAP Deviation ---
            # Requires low realized vol, VWAP deviation, less extreme RSI, no vol spike needed
            if is_low_vol:
                if (
                    prev_low < prev_lower * MR_CLOSE_BELOW_BAND_FACTOR
                    and current_close > current_lower * MR_CLOSE_ABOVE_BAND_FACTOR
                    and current_rsi < MR_QUIET_RSI_MAX
                    and current_close < current_vwap * MR_VWAP_DISCOUNT_FACTOR
                ):
                    direction = "BUY"
                    playbook = "B_QUIET"
                    target_price = current_vwap
                    break
                    
                if (
                    prev_high > prev_upper * MR_CLOSE_ABOVE_BAND_FACTOR
                    and current_close < current_upper * MR_CLOSE_BELOW_BAND_FACTOR
                    and current_rsi > MR_QUIET_RSI_MIN
                    and current_close > current_vwap * MR_VWAP_PREMIUM_FACTOR
                ):
                    direction = "SELL"
                    playbook = "B_QUIET"
                    target_price = current_vwap
                    break

        if direction is None:
            return None

        # -- Entry / Exit -----------------------------------------------
        entry = current_close
        order_type = "LIMIT" if config.USE_LIMIT_ORDERS_MR else "MARKET"

        # Tighter stop for Quiet Playbook
        stop_mult = config.MR_STOP_MULT if playbook == "A_CLIMAX" else config.MR_STOP_MULT * MR_QUIET_STOP_ADJ
        stop_distance = stop_mult * atr_val
        stop_loss = entry - stop_distance if direction == "BUY" else entry + stop_distance

        take_profit = entry + (target_price - entry) * config.MR_TP_TARGET_MULT

        # R/R gate: TP must meet the minimum R/R ratio to survive slippage
        tp_distance = abs(take_profit - entry)
        sl_distance = abs(entry - stop_loss)
        
        # Ensure target is logically in front of entry
        if (direction == "BUY" and take_profit <= entry) or (direction == "SELL" and take_profit >= entry):
            return None
            
        if tp_distance < sl_distance * config.MR_MIN_RR:
            logger.debug(
                "%s %s: bad R/R (entry=%.2f, SL=%.2f, TP=%.2f)",
                self.name, ticker, entry, stop_loss, take_profit,
            )
            return None

        # -- Confidence (0–90) ------------------------------------------
        confidence = 40.0

        if playbook == "A_CLIMAX":
            # Climax rewards extreme RSI, high volume, and deep wicks
            if direction == "BUY" and current_rsi < MR_CLIMAX_RSI_EXTREME_LONG:
                confidence += 15.0
            elif direction == "SELL" and current_rsi > MR_CLIMAX_RSI_EXTREME_SHORT:
                confidence += 15.0
                
            if vol_ratio > MR_CLIMAX_VOL_RATIO:
                confidence += 10.0
                
        elif playbook == "B_QUIET":
            # Quiet rewards low vol and tight price action, but caps confidence slightly lower
            # since it is not a major structural reversal
            confidence = 35.0
            if atr_pct < MR_QUIET_LOW_VOL_ATR_PCT:
                confidence += 15.0
            if direction == "BUY" and current_close < current_vwap * MR_VWAP_DEEP_DISCOUNT:
                confidence += 15.0
            elif direction == "SELL" and current_close > current_vwap * MR_VWAP_DEEP_PREMIUM:
                confidence += 15.0

        # Wick on the failed-extension bar (reversal confirmation) applies to both
        for offset in range(2, min(len(df), 5)):
            if direction == "BUY" and float(close.iloc[-offset]) > float(low.iloc[-offset]) + (float(close.iloc[-offset]) - float(low.iloc[-offset])) * MR_WICK_CONFIRMATION_FRAC:
                confidence += 10.0
                break
            elif direction == "SELL" and float(close.iloc[-offset]) < float(high.iloc[-offset]) - (float(high.iloc[-offset]) - float(close.iloc[-offset])) * MR_WICK_CONFIRMATION_FRAC:
                confidence += 10.0
                break

        confidence = min(90.0, confidence)

        # -- Reason -----------------------------------------------------
        playbook_name = "Climax Fade" if playbook == "A_CLIMAX" else "Quiet VWAP Reversion"
        reason = (
            f"Mean reversion {direction} ({playbook_name}): "
            f"RSI={current_rsi:.1f}, "
            f"VolRatio={vol_ratio:.1f}x"
        )

        # -- Trailing stop logic coupled to playbook --------------------
        #   A_CLIMAX → "vwap"      (trail along VWAP as it moves)
        #   B_QUIET  → "vwap"      (full exit via VWAP; could become "no_runner")
        trail_logic = "vwap"

        # -- OU Half-Life Dynamic Time Stop -----------------------------
        # Use np.log(close) as the input series for the AR(1) state-space model
        # S_t = α + β·S_{t-1} + ε_t  (raw log-price, no detrending needed)
        detrended = np.log(close).dropna().to_numpy()
        half_life = _estimate_ou_half_life(detrended)
        if half_life is not None:
            # 2H time stop: gives 2 half-lives for the trade to revert
            time_stop_bars = max(5, min(100, int(np.ceil(2.0 * half_life))))
            logger.debug(
                "%s %s: OU half-life=%.1f bars → time_stop=%d bars",
                self.name, ticker, half_life, time_stop_bars,
            )
        else:
            time_stop_bars = 6  # sensible fallback (existing default)
            logger.debug(
                "%s %s: OU half-life N/A → time_stop=%d bars (fallback)",
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
