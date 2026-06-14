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
    """Estimate the OU mean-reversion half-life via AR(1) regression.

    Fits  ΔS_t = α + β·S_{t-1} + ε_t  using ``numpy.polyfit`` (deg=1).

    The discrete-time half-life is computed as:

        H = -ln(2) / ln(1 + β)

    where *β* is the slope (theta) of the regression.  This formula is
    exact for an AR(1) process, unlike the continuous approximation
    ln(2)/|θ|.

    Parameters
    ----------
    prices : np.ndarray
        Detrended price series (e.g. price - SMA) with at least
        *lookback* + 2 observations.
    lookback : int
        How many price-difference observations to regress over (default 80).

    Returns
    -------
    float or None
        Half-life in bars, or None if the estimate is not mathematically
        valid (β ≥ 0 — not mean-reverting; β ≤ -1 — non-stationary /
        oscillating explosive; or half-life outside the 5–50 bar sanity
        window).
    """
    if len(prices) < lookback + 2:
        return None

    # Take the trailing segment (avoid copying the full array)
    seg = prices[-(lookback + 1):]

    x = seg[:-1]          # S_{t-1}
    y = np.diff(seg)      # ΔS_t

    # Demean the regressor for numerical stability
    x_mean = np.mean(x)
    x_demeaned = x - x_mean

    # Guard against degenerate series (all identical prices)
    var_x = np.var(x_demeaned)
    if var_x < 1e-12:
        return None

    # polyfit(x, y, 1) → [slope, intercept]  (y = slope·x + intercept)
    try:
        beta, _alpha = np.polyfit(x_demeaned, y, 1)
    except np.linalg.LinAlgError:
        return None

    # Edge cases — non-stationary or explosive dynamics
    #   β >= 0   → unit root or explosive (not mean-reverting)
    #   β <= -1  → oscillating explosive (non-stationary)
    if beta >= 0:
        return None
    if beta <= -1.0:
        return None

    # Exact discrete-time half-life for an AR(1): β^{H} = 0.5
    #   H = -ln(2) / ln(1 + β)
    half_life = -np.log(2.0) / np.log(1.0 + beta)

    # Bounds check — half-life must be meaningful for intraday mean reversion
    if half_life < 5.0 or half_life > 50.0:
        return None

    return half_life

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

    def compute_stop_loss(
        self,
        entry: float,
        direction: str,
        atr: float,
    ) -> float:
        stop_distance = config.MR_STOP_MULT * atr
        if direction == "BUY":
            return entry - stop_distance
        else:
            return entry + stop_distance

    def analyze(self, df: pd.DataFrame, ticker: str) -> Optional[StrategySignal]:
        # -- Minimum bars ------------------------------------------------
        if len(df) < 30:
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

        # -- VWAP --------------------------------------------------------
        vwap = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

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
        is_low_vol = atr_pct < 0.003  # less than 0.3% ATR
        
        for offset in range(2, min(len(df), 5)):
            prev_low = float(low.iloc[-offset])
            prev_high = float(high.iloc[-offset])
            prev_lower = float(lower.iloc[-offset])
            prev_upper = float(upper.iloc[-offset])
            
            # --- Playbook A: Climax Failed-Extension ---
            # Requires elevated volume, deep band break, and extreme RSI
            is_deep_break_long = prev_low < prev_lower * 0.998
            is_deep_break_short = prev_high > prev_upper * 1.002
            
            if (
                is_deep_break_long
                and current_close > current_lower * 0.99
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
                and current_close < current_upper * 1.01
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
                    prev_low < prev_lower * 1.01
                    and current_close > current_lower * 0.99
                    and current_rsi < 45.0
                    and current_close < current_vwap * 0.998
                ):
                    direction = "BUY"
                    playbook = "B_QUIET"
                    target_price = current_vwap
                    break
                    
                if (
                    prev_high > prev_upper * 0.99
                    and current_close < current_upper * 1.01
                    and current_rsi > 55.0
                    and current_close > current_vwap * 1.002
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
        stop_mult = config.MR_STOP_MULT if playbook == "A_CLIMAX" else config.MR_STOP_MULT * 0.7
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
            if direction == "BUY" and current_rsi < 25:
                confidence += 15.0
            elif direction == "SELL" and current_rsi > 75:
                confidence += 15.0
                
            if vol_ratio > 2.0:
                confidence += 10.0
                
        elif playbook == "B_QUIET":
            # Quiet rewards low vol and tight price action, but caps confidence slightly lower
            # since it is not a major structural reversal
            confidence = 35.0
            if atr_pct < 0.002:
                confidence += 15.0
            if direction == "BUY" and current_close < current_vwap * 0.996:
                confidence += 15.0
            elif direction == "SELL" and current_close > current_vwap * 1.004:
                confidence += 15.0

        # Wick on the failed-extension bar (reversal confirmation) applies to both
        for offset in range(2, min(len(df), 5)):
            if direction == "BUY" and float(close.iloc[-offset]) > float(low.iloc[-offset]) + (float(close.iloc[-offset]) - float(low.iloc[-offset])) * 0.3:
                confidence += 10.0
                break
            elif direction == "SELL" and float(close.iloc[-offset]) < float(high.iloc[-offset]) - (float(high.iloc[-offset]) - float(close.iloc[-offset])) * 0.3:
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
        # Use detrended price (close - SMA) to isolate mean-reverting component
        detrended = (close - sma20).dropna().to_numpy()
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
