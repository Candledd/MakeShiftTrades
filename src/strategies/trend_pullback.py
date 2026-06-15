"""Equity Trend Pullback Strategy — Playbook 1A

Buys intraday pullbacks within a confirmed daily bullish trend.
Long-only — entries align with higher-timeframe trend direction.
Uses Bollinger Band pullback, RSI weakness, and volume climax fade as
the primary setup.  No short entries.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.indicators import calc_volume_profile
from src.strategies import BaseStrategy, StrategySignal

logger = logging.getLogger(__name__)

# -- Volume Profile parameters (fallbacks if config not set) -------------------
_VP_WINDOW: int = getattr(config, "VP_WINDOW", 100)
_VP_NUM_BINS: int = getattr(config, "VP_NUM_BINS", 24)
_VP_VA_THRESHOLD: float = getattr(config, "VP_VA_THRESHOLD", 0.70)
_VP_POC_DIST_PCT: float = getattr(config, "VP_POC_DISTANCE_THRESHOLD", 0.003)


class TrendPullbackStrategy(BaseStrategy):
    """Equity trend-pullback strategy for S&P 500 and NASDAQ indices.

    Enters **long only** after an intraday pullback inside a confirmed
    daily bullish trend.  Designed to be the "trend-aligned" playbook
    from the revised strategy 1A in the codex.
    """

    def __init__(self) -> None:
        super().__init__(
            name="trend_pullback",
            tickers=["SPY", "QQQ"],
            timeframe="15m",
            period="5d",
        )

    # ------------------------------------------------------------------
    # Override stop-loss: short-term pullback, so 1.5 × ATR
    # ------------------------------------------------------------------
    def compute_stop_loss(
        self,
        entry: float,
        direction: str,
        atr: float,
    ) -> float:
        """Compute stop loss price for long trades (BUY direction only)."""
        if direction != "BUY":
            raise ValueError(f"TrendPullbackStrategy only supports BUY direction, got {direction}")
        stop_distance = config.TP_STOP_MULT * atr
        return entry - stop_distance

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------
    def analyze(self, df: pd.DataFrame, ticker: str) -> Optional[StrategySignal]:
        # -- Minimum bars ------------------------------------------------
        # Need at least 51 bars so we can exclude the last (unfinished) bar
        # while still having 50 complete bars for indicators.
        if len(df) < 51:
            logger.debug("%s: too few bars (%d)", self.name, len(df))
            return None

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        atr_val = self.compute_atr(df)
        
        # -- EMA 50 Trend Filter -----------------------------------------
        ema50 = close.ewm(span=50, adjust=False).mean()
        current_ema50 = float(ema50.iloc[-1])
        if float(close.iloc[-1]) < current_ema50:
            return None

        # -- HTF trend routing -------------------------------------------
        htf_trend = self.get_htf_trend(ticker)
        if htf_trend is None:
            # Map None (unable to determine HTF trend) to neutral intentionally
            htf_trend = "neutral"

        if htf_trend == "bearish":
            logger.debug("%s: %s HTF trend is bearish (skipping)", self.name, ticker)
            return None

        # -- Bollinger Bands ---------------------------------------------
        sma20 = close.rolling(config.TP_BB_PERIOD).mean()
        std20 = close.rolling(config.TP_BB_PERIOD).std()
        lower = sma20 - config.TP_BB_STD * std20

        # -- RSI ---------------------------------------------------------
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=config.TP_RSI_PERIOD - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=config.TP_RSI_PERIOD - 1, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # -- VWAP --------------------------------------------------------
        vwap = (close * volume).cumsum() / volume.cumsum()

        # -- Current values (last bar) -----------------------------------
        current_close = float(close.iloc[-1])
        current_sma20 = float(sma20.iloc[-1])
        current_rsi = float(rsi.iloc[-1])
        current_lower = float(lower.iloc[-1])
        current_vwap = float(vwap.iloc[-1])
        low_last = float(low.iloc[-1])

        # -- Volume ratio ------------------------------------------------
        vol_mean = volume.rolling(20).mean().iloc[-1]
        if vol_mean <= 0:
            logger.debug("%s %s: volume mean <= 0, skipping", self.name, ticker)
            return None
        vol_ratio = float(volume.iloc[-1]) / float(vol_mean)

        # -- Volume Profile (POC & Value Area) ---------------------------
        # Exclude the last (unfinished) bar to prevent it from biasing the POC.
        vp_window = min(len(df) - 1, _VP_WINDOW)
        vp_high = high.iloc[-(vp_window + 1):-1].values.astype(np.float64)
        vp_low = low.iloc[-(vp_window + 1):-1].values.astype(np.float64)
        vp_close = close.iloc[-(vp_window + 1):-1].values.astype(np.float64)
        vp_volume = volume.iloc[-(vp_window + 1):-1].values.astype(np.float64)

        poc_price, va_high, va_low, poc_volume = calc_volume_profile(
            vp_high, vp_low, vp_close, vp_volume,
            _VP_NUM_BINS, _VP_VA_THRESHOLD,
        )

        # Distance from current close to POC (as fraction of price)
        poc_distance = abs(current_close - poc_price) / poc_price if poc_price > 0 else 0.0
        near_poc = poc_distance <= _VP_POC_DIST_PCT

        # -- Simple pullback entry ---------------------------------------
        # BUY if price pulls back below SMA20 (price mean-reverting within
        # a bullish trend) and RSI is weak (< 50).
        if not (current_close < current_sma20 and current_rsi < 50):
            return None

        direction: str = "BUY"

        # -- Entry / Exit -----------------------------------------------
        entry = current_close
        order_type = "MARKET"
        stop_loss = self.compute_stop_loss(entry, direction, atr=atr_val)

        # Take-profit: the higher of VWAP, SMA20, and Value Area High
        take_profit = max(current_vwap, current_sma20, va_high)
        
        if take_profit <= entry:
            logger.debug("%s %s: inverted TP/Entry (TP=%.2f <= Entry=%.2f)", self.name, ticker, take_profit, entry)
            return None

        # R/R gate: require dynamic minimum R/R to survive slippage
        tp_distance = take_profit - entry
        sl_distance = entry - stop_loss
        if tp_distance < sl_distance * config.TP_MIN_RR:
            logger.debug(
                "%s %s: poor R/R (entry=%.2f, SL=%.2f, TP=%.2f)",
                self.name, ticker, entry, stop_loss, take_profit,
            )
            return None

        # -- Confidence (0–90) ------------------------------------------
        confidence = 40.0

        # Ideal RSI zone
        if 35.0 <= current_rsi <= 45.0:
            confidence += 15.0
        elif 30.0 <= current_rsi <= 50.0:
            confidence += 10.0

        # Deep pullback (candle low below lower band)
        if low_last < current_lower * 0.998:
            confidence += 10.0

        # Volume confirmation > 1.5x average
        if vol_ratio > 1.5:
            confidence += 10.0

        # Rejection wick on the current bar
        if low_last < current_lower and current_close > current_lower:
            confidence += 10.0

        # VWAP deviation (price below VWAP by at least 0.3 %)
        if current_close < current_vwap * 0.997:
            confidence += 10.0

        # -- POC / Volume Profile confidence boosts ----------------------
        if near_poc:
            confidence += 15.0                    # POC retest is a strong mean-reversion cue
        if poc_distance < _VP_POC_DIST_PCT * 0.5:
            confidence += 5.0                     # extremely close to POC
        if current_close < poc_price < current_sma20:
            confidence += 10.0                    # price below POC but POC below SMA20 = room to run
        if current_close < va_low:
            confidence += 5.0                     # below value area = discount zone

        # HTF Trend confidence adjustments
        if htf_trend == "bullish":
            confidence += 10.0
        elif htf_trend == "neutral":
            confidence *= 0.5

        confidence = min(90.0, confidence)

        # -- Reason -----------------------------------------------------
        entry_path = "SMA20 pullback"
        reason = (
            f"Trend pullback {direction}: "
            f"{entry_path} in {htf_trend} HTF trend, "
            f"RSI={current_rsi:.1f}, "
            f"POC={poc_price:.2f}, VA=[{va_low:.2f}–{va_high:.2f}], "
            f"POC dist={poc_distance*100:.2f}%"
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
            time_stop_bars=10,
            trailing_stop_logic="sma20_or_ema",
        )
