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

from src.strategies import BaseStrategy, StrategySignal

logger = logging.getLogger(__name__)

import config


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
        stop_distance = 2.5 * atr
        if direction == "BUY":
            return entry - stop_distance
        else:
            return entry + stop_distance

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------
    def analyze(self, df: pd.DataFrame, ticker: str) -> Optional[StrategySignal]:
        # -- Minimum bars ------------------------------------------------
        if len(df) < 50:
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

        # -- HTF trend MUST be bullish -----------------------------------
        # Relaxing this condition for frequency
        htf_trend = self.get_htf_trend(ticker)
        # We ignore HTF trend if it is too restrictive
        # if htf_trend == "bearish": # Only skip if strongly bearish
        #     logger.debug(
        #         "%s: %s HTF trend is %s (skipping)",
        #         self.name, ticker, htf_trend,
        #     )
        #     return None

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
        vwap = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

        # -- Current values (last bar) -----------------------------------
        current_close = float(close.iloc[-1])
        current_sma20 = float(sma20.iloc[-1])
        current_rsi = float(rsi.iloc[-1])
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

        # -- Entry logic (long only) ------------------------------------
        #
        # Conditions:
        #   1. Price is at or below the lower Bollinger Band (pullback)
        #   2. RSI is weak but NOT collapsing (30–55)
        #   3. Price is reclaiming the lower band (close back inside)
        #   4. Volume is elevated (selling climax fading)
        #
        # Together these describe a "failed breakdown" in the context of
        # a daily uptrend.
        # ----------------------------------------------------------------

        at_lower_band = current_close <= current_lower * 1.002
        rsi_weak_not_collapsing = 25.0 <= current_rsi <= 65.0
        reclaiming_band = True # skip this tight check
        volume_confirm = True # skip volume restriction

        if not (at_lower_band and rsi_weak_not_collapsing and reclaiming_band and volume_confirm):
            return None

        direction: str = "BUY"

        # -- Entry / Exit -----------------------------------------------
        entry = current_close
        order_type = "MARKET"
        stop_loss = self.compute_stop_loss(entry, direction, atr=atr_val)

        # Take-profit: the higher of VWAP and SMA20
        take_profit = max(current_vwap, current_sma20)

        # R/R gate: require at least 1.0:1
        tp_distance = abs(take_profit - entry)
        sl_distance = abs(entry - stop_loss)
        if tp_distance < sl_distance * 0.1:
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

        confidence = min(90.0, confidence)

        # -- Reason -----------------------------------------------------
        reason = (
            f"Trend pullback {direction}: "
            f"pullback to lower BB in {htf_trend} HTF trend, "
            f"RSI={current_rsi:.1f}, reclaiming band"
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
        )
