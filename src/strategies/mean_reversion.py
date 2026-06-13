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

        # -- Volume spike filter -----------------------------------------
        if vol_ratio <= config.MR_VOL_SPIKE_MULT:
            logger.debug(
                "%s %s: volume too low (%.2f < %.2f)",
                self.name, ticker, vol_ratio, config.MR_VOL_SPIKE_MULT,
            )
            return None

        # -- Failed-extension entry logic --------------------------------
        #
        # Look 1–3 bars back for a close *outside* the band, then require
        # that the most recent bar shows reclamation (close back inside).
        # This avoids catching momentum on the first band break.
        #
        # LONG  setup: earlier bar closed below lower band, current bar
        #              closes back above the lower band.
        # SHORT setup: earlier bar closed above upper band, current bar
        #              closes back below the upper band.
        # ----------------------------------------------------------------
        direction: Optional[str] = None

        # Scan recent bars (index -1 is current, -2 is previous, …)
        for offset in range(2, min(len(df), 5)):  # look back 2–4 bars
            prev_close = float(close.iloc[-offset])
            prev_close_lower = float(lower.iloc[-offset])
            prev_close_upper = float(upper.iloc[-offset])

            # -- Long signal candidate --
            if (
                float(low.iloc[-offset]) < prev_close_lower * 1.01                       # was outside lower band
                and current_close > current_lower * 0.99                   # reclaimed the band
                and current_rsi < config.MR_RSI_OVERSOLD            # RSI exhaustion
            ):
                # Volume must show climax fading (elevated but not extreme)
                if vol_ratio <= 10.0:      # volume still above min
                    direction = "BUY"
                    break

            # -- Short signal candidate --
            if (
                float(high.iloc[-offset]) > prev_close_upper * 0.99                       # was outside upper band
                and current_close < current_upper * 1.01                   # reclaimed the band
                and current_rsi > config.MR_RSI_OVERBOUGHT          # RSI exhaustion
            ):
                if vol_ratio <= 10.0:
                    direction = "SELL"
                    break

        if direction is None:
            return None

        # -- Entry / Exit -----------------------------------------------
        # Even for limit orders, we use the current_close (the reclaim price) 
        # to avoid missing the trade if price continues reverting.
        entry = current_close
        if config.USE_LIMIT_ORDERS_MR:
            order_type = "LIMIT"
        else:
            order_type = "MARKET"

        stop_loss = self.compute_stop_loss(entry, direction, atr=atr_val)

        # Take-profit: dynamic target based on distance to SMA
        take_profit = entry + (current_sma20 - entry) * getattr(config, 'MR_TP_TARGET_MULT', 1.0)

        # R/R gate: TP must meet the minimum R/R ratio to survive slippage
        tp_distance = abs(take_profit - entry)
        sl_distance = abs(entry - stop_loss)
        
        # Ensure target is logically in front of entry
        if (direction == "BUY" and take_profit <= entry) or (direction == "SELL" and take_profit >= entry):
            return None
            
        if tp_distance < sl_distance * getattr(config, 'MR_MIN_RR', 1.0):
            logger.debug(
                "%s %s: bad R/R (entry=%.2f, SL=%.2f, TP=%.2f)",
                self.name, ticker, entry, stop_loss, take_profit,
            )
            return None

        # -- Confidence (0–90) ------------------------------------------
        confidence = 40.0

        # RSI extreme
        if direction == "BUY" and current_rsi < 25:
            confidence += 15.0
        elif direction == "SELL" and current_rsi > 75:
            confidence += 15.0

        # Deep band extension (the offset bar)
        for offset in range(2, min(len(df), 5)):
            if direction == "BUY" and float(low.iloc[-offset]) < float(lower.iloc[-offset]) * 0.995:
                confidence += 10.0
                break
            elif direction == "SELL" and float(high.iloc[-offset]) > float(upper.iloc[-offset]) * 1.005:
                confidence += 10.0
                break

        # Volume confirmation > 1.5x
        if vol_ratio > 1.5:
            confidence += 10.0

        # Wick on the failed-extension bar (reversal confirmation)
        for offset in range(2, min(len(df), 5)):
            if direction == "BUY" and float(close.iloc[-offset]) > float(low.iloc[-offset]) + (float(close.iloc[-offset]) - float(low.iloc[-offset])) * 0.3:
                confidence += 10.0
                break
            elif direction == "SELL" and float(close.iloc[-offset]) < float(high.iloc[-offset]) - (float(high.iloc[-offset]) - float(close.iloc[-offset])) * 0.3:
                confidence += 10.0
                break

        # VWAP deviation
        if direction == "BUY" and current_close < current_vwap * 0.997:
            confidence += 10.0
        elif direction == "SELL" and current_close > current_vwap * 1.003:
            confidence += 10.0

        confidence = min(90.0, confidence)

        # -- Reason -----------------------------------------------------
        reason = (
            f"Mean reversion {direction}: "
            f"failed extension, RSI={current_rsi:.1f}, "
            f"reclaimed BB"
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
