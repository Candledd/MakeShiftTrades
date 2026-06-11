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
    """Mean reversion strategy for S&P 500 and NASDAQ indices.

    Catches small overextensions using Bollinger Bands and RSI
    on 15-minute candles.
    """

    def __init__(self) -> None:
        super().__init__(
            name="mean_reversion",
            tickers=["SPY", "QQQ"],
            timeframe="15m",
            period="5d",
        )

    def analyze(self, df: pd.DataFrame, ticker: str) -> Optional[StrategySignal]:
        # ── Minimum bars ──────────────────────────────────────────────
        if len(df) < 30:
            logger.debug("%s: too few bars (%d)", self.name, len(df))
            return None

        # ── Cooldown check (3a) ──────────────────────────────────────
        if self.is_on_cooldown(ticker):
            logger.debug("%s: %s on cooldown, skipping", self.name, ticker)
            return None

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # ── ATR ──────────────────────────────────────────────────────
        atr_val = self.compute_atr(df)

        # ── Bollinger Bands ──────────────────────────────────────────
        sma20 = close.rolling(config.MR_BB_PERIOD).mean()
        std20 = close.rolling(config.MR_BB_PERIOD).std()
        upper = sma20 + config.MR_BB_STD * std20
        lower = sma20 - config.MR_BB_STD * std20

        # ── RSI ──────────────────────────────────────────────────────
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=config.MR_RSI_PERIOD - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=config.MR_RSI_PERIOD - 1, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # ── Stochastic RSI (3c) ──────────────────────────────────────
        rsi_min = rsi.rolling(config.MR_STOCH_RSI_PERIOD).min()
        rsi_max = rsi.rolling(config.MR_STOCH_RSI_PERIOD).max()
        stoch_rsi = ((rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)) * 100
        stoch_rsi = stoch_rsi.fillna(50.0)
        current_stoch_rsi = float(stoch_rsi.iloc[-1])

        # ── Z-score ──────────────────────────────────────────────────
        std20_safe = std20.replace(0, np.nan)
        zscore = (close - sma20) / std20_safe

        # ── Current values (last bar) ────────────────────────────────
        current_close = close.iloc[-1]
        current_sma20 = sma20.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_zscore = zscore.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]

        # ── Volume ratio ─────────────────────────────────────────────
        vol_mean = volume.rolling(20).mean().iloc[-1]
        if vol_mean <= 0:
            logger.debug("%s %s: volume mean <= 0, skipping", self.name, ticker)
            return None
        vol_ratio = volume.iloc[-1] / vol_mean

        # ── Signal logic ─────────────────────────────────────────────
        direction: Optional[str] = None

        if current_close < current_lower and current_rsi < config.MR_RSI_OVERSOLD and current_zscore < -1.5 and current_stoch_rsi < config.MR_STOCH_RSI_OVERSOLD:
            direction = "BUY"
        elif current_close > current_upper and current_rsi > config.MR_RSI_OVERBOUGHT and current_zscore > 1.5 and current_stoch_rsi > config.MR_STOCH_RSI_OVERBOUGHT:
            direction = "SELL"

        if direction is None:
            return None

        # ── HTF trend confirmation (3b) ──────────────────────────────
        htf_trend = self.get_htf_trend(ticker)

        # ── Volume spike filter (3e) ──────────────────────────────────
        if vol_ratio <= config.MR_VOL_SPIKE_MULT:
            logger.debug(
                "%s %s: volume too low (%.2f < %.2f)",
                self.name, ticker, vol_ratio, config.MR_VOL_SPIKE_MULT,
            )
            return None

        # ── MTF trend confirmation (3b) ──────────────────────────────
        if config.MTF_CONFIRMATION_ENABLED and htf_trend is not None:
            if (direction == "BUY" and htf_trend != "bullish") or (direction == "SELL" and htf_trend != "bearish"):
                logger.debug(
                    "%s %s: HTF trend %s does not match %s direction",
                    self.name, ticker, htf_trend, direction,
                )
                return None

        # ── Entry / Exit ─────────────────────────────────────────────
        if config.USE_LIMIT_ORDERS_MR:
            entry = current_lower if direction == "BUY" else current_upper
            order_type = "LIMIT"
        else:
            entry = current_close
            order_type = "MARKET"
        stop_loss = self.compute_stop_loss(entry, direction, atr=atr_val)

        # ── VWAP reversion target (3d) ───────────────────────────────
        current_vwap = None
        if config.MR_VWAP_ENABLED:
            vwap = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
            current_vwap = float(vwap.iloc[-1])
            if direction == "BUY":
                take_profit = max(current_sma20, current_vwap)
            else:
                take_profit = min(current_sma20, current_vwap)
        else:
            take_profit = current_sma20

        # R/R gate: TP must be at least as far as the stop
        tp_distance = abs(take_profit - entry)
        sl_distance = abs(entry - stop_loss)
        if tp_distance < sl_distance:
            logger.debug(
                "%s %s: bad R/R (entry=%.2f, SL=%.2f, TP=%.2f)",
                self.name, ticker, entry, stop_loss, take_profit,
            )
            return None

        # ── Confidence (0–90) ────────────────────────────────────────
        confidence = 40.0

        # RSI extreme
        if direction == "BUY" and current_rsi < 25:
            confidence += 15.0
        elif direction == "SELL" and current_rsi > 75:
            confidence += 15.0

        # Z-score extreme
        if direction == "BUY" and current_zscore < -2.0:
            confidence += 15.0
        elif direction == "SELL" and current_zscore > 2.0:
            confidence += 15.0

        # Volume confirmation (> 1.5x average)
        if vol_ratio > 1.5:
            confidence += 10.0

        # Rejection wick
        if direction == "BUY" and low.iloc[-1] < current_lower and current_close > current_lower:
            confidence += 10.0
        elif direction == "SELL" and high.iloc[-1] > current_upper and current_close < current_upper:
            confidence += 10.0

        # StochRSI extreme (3f)
        if direction == "BUY" and current_stoch_rsi < 10:
            confidence += 10.0
        elif direction == "SELL" and current_stoch_rsi > 90:
            confidence += 10.0

        # VWAP alignment (3f)
        if current_vwap is not None:
            if direction == "BUY" and current_close < current_vwap:
                confidence += 10.0
            elif direction == "SELL" and current_close > current_vwap:
                confidence += 10.0

        # HTF trend alignment (3f)
        if htf_trend is not None:
            if (direction == "BUY" and htf_trend == "bullish") or (direction == "SELL" and htf_trend == "bearish"):
                confidence += 10.0

        confidence = min(90.0, confidence)

        # ── Reason string ────────────────────────────────────────────
        reason = (
            f"Mean reversion {direction}: "
            f"RSI={current_rsi:.1f}, Z={current_zscore:.2f}, "
            f"{'close below lower BB' if direction == 'BUY' else 'close above upper BB'}"
        )

        # ── Record signal (3g) ──────────────────────────────────────
        self.record_signal(ticker)

        # ── Assemble signal ──────────────────────────────────────────
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
