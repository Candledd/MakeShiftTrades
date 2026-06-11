from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Optional
import pandas as pd
import numpy as np
import config
from src.strategies import BaseStrategy, StrategySignal


logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """Trend-following strategy for commodities (Gold, Oil) on 4h candles.

    Uses EMA crossover + MACD confirmation to follow commodity trends
    with wide ATR-based targets.
    """

    name = "trend_following"
    tickers = ["GLD", "USO"]
    timeframe = "4h"
    period = "3mo"

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            tickers=self.tickers,
            timeframe=self.timeframe,
            period=self.period,
        )

    def analyze(self, df: pd.DataFrame, ticker: str) -> Optional[StrategySignal]:
        """Analyze 4h OHLCV data and return a trend-following signal or None."""
        # Require at least 60 bars for EMA50 to stabilize
        if df is None or len(df) < 60:
            logger.debug(
                "%s: insufficient data (%d bars, need 60)",
                ticker,
                len(df) if df is not None else 0,
            )
            return None

        # 5a. Cooldown check
        if self.is_on_cooldown(ticker):
            logger.debug("%s: %s on cooldown, skipping", self.name, ticker)
            return None

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # ── Indicators ──
        ema20 = close.ewm(span=config.TF_EMA_FAST, adjust=False).mean()
        ema50 = close.ewm(span=config.TF_EMA_SLOW, adjust=False).mean()

        # MACD
        macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        # ATR series for ADX / indicators (full Series)
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_series = tr.ewm(span=14, adjust=False).mean()

        # ATR for take-profit target (scalar)
        atr_val = float(atr_series.iloc[-1])

        # 5b. ADX Trend Strength
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0.0
        minus_dm[minus_dm < 0] = 0.0
        both_positive = (plus_dm > 0) & (minus_dm > 0)
        plus_dm[both_positive & (plus_dm < minus_dm)] = 0.0
        minus_dm[both_positive & (minus_dm < plus_dm)] = 0.0
        smoothed_plus_dm = plus_dm.ewm(span=config.TF_ADX_PERIOD, adjust=False).mean()
        smoothed_minus_dm = minus_dm.ewm(span=config.TF_ADX_PERIOD, adjust=False).mean()
        atr_safe = atr_series.replace(0, np.nan)
        plus_di = 100.0 * smoothed_plus_dm / atr_safe
        minus_di = 100.0 * smoothed_minus_dm / atr_safe
        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        adx = dx.ewm(span=config.TF_ADX_PERIOD, adjust=False).mean()
        current_adx = float(adx.iloc[-1])

        # ADX trend strength filter
        if current_adx < config.TF_ADX_MIN_STRENGTH:
            logger.debug(
                "%s: ADX too weak (%.1f < %.1f), skipping",
                ticker, current_adx, config.TF_ADX_MIN_STRENGTH,
            )
            return None

        # 5c. RSI for trend exhaustion
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(com=config.TF_RSI_PERIOD - 1, adjust=False).mean()
        loss_s = (-delta).clip(lower=0).ewm(com=config.TF_RSI_PERIOD - 1, adjust=False).mean()
        rsi = 100.0 - (100.0 / (1.0 + gain / loss_s))
        current_rsi = float(rsi.iloc[-1])

        # Current values (last bar)
        current_close = float(close.iloc[-1])
        current_ema20 = float(ema20.iloc[-1])
        current_ema50 = float(ema50.iloc[-1])
        current_histogram = float(histogram.iloc[-1])
        ema_diff = ema20 - ema50
        current_ema_diff = float(ema_diff.iloc[-1])

        # ── Crossover Detection ──
        # Check if sign of (ema20 - ema50) changed in last 3 bars
        fresh_cross_up = bool(
            any(ema_diff.iloc[-3:] > 0) and any(ema_diff.iloc[-4:-1] <= 0)
        )
        fresh_cross_down = bool(
            any(ema_diff.iloc[-3:] < 0) and any(ema_diff.iloc[-4:-1] >= 0)
        )

        # ── Volume check ──
        volume_series = df["Volume"]
        vol_avg_20 = float(volume_series.rolling(window=20).mean().iloc[-1])
        current_volume = float(volume_series.iloc[-1])
        volume_above_avg = current_volume > vol_avg_20 if vol_avg_20 > 0 else False

        # ── Signal Direction ──
        direction = None
        reason_parts: list[str] = []

        # BUY conditions: uptrend with positive momentum
        buy_conditions = (
            current_ema20 > current_ema50
            and current_histogram > 0
            and current_close > current_ema20
        )

        # SELL conditions: downtrend with negative momentum
        sell_conditions = (
            current_ema20 < current_ema50
            and current_histogram < 0
            and current_close < current_ema20
        )

        if buy_conditions:
            direction = "BUY"
            reason_parts.append("EMA20 > EMA50")
            reason_parts.append("MACD histogram positive")
            reason_parts.append("Close > EMA20")
        elif sell_conditions:
            direction = "SELL"
            reason_parts.append("EMA20 < EMA50")
            reason_parts.append("MACD histogram negative")
            reason_parts.append("Close < EMA20")
        else:
            logger.debug("%s: no signal conditions met", ticker)
            return None

        # 5e. MTF trend confirmation
        htf_trend = self.get_htf_trend(ticker)
        if config.MTF_CONFIRMATION_ENABLED and htf_trend is not None:
            if (direction == "BUY" and htf_trend != "bullish") or (
                direction == "SELL" and htf_trend != "bearish"
            ):
                logger.debug(
                    "%s %s: HTF trend (%s) conflicts with %s",
                    self.name, ticker, htf_trend, direction,
                )
                return None

        # ── Entry / Exit Prices ──
        entry = current_close
        stop_loss = self.compute_stop_loss(entry, direction, atr=atr_val)

        if direction == "BUY":
            take_profit = entry + config.TF_ATR_TARGET_MULT * atr_val
        else:
            take_profit = entry - config.TF_ATR_TARGET_MULT * atr_val

        # 5d. Pullback entry optimization
        pullback_entry = False
        if config.TF_PULLBACK_ENABLED:
            if direction == "BUY":
                # Check if any of last 3 bars pulled back to EMA20
                for i in range(-3, 0):
                    if low.iloc[i] <= ema20.iloc[i] and close.iloc[i] > ema20.iloc[i]:
                        pullback_entry = True
                        break
            else:  # SELL
                for i in range(-3, 0):
                    if high.iloc[i] >= ema20.iloc[i] and close.iloc[i] < ema20.iloc[i]:
                        pullback_entry = True
                        break

        # ── Confidence Calculation (0-100) ──
        confidence = 35  # base for valid trend setup

        # +15 if MACD histogram has been increasing/decreasing for 2+ bars
        # (momentum accelerating)
        if direction == "BUY":
            if (
                histogram.iloc[-1] > histogram.iloc[-2]
                > histogram.iloc[-3]
            ):
                confidence += 15
                reason_parts.append("MACD momentum accelerating")
        else:  # SELL
            if (
                histogram.iloc[-1] < histogram.iloc[-2]
                < histogram.iloc[-3]
            ):
                confidence += 15
                reason_parts.append("MACD momentum accelerating")

        # +10 if EMA gap is widening (trend strengthening)
        if len(ema_diff) >= 2:
            if abs(current_ema_diff) > abs(float(ema_diff.iloc[-2])):
                confidence += 10
                reason_parts.append("EMA gap widening")

        # +15 if a fresh crossover happened in the last 3 bars
        if direction == "BUY" and fresh_cross_up:
            confidence += 15
            reason_parts.append("Fresh EMA crossover up")
        elif direction == "SELL" and fresh_cross_down:
            confidence += 15
            reason_parts.append("Fresh EMA crossover down")

        # +10 if price is > 1 ATR from EMA(50) in trend direction
        if direction == "BUY":
            if current_close > current_ema50 + atr_val:
                confidence += 10
                reason_parts.append("Price > 1 ATR above EMA50")
        else:  # SELL
            if current_close < current_ema50 - atr_val:
                confidence += 10
                reason_parts.append("Price > 1 ATR below EMA50")

        # +10 if volume > 20-bar average
        if volume_above_avg:
            confidence += 10
            reason_parts.append("Volume above average")

        # 5c. RSI exhaustion penalty (trend may be overextended)
        if direction == "BUY" and current_rsi > config.TF_RSI_EXHAUSTION_HIGH:
            confidence -= 15
            reason_parts.append("RSI exhaustion warning")
        elif direction == "SELL" and current_rsi < config.TF_RSI_EXHAUSTION_LOW:
            confidence -= 15
            reason_parts.append("RSI exhaustion warning")

        # 5f. ADX strength bonuses
        if current_adx > 30:
            confidence += 10
            reason_parts.append("ADX > 30")
        if current_adx > 40:
            confidence += 5
            reason_parts.append("ADX > 40")

        # 5d. Pullback entry bonus
        if pullback_entry:
            confidence += 15
            reason_parts.append("Pullback to EMA20")

        # 5e. HTF alignment bonus
        if htf_trend is not None:
            if (direction == "BUY" and htf_trend == "bullish") or (
                direction == "SELL" and htf_trend == "bearish"
            ):
                confidence += 10
                reason_parts.append(f"HTF {htf_trend}")

        confidence = min(confidence, 90)

        # ── Build Signal ──
        reason = " | ".join(reason_parts)

        # 5g. Record signal
        self.record_signal(ticker)

        return StrategySignal(
            ticker=ticker,
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=float(confidence),
            strategy_name=self.name,
            timeframe=self.timeframe,
            reason=reason,
            atr=atr_val,
            timestamp=datetime.now(timezone.utc),
            order_type="MARKET",
        )
