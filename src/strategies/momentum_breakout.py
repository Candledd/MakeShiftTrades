from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np

import config
from src.strategies import BaseStrategy, StrategySignal

logger = logging.getLogger(__name__)


class MomentumBreakoutStrategy(BaseStrategy):
    """Ride momentum breakouts on Bitcoin using 1-hour candles.

    Uses a Donchian Channel (20-bar) breakout filter combined with ADX(14)
    for trend-strength confirmation and ATR(14) for position sizing and
    wide profit targets.
    """

    name = "momentum_breakout"
    tickers = ["BTC-USD"]
    timeframe = "1h"
    period = "1mo"

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            tickers=self.tickers,
            timeframe=self.timeframe,
            period=self.period,
        )

    def analyze(self, df: pd.DataFrame, ticker: str) -> Optional[StrategySignal]:
        # ── Data quality guard ──────────────────────────────────────────────
        if df is None or len(df) < 30:
            return None

        # ── 4a: Cooldown check ──────────────────────────────────────────────
        if self.is_on_cooldown(ticker):
            logger.debug("%s: %s on cooldown, skipping", self.name, ticker)
            return None

        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)
        volume = df["Volume"].astype(float)

        # ── Donchian Channel (20 bars, PREVIOUS bars only) ──────────────────
        upper_channel = high.shift(1).rolling(config.MB_DONCHIAN_PERIOD).max()
        lower_channel = low.shift(1).rolling(config.MB_DONCHIAN_PERIOD).min()

        # ── ATR(14) — True Range smoothed with ewm ──────────────────────────
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
        atr_val = float(atr_series.iloc[-1])

        # ── ADX(14) ──────────────────────────────────────────────────────────
        plus_dm = high.diff()
        minus_dm = -low.diff()

        # Zero out negative directional movement
        plus_dm[plus_dm < 0] = 0.0
        minus_dm[minus_dm < 0] = 0.0

        # Cross-check: only keep the larger directional movement
        both_positive = (plus_dm > 0) & (minus_dm > 0)
        plus_dm[both_positive & (plus_dm < minus_dm)] = 0.0
        minus_dm[both_positive & (minus_dm < plus_dm)] = 0.0

        # Smooth with ewm(span=14)
        smoothed_plus_dm = plus_dm.ewm(span=14, adjust=False).mean()
        smoothed_minus_dm = minus_dm.ewm(span=14, adjust=False).mean()

        # Directional indicators
        # Guard against division by zero in ATR
        atr_safe = atr_series.replace(0, np.nan)
        plus_di = 100.0 * smoothed_plus_dm / atr_safe
        minus_di = 100.0 * smoothed_minus_dm / atr_safe

        # DX and ADX
        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        adx = dx.ewm(span=14, adjust=False).mean()

        # ── Volume ratio ────────────────────────────────────────────────────
        volume_ma = volume.rolling(20).mean()
        volume_ratio = volume / volume_ma.replace(0, np.nan)

        # ── Current values (last bar) ───────────────────────────────────────
        current_close = float(close.iloc[-1])
        current_upper = float(upper_channel.iloc[-1])
        current_lower = float(lower_channel.iloc[-1])
        current_adx = float(adx.iloc[-1])
        current_atr = atr_val
        current_volume_ratio = float(volume_ratio.iloc[-1])

        # Validate that computed values are sane
        if any(
            pd.isna(v) or v == 0.0
            for v in [current_upper, current_lower, current_adx, current_atr]
        ):
            return None

        # ── 4b: Bollinger Band Squeeze detection ───────────────────────────
        bb_sma = close.rolling(config.MB_SQUEEZE_LOOKBACK).mean()
        bb_std = close.rolling(config.MB_SQUEEZE_LOOKBACK).std()
        bb_upper = bb_sma + config.MB_SQUEEZE_BB_MULT * bb_std
        bb_lower = bb_sma - config.MB_SQUEEZE_BB_MULT * bb_std

        # Keltner Channel for squeeze comparison
        kc_upper = bb_sma + config.MB_SQUEEZE_KC_MULT * atr_series
        kc_lower = bb_sma - config.MB_SQUEEZE_KC_MULT * atr_series

        # Squeeze: BB inside KC means consolidation (coiling)
        is_squeeze = (bb_lower.iloc[-1] > kc_lower.iloc[-1]) and (bb_upper.iloc[-1] < kc_upper.iloc[-1])
        was_squeeze = any(
            (bb_lower.iloc[i] > kc_lower.iloc[i]) and (bb_upper.iloc[i] < kc_upper.iloc[i])
            for i in range(-5, -1)
        ) if len(bb_lower) >= 5 else False

        # Squeeze release = was in squeeze recently but no longer
        squeeze_release = was_squeeze and not is_squeeze

        # ── Signal logic ─────────────────────────────────────────────────────
        # Check upward / downward momentum on the last candle
        last_bar_up = close.iloc[-1] > close.iloc[-2] if len(close) >= 2 else False

        is_buy = (
            current_close > current_upper
            and current_adx > config.MB_ADX_THRESHOLD
            and last_bar_up
        )

        is_sell = (
            current_close < current_lower
            and current_adx > config.MB_ADX_THRESHOLD
            and not last_bar_up
        )

        if not (is_buy or is_sell):
            return None

        direction: str = "BUY" if is_buy else "SELL"

        # ── 4c: False breakout filter ───────────────────────────────────────
        if is_buy:
            bars_above = sum(
                1 for i in range(-config.MB_FALSE_BREAKOUT_BARS, 0)
                if close.iloc[i] > upper_channel.iloc[i]
            )
            if bars_above < 1:
                logger.debug("%s %s: false breakout filter (only %d bars above channel)", self.name, ticker, bars_above)
                return None
        else:  # SELL
            bars_below = sum(
                1 for i in range(-config.MB_FALSE_BREAKOUT_BARS, 0)
                if close.iloc[i] < lower_channel.iloc[i]
            )
            if bars_below < 1:
                logger.debug("%s %s: false breakout filter (only %d bars below channel)", self.name, ticker, bars_below)
                return None

        # ── 4d: Minimum volume ratio filter ─────────────────────────────────
        if current_volume_ratio < config.MB_MIN_VOLUME_RATIO:
            logger.debug("%s %s: volume too low (%.2f < %.2f)", self.name, ticker, current_volume_ratio, config.MB_MIN_VOLUME_RATIO)
            return None

        # ── 4e: MTF trend confirmation ──────────────────────────────────────
        if config.MTF_CONFIRMATION_ENABLED:
            htf_trend = self.get_htf_trend(ticker)
            if htf_trend is not None:
                if (direction == "BUY" and htf_trend != "bullish") or (direction == "SELL" and htf_trend != "bearish"):
                    logger.debug("%s %s: MTF trend mismatch (%s), skipping", self.name, ticker, htf_trend)
                    return None

        # ── 4f: Momentum divergence detection (RSI divergence) ──────────────
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(com=config.MB_RSI_PERIOD - 1, adjust=False).mean()
        loss_s = (-delta).clip(lower=0).ewm(com=config.MB_RSI_PERIOD - 1, adjust=False).mean()
        rsi = 100.0 - (100.0 / (1.0 + gain / loss_s))

        divergence_penalty = 0
        if is_buy and len(close) >= 5:
            if close.iloc[-1] > close.iloc[-5:].max() and rsi.iloc[-1] < rsi.iloc[-5:-1].max():
                divergence_penalty = 10
        elif is_sell and len(close) >= 5:
            if close.iloc[-1] < close.iloc[-5:].min() and rsi.iloc[-1] > rsi.iloc[-5:-1].min():
                divergence_penalty = 10

        # ── Entry / Exit ─────────────────────────────────────────────────────
        if direction == "BUY":
            entry = current_close * (1 + config.SLIPPAGE_CAP_PCT)
        else:
            entry = current_close * (1 - config.SLIPPAGE_CAP_PCT)
        stop_loss = self.compute_stop_loss(entry, direction, atr=atr_val)

        if direction == "BUY":
            take_profit = entry + config.MB_ATR_TARGET_MULT * current_atr
        else:
            take_profit = entry - config.MB_ATR_TARGET_MULT * current_atr

        # ── Confidence (0–100, capped at 90) ─────────────────────────────────
        confidence = 35  # base for a valid breakout

        # ADX trend strength bonuses
        if current_adx > 30:
            confidence += 15
        if current_adx > 40:
            confidence += 10

        # Volume-confirmed breakout
        if current_volume_ratio > 2.0:
            confidence += 15

        # Clean break — price more than 0.5 % beyond the channel
        if direction == "BUY" and current_close > current_upper * 1.005:
            confidence += 10
        elif direction == "SELL" and current_close < current_lower * 0.995:
            confidence += 10

        # All of the last 3 candles closed in the same direction as the signal
        if len(close) >= 4:
            # diffs at indices -3, -2, -1 (includes the signal candle)
            recent_diffs = close.diff().iloc[-3:]
            if direction == "BUY":
                if all(d > 0 for d in recent_diffs if pd.notna(d)):
                    confidence += 10
            else:
                if all(d < 0 for d in recent_diffs if pd.notna(d)):
                    confidence += 10

        # ── 4g: Squeeze release bonus ──────────────────────────────────────
        if squeeze_release:
            confidence += 15

        # ── 4g: ATR increasing (volatility expanding) ──────────────────────
        if len(atr_series) >= 5 and atr_series.iloc[-1] > atr_series.iloc[-5]:
            confidence += 5

        # ── 4f: Divergence penalty ─────────────────────────────────────────
        confidence -= divergence_penalty

        confidence = min(confidence, 90)

        # ── Reason string ────────────────────────────────────────────────────
        channel_direction = "above upper channel" if direction == "BUY" else "below lower channel"
        reason = (
            f"{direction} breakout on {ticker} | "
            f"close ${current_close:.2f} {channel_direction} | "
            f"ADX {current_adx:.1f} | "
            f"ATR ${current_atr:.2f} | "
            f"vol ratio {current_volume_ratio:.2f} | "
            f"confidence {confidence}/90"
        )

        # ── 4h: Record signal ───────────────────────────────────────────────
        self.record_signal(ticker)

        return StrategySignal(
            ticker=ticker,
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            strategy_name=self.name,
            timeframe=self.timeframe,
            reason=reason,
            atr=current_atr,
            timestamp=datetime.now(timezone.utc),
            order_type="LIMIT",
        )
