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
    """Two-stage BTC breakout strategy using 1-hour candles.

    Stage 1 — Compression:
      ATR, Bollinger Bandwidth, and normalised range must be at low
      percentiles (the market is coiling / consolidating).  A
      composite compression score (0-100) summarises the four metrics.

    Stage 2 — Expansion:
      Price closes outside the Donchian channel *after* compression,
      confirmed by a volume expansion.  A 4-hour EMA trend filter
      provides directional bias.

    Exit:
      Partial take-profit at 1.5 R (configurable via
      ``MB_PARTIAL_TP_RISK_MULT``).  The runner is trailed by the
      engine's existing trailing-stop logic — no hard 2.5 ATR cap.
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

    # ────────────────────────────────────────────────────────────────────────
    # Compression-score helpers
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _percentile_rank(series: pd.Series, window: int) -> float:
        """Return the percentile rank (0-100) of the *last* element of
        *series* relative to the most recent *window* values."""
        n = len(series)
        if n < window:
            return 50.0  # neutral default when data is insufficient
        subset = series.iloc[-window:]
        last_val = float(series.iloc[-1])
        count_below = float((subset < last_val).sum())
        return count_below / window * 100.0

    def _compression_score(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        atr_series: pd.Series,
    ) -> float:
        """Return a composite compression score from 0 (no compression) to
        100 (maximum compression / coiling).

        Four equally-weighted components:
          1. ATR percentile             (inverted — low ATR → high score)
          2. Bollinger Bandwidth %tile  (inverted — narrow → high score)
          3. Normalised range %tile     (inverted — tight range → high score)
          4. Volume stability score     (low rel. volume → high score)
        """
        lookback = config.MB_ATR_PERCENTILE_LOOKBACK
        n = len(close)
        if n < lookback:
            return 0.0

        # 1. ATR percentile ── low ATR means high compression
        atr_pctile = self._percentile_rank(atr_series, lookback)
        atr_score = 100.0 - atr_pctile  # invert

        # 2. Bollinger Bandwidth percentile ── narrow width = high compression
        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std(ddof=0)
        bbw = 2.0 * bb_std / bb_sma.replace(0, np.nan)
        bbw_pctile = self._percentile_rank(bbw, lookback)
        bbw_score = 100.0 - bbw_pctile

        # 3. Normalised candle-range percentile ── tight range = compression
        norm_range = (high - low) / close
        range_pctile = self._percentile_rank(norm_range, lookback)
        range_score = 100.0 - range_pctile

        # 4. Volume stability ── low relative volume suggests coiling
        vol_ma = volume.rolling(20).mean().replace(0, np.nan)
        vol_ratio = volume / vol_ma
        current_vr = float(vol_ratio.iloc[-1])
        if pd.notna(current_vr):
            vol_score = max(0.0, 100.0 - current_vr * 50.0)
            vol_score = min(vol_score, 100.0)
        else:
            vol_score = 50.0

        composite = (atr_score + bbw_score + range_score + vol_score) / 4.0
        return composite

    # ────────────────────────────────────────────────────────────────────────
    # Main analysis entry-point
    # ────────────────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame, ticker: str) -> Optional[StrategySignal]:
        # ── Data quality guard ──────────────────────────────────────────
        if df is None or len(df) < config.MB_ATR_PERCENTILE_LOOKBACK + 10:
            return None

        # ── Cooldown check ─────────────────────────────────────────────
        if self.is_on_cooldown(ticker):
            logger.debug("%s: %s on cooldown, skipping", self.name, ticker)
            return None

        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)
        volume = df["Volume"].astype(float)

        # ── Indicators ─────────────────────────────────────────────────

        # Donchian Channel (20 bars, using PREVIOUS bars only)
        donchian_period = config.MB_DONCHIAN_PERIOD
        upper_channel = high.shift(1).rolling(donchian_period).max()
        lower_channel = low.shift(1).rolling(donchian_period).min()

        # ATR(14)
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

        # Volume MA(20)
        volume_ma = volume.rolling(20).mean().replace(0, np.nan)

        # ────────────────────────────────────────────────────────────────
        # Stage 1: Compression detection
        # ────────────────────────────────────────────────────────────────
        compression = self._compression_score(close, high, low, volume, atr_series)
        logger.debug(
            "%s %s: compression score = %.1f/100 (threshold %.0f)",
            self.name, ticker, compression, config.MB_COMPRESSION_THRESHOLD,
        )
        if compression < config.MB_COMPRESSION_THRESHOLD:
            return None

        # ────────────────────────────────────────────────────────────────
        # Stage 2: Expansion confirmation
        # ────────────────────────────────────────────────────────────────
        current_close = float(close.iloc[-1])
        current_upper = float(upper_channel.iloc[-1])
        current_lower = float(lower_channel.iloc[-1])
        current_atr = atr_val
        current_volume_ratio = float((volume / volume_ma).iloc[-1])

        if any(
            pd.isna(v) or v == 0.0
            for v in [current_upper, current_lower, current_atr]
        ):
            return None

        # Directional bias from last bar's close
        last_bar_up = close.iloc[-1] > close.iloc[-2] if len(close) >= 2 else False

        is_buy = current_close > current_upper and last_bar_up
        is_sell = current_close < current_lower and not last_bar_up

        if not (is_buy or is_sell):
            return None

        direction: str = "BUY" if is_buy else "SELL"

        # ── Volume expansion confirmation ──────────────────────────────
        if current_volume_ratio < config.MB_EXPANSION_VOLUME_RATIO:
            logger.debug(
                "%s %s: volume expansion insufficient (%.2f < %.2f)",
                self.name, ticker,
                current_volume_ratio, config.MB_EXPANSION_VOLUME_RATIO,
            )
            return None

        # ── False breakout filter ──────────────────────────────────────
        if direction == "BUY":
            bars_above = sum(
                1 for i in range(-config.MB_FALSE_BREAKOUT_BARS, 0)
                if close.iloc[i] > upper_channel.iloc[i]
            )
            if bars_above < 1:
                logger.debug(
                    "%s %s: false breakout (only %d bars above channel)",
                    self.name, ticker, bars_above,
                )
                return None
        else:
            bars_below = sum(
                1 for i in range(-config.MB_FALSE_BREAKOUT_BARS, 0)
                if close.iloc[i] < lower_channel.iloc[i]
            )
            if bars_below < 1:
                logger.debug(
                    "%s %s: false breakout (only %d bars below channel)",
                    self.name, ticker, bars_below,
                )
                return None

        # ── 4-hour trend filter (mandatory for BTC) ────────────────────
        htf_trend = self.get_htf_trend(ticker, htf_interval="4h", htf_period="1mo")
        if htf_trend is not None:
            if (direction == "BUY" and htf_trend != "bullish") or (
                direction == "SELL" and htf_trend != "bearish"
            ):
                logger.debug(
                    "%s %s: 4h trend mismatch (%s), skipping",
                    self.name, ticker, htf_trend,
                )
                return None

        # ────────────────────────────────────────────────────────────────
        # Entry / Stop / Partial Take-Profit
        # ────────────────────────────────────────────────────────────────

        # Entry with slippage buffer
        if direction == "BUY":
            entry = current_close * (1 + config.SLIPPAGE_CAP_PCT)
        else:
            entry = current_close * (1 - config.SLIPPAGE_CAP_PCT)

        # Stop loss (base class uses 2.0 ATR for momentum_breakout)
        stop_loss = self.compute_stop_loss(entry, direction, atr=atr_val)

        # Partial take-profit at MB_PARTIAL_TP_RISK_MULT × risk (no hard
        # 2.5 ATR cap).  The runner is trailed by the engine's trailing
        # stop, giving uncapped upside on the remaining position.
        risk = abs(entry - stop_loss)
        tp_distance = config.MB_PARTIAL_TP_RISK_MULT * risk
        if direction == "BUY":
            take_profit = entry + tp_distance
        else:
            take_profit = entry - tp_distance

        # ── RSI divergence penalty ─────────────────────────────────────
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(
            com=config.MB_RSI_PERIOD - 1, adjust=False
        ).mean()
        loss_s = (-delta).clip(lower=0).ewm(
            com=config.MB_RSI_PERIOD - 1, adjust=False
        ).mean()
        rsi = 100.0 - (100.0 / (1.0 + gain / loss_s))

        divergence_penalty = 0
        if is_buy and len(close) >= 5:
            if (
                close.iloc[-1] > close.iloc[-5:].max()
                and rsi.iloc[-1] < rsi.iloc[-5:-1].max()
            ):
                divergence_penalty = 10
        elif is_sell and len(close) >= 5:
            if (
                close.iloc[-1] < close.iloc[-5:].min()
                and rsi.iloc[-1] > rsi.iloc[-5:-1].min()
            ):
                divergence_penalty = 10

        # ── Confidence (0–100, capped at 90) ───────────────────────────
        confidence = 40  # base for valid compression + expansion

        # Stronger compression warrants higher confidence
        if compression > 75:
            confidence += 15
        elif compression > 65:
            confidence += 10

        # Volume well above the expansion threshold
        if current_volume_ratio > 2.5:
            confidence += 15
        elif current_volume_ratio > 2.0:
            confidence += 10

        # Clean break — price more than 0.5 % beyond the channel
        if direction == "BUY" and current_close > current_upper * 1.005:
            confidence += 10
        elif direction == "SELL" and current_close < current_lower * 0.995:
            confidence += 10

        # All of the last 3 candles closed in the same direction
        if len(close) >= 4:
            recent_diffs = close.diff().iloc[-3:]
            if direction == "BUY":
                if all(d > 0 for d in recent_diffs if pd.notna(d)):
                    confidence += 10
            else:
                if all(d < 0 for d in recent_diffs if pd.notna(d)):
                    confidence += 10

        # Volatility expansion after compression
        if len(atr_series) >= 5 and atr_series.iloc[-1] > atr_series.iloc[-5]:
            confidence += 5

        # Divergence penalty
        confidence -= divergence_penalty
        confidence = min(confidence, 90)

        # ── Reason string ─────────────────────────────────────────────
        channel_side = (
            "above upper channel" if direction == "BUY"
            else "below lower channel"
        )
        reason = (
            f"{direction} BTC compression breakout | "
            f"close ${current_close:.2f} {channel_side} | "
            f"compression {compression:.0f}/100 | "
            f"vol ratio {current_volume_ratio:.2f} | "
            f"tp {config.MB_PARTIAL_TP_RISK_MULT:.1f}R partial + trail | "
            f"confidence {confidence}/90"
        )

        # ── Record signal ─────────────────────────────────────────────
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
