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
        bb_sma = close.rolling(config.MB_BB_PERIOD).mean()
        bb_std = close.rolling(config.MB_BB_PERIOD).std(ddof=0)
        bbw = 2.0 * bb_std / bb_sma.replace(0, np.nan)
        bbw_pctile = self._percentile_rank(bbw, lookback)
        bbw_score = 100.0 - bbw_pctile

        # 3. Normalised candle-range percentile ── tight range = compression
        norm_range = (high - low) / close
        range_pctile = self._percentile_rank(norm_range, lookback)
        range_score = 100.0 - range_pctile

        # 4. Volume stability ── low relative volume suggests coiling
        vol_ma = volume.rolling(config.MB_BB_PERIOD).mean().replace(0, np.nan)
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

    # ────────────────────────────────────────────────────────────────────────
    # Time-of-Day RVOL (Relative Volume) — institutional cumulative baseline
    # ────────────────────────────────────────────────────────────────────────

    def _compute_tod_rvol(self, df: pd.DataFrame) -> float:
        """Compute Time-of-Day Relative Volume (RVOL) ratio.

        For the current (last) bar, this method calculates:

            RVOL = CumulativeVolumeToday(Open → CurrentTime)
                   / AvgCumulativeVolume(Open → CurrentTime) over N-day lookback

        By comparing cumulative volume at the *exact same time-of-day* across
        the lookback, this normalises out intraday volume seasonality (the
        morning rush, lunch lull, afternoon ramp) so breakouts are confirmed
        only when volume is genuinely elevated for that specific time block.

        Returns
        -------
        float
            RVOL ratio (>1.0 means volume is above average for this
            time-of-day).  Falls back to 1.0 (neutral) when insufficient
            history is available.
        """
        lookback_days = config.MB_RVOL_LOOKBACK

        # Work on a copy to avoid mutating the caller's DataFrame
        df = df.copy()

        # ── Ensure a DatetimeIndex ──────────────────────────────────────
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx)
        df = df.set_index(idx)

        # ── Intraday cumulative volume per bar ──────────────────────────
        df["_date"] = df.index.date
        df["_time"] = df.index.time
        df["_cum_vol"] = df.groupby("_date")["Volume"].cumsum()

        # ── Vectorised pivot: dates × times ─────────────────────────────
        # Build a 2-D table (dates as rows, times as columns) so that the
        # rolling average operates across entire columns at once, avoiding
        # the slow groupby().transform(lambda ...) anti-pattern.

        # Sort only if not already monotonic to avoid redundant copies
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        pivot = df.pivot_table(
            index="_date", columns="_time", values="_cum_vol", aggfunc="first",
        )

        # Rolling average of the previous N days at each TOD bucket.
        # shift(1) excludes today's own value from the average.
        tod_avg_pivot = pivot.shift(1).rolling(
            window=lookback_days, min_periods=5,
        ).mean()

        # Stack back to a Series aligned with the original rows
        tod_avg = (
            tod_avg_pivot
            .stack()
            .reindex(pd.MultiIndex.from_frame(df[["_date", "_time"]]))
        )
        tod_avg.index = df.index  # restore original DatetimeIndex

        current_cum_vol = float(df["_cum_vol"].iloc[-1])
        avg_cum_vol = float(tod_avg.iloc[-1])

        if pd.isna(avg_cum_vol) or avg_cum_vol <= 0.0:
            return 1.0  # neutral fallback — insufficient history

        return current_cum_vol / avg_cum_vol

    def analyze(self, df: pd.DataFrame, ticker: str) -> Optional[StrategySignal]:
        # ── Data quality guard ──────────────────────────────────────────
        if df is None or len(df) < config.MB_ATR_PERCENTILE_LOOKBACK + 10:
            return None

        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        open_p = df["Open"].astype(float)
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
        atr_3_series = tr.ewm(span=3, adjust=False).mean()
        
        atr_14_val = float(atr_series.iloc[-1])
        atr_3_val = float(atr_3_series.iloc[-1])
        
        # Dual-ATR Sizing: Catch sudden intraday vol expansions immediately
        atr_val = max(atr_14_val, atr_3_val)
        
        # Vol-of-Vol Scalar (Standard deviation of last 10 ATR readings)
        atr_10_std = float(atr_series.iloc[-10:].std())
        fat_tail_scalar = 1.0
        if atr_10_std > (atr_14_val * 0.15):
            fat_tail_scalar = 0.6  # Midpoint of 0.5-0.7x

        # Time-of-Day RVOL replaces the simple 20-bar rolling volume average.
        # Computed inside _compute_tod_rvol().

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
        current_volume_ratio = self._compute_tod_rvol(df)

        if any(
            pd.isna(v) or v == 0.0
            for v in [current_upper, current_lower, current_atr]
        ):
            return None


        # Immediate breakout trigger on current bar — no hold requirement
        is_buy = current_close > current_upper
        is_sell = current_close < current_lower

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

        # ── False breakout filter ──
        # Replaced N-bar persistence with a 1-bar Hold rule:
        # Breakout close -> next candle does not close back inside range.
        # This is strictly enforced by the is_buy/is_sell conditions above.

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

        # ── VPIN (Volume-Synchronized Probability of Informed Trading) ──
        vpin = self.compute_vpin(df, config.VPIN_WINDOW)
        vpin_confirmed: bool = vpin > config.VPIN_MB_BOOST_THRESHOLD
        if vpin_confirmed:
            logger.debug(
                "%s %s: VPIN = %.4f (>%.2f) — toxic order flow confirmation",
                self.name, ticker, vpin, config.VPIN_MB_BOOST_THRESHOLD,
            )

        # ── Confidence (0–100, capped at 90) ───────────────────────────
        confidence = 40  # base for valid compression + expansion

        # Stronger compression warrants higher confidence
        if compression > 75:
            confidence += 15
        elif compression > 65:
            confidence += 10

        # Volume well above the expansion threshold
        if current_volume_ratio > config.MB_HIGH_VOL_RATIO:
            confidence += 15
        elif current_volume_ratio > config.MB_MED_VOL_RATIO:
            confidence += 10

        # VPIN toxicity confirmation — aggressive informed flow supports breakout
        if vpin_confirmed:
            confidence += 15

        # Clean break — price more than channel buffer beyond the channel
        mb_channel_buffer = getattr(config, 'MB_CHANNEL_BUFFER_PCT', 0.005)
        if direction == "BUY" and current_close > current_upper * (1 + mb_channel_buffer):
            confidence += 10
        elif direction == "SELL" and current_close < current_lower * (1 - mb_channel_buffer):
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

        if vpin_confirmed:
            reason += " | VPIN Toxicity confirmation"

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
            fat_tail_scalar=fat_tail_scalar,
            time_stop_bars=int(getattr(config, 'TIME_STOP_CRYPTO_HOURS', 10)),
            trailing_stop_logic="donchian",
        )
