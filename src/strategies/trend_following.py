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

    GLD: trend filter (HTF) + pullback/breakout entries, lower turnover,
         structure-based trailing exits. Avoids short-term MACD flips.

    PDBC: wider volatility / ADX filters, chop avoidance via BB width,
         trend-following only when range expansion is real.
         Initial bracket target with trailing for uncapped upside.
    """

    name = "trend_following"
    tickers = ["GLD", "PDBC"]
    timeframe = "4h"
    period = "3mo"

    # ── Ticker-specific thresholds ──────────────────────────────────────
    # GLD: tighter stops, trend-confirmation required
    GLD_TP_MULT = 3.0         # ATR multiplier for initial bracket target
    GLD_ADX_MIN = 25.0        # minimum ADX strength

    # PDBC: wider stops for oil volatility, stricter ADX to avoid chop
    PDBC_TP_MULT = 4.0         # ATR multiplier for initial bracket target

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            tickers=self.tickers,
            timeframe=self.timeframe,
            period=self.period,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public entry point — dispatches to ticker-specific analyzers
    # ──────────────────────────────────────────────────────────────────────

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

        # Cooldown check
        if self.is_on_cooldown(ticker):
            logger.debug("%s: %s on cooldown, skipping", self.name, ticker)
            return None

        # ── Common indicators ───────────────────────────────────────────
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        fast_ema_span = config.GLD_EMA_FAST if ticker == "GLD" else config.PDBC_EMA_FAST
        ema20 = close.ewm(span=fast_ema_span, adjust=False).mean()
        ema50 = close.ewm(span=config.TF_EMA_SLOW, adjust=False).mean()

        # MACD
        macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        # ATR series
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

        # ADX
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

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(com=config.TF_RSI_PERIOD - 1, adjust=False).mean()
        loss_s = (-delta).clip(lower=0).ewm(com=config.TF_RSI_PERIOD - 1, adjust=False).mean()
        rsi = 100.0 - (100.0 / (1.0 + gain / loss_s))
        current_rsi = float(rsi.iloc[-1])

        # Current values
        current_close = float(close.iloc[-1])
        current_ema20 = float(ema20.iloc[-1])
        current_ema50 = float(ema50.iloc[-1])
        current_histogram = float(histogram.iloc[-1])
        ema_diff = ema20 - ema50
        current_ema_diff = float(ema_diff.iloc[-1])

        # Crossover detection in last 3 bars
        fresh_cross_up = bool(
            any(ema_diff.iloc[-3:] > 0) and any(ema_diff.iloc[-4:-1] <= 0)
        )
        fresh_cross_down = bool(
            any(ema_diff.iloc[-3:] < 0) and any(ema_diff.iloc[-4:-1] >= 0)
        )

        # Volume
        volume_series = df["Volume"]
        vol_avg_20 = float(volume_series.rolling(window=20).mean().iloc[-1])
        current_volume = float(volume_series.iloc[-1])
        volume_above_avg = current_volume > vol_avg_20 if vol_avg_20 > 0 else False

        # HTF trend
        htf_trend = self.get_htf_trend(ticker)

        # Build common indicators dict
        indicators = {
            "close": close,
            "high": high,
            "low": low,
            "ema20": ema20,
            "ema50": ema50,
            "ema_diff": ema_diff,
            "histogram": histogram,
            "atr_series": atr_series,
            "adx": adx,
            "rsi": rsi,
        }

        # ── Dispatch to ticker-specific logic ──
        if ticker == "GLD":
            return self._analyze_gld(
                indicators=indicators,
                atr_val=atr_val,
                current_close=current_close,
                current_ema20=current_ema20,
                current_ema50=current_ema50,
                current_histogram=current_histogram,
                current_ema_diff=current_ema_diff,
                current_adx=current_adx,
                current_rsi=current_rsi,
                fresh_cross_up=fresh_cross_up,
                fresh_cross_down=fresh_cross_down,
                volume_above_avg=volume_above_avg,
                htf_trend=htf_trend,
                ticker=ticker,
            )
        elif ticker == "PDBC":
            return self._analyze_pdbc(
                indicators=indicators,
                atr_val=atr_val,
                current_close=current_close,
                current_ema20=current_ema20,
                current_ema50=current_ema50,
                current_histogram=current_histogram,
                current_ema_diff=current_ema_diff,
                current_adx=current_adx,
                current_rsi=current_rsi,
                fresh_cross_up=fresh_cross_up,
                fresh_cross_down=fresh_cross_down,
                volume_above_avg=volume_above_avg,
                htf_trend=htf_trend,
                ticker=ticker,
            )
        else:
            logger.warning("%s: unsupported ticker %s", self.name, ticker)
            return None

    # ──────────────────────────────────────────────────────────────────────
    # GLD: trend filter + pullback / breakout, lower turnover
    # ──────────────────────────────────────────────────────────────────────

    def _analyze_gld(
        self,
        indicators: dict,
        atr_val: float,
        current_close: float,
        current_ema20: float,
        current_ema50: float,
        current_histogram: float,
        current_ema_diff: float,
        current_adx: float,
        current_rsi: float,
        fresh_cross_up: bool,
        fresh_cross_down: bool,
        volume_above_avg: bool,
        htf_trend: Optional[str],
        ticker: str,
    ) -> Optional[StrategySignal]:
        """GLD-specific analysis: trend-confirmed pullback/breakout entries."""

        close = indicators["close"]
        low = indicators["low"]
        high = indicators["high"]
        ema20 = indicators["ema20"]
        ema50 = indicators["ema50"]
        ema_diff = indicators["ema_diff"]
        histogram = indicators["histogram"]

        reason_parts: list[str] = []

        # ── GLD: HTF trend is REQUIRED (not just a bonus) ──────────────
        # GLD moves are driven by macro direction; trade with the daily trend.
        # Controlled by config.GLD_TREND_FILTER.
        if config.GLD_TREND_FILTER == "HTF" and htf_trend is None:
            logger.debug("%s: GLD — no clear HTF trend, skipping", ticker)
            return None

        gld_direction: Optional[str] = None
        if htf_trend == "bullish":
            gld_direction = "BUY"
        elif htf_trend == "bearish":
            gld_direction = "SELL"
        else:
            return None

        reason_parts.append(f"HTF {htf_trend}")

        # ── ADX strength filter ────────────────────────────────────────
        if current_adx < self.GLD_ADX_MIN:
            logger.debug(
                "%s: GLD — ADX too weak (%.1f < %.1f), skipping",
                ticker, current_adx, self.GLD_ADX_MIN,
            )
            return None
        reason_parts.append(f"ADX {current_adx:.0f}")

        # ── Price must respect the HTF trend on 4h ─────────────────────
        if gld_direction == "BUY":
            if current_close <= current_ema50:
                logger.debug("%s: GLD BUY — close below EMA50, skipping", ticker)
                return None
            if current_ema20 <= current_ema50:
                logger.debug("%s: GLD BUY — EMA20 below EMA50 (no trend), skipping", ticker)
                return None
        else:  # SELL
            if current_close >= current_ema50:
                logger.debug("%s: GLD SELL — close above EMA50, skipping", ticker)
                return None
            if current_ema20 >= current_ema50:
                logger.debug("%s: GLD SELL — EMA20 above EMA50 (no trend), skipping", ticker)
                return None

        # ── MACD must agree (histogram sign matches direction) ─────────
        # Avoid short-term MACD flips: require consistency.
        if gld_direction == "BUY" and current_histogram <= 0:
            logger.debug("%s: GLD BUY — MACD histogram not positive, skipping", ticker)
            return None
        if gld_direction == "SELL" and current_histogram >= 0:
            logger.debug("%s: GLD SELL — MACD histogram not negative, skipping", ticker)
            return None

        # ── Entry trigger: pullback to EMA20 OR fresh EMA crossover ────
        # Two paths to enter:
        #   A) Pullback — price touched/near EMA20 and bounced
        #   B) Fresh breakout — EMA crossover just happened + momentum
        pullback_trigger = False
        breakout_trigger = False

        # A) Pullback check: any of last 3 bars pulled back to EMA20
        # Controlled by config.GLD_PULLBACK_TRIGGER.
        pullback_enabled = config.GLD_PULLBACK_TRIGGER == "enabled"
        if pullback_enabled:
            if gld_direction == "BUY":
                for i in range(-3, 0):
                    if low.iloc[i] <= ema20.iloc[i] and close.iloc[i] > ema20.iloc[i]:
                        pullback_trigger = True
                        break
            else:  # SELL
                for i in range(-3, 0):
                    if high.iloc[i] >= ema20.iloc[i] and close.iloc[i] < ema20.iloc[i]:
                        pullback_trigger = True
                        break

        # B) Fresh crossover in last 3 bars + histogram accelerating
        if gld_direction == "BUY" and fresh_cross_up:
            # Confirm with MACD momentum: histogram rising for 2+ bars
            if len(histogram) >= 3 and histogram.iloc[-1] > histogram.iloc[-2] > histogram.iloc[-3]:
                breakout_trigger = True
                reason_parts.append("Fresh EMA crossover up")
        elif gld_direction == "SELL" and fresh_cross_down:
            if len(histogram) >= 3 and histogram.iloc[-1] < histogram.iloc[-2] < histogram.iloc[-3]:
                breakout_trigger = True
                reason_parts.append("Fresh EMA crossover down")

        if not pullback_trigger and not breakout_trigger:
            logger.debug(
                "%s: GLD — no pullback or breakout trigger, skipping", ticker
            )
            return None

        if pullback_trigger:
            reason_parts.append("Pullback to EMA20")

        # ── Stop loss & initial target ─────────────────────────────
        stop_mult = config.GLD_STOP_MULT
        tp_mult = self.GLD_TP_MULT

        if gld_direction == "BUY":
            entry = current_close
            stop_loss = entry - stop_mult * atr_val
            take_profit = entry + tp_mult * atr_val
        else:
            entry = current_close
            stop_loss = entry + stop_mult * atr_val
            take_profit = entry - tp_mult * atr_val

        # ── Confidence calculation ──────────────────────────────────
        confidence = 40  # higher base — GLD is more selective

        # MACD momentum accelerating (+15)
        if gld_direction == "BUY":
            if len(histogram) >= 3 and histogram.iloc[-1] > histogram.iloc[-2] > histogram.iloc[-3]:
                confidence += 15
                reason_parts.append("MACD momentum accelerating")
        else:
            if len(histogram) >= 3 and histogram.iloc[-1] < histogram.iloc[-2] < histogram.iloc[-3]:
                confidence += 15
                reason_parts.append("MACD momentum accelerating")

        # EMA gap widening (+10)
        if len(ema_diff) >= 2:
            if abs(current_ema_diff) > abs(float(ema_diff.iloc[-2])):
                confidence += 10
                reason_parts.append("EMA gap widening")

        # Fresh crossover bonus (+15) — already counted above if breakout
        if breakout_trigger:
            confidence += 10  # smaller increment since we already checked

        # Price > 1 ATR from EMA50 in trend direction (+10)
        if gld_direction == "BUY":
            if current_close > current_ema50 + atr_val:
                confidence += 10
                reason_parts.append("Price > 1 ATR above EMA50")
        else:
            if current_close < current_ema50 - atr_val:
                confidence += 10
                reason_parts.append("Price > 1 ATR below EMA50")

        # Volume above average (+10)
        if volume_above_avg:
            confidence += 10
            reason_parts.append("Volume above average")

        # RSI exhaustion penalty (-15)
        if gld_direction == "BUY" and current_rsi > config.TF_RSI_EXHAUSTION_HIGH:
            confidence -= 15
            reason_parts.append("RSI exhaustion warning")
        elif gld_direction == "SELL" and current_rsi < config.TF_RSI_EXHAUSTION_LOW:
            confidence -= 15
            reason_parts.append("RSI exhaustion warning")

        # ADX strength bonuses (+10 / +5)
        if current_adx > 30:
            confidence += 10
            reason_parts.append("ADX > 30")
        if current_adx > 40:
            confidence += 5
            reason_parts.append("ADX > 40")

        # Pullback bonus (+15)
        if pullback_trigger:
            confidence += 15
            reason_parts.append("Pullback to EMA20")

        # HTF alignment bonus (already required, but note it)
        if htf_trend is not None:
            confidence += 10
            reason_parts.append(f"HTF {htf_trend}")

        confidence = min(confidence, 90)

        reason = " | ".join(reason_parts)
        reason += " | Trailing exit active"

        self.record_signal(ticker)

        return StrategySignal(
            ticker=ticker,
            direction=gld_direction,
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
            time_stop_bars=15,
            trailing_stop_logic="atr",
        )

    # ──────────────────────────────────────────────────────────────────────
    # PDBC: wider volatility filters, chop avoidance, range expansion
    # ──────────────────────────────────────────────────────────────────────

    def _analyze_pdbc(
        self,
        indicators: dict,
        atr_val: float,
        current_close: float,
        current_ema20: float,
        current_ema50: float,
        current_histogram: float,
        current_ema_diff: float,
        current_adx: float,
        current_rsi: float,
        fresh_cross_up: bool,
        fresh_cross_down: bool,
        volume_above_avg: bool,
        htf_trend: Optional[str],
        ticker: str,
    ) -> Optional[StrategySignal]:
        """PDBC-specific analysis: chop avoidance, range expansion, wider stops."""

        close = indicators["close"]
        high = indicators["high"]
        low = indicators["low"]
        ema20 = indicators["ema20"]
        ema50 = indicators["ema50"]
        ema_diff = indicators["ema_diff"]
        histogram = indicators["histogram"]

        reason_parts: list[str] = []

        # ── Chop detection: Bollinger Bandwidth ─────────────────────────
        # If BB width is below its 20-period median, market is choppy —
        # skip EMA/MACD entries that would whipsaw.
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        bb_width = (2.0 * std20) / sma20
        bb_width_median = bb_width.rolling(window=20).median().iloc[-1]
        current_bb_width = float(bb_width.iloc[-1])

        # Normalise to avoid NaN on early bars
        if not np.isfinite(bb_width_median) or bb_width_median <= 0:
            bb_width_median = current_bb_width

        chop_zone = current_bb_width < bb_width_median * 0.85

        if chop_zone:
            logger.debug(
                "%s: PDBC — chop zone detected (BB width %.4f < median %.4f), skipping",
                ticker, current_bb_width, bb_width_median,
            )
            return None

        # ── Stricter ADX for PDBC — avoid weak trends ───────────────────
        if current_adx < config.PDBC_ADX_MIN:
            logger.debug(
                "%s: PDBC — ADX too weak (%.1f < %.1f), skipping",
                ticker, current_adx, config.PDBC_ADX_MIN,
            )
            return None
        reason_parts.append(f"ADX {current_adx:.0f}")

        # ── Primary trend direction ─────────────────────────────────────
        # PDBC needs unambiguous EMA alignment (not just any crossover)
        bullish_aligned = (current_ema20 > current_ema50
                           and current_close > current_ema20)
        bearish_aligned = (current_ema20 < current_ema50
                           and current_close < current_ema20)

        direction: Optional[str] = None

        if bullish_aligned and current_histogram > 0:
            direction = "BUY"
            reason_parts.append("EMA20 > EMA50")
            reason_parts.append("MACD histogram positive")
            reason_parts.append("Close > EMA20")
        elif bearish_aligned and current_histogram < 0:
            direction = "SELL"
            reason_parts.append("EMA20 < EMA50")
            reason_parts.append("MACD histogram negative")
            reason_parts.append("Close < EMA20")
        else:
            logger.debug(
                "%s: PDBC — no aligned trend with MACD confirmation", ticker
            )
            return None

        # ── Range expansion check ───────────────────────────────────────
        # Require that the 5-bar range is expanding relative to ATR
        # (avoids entering a narrow-range / grinding market)
        recent_range = (high.rolling(window=5).max() - low.rolling(window=5).min())
        current_range = float(recent_range.iloc[-1])
        avg_range_20 = float(recent_range.rolling(window=20).mean().iloc[-1])
        atr_20_avg = float(indicators["atr_series"].rolling(window=20).mean().iloc[-1])

        # Only enter if range is not compressed (at least config.PDBC_RANGE_EXPANSION_THRESHOLD of typical)
        range_threshold = config.PDBC_RANGE_EXPANSION_THRESHOLD
        range_expanding = False
        if atr_20_avg > 0 and current_range >= avg_range_20 * range_threshold:
            range_expanding = True
            reason_parts.append("Range expanding")

        if not range_expanding:
            logger.debug(
                "%s: PDBC — range too narrow (%.2f < %.2f * %.2f), skipping",
                ticker, current_range, avg_range_20, range_threshold,
            )
            return None

        # ── MTF trend confirmation ──────────────────────────────────────
        if config.MTF_CONFIRMATION_ENABLED and htf_trend is not None:
            if (direction == "BUY" and htf_trend != "bullish") or (
                direction == "SELL" and htf_trend != "bearish"
            ):
                logger.debug(
                    "%s: PDBC — HTF trend (%s) conflicts with %s",
                    ticker, htf_trend, direction,
                )
                return None

        # ── Stop loss & initial target (wider for PDBC) ─────────────────
        stop_mult = config.PDBC_STOP_MULT
        tp_mult = self.PDBC_TP_MULT

        if direction == "BUY":
            entry = current_close
            stop_loss = entry - stop_mult * atr_val
            take_profit = entry + tp_mult * atr_val
        else:
            entry = current_close
            stop_loss = entry + stop_mult * atr_val
            take_profit = entry - tp_mult * atr_val

        # ── Confidence calculation ──────────────────────────────────
        confidence = 30  # base

        # MACD momentum accelerating (+15)
        if direction == "BUY":
            if len(histogram) >= 3 and histogram.iloc[-1] > histogram.iloc[-2] > histogram.iloc[-3]:
                confidence += 15
                reason_parts.append("MACD momentum accelerating")
        else:
            if len(histogram) >= 3 and histogram.iloc[-1] < histogram.iloc[-2] < histogram.iloc[-3]:
                confidence += 15
                reason_parts.append("MACD momentum accelerating")

        # EMA gap widening (+10)
        if len(ema_diff) >= 2:
            if abs(current_ema_diff) > abs(float(ema_diff.iloc[-2])):
                confidence += 10
                reason_parts.append("EMA gap widening")

        # Fresh crossover bonus (+15)
        if direction == "BUY" and fresh_cross_up:
            confidence += 15
            reason_parts.append("Fresh EMA crossover up")
        elif direction == "SELL" and fresh_cross_down:
            confidence += 15
            reason_parts.append("Fresh EMA crossover down")

        # Price > 1 ATR from EMA50 in trend direction (+10)
        if direction == "BUY":
            if current_close > current_ema50 + atr_val:
                confidence += 10
                reason_parts.append("Price > 1 ATR above EMA50")
        else:
            if current_close < current_ema50 - atr_val:
                confidence += 10
                reason_parts.append("Price > 1 ATR below EMA50")

        # Volume above average (+10)
        if volume_above_avg:
            confidence += 10
            reason_parts.append("Volume above average")

        # RSI exhaustion penalty (-15)
        if direction == "BUY" and current_rsi > config.TF_RSI_EXHAUSTION_HIGH:
            confidence -= 15
            reason_parts.append("RSI exhaustion warning")
        elif direction == "SELL" and current_rsi < config.TF_RSI_EXHAUSTION_LOW:
            confidence -= 15
            reason_parts.append("RSI exhaustion warning")

        # ADX strength bonuses (+10 / +5)
        if current_adx > 30:
            confidence += 10
            reason_parts.append("ADX > 30")
        if current_adx > 40:
            confidence += 5
            reason_parts.append("ADX > 40")

        # HTF alignment bonus (+10)
        if htf_trend is not None:
            if (direction == "BUY" and htf_trend == "bullish") or (
                direction == "SELL" and htf_trend == "bearish"
            ):
                confidence += 10
                reason_parts.append(f"HTF {htf_trend}")

        confidence = min(confidence, 90)

        reason = " | ".join(reason_parts)
        reason += " | Trailing exit active"

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
            time_stop_bars=9,
            trailing_stop_logic="tighter_atr",
        )
