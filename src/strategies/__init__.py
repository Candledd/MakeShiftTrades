from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class StrategySignal:
    """Universal signal format that all strategies emit."""

    ticker: str  # e.g. "SPY", "BTC-USD"
    direction: Literal["BUY", "SELL"]
    entry: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0–100
    strategy_name: str  # e.g. "mean_reversion"
    timeframe: str  # e.g. "15m", "1h", "4h"
    reason: str  # human-readable explanation
    atr: float  # ATR(14) at signal time
    timestamp: datetime
    sizing_multiplier: float = 1.0
    vol_multiplier: float = 1.0
    order_type: str = "MARKET"
    fat_tail_scalar: float = 1.0
    time_stop_bars: int = 10
    trailing_stop_logic: str = "default"


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(
        self,
        name: str,
        tickers: list[str],
        timeframe: str,
        period: str,
    ) -> None:
        self.name = name
        self.tickers = tickers
        self.timeframe = timeframe
        self.period = period
        self._last_signal_time: dict[str, float] = {}  # ticker -> timestamp of last signal
        self._htf_cache: dict[str, tuple[float, Optional[str]]] = {}  # ticker -> (timestamp, trend)

    @abstractmethod
    def analyze(self, df: pd.DataFrame, ticker: str) -> Optional[StrategySignal]:
        """Analyze OHLCV data and return a signal, or None if no valid signal."""
        ...

    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Compute ATR(period) from High/Low/Close columns.

        Returns the last ATR value as a float.
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()
        return float(atr.iloc[-1])

    def compute_stop_loss(
        self,
        entry: float,
        direction: str,
        atr: float,
    ) -> float:
        """Compute a stop-loss price using ATR-based multipliers per strategy type.

        * mean_reversion      →  1.5 × ATR
        * momentum_breakout   →  2.0 × ATR
        * trend_following     →  2.0 × ATR
        """
        # ATR multiplier depends on strategy type
        if self.name == "mean_reversion":
            mult = 1.5
        elif self.name in ("momentum_breakout", "trend_following"):
            mult = 2.0
        else:
            mult = 2.0  # safe default for unknown strategies

        stop_distance = mult * atr

        if direction == "BUY":
            return entry - stop_distance
        elif direction == "SELL":
            return entry + stop_distance
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def compute_vpin(self, df: pd.DataFrame, window: int) -> float:
        """Compute VPIN over the last *window* bars using Bulk Volume Classification.

        Drops any rows with NaN in required columns, extracts the last
        *window* bars, and delegates to the Numba-compiled ``calc_vpin``.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with at least *window* rows.
        window : int
            Rolling window length for standard deviation and VPIN computation.

        Returns
        -------
        float
            The VPIN value (0–1).  Returns 0.0 if insufficient data.
        """
        from src.indicators import calc_vpin

        clean = df.dropna(subset=["Open", "Close", "Volume"])
        if len(clean) < 2:
            return 0.0

        open_np = clean["Open"].values[-window:]
        close_np = clean["Close"].values[-window:]
        volume_np = clean["Volume"].values[-window:]
        return calc_vpin(open_np, close_np, volume_np, window)

    def is_on_cooldown(
        self, ticker: str, current_time: Optional[float | datetime | pd.Timestamp] = None
    ) -> bool:
        """Check if the given ticker is currently in cooldown.

        If *current_time* is provided (e.g. during backtesting), compares
        against the simulated bar timestamp. Otherwise falls back to system
        wall-clock time.
        """
        import config

        if ticker not in self._last_signal_time:
            return False

        last_time = self._last_signal_time[ticker]
        if current_time is not None:
            curr_sec = (
                current_time.timestamp()
                if isinstance(current_time, (datetime, pd.Timestamp))
                else float(current_time)
            )
            last_sec = (
                last_time.timestamp()
                if isinstance(last_time, (datetime, pd.Timestamp))
                else float(last_time)
            )
            elapsed = curr_sec - last_sec
        else:
            now_sec = time.time()
            last_sec = (
                last_time.timestamp()
                if isinstance(last_time, (datetime, pd.Timestamp))
                else float(last_time)
            )
            elapsed = now_sec - last_sec

        return elapsed < config.SIGNAL_COOLDOWN_SECONDS

    def record_signal(
        self, ticker: str, timestamp: Optional[float | datetime | pd.Timestamp] = None
    ) -> None:
        """Record that a signal was generated for *ticker* at the given time or now."""
        self._last_signal_time[ticker] = timestamp if timestamp is not None else time.time()

    def get_htf_trend(
        self,
        ticker: str,
        htf_interval: str = "1d",
        htf_period: str = "2y",
        as_of: Optional[datetime | pd.Timestamp] = None,
    ) -> Optional[str]:
        """Fetch higher-timeframe data and determine the macro trend.

        Uses daily OHLCV data to compute EMA(20) / EMA(50). Returns
        ``"bullish"`` if EMA(20) > EMA(50) and close > EMA(20),
        ``"bearish"`` if EMA(20) < EMA(50) and close < EMA(20),
        or ``None`` otherwise (neutral / error).

        When *as_of* is provided (e.g. during backtesting), evaluates
        trend strictly on or before *as_of* to prevent lookahead bias.
        """
        now = time.time()
        # In live mode (no as_of), use 6-hour cache
        if as_of is None and ticker in self._htf_cache:
            cache_time, cached_trend = self._htf_cache[ticker]
            if now - cache_time < 21600:
                return cached_trend

        try:
            from charts.data import fetch_ohlcv

            end_date_arg = str(as_of) if as_of is not None else None
            df = fetch_ohlcv(ticker, interval=htf_interval, period=htf_period, end_date=end_date_arg)
            if df is None or len(df) < 50:
                if as_of is None:
                    self._htf_cache[ticker] = (now, None)
                return None

            if as_of is not None:
                # Ensure timezone compatibility for filtering
                if df.index.tz is not None and getattr(as_of, "tzinfo", None) is None:
                    as_of_tz = pd.to_datetime(as_of).tz_localize("UTC")
                elif df.index.tz is None and getattr(as_of, "tzinfo", None) is not None:
                    as_of_tz = pd.to_datetime(as_of).tz_localize(None)
                else:
                    as_of_tz = as_of
                df = df[df.index <= as_of_tz]
                if len(df) < 50:
                    return None

            close = df["Close"]
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()

            current_close = float(close.iloc[-1])
            current_ema20 = float(ema20.iloc[-1])
            current_ema50 = float(ema50.iloc[-1])

            if current_ema20 > current_ema50 and current_close > current_ema20:
                trend: Optional[str] = "bullish"
            elif current_ema20 < current_ema50 and current_close < current_ema20:
                trend = "bearish"
            else:
                trend = None

            if as_of is None:
                self._htf_cache[ticker] = (now, trend)
            return trend
        except Exception:
            logger.exception("get_htf_trend failed for %s", ticker)
            if as_of is None:
                self._htf_cache[ticker] = (now, None)
            return None
