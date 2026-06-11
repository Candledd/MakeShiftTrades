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
    order_type: str = "MARKET"


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

    def is_on_cooldown(self, ticker: str) -> bool:
        """Check if the given ticker is currently in cooldown.

        Returns True if a signal was recorded recently enough that the
        cooldown period has not yet elapsed.
        """
        import config

        if ticker not in self._last_signal_time:
            return False
        elapsed = time.time() - self._last_signal_time[ticker]
        return elapsed < config.SIGNAL_COOLDOWN_SECONDS

    def record_signal(self, ticker: str) -> None:
        """Record that a signal was generated for *ticker* at the current time."""
        self._last_signal_time[ticker] = time.time()

    def get_htf_trend(
        self,
        ticker: str,
        htf_interval: str = "1d",
        htf_period: str = "3mo",
    ) -> Optional[str]:
        """Fetch higher-timeframe data and determine the macro trend.

        Uses daily OHLCV data to compute EMA(20) / EMA(50). Returns
        ``"bullish"`` if EMA(20) > EMA(50) and close > EMA(20),
        ``"bearish"`` if EMA(20) < EMA(50) and close < EMA(20),
        or ``None`` otherwise (neutral / error).

        Results are cached for up to 6 hours (21600 seconds) per ticker.
        """
        # Check cache first
        if ticker in self._htf_cache:
            cache_time, cached_trend = self._htf_cache[ticker]
            if time.time() - cache_time < 21600:
                return cached_trend

        try:
            from charts.data import fetch_ohlcv

            df = fetch_ohlcv(ticker, interval=htf_interval, period=htf_period)
            if df is None or len(df) < 50:
                self._htf_cache[ticker] = (time.time(), None)
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

            self._htf_cache[ticker] = (time.time(), trend)
            return trend
        except Exception:
            logger.exception("get_htf_trend failed for %s", ticker)
            self._htf_cache[ticker] = (time.time(), None)
            return None
