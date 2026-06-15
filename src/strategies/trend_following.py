from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Optional
import pandas as pd
import numpy as np
from src.strategies import BaseStrategy, StrategySignal


logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """Classic Turtle Donchian Breakout trend-following strategy.

    Enters long when price breaks above the 55-bar Donchian channel
    and short when it breaks below.  The initial stop is placed at
    the opposite 20-bar Donchian level.  Take-profit is set extremely
    wide (50 × ATR) so the position is ridden via the trailing stop.
    """

    name = "trend_following"
    tickers = ["GLD", "PDBC"]
    timeframe = "4h"
    period = "3mo"

    # ── EWMA Realized Volatility Regime Sizing ────────────────────────
    # bars_per_day is inferred dynamically from the DataFrame index below.
    TARGET_VOL_ANNUAL = 0.15     # 15 % target annualised volatility
    VOL_REGIME_CAP = 3.0         # maximum inverse-vol multiplier

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            tickers=self.tickers,
            timeframe=self.timeframe,
            period=self.period,
        )

    # ──────────────────────────────────────────────────────────────────────
    # EWMA Realized Volatility — inverse-vol regime sizing
    # ──────────────────────────────────────────────────────────────────────

    def _compute_vol_regime_multiplier(self, df: pd.DataFrame) -> float:
        """Compute an inverse-volatility sizing multiplier using EWMA.

        Bars-per-day is inferred dynamically from the DataFrame index
        frequency (or approximated via the median bar spacing).

        Steps
        -----
        1. Daily returns  →  ``Close.pct_change()``
        2. EWMA std(60)   →  responsive to regime shifts
        3. Annualise      →  ``ewma_std * sqrt(252 * bars_per_day)``
        4. Inverse weight →  ``TARGET_VOL_ANNUAL / annualised_vol``
        5. Cap            →  ``min(weight, VOL_REGIME_CAP)``

        Returns
        -------
        float
            Multiplier in ``[0, VOL_REGIME_CAP]``.  Falls back to 1.0
            when there is insufficient data or volatility is zero.
        """
        if df is None or len(df) < 62:
            return 1.0

        try:
            # ── Dynamically infer bars per day ────────────────────────
            bars_per_day = 1.0  # fallback
            if isinstance(df.index, pd.DatetimeIndex) and len(df) >= 2:
                delta_seconds = (
                    df.index.to_series().diff().dt.total_seconds().median()
                )
                if pd.notna(delta_seconds) and delta_seconds > 0:
                    bars_per_day = 86400.0 / delta_seconds

            returns = df["Close"].pct_change()
            ewma_std = returns.ewm(span=60, adjust=False).std()

            current_ewma_std = float(ewma_std.iloc[-1])
            if pd.isna(current_ewma_std) or current_ewma_std <= 0.0:
                return 1.0

            annualisation_factor = np.sqrt(252 * bars_per_day)
            ewma_realized_vol_annual = current_ewma_std * annualisation_factor

            multiplier = self.TARGET_VOL_ANNUAL / ewma_realized_vol_annual
            multiplier = min(multiplier, self.VOL_REGIME_CAP)

            logger.debug(
                "vol_regime: ewma_std=%.6f  annual_vol=%.4f  mult=%.2f  bars_day=%.1f",
                current_ewma_std, ewma_realized_vol_annual, multiplier, bars_per_day,
            )
            return multiplier
        except Exception:
            logger.exception("_compute_vol_regime_multiplier failed")
            return 1.0

    def analyze(self, df: pd.DataFrame, ticker: str) -> Optional[StrategySignal]:
        """Turtle Donchian Breakout trend-following analysis.

        Enters on a 55-bar Donchian breakout and places the initial stop
        at the opposite 20-bar Donchian level.  Take-profit is set at
        50 × ATR so the trend is ridden via the trailing stop instead.
        """
        # Require at least 60 bars for indicators to stabilise
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

        # ── EWMA vol regime multiplier ──────────────────────────────────
        vol_regime_multiplier = self._compute_vol_regime_multiplier(df)

        # ── ATR (Wilder's EMA, span=14) ─────────────────────────────────
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

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
        if pd.isna(atr_val) or atr_val <= 0:
            return None

        # ── Donchian Channels ───────────────────────────────────────────
        # Slow channel (period=55) — entry signals
        donchian_upper_55 = high.shift(1).rolling(55).max()
        donchian_lower_55 = low.shift(1).rolling(55).min()

        # Fast channel (period=20) — initial stop loss
        donchian_upper_20 = high.shift(1).rolling(20).max()
        donchian_lower_20 = low.shift(1).rolling(20).min()

        current_close = float(close.iloc[-1])
        current_donchian_upper_55 = float(donchian_upper_55.iloc[-1])
        current_donchian_lower_55 = float(donchian_lower_55.iloc[-1])
        current_donchian_upper_20 = float(donchian_upper_20.iloc[-1])
        current_donchian_lower_20 = float(donchian_lower_20.iloc[-1])

        # ── Entry logic ─────────────────────────────────────────────────
        direction: Optional[str] = None
        reason_parts: list[str] = []

        if current_close > current_donchian_upper_55:
            direction = "BUY"
            reason_parts.append(
                f"Close {current_close:.2f} > Donchian(55) upper {current_donchian_upper_55:.2f}"
            )
        elif current_close < current_donchian_lower_55:
            direction = "SELL"
            reason_parts.append(
                f"Close {current_close:.2f} < Donchian(55) lower {current_donchian_lower_55:.2f}"
            )
        else:
            logger.debug(
                "%s: no Donchian(55) breakout (close=%.2f, upper=%.2f, lower=%.2f)",
                ticker, current_close, current_donchian_upper_55, current_donchian_lower_55,
            )
            return None

        entry = current_close

        # ── Stop loss ───────────────────────────────────────────────────
        if direction == "BUY":
            stop_loss = current_donchian_lower_20
            # Fallback if SL >= entry or is NaN
            if pd.isna(stop_loss) or stop_loss >= entry:
                stop_loss = entry - (atr_val * 2.0)
        else:  # SELL
            stop_loss = current_donchian_upper_20
            # Fallback if SL <= entry or is NaN
            if pd.isna(stop_loss) or stop_loss <= entry:
                stop_loss = entry + (atr_val * 2.0)

        # ── Take profit (effectively never hits — ride the trend) ───────
        if direction == "BUY":
            take_profit = entry + (atr_val * 50.0)
        else:
            take_profit = entry - (atr_val * 50.0)

        confidence = 50.0

        reason = " | ".join(reason_parts)
        reason += " | Trail: Donchian(20)"

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
            atr=atr_val,
            timestamp=datetime.now(timezone.utc),
            order_type="MARKET",
            time_stop_bars=999999,
            trailing_stop_logic="donchian",
            vol_multiplier=vol_regime_multiplier,
        )

