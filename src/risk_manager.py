from __future__ import annotations
import logging
from typing import Optional
from src.strategies import StrategySignal

RISK_ON_ASSETS = {"SPY", "QQQ", "IWM", "DIA"}
CRYPTO_ASSETS = {"BTC-USD", "ETH-USD", "BTCUSD", "ETHUSD"}
COMMODITY_ASSETS = {"GLD", "USO", "GC=F", "CL=F"}

# ── Validation Constants ─────────────────────────────────────────────
MAX_STOP_PCT = 0.10      # Maximum stop distance as fraction of entry (10%)
MIN_STOP_PCT = 0.001     # Minimum stop distance as fraction of entry (0.1%)
MIN_RR_RATIO = 1.5       # Minimum reward/risk ratio


class RiskManager:
    """Strategy-agnostic risk management: position sizing, validation, and correlation filtering."""

    def __init__(
        self,
        max_risk_pct: float = 0.01,
        max_position_pct: float = 0.20,
        max_notional: float = 10000.0,
        max_positions: int = 5,
    ) -> None:
        self.max_risk_pct = max_risk_pct
        self.max_position_pct = max_position_pct
        self.max_notional = max_notional
        self.max_positions = max_positions
        
        # Adaptive tracking state
        self.peak_equity = 0.0
        
        self.logger = logging.getLogger(__name__)

        # Ticker mapping: Yahoo Finance -> Alpaca format
        self._ticker_map: dict[str, str] = {
            "BTC-USD": "BTCUSD",
            "ETH-USD": "ETHUSD",
        }

    # ── Position Sizing ────────────────────────────────────────────────

    def calculate_position_size(
        self, signal: StrategySignal, account_equity: float
    ) -> float:
        """Adaptive Stop-distance-based position sizing.

        Returns the notional dollar amount to invest, scaled dynamically by
        signal confidence and account drawdown protection.
        """
        # 1. Update Peak Equity (High-Water Mark)
        if account_equity > self.peak_equity:
            self.peak_equity = account_equity

        # 2. Drawdown Penalty Multiplier (Equity-Adaptive)
        dd_multiplier = 1.0
        if self.peak_equity > 0 and account_equity < self.peak_equity:
            drawdown_pct = (self.peak_equity - account_equity) / self.peak_equity
            dd_multiplier = max(0.2, 1.0 - (drawdown_pct * 10.0))  # Floor risk at 20% of normal

        # 3. Confidence Scaling (Signal-Adaptive)
        confidence_clamped = max(10.0, min(100.0, signal.confidence))
        conf_multiplier = confidence_clamped / 100.0

        # 4. Sector-Specific AI Multiplier
        import config
        if signal.strategy_name == "mean_reversion":
            ai_mult = getattr(config, 'AI_RISK_MULTIPLIER_EQUITY', 1.0)
        elif signal.strategy_name == "momentum_breakout":
            ai_mult = getattr(config, 'AI_RISK_MULTIPLIER_CRYPTO', 1.0)
        elif signal.strategy_name == "trend_following":
            ai_mult = getattr(config, 'AI_RISK_MULTIPLIER_COMMODITY', 1.0)
        else:
            ai_mult = 1.0

        # 5. Final Adaptive Risk Calculation
        adaptive_risk_pct = self.max_risk_pct * ai_mult * dd_multiplier * conf_multiplier
        risk_dollars = account_equity * adaptive_risk_pct

        # 6. Stop Distance Translation
        stop_distance = abs(signal.entry - signal.stop_loss)
        if stop_distance <= 0:
            return 0.0

        position_size_shares = risk_dollars / stop_distance
        notional = position_size_shares * signal.entry

        # 7. Apply Absolute Caps
        notional = min(notional, account_equity * self.max_position_pct)
        notional = min(notional, self.max_notional)
        notional = max(1.0, round(notional, 2))

        self.logger.info(
            "Adaptive Sizing | Conf: %.1f%% | Regime Mult: %.2fx | "
            "Drawdown Mult: %.2fx | Risk: %.3f%% ($%.2f) | Notional: $%.2f",
            confidence_clamped,
            ai_mult,
            dd_multiplier,
            adaptive_risk_pct * 100,
            risk_dollars,
            notional,
        )
        return notional

    # ── Correlation Filter ─────────────────────────────────────────────

    def check_correlation_filter(
        self, signal: StrategySignal, current_positions: list[dict], active_orders: list[dict] = None
    ) -> tuple[bool, str]:
        """Enforce correlation and diversification rules.

        *current_positions* format::
            [{'symbol': str, 'side': str ('long'|'short'), 'market_value': float}, ...]

        *active_orders* format::
            [{'symbol': str, 'status': str}, ...]

        Returns ``(True, "OK")`` if the signal passes, or ``(False, reason)``
        if it is blocked.
        """
        if active_orders is None:
            active_orders = []

        # ── Build set of existing position symbols (including mapped aliases) ──
        position_symbols: set[str] = set()
        for pos in current_positions:
            sym = pos.get("symbol", "")
            position_symbols.add(sym)
            if sym in self._ticker_map:
                position_symbols.add(self._ticker_map[sym])
            if sym in self._ticker_map.values():
                for k, v in self._ticker_map.items():
                    if v == sym:
                        position_symbols.add(k)

        # ── Build set of symbols that have pending unfilled orders ──
        pending_order_symbols: set[str] = set()
        for o in active_orders:
            sym = o.get("symbol", "")
            pending_order_symbols.add(sym)
            if sym in self._ticker_map:
                pending_order_symbols.add(self._ticker_map[sym])
            if sym in self._ticker_map.values():
                for k, v in self._ticker_map.items():
                    if v == sym:
                        pending_order_symbols.add(k)

        # 1A. Max 1 active position per symbol
        if signal.ticker in position_symbols:
            return (False, f"Correlation filter: {signal.ticker} already has an open position")
        if signal.ticker in self._ticker_map and self._ticker_map[signal.ticker] in position_symbols:
            return (
                False,
                f"Correlation filter: {self._ticker_map[signal.ticker]} already has an open position",
            )

        # 1B. Prevent double-dipping: Block if a pending order already exists for this symbol
        if signal.ticker in pending_order_symbols:
            return (False, f"Pending order guard: {signal.ticker} already has an unfilled active order")
        if signal.ticker in self._ticker_map and self._ticker_map[signal.ticker] in pending_order_symbols:
            return (False, f"Pending order guard: {self._ticker_map[signal.ticker]} already has an unfilled active order")

        # 4. Total positions cap
        if len(current_positions) >= self.max_positions:
            return (
                False,
                f"Correlation filter: max positions ({self.max_positions}) reached",
            )

        # 5. Commodities are always allowed through risk-on rules 2 & 3
        if signal.ticker in COMMODITY_ASSETS:
            return (True, "OK")

        # 2 & 3. Risk-on correlation — count long / short positions in both
        #         RISK_ON_ASSETS and CRYPTO_ASSETS (symmetric for all risk assets)
        risk_on_long = 0
        risk_on_short = 0
        for pos in current_positions:
            sym = pos.get("symbol", "")
            side = pos.get("side", "").lower()
            if sym in RISK_ON_ASSETS or sym in CRYPTO_ASSETS:
                if side == "long":
                    risk_on_long += 1
                elif side == "short":
                    risk_on_short += 1

        signal_normalized = self._ticker_map.get(signal.ticker, signal.ticker)
        signal_is_risk_or_crypto = (
            signal.ticker in RISK_ON_ASSETS
            or signal.ticker in CRYPTO_ASSETS
            or signal_normalized in RISK_ON_ASSETS
            or signal_normalized in CRYPTO_ASSETS
        )

        if signal.direction == "BUY" and signal_is_risk_or_crypto:
            if risk_on_long >= 2:
                return (False, "Correlation filter: 2+ risk-on longs already open")

        if signal.direction == "SELL" and signal_is_risk_or_crypto:
            if risk_on_short >= 2:
                return (False, "Correlation filter: 2+ risk-on shorts already open")

        return (True, "OK")

    # ── Signal Validation ──────────────────────────────────────────────

    def validate_signal(self, signal: StrategySignal) -> tuple[bool, str]:
        """Validate the signal's price, stop, R/R and direction consistency.

        Returns ``(True, "OK")`` or ``(False, reason)``.
        """
        # 1. Entry price
        if signal.entry <= 0:
            return (False, "Invalid entry price")

        # 2. Stop distance
        stop_pct = abs(signal.entry - signal.stop_loss) / signal.entry
        if stop_pct > MAX_STOP_PCT:
            return (False, f"Stop too wide (>{MAX_STOP_PCT*100:.1f}%)")
        if stop_pct < MIN_STOP_PCT:
            return (False, f"Stop too tight (<{MIN_STOP_PCT*100:.2f}%)")

        # 3. R/R check
        risk = abs(signal.entry - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry)
        if risk <= 0:
            return (False, "Zero risk")
        # Add a tiny 0.01 tolerance to account for floating point inaccuracies (e.g. 1.499999 < 1.5)
        if (reward / risk) < (MIN_RR_RATIO - 0.01):
            return (False, f"R/R below {MIN_RR_RATIO}")

        # 4. Direction consistency
        if signal.direction == "BUY":
            if signal.stop_loss >= signal.entry:
                return (False, "BUY stop loss must be below entry")
            if signal.take_profit <= signal.entry:
                return (False, "BUY take profit must be above entry")
        elif signal.direction == "SELL":
            if signal.stop_loss <= signal.entry:
                return (False, "SELL stop loss must be above entry")
            if signal.take_profit >= signal.entry:
                return (False, "SELL take profit must be below entry")
        else:
            return (False, f"Unknown direction: {signal.direction}")

        # 5. Confidence must be positive
        if signal.confidence <= 0:
            return (False, "Confidence must be > 0")

        return (True, "OK")

    # ── Full Approval Pipeline ─────────────────────────────────────────

    def approve(
        self,
        signal: StrategySignal,
        account_equity: float,
        current_positions: list[dict],
        active_orders: list[dict] = None,
    ) -> tuple[bool, float, str]:
        """Run the full risk pipeline: validate → correlation → size.

        Returns ``(approved, notional, reason)``.
        """
        if active_orders is None:
            active_orders = []

        # 1. Validate signal
        valid, reason = self.validate_signal(signal)
        if not valid:
            self.logger.info("Signal rejected: %s %s — %s", signal.direction, signal.ticker, reason)
            return (False, 0.0, reason)

        # 2. Correlation filter (now includes double-dip guard against active orders)
        allowed, reason = self.check_correlation_filter(signal, current_positions, active_orders)
        if not allowed:
            self.logger.info("Signal rejected: %s %s — %s", signal.direction, signal.ticker, reason)
            return (False, 0.0, reason)

        # 3. Position sizing
        notional = self.calculate_position_size(signal, account_equity)
        if notional <= 0:
            self.logger.info(
                "Signal rejected: %s %s — Position size is zero",
                signal.direction,
                signal.ticker,
            )
            return (False, 0.0, "Position size is zero")

        self.logger.info(
            "Signal approved: %s %s $%.2f", signal.direction, signal.ticker, notional
        )
        return (True, notional, "Approved")
