from __future__ import annotations
import logging
from typing import Optional
from src.strategies import StrategySignal
from src.database import get_realized_pnl, get_strategy_expectancy
import config as _cfg

RISK_ON_ASSETS = {"SPY", "QQQ", "IWM", "DIA"}
CRYPTO_ASSETS = {"BTC-USD", "ETH-USD", "BTCUSD", "ETHUSD"}
COMMODITY_ASSETS = {"GLD", "USO", "GC=F", "CL=F"}

# ── Validation Constants ─────────────────────────────────────────────
# The SPY and QQQ tradeable unit — they are ~0.95+ correlated and should
# never be held simultaneously.
SPY_QQQ_UNIT = {"SPY", "QQQ"}

MAX_STOP_PCT = 0.10      # Maximum stop distance as fraction of entry (10%)
MIN_STOP_PCT = 0.001     # Minimum stop distance as fraction of entry (0.1%)
MIN_RR_RATIO = 1.5       # Minimum reward/risk ratio


class RiskManager:
    """Strategy-agnostic risk management: position sizing, validation,
    tiered risk, spread/liquidity caps, gap buffer, and volatility shock adjustment."""

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

        # ── Tiered risk overrides (loaded from config) ────────────────
        self._tier_equity_pct = getattr(_cfg, 'RISK_TIER_EQUITY_PCT', 0.0025)
        self._tier_crypto_pct = getattr(_cfg, 'RISK_TIER_CRYPTO_PCT', 0.0050)
        self._tier_commodity_pct = getattr(_cfg, 'RISK_TIER_COMMODITY_PCT', 0.0035)

        # ── Spread / liquidity cap ────────────────────────────────────
        self._spread_atr_cap_pct = getattr(_cfg, 'SPREAD_ATR_CAP_PCT', 0.05)

        # ── Gap / slippage buffer ─────────────────────────────────────
        self._gap_buffer_pct = getattr(_cfg, 'GAP_SLIPPAGE_BUFFER_PCT', 0.001)

        # ── Volatility shock ──────────────────────────────────────────
        self._shock_atr_pct = getattr(_cfg, 'VOLATILITY_SHOCK_ATR_PCT', 0.03)
        self._shock_reduction = getattr(_cfg, 'VOLATILITY_SHOCK_REDUCTION', 0.50)

        self.logger = logging.getLogger(__name__)

        # Ticker mapping: Yahoo Finance -> Alpaca format
        self._ticker_map: dict[str, str] = {
            "BTC-USD": "BTCUSD",
            "ETH-USD": "ETHUSD",
        }

    # ── Asset Tier Helpers ────────────────────────────────────────────

    def _resolve_ticker(self, raw: str) -> str:
        """Resolve a ticker to its canonical (Alpaca) form."""
        return self._ticker_map.get(raw, raw)

    def _resolve_symbol_set(self, items: list, key_field: str = "symbol") -> set[str]:
        """Build a set of symbols from *items*, including resolved aliases."""
        symbols: set[str] = set()
        for item in items:
            sym = item.get(key_field, "")
            if not sym:
                continue
            symbols.add(sym)
            if sym in self._ticker_map:
                symbols.add(self._ticker_map[sym])
            if sym in self._ticker_map.values():
                for k, v in self._ticker_map.items():
                    if v == sym:
                        symbols.add(k)
        return symbols

    def _get_tier_risk_pct(self, signal: StrategySignal) -> float:
        """Return the risk-per-trade percentage for the signal's asset class.

        Falls back to *self.max_risk_pct* for tickers not in any known tier.
        """
        ticker = self._resolve_ticker(signal.ticker)

        if ticker in COMMODITY_ASSETS or signal.ticker in COMMODITY_ASSETS:
            return self._tier_commodity_pct
        if ticker in CRYPTO_ASSETS or signal.ticker in CRYPTO_ASSETS:
            return self._tier_crypto_pct
        if ticker in RISK_ON_ASSETS or signal.ticker in RISK_ON_ASSETS:
            return self._tier_equity_pct

        # Unknown asset — use general cap
        return self.max_risk_pct

    def _get_tier_name(self, signal: StrategySignal) -> str:
        """Human-readable tier name for log messages."""
        ticker = self._resolve_ticker(signal.ticker)
        if ticker in COMMODITY_ASSETS or signal.ticker in COMMODITY_ASSETS:
            return "commodity"
        if ticker in CRYPTO_ASSETS or signal.ticker in CRYPTO_ASSETS:
            return "crypto"
        if ticker in RISK_ON_ASSETS or signal.ticker in RISK_ON_ASSETS:
            return "equity"
        return "unknown"

    # ── Spread / Liquidity Validation ─────────────────────────────────

    def _check_atr_spread(self, signal: StrategySignal) -> tuple[bool, str]:
        """Reject trades where ATR/price exceeds the spread cap.

        A high ATR‑to‑price ratio is a proxy for wide bid‑ask spreads or
        illiquid markets.  Returns ``(True, "OK")`` or ``(False, reason)``.
        """
        if signal.atr <= 0 or signal.entry <= 0:
            return (True, "OK")  # can't judge — pass

        atr_pct = signal.atr / signal.entry
        if atr_pct > self._spread_atr_cap_pct:
            return (
                False,
                f"ATR spread cap: ATR/price {atr_pct:.4f} exceeds "
                f"{self._spread_atr_cap_pct:.4f}",
            )
        return (True, "OK")

    # ── Volatility Shock Adjustment ───────────────────────────────────

    def _volatility_shock_factor(self, signal: StrategySignal) -> float:
        """Return a multiplier (0..1) for positions during high volatility.

        When ATR/price exceeds *self._shock_atr_pct*, return the configured
        reduction factor (default 0.50).  Otherwise return 1.0 (no change).
        """
        if signal.atr <= 0 or signal.entry <= 0:
            return 1.0

        atr_pct = signal.atr / signal.entry
        if atr_pct > self._shock_atr_pct:
            self.logger.info(
                "Volatility shock active: ATR/price %.4f > %.4f — reducing by %.0f%%",
                atr_pct,
                self._shock_atr_pct,
                (1.0 - self._shock_reduction) * 100,
            )
            return self._shock_reduction
        return 1.0

    # ── Gap / Slippage Buffer ─────────────────────────────────────────

    def _apply_gap_buffer(self, notional: float) -> float:
        """Reduce notional by the gap/slippage buffer fraction."""
        reduced = notional * (1.0 - self._gap_buffer_pct)
        return max(1.0, round(reduced, 2))

    # ── Position Sizing ────────────────────────────────────────────────

    def calculate_position_size(
        self, signal: StrategySignal, account_equity: float
    ) -> float:
        """Adaptive Stop-distance-based position sizing with tiered risk,
        volatility shock protection, and gap/slippage buffer.

        Returns the notional dollar amount to invest, scaled dynamically by
        asset-class tier, signal confidence, drawdown, and volatility.
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

        # 4. Tiered Risk Percentage (asset-class-aware)
        tier_risk_pct = self._get_tier_risk_pct(signal)
        tier_name = self._get_tier_name(signal)

        # 5. Sector-Specific AI Multiplier
        if signal.strategy_name == "mean_reversion":
            ai_mult = getattr(_cfg, 'AI_RISK_MULTIPLIER_EQUITY', 1.0)
        elif signal.strategy_name == "momentum_breakout":
            ai_mult = getattr(_cfg, 'AI_RISK_MULTIPLIER_CRYPTO', 1.0)
        elif signal.strategy_name == "trend_following":
            ai_mult = getattr(_cfg, 'AI_RISK_MULTIPLIER_COMMODITY', 1.0)
        else:
            ai_mult = 1.0

        # 6. Volatility Shock Factor
        shock_mult = self._volatility_shock_factor(signal)

        # 7. Final Adaptive Risk Calculation
        adaptive_risk_pct = (
            tier_risk_pct * ai_mult * dd_multiplier * conf_multiplier * shock_mult
        )

        # Enforce absolute 1% maximum hard limit regardless of multipliers
        adaptive_risk_pct = min(adaptive_risk_pct, 0.01)

        risk_dollars = account_equity * adaptive_risk_pct

        # 8. Stop Distance Translation
        stop_distance = abs(signal.entry - signal.stop_loss)
        if stop_distance <= 0:
            return 0.0

        position_size_shares = risk_dollars / stop_distance
        notional = position_size_shares * signal.entry

        # 9. Apply Absolute Caps
        notional = min(notional, account_equity * self.max_position_pct)
        notional = min(notional, self.max_notional)
        notional = max(1.0, round(notional, 2))

        # 10. Gap / Slippage Buffer
        notional = self._apply_gap_buffer(notional)

        self.logger.info(
            "Adaptive Sizing | Tier: %s @ %.4f%% | Conf: %.1f%% | "
            "Shock: %.2fx | DD: %.2fx | Risk: %.3f%% ($%.2f) | Notional: $%.2f",
            tier_name,
            tier_risk_pct * 100,
            confidence_clamped,
            shock_mult,
            dd_multiplier,
            adaptive_risk_pct * 100,
            risk_dollars,
            notional,
        )
        return notional

    # ── Correlation Filter ─────────────────────────────────────────────

    def check_correlation_filter(
        self,
        signal: StrategySignal,
        current_positions: list[dict],
        active_orders: list[dict] = None,
        position_state: dict = None,
        account_equity: float = 0.0,
        proposed_notional: float = 0.0,
    ) -> tuple[bool, str]:
        """Enforce correlation and diversification rules.

        *current_positions* format::
            [{'symbol': str, 'side': str ('long'|'short'), 'market_value': float}, ...]

        *active_orders* format::
            [{'symbol': str, 'status': str}, ...]

        *position_state*::
            {ticker: {'entry_price': float, 'stop_loss': float, 'qty': float}, ...}

        Returns ``(True, "OK")`` if the signal passes, or ``(False, reason)``
        if it is blocked.
        """
        if active_orders is None:
            active_orders = []
        if position_state is None:
            position_state = {}

        # ── Portfolio Heat Check — Total Open Risk ──────────────────────────
        # Include both open positions AND pending orders in the risk calculation
        total_risk_usd = 0.0
        heat_symbols: set[str] = set()
        for pos in current_positions:
            sym = pos.get("symbol", "")
            if sym:
                heat_symbols.add(sym)
        for o in active_orders:
            sym = o.get("symbol", "")
            if sym:
                heat_symbols.add(sym)
        for sym in heat_symbols:
            state = position_state.get(sym)
            if state is not None and "entry_price" in state and "stop_loss" in state:
                risk_usd = abs(state["entry_price"] - state["stop_loss"]) * state.get("qty", 0.0)
                total_risk_usd += risk_usd
            else:
                # Fallback after restart if state is missing
                pos = next((p for p in current_positions if p.get('symbol') == sym), None)
                if pos:
                    total_risk_usd += pos.get('market_value', 0.0) * 0.05

        # ── Include proposed trade risk in portfolio heat ─────────────
        if proposed_notional > 0 and signal.entry > 0:
            proposed_qty = proposed_notional / signal.entry
            proposed_risk_usd_val = abs(signal.entry - signal.stop_loss) * proposed_qty
            total_risk_usd += proposed_risk_usd_val
        else:
            proposed_risk_usd_val = 0.0

        if account_equity > 0 and total_risk_usd > 0:
            max_portfolio_risk_pct = _cfg.MAX_OPEN_PORTFOLIO_RISK_PCT
            if total_risk_usd / account_equity > max_portfolio_risk_pct:
                return (
                    False,
                    f"Portfolio heat limit: total open risk ${total_risk_usd:.2f} "
                    f"({total_risk_usd/account_equity:.2%}) exceeds "
                    f"{max_portfolio_risk_pct:.2%}",
                )

        # ── Cluster Risk Check ──────────────────────────────────────────────
        def _cluster_risk(cluster_set: set[str]) -> float:
            risk = 0.0
            for sym in heat_symbols:
                if sym in cluster_set or self._ticker_map.get(sym, sym) in cluster_set:
                    state = position_state.get(sym)
                    if state is not None and "entry_price" in state and "stop_loss" in state:
                        risk += abs(state["entry_price"] - state["stop_loss"]) * state.get("qty", 0.0)
                    else:
                        pos = next((p for p in current_positions if p.get('symbol') == sym), None)
                        if pos:
                            risk += pos.get('market_value', 0.0) * 0.05
            return risk

        if account_equity > 0:
            max_cluster_pct = _cfg.MAX_CLUSTER_RISK_PCT
            signal_resolved_ticker = self._resolve_ticker(signal.ticker)
            for cluster_name, cluster_set in [
                ("risk-on", RISK_ON_ASSETS),
                ("crypto", CRYPTO_ASSETS),
                ("commodity", COMMODITY_ASSETS),
            ]:
                cluster_risk = _cluster_risk(cluster_set)
                # Include proposed trade risk if signal belongs to this cluster
                if proposed_risk_usd_val > 0 and (
                    signal.ticker in cluster_set or signal_resolved_ticker in cluster_set
                ):
                    cluster_risk += proposed_risk_usd_val
                if cluster_risk > 0 and (cluster_risk / account_equity) > max_cluster_pct:
                    return (
                        False,
                        f"Cluster heat limit ({cluster_name}): cluster risk "
                        f"${cluster_risk:.2f} ({cluster_risk/account_equity:.2%}) "
                        f"exceeds {max_cluster_pct:.2%}",
                    )

        # ── Build set of existing position symbols (including mapped aliases) ──
        position_symbols = self._resolve_symbol_set(current_positions)

        # ── Build set of symbols that have pending unfilled orders ──
        pending_order_symbols = self._resolve_symbol_set(active_orders)

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

        # 1C. SPY/QQQ tradeable unit guard — highly correlated (0.95+), treat as single position
        signal_resolved = self._resolve_ticker(signal.ticker)
        if signal_resolved in SPY_QQQ_UNIT:
            counterpart = "QQQ" if signal_resolved == "SPY" else "SPY"
            if counterpart in position_symbols:
                return (False, f"SPY/QQQ guard: {counterpart} already open; rejecting {signal.ticker} (0.95+ correlated)")
            if counterpart in pending_order_symbols:
                return (False, f"SPY/QQQ guard: {counterpart} has pending order; rejecting {signal.ticker}")

        # 4. Total positions cap
        if len(current_positions) >= self.max_positions:
            return (
                False,
                f"Correlation filter: max positions ({self.max_positions}) reached",
            )

        # 2 & 3. Risk-on correlation — count long / short positions across all
        #         major asset classes (equity, crypto, commodity)
        risk_on_long = 0
        risk_on_short = 0
        for pos in current_positions:
            sym = pos.get("symbol", "")
            side = pos.get("side", "").lower()
            if sym in RISK_ON_ASSETS or sym in CRYPTO_ASSETS or sym in COMMODITY_ASSETS:
                if side == "long":
                    risk_on_long += 1
                elif side == "short":
                    risk_on_short += 1

        signal_normalized = self._ticker_map.get(signal.ticker, signal.ticker)
        signal_is_risk_asset = (
            signal.ticker in RISK_ON_ASSETS
            or signal.ticker in CRYPTO_ASSETS
            or signal.ticker in COMMODITY_ASSETS
            or signal_normalized in RISK_ON_ASSETS
            or signal_normalized in CRYPTO_ASSETS
            or signal_normalized in COMMODITY_ASSETS
        )

        if signal.direction == "BUY" and signal_is_risk_asset:
            if risk_on_long >= 2:
                return (False, "Correlation filter: 2+ risk-on longs already open")

        if signal.direction == "SELL" and signal_is_risk_asset:
            if risk_on_short >= 2:
                return (False, "Correlation filter: 2+ risk-on shorts already open")

        return (True, "OK")

    # ── Signal Validation ──────────────────────────────────────────────

    def validate_signal(
        self, signal: StrategySignal, signal_regime: str | None = None
    ) -> tuple[bool, str]:
        """Validate the signal's price, stop, R/R, direction consistency,
        ATR-based spread/liquidity cap, and strategy expectancy.

        ``signal_regime`` is an optional regime label used for granular
        expectancy filtering (e.g. ``"bullish_calm"``).

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

        # 6. ATR-based spread / liquidity cap
        spread_ok, spread_reason = self._check_atr_spread(signal)
        if not spread_ok:
            return (False, spread_reason)

        # 7. Strategy expectancy gate — reject trades with negative expected value
        #    Two-tier granular check: first with symbol + regime, fall back to
        #    broader strategy+direction when the sample is too small.
        ev_r, sample_size = get_strategy_expectancy(
            signal.strategy_name, signal.direction,
            symbol=signal.ticker, regime=signal_regime,
        )
        if sample_size < _cfg.MIN_EXPECTANCY_SAMPLES:
            ev_r, sample_size = get_strategy_expectancy(signal.strategy_name, signal.direction)
        if sample_size >= _cfg.MIN_EXPECTANCY_SAMPLES and ev_r < _cfg.MIN_EXPECTANCY_R:
            return (
                False,
                f"Expectancy gate: ev_r={ev_r:.3f} (sample={sample_size}) "
                f"below {_cfg.MIN_EXPECTANCY_R} threshold",
            )

        return (True, "OK")

    # ── Full Approval Pipeline ─────────────────────────────────────────

    def approve(
        self,
        signal: StrategySignal,
        account_equity: float,
        current_positions: list[dict],
        active_orders: list[dict] = None,
        position_state: dict = None,
        signal_regime: Optional[str] = None,
    ) -> tuple[bool, float, str]:
        """Run the full risk pipeline: validate → correlation → size.

        ``signal_regime`` is forwarded to ``validate_signal`` for granular
        expectancy filtering.

        Returns ``(approved, notional, reason)``.
        """
        if active_orders is None:
            active_orders = []
        if position_state is None:
            position_state = {}

        # 0. P&L Kill Switch — daily and weekly loss limits
        daily_pnl = get_realized_pnl(1)
        if daily_pnl < account_equity * -abs(_cfg.MAX_DAILY_LOSS_PCT):
            self.logger.info(
                "Signal rejected: %s %s — Daily loss limit hit ($%.2f < $%.2f)",
                signal.direction, signal.ticker,
                daily_pnl, account_equity * -abs(_cfg.MAX_DAILY_LOSS_PCT),
            )
            return (False, 0.0, f"Daily P&L kill switch: ${daily_pnl:.2f} exceeds limit")

        weekly_pnl = get_realized_pnl(7)
        if weekly_pnl < account_equity * -abs(_cfg.MAX_WEEKLY_LOSS_PCT):
            self.logger.info(
                "Signal rejected: %s %s — Weekly loss limit hit ($%.2f < $%.2f)",
                signal.direction, signal.ticker,
                weekly_pnl, account_equity * -abs(_cfg.MAX_WEEKLY_LOSS_PCT),
            )
            return (False, 0.0, f"Weekly P&L kill switch: ${weekly_pnl:.2f} exceeds limit")

        # 1. Validate signal
        valid, reason = self.validate_signal(signal, signal_regime=signal_regime)
        if not valid:
            self.logger.info("Signal rejected: %s %s — %s", signal.direction, signal.ticker, reason)
            return (False, 0.0, reason)

        # 2. Position sizing (calculated early for portfolio heat check)
        notional = self.calculate_position_size(signal, account_equity)
        if notional <= 0:
            self.logger.info(
                "Signal rejected: %s %s — Position size is zero",
                signal.direction,
                signal.ticker,
            )
            return (False, 0.0, "Position size is zero")

        # 3. Correlation filter (now includes double-dip guard against active orders
        #    and portfolio heat check with proposed notional)
        allowed, reason = self.check_correlation_filter(
            signal, current_positions, active_orders,
            position_state=position_state, account_equity=account_equity,
            proposed_notional=notional,
        )
        if not allowed:
            self.logger.info("Signal rejected: %s %s — %s", signal.direction, signal.ticker, reason)
            return (False, 0.0, reason)

        self.logger.info(
            "Signal approved: %s %s $%.2f", signal.direction, signal.ticker, notional
        )
        return (True, notional, "Approved")
