from __future__ import annotations
import logging
import os
from typing import Optional
import pandas as pd
from src.strategies import StrategySignal
from src.database import get_realized_pnl, get_strategy_expectancy
import config as _cfg

import time

RISK_ON_ASSETS = {"SPY", "QQQ", "IWM", "DIA"}
CRYPTO_ASSETS = {"BTC-USD", "ETH-USD", "BTCUSD", "ETHUSD"}
COMMODITY_ASSETS = {"GLD", "PDBC", "GC=F", "CL=F"}

# ── Validation Constants ─────────────────────────────────────────────
# The SPY and QQQ tradeable unit — they are ~0.95+ correlated and should
# never be held simultaneously.
SPY_QQQ_UNIT = {"SPY", "QQQ"}

MAX_STOP_PCT = 0.10      # Maximum stop distance as fraction of entry (10%)
MIN_STOP_PCT = 0.001     # Minimum stop distance as fraction of entry (0.1%)
MIN_RR_RATIO = 1.0       # Universal minimum reward/risk ratio (fallback)

# Cache TTL for correlation price data (1 hour in seconds)
CORR_CACHE_TTL_SECONDS = 3600


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

        # ── In-memory price cache for correlation penalty (value = (Series, timestamp)) ──
        self._price_cache: dict[str, tuple[pd.Series, float]] = {}

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

    # ── Asset-Specific Fallback Risk Mapping (Position Restart Recovery) ──
    # Midpoints of the risk ranges specified for each asset:
    #   SPY/QQQ  →  2.5%  (range 2-3%)
    #   BTC      → 10.0%  (range 8-12%)
    #   GLD      →  4.0%  (range 3-5%)
    #   PDBC     →  6.5%  (range 5-8%)
    # Used as the default stop-distance percentage when reconstructing
    # position state after a restart and the original stop-loss is unknown.
    _FALLBACK_RISK_MAP: dict[str, float] = {
        # SPY & QQQ — large-cap equity ETFs (tightest range 2-3%)
        "SPY":   0.025,  # 2.5% — midpoint of 2-3%
        "QQQ":   0.025,  # 2.5% — midpoint of 2-3%
        # Crypto — highest volatility, widest range 8-12%
        "BTC-USD": 0.10,  # 10.0% — midpoint of 8-12%
        "BTCUSD":  0.10,  # 10.0% — midpoint of 8-12%
        # Gold — moderate volatility, range 3-5%
        "GLD":   0.04,   # 4.0% — midpoint of 3-5%
        # Broad commodities — medium-high volatility, range 5-8%
        "PDBC":  0.065,  # 6.5% — midpoint of 5-8%
    }

    # Fallback percentages per asset class (used when the exact symbol
    # is not in _FALLBACK_RISK_MAP but the asset class is known).
    _CLASS_FALLBACK_RISK: dict[str, float] = {
        "equity":    0.025,  # 2.5%
        "crypto":    0.10,   # 10.0%
        "commodity": 0.05,   # 5.0% (general commodity fallback)
    }

    def get_fallback_risk_pct(self, symbol: str) -> float:
        """Return the asset-specific fallback risk percentage for *symbol*.

        Risk ranges (midpoint used):
          SPY/QQQ  →  2–3%    (2.5%)
          BTC      →  8–12%   (10.0%)
          GLD      →  3–5%    (4.0%)
          PDBC     →  5–8%    (6.5%)

        Falls back to the asset-class average for other known assets,
        or a general 5% for unrecognised tickers.
        """
        resolved = self._resolve_ticker(symbol)

        # 1. Exact symbol match in the mapping (most specific)
        fallback = self._FALLBACK_RISK_MAP.get(resolved)
        if fallback is not None:
            return fallback

        # 2. Asset-class level matching (broader fallback for related tickers)
        if resolved in RISK_ON_ASSETS or symbol in RISK_ON_ASSETS:
            return self._CLASS_FALLBACK_RISK["equity"]
        if resolved in CRYPTO_ASSETS or symbol in CRYPTO_ASSETS:
            return self._CLASS_FALLBACK_RISK["crypto"]
        if resolved in COMMODITY_ASSETS or symbol in COMMODITY_ASSETS:
            return self._CLASS_FALLBACK_RISK["commodity"]

        # 3. Catch-all — 5% for completely unknown tickers
        return 0.05

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

    def _resolve_asset_tier(self, ticker: str) -> str:
        """Return the asset tier name for *ticker*: 'equity', 'crypto', 'commodity', or 'unknown'."""
        resolved = self._resolve_ticker(ticker)
        if resolved in COMMODITY_ASSETS or ticker in COMMODITY_ASSETS:
            return "commodity"
        if resolved in CRYPTO_ASSETS or ticker in CRYPTO_ASSETS:
            return "crypto"
        if resolved in RISK_ON_ASSETS or ticker in RISK_ON_ASSETS:
            return "equity"
        return "unknown"

    def _get_tier_risk_pct(self, signal: StrategySignal) -> float:
        """Return the risk-per-trade percentage for the signal's asset class.

        Falls back to *self.max_risk_pct* for tickers not in any known tier.
        """
        tier = self._resolve_asset_tier(signal.ticker)
        mapping = {
            "commodity": self._tier_commodity_pct,
            "crypto":    self._tier_crypto_pct,
            "equity":    self._tier_equity_pct,
        }
        return mapping.get(tier, self.max_risk_pct)

    def _get_tier_name(self, signal: StrategySignal) -> str:
        """Human-readable tier name for log messages."""
        return self._resolve_asset_tier(signal.ticker)

    # ── Spread / Liquidity Validation ─────────────────────────────────

    def _check_execution_metrics(
        self, signal: StrategySignal, df: pd.DataFrame = None, quote: dict = None, notional: float = 0.0
    ) -> tuple[bool, str]:
        """Reject trades based on true execution liquidity, not just volatility.
        
        Evaluates:
        - Spread relative to target (if quote available)
        - Market impact / Participation rate (if DF available)
        - ATR/Price as a volatility shock filter
        """
        # 1. Volatility Shock Filter (ATR/Price)
        if signal.atr > 0 and signal.entry > 0:
            atr_pct = signal.atr / signal.entry
            if atr_pct > self._spread_atr_cap_pct:
                return (
                    False,
                    f"Volatility shock cap: ATR/price {atr_pct:.4f} exceeds "
                    f"{self._spread_atr_cap_pct:.4f}",
                )
                
        # 2. Market Impact / Participation Rate
        if df is not None and len(df) >= 20 and notional > 0:
            # 20-bar rolling dollar volume
            dollar_vol = (df["Close"] * df["Volume"]).rolling(20).mean().iloc[-1]
            if dollar_vol > 0:
                participation_rate = notional / dollar_vol
                # Soft cap at 5% of recent 20-bar average volume to prevent moving the market
                if participation_rate > 0.05:
                    return (
                        False,
                        f"Liquidity rejected: Order size ${notional:.0f} exceeds 5% of "
                        f"avg bar volume ${dollar_vol:.0f} (PR={participation_rate:.3f})"
                    )
                    
        # 3. Live Bid/Ask Spread vs Expected Target
        if quote is not None:
            bid = quote.get("bid", 0.0)
            ask = quote.get("ask", 0.0)
            if bid > 0 and ask > 0 and ask > bid:
                spread = ask - bid
                target_dist = abs(signal.take_profit - signal.entry)
                # Reject if crossing the spread eats more than 10% of the gross target
                if target_dist > 0 and (spread / target_dist) > 0.10:
                    return (
                        False,
                        f"Spread rejected: Bid/Ask spread ${spread:.3f} eats "
                        f"{(spread/target_dist)*100:.1f}% of gross target ${target_dist:.3f}"
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
        self, signal: StrategySignal, account_equity: float, signal_regime: Optional[str] = None,
        ml_multiplier: float = 1.0, ev_r: float = 0.0, sample_size: int = 0,
        current_positions: Optional[list[dict]] = None,
    ) -> float:
        """Adaptive Stop-distance-based position sizing with tiered risk,
        volatility shock protection, gap/slippage buffer, ML multiplier,
        and expectancy-based soft penalties.

        Returns the notional dollar amount to invest, scaled dynamically by
        asset-class tier, signal confidence, drawdown, volatility, ML, and
        expectancy ratio.
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
        elif signal.ticker == "GLD":
            ai_mult = getattr(_cfg, 'AI_RISK_MULTIPLIER_GOLD', 1.0)
        elif signal.ticker == "PDBC":
            ai_mult = getattr(_cfg, 'AI_RISK_MULTIPLIER_BROAD_COMMODITY', 1.0)
        else:
            ai_mult = 1.0

        # 6. Volatility Shock Factor
        shock_mult = self._volatility_shock_factor(signal)

        # 7. Final Adaptive Risk Calculation
        adaptive_risk_pct = (
            tier_risk_pct * ai_mult * dd_multiplier * conf_multiplier * shock_mult
        )
        
        # 8. Correlation Penalty (Portfolio-Level)
        correlation_multiplier = 1.0
        if current_positions is not None:
            correlation_multiplier = self._compute_correlation_penalty(
                signal.ticker, current_positions, signal.direction,
            )
            adaptive_risk_pct *= correlation_multiplier

        # Fat-Tail Vol-of-Vol Scalar (Injected directly from strategy)
        if hasattr(signal, 'fat_tail_scalar') and signal.fat_tail_scalar != 1.0:
            adaptive_risk_pct *= signal.fat_tail_scalar
            
        # Hard Regime Cap for Bearish Volatile
        if signal_regime == "Bearish Volatile":
            adaptive_risk_pct *= 0.50  # Cap nominal exposure at 50% in panic regimes
            
        # Enforce absolute 4% maximum hard limit regardless of multipliers
        adaptive_risk_pct = min(adaptive_risk_pct, 0.04)

        risk_dollars = account_equity * adaptive_risk_pct

        # 8. Stop Distance Translation
        stop_distance = abs(signal.entry - signal.stop_loss)
        if stop_distance <= 0:
            return 0.0

        position_size_shares = risk_dollars / stop_distance
        notional = position_size_shares * signal.entry

        # 9. ML Multiplier (applied before absolute caps so portfolio heat checks
        #    see the boosted notional — must precede max_notional hard cap)
        notional *= ml_multiplier
        notional = max(1.0, round(notional, 2))

        # 10. Expectancy-Based Soft Penalty
        #     ev_r / sample_size are already fetched once in approve().
        #     Apply a tapered notional reduction when sample coverage exists but
        #     is insufficient for a hard disable.
        expectancy_multiplier = 1.0
        if ev_r < _cfg.MIN_EXPECTANCY_R:
            band_1 = _cfg.EXPECTANCY_SOFT_BAND_1_MIN
            band_2 = _cfg.EXPECTANCY_SOFT_BAND_2_MIN
            hard_disable = _cfg.EXPECTANCY_HARD_DISABLE_SAMPLES
            if band_1 <= sample_size < band_2:
                expectancy_multiplier = _cfg.EXPECTANCY_SOFT_BAND_1_MULT
            elif band_2 <= sample_size < hard_disable:
                expectancy_multiplier = _cfg.EXPECTANCY_SOFT_BAND_2_MULT

        notional *= expectancy_multiplier
        notional = max(1.0, round(notional, 2))

        # 11. Gap / Slippage Buffer
        notional = self._apply_gap_buffer(notional)

        # 12. Apply Absolute Caps (max_notional must be the very last sizing step)
        notional = min(notional, account_equity * self.max_position_pct)
        notional = min(notional, self.max_notional)
        notional = max(1.0, round(notional, 2))

        self.logger.info(
            "Adaptive Sizing | Tier: %s @ %.4f%% | Conf: %.1f%% | "
            "Shock: %.2fx | DD: %.2fx | Corr: %.2fx | Risk: %.3f%% ($%.2f) | "
            "Notional: $%.2f | ML: %.2fx | Exp: %.2fx (ev_r=%.3f, n=%d)",
            tier_name,
            tier_risk_pct * 100,
            confidence_clamped,
            shock_mult,
            dd_multiplier,
            correlation_multiplier,
            adaptive_risk_pct * 100,
            risk_dollars,
            notional,
            ml_multiplier,
            expectancy_multiplier,
            ev_r,
            sample_size,
        )
        return notional

    # ── Correlation Filter ─────────────────────────────────────────────

    def _compute_position_risk_usd(
        self, symbols: set[str], position_state: dict, current_positions: list[dict],
    ) -> float:
        """Compute total portfolio risk USD for the given set of symbols.

        Uses *position_state* (entry_price, stop_loss, qty) when available,
        otherwise falls back to ``market_value * get_fallback_risk_pct()``.
        """
        total = 0.0
        for sym in symbols:
            state = position_state.get(sym)
            if state is not None and "entry_price" in state and "stop_loss" in state:
                total += abs(state["entry_price"] - state["stop_loss"]) * state.get("qty", 0.0)
            else:
                pos = next((p for p in current_positions if p.get('symbol') == sym), None)
                if pos:
                    total += pos.get('market_value', 0.0) * self.get_fallback_risk_pct(sym)
        return total

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
        heat_symbols: set[str] = set()
        for pos in current_positions:
            sym = pos.get("symbol", "")
            if sym:
                heat_symbols.add(sym)
        for o in active_orders:
            sym = o.get("symbol", "")
            if sym:
                heat_symbols.add(sym)
        total_risk_usd = self._compute_position_risk_usd(heat_symbols, position_state, current_positions)

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
            cluster_symbols = {
                sym for sym in heat_symbols
                if sym in cluster_set or self._ticker_map.get(sym, sym) in cluster_set
            }
            return self._compute_position_risk_usd(cluster_symbols, position_state, current_positions)

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

    # ── Correlation Penalty (Portfolio-Level Sizing) ──────────────────

    def _compute_correlation_penalty(
        self, signal_ticker: str, current_positions: list[dict], signal_direction: str,
    ) -> float:
        """Compute a direction-aware, notional-weighted correlation penalty multiplier.

        Fetches the last 60 days of daily Close prices for the signal ticker
        and all currently open tickers using ``yfinance``.  Calculates the
        Pearson correlation matrix of their daily returns and computes the
        notional-weighted average correlation of the new ticker against the
        existing portfolio.

        The weight of each existing position is proportional to its *market_value*.
        Position *side* (``"long"`` / ``"short"``) is compared to the signal
        *direction* (``"BUY"`` / ``"SELL"``).  Opposite-direction positions are
        treated as hedges — their correlation contributes *negatively* to the
        weighted average, reducing the penalty.

        The penalty multiplier is::

            multiplier = 1.0 - max(0, avg_weighted_corr * 0.8)   [floored at 0.2]

        Price data is cached in ``self._price_cache`` with a 1-hour TTL.
        Returns ``1.0`` when there are fewer than two tickers, when data
        cannot be fetched, or when the notional-weighted average correlation
        is zero or negative.
        """
        import yfinance as yf

        # 1. Collect all unique tickers
        tickers: set[str] = {signal_ticker}
        for pos in current_positions:
            sym = pos.get("symbol", "")
            if sym:
                tickers.add(sym)

        if len(tickers) < 2:
            return 1.0

        ticker_list = sorted(tickers)
        now = time.time()

        # 2. Download price data for uncached or expired tickers
        uncached: list[str] = []
        for t in ticker_list:
            entry = self._price_cache.get(t)
            if entry is None or (now - entry[1]) > CORR_CACHE_TTL_SECONDS:
                uncached.append(t)

        if uncached:
            try:
                data = yf.download(uncached, period="60d", progress=False)
                if data.empty:
                    return 1.0

                if isinstance(data.columns, pd.MultiIndex):
                    for t in uncached:
                        if 'Close' in data and t in data['Close'].columns:
                            self._price_cache[t] = (data['Close'][t].dropna(), now)
                else:
                    if 'Close' in data.columns:
                        self._price_cache[uncached[0]] = (data['Close'].dropna(), now)
            except Exception:
                self.logger.warning(
                    "Correlation penalty: yfinance download failed for %s", uncached,
                    exc_info=True,
                )
                return 1.0

        # 3. Build DataFrame of Close prices from cache
        close_series: dict[str, pd.Series] = {}
        for t in ticker_list:
            entry = self._price_cache.get(t)
            if entry is not None:
                close_series[t] = entry[0]

        if len(close_series) < 2:
            return 1.0

        close_df = pd.DataFrame(close_series).dropna(how="any")
        if len(close_df) < 5:
            return 1.0  # Not enough overlapping data points

        # 4. Daily returns & Pearson correlation matrix
        returns_df = close_df.pct_change().dropna()
        if returns_df.empty:
            return 1.0

        corr_matrix = returns_df.corr(method="pearson")

        # 5. Notional-weighted, direction-aware average correlation
        other_tickers = [t for t in ticker_list if t != signal_ticker and t in close_series]
        if not other_tickers:
            return 1.0

        # Map signal direction to canonical side
        signal_is_long = signal_direction.upper() == "BUY"

        total_notional = 0.0
        weighted_sum = 0.0

        for pos in current_positions:
            sym = pos.get("symbol", "")
            if sym not in other_tickers:
                continue
            notional = pos.get("market_value", 0.0)
            if notional <= 0:
                continue

            pos_side = pos.get("side", "").lower()
            same_direction = (signal_is_long and pos_side == "long") or (
                not signal_is_long and pos_side == "short"
            )
            direction_sign = 1.0 if same_direction else -1.0

            corr_val = corr_matrix.loc[signal_ticker, sym]
            weighted_sum += corr_val * direction_sign * notional
            total_notional += notional

        if total_notional <= 0:
            return 1.0

        avg_weighted_corr = weighted_sum / total_notional

        # 6. Scale down only for positive net weighted correlation
        multiplier = 1.0 - max(0.0, avg_weighted_corr * 0.8)
        return max(0.2, multiplier)

    # ── VIX & Market Stress Gating ─────────────────────────────────────

    def get_vix_level(self) -> float:
        """Fetch or mock current VIX levels.
        Ready to plug into a live data feed or broker API.
        """
        try:
            return float(os.getenv("MOCK_VIX_LEVEL", "20.0"))
        except ValueError:
            return 20.0

    def check_trade_viability(self, signal: StrategySignal) -> tuple[bool, str]:
        """Perform additional macro/market-stress viability checks.
        
        Specifically, rejects mean reversion trades if VIX is above MAX_VIX_THRESHOLD,
        as high VIX environments represent extreme market stress where mean reversion
        strategies tend to suffer from tail correlation blowups.
        """
        if signal.strategy_name == "mean_reversion":
            vix = self.get_vix_level()
            max_vix = getattr(_cfg, 'MAX_VIX_THRESHOLD', 30.0)
            if vix > max_vix:
                return (
                    False,
                    f"VIX Circuit Breaker: Current VIX {vix:.2f} exceeds "
                    f"MAX_VIX_THRESHOLD {max_vix:.2f} for mean reversion"
                )
        return (True, "OK")

    # ── Signal Validation ──────────────────────────────────────────────

    def validate_signal(
        self, signal: StrategySignal, signal_regime: str | None = None,
        df: pd.DataFrame = None, quote: dict = None, notional: float = 0.0,
        ev_r: float = 0.0, sample_size: int = 0,
    ) -> tuple[bool, str]:
        """Validate the signal's price, stop, R/R, direction consistency,
        ATR-based spread/liquidity cap, and strategy expectancy.

        ``signal_regime`` is an optional regime label used for granular
        expectancy filtering (e.g. ``"bullish_calm"``).

        Returns ``(True, "OK")`` or ``(False, reason)``.
        """
        # 0. Viability check (VIX circuit breaker)
        viable, viability_reason = self.check_trade_viability(signal)
        if not viable:
            return (False, viability_reason)

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
        
        # Strategy-specific R/R minimums as the ultimate safety net
        if signal.strategy_name == "mean_reversion":
            # Strict fixed R/R for mean reversion
            min_rr = getattr(_cfg, 'MR_MIN_RR', 1.2)
        elif signal.strategy_name == "trend_pullback":
            # Higher structural R/R required for pullback setups
            min_rr = getattr(_cfg, 'TP_MIN_RR', 1.5)
        elif signal.strategy_name == "momentum_breakout":
            # Momentum breakout expects uncapped runner expectancy; ensure initial partial covers risk
            min_rr = getattr(_cfg, 'MB_PARTIAL_TP_RISK_MULT', 1.0)
        elif signal.strategy_name == "trend_following":
            # Trend following relies on trailing stops; do not penalize fixed target too aggressively
            min_rr = getattr(_cfg, 'TF_MIN_RR', 0.8)
        else:
            min_rr = MIN_RR_RATIO
            
        # Add a tiny 0.01 tolerance to account for floating point inaccuracies (e.g. 1.499999 < 1.5)
        if (reward / risk) < (min_rr - 0.01):
            return (False, f"R/R {(reward/risk):.2f} below {min_rr} for {signal.strategy_name}")

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

        # 6. Execution liquidity and spread checks
        spread_ok, spread_reason = self._check_execution_metrics(signal, df=df, quote=quote, notional=notional)
        if not spread_ok:
            return (False, spread_reason)

        # 7. Strategy expectancy gate — reject trades with negative expected value
        #    ev_r / sample_size are already fetched once in approve().
        #    Hard-disable only when we have enough samples and the expectancy
        #    ratio is below the minimum threshold.
        if sample_size >= _cfg.EXPECTANCY_HARD_DISABLE_SAMPLES and ev_r < _cfg.MIN_EXPECTANCY_R:
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
        df: pd.DataFrame = None,
        quote: dict = None,
        ml_multiplier: float = 1.0,
    ) -> tuple[bool, float, str]:
        """Run the full risk pipeline: validate → correlation → size.

        ``signal_regime`` is forwarded to ``validate_signal`` for granular
        expectancy filtering.

        ``ml_multiplier`` is forwarded to ``calculate_position_size`` so that
        ML boost/reduce is baked into the notional before portfolio heat checks.

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

        # 0b. Two-tier expectancy lookup (used by sizing AND validation)
        alpaca_ticker = self._resolve_ticker(signal.ticker)
        ev_r, sample_size = get_strategy_expectancy(
            signal.strategy_name, signal.direction,
            symbol=alpaca_ticker, regime=signal_regime,
        )
        if sample_size < _cfg.MIN_EXPECTANCY_SAMPLES:
            ev_r, sample_size = get_strategy_expectancy(signal.strategy_name, signal.direction)

        # 1. Position sizing (ML multiplier baked in for portfolio heat)
        notional = self.calculate_position_size(
            signal, account_equity, signal_regime=signal_regime,
            ml_multiplier=ml_multiplier,
            ev_r=ev_r, sample_size=sample_size,
            current_positions=current_positions,
        )
        if notional <= 0:
            self.logger.info(
                "Signal rejected: %s %s — Position size is zero",
                signal.direction,
                signal.ticker,
            )
            return (False, 0.0, "Position size is zero")

        # 2. Validate signal with full liquidity awareness
        valid, reason = self.validate_signal(
            signal, signal_regime=signal_regime, df=df, quote=quote, notional=notional,
            ev_r=ev_r, sample_size=sample_size,
        )
        if not valid:
            self.logger.info("Signal rejected: %s %s — %s", signal.direction, signal.ticker, reason)
            return (False, 0.0, reason)

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


