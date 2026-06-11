"""MakeShiftTrades — Autonomous Multi-Asset Trading Bot Engine
==============================================================

Orchestrates periodic strategy scans across instruments, executes signals
through the RiskManager → AlpacaTrader pipeline, and logs all activity.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from datetime import datetime, timezone

import pandas as pd

import config
from charts.data import fetch_ohlcv
from src.strategies import StrategySignal
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum_breakout import MomentumBreakoutStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.trend_pullback import TrendPullbackStrategy
from src.risk_manager import RiskManager
from src.alpaca_trader import AlpacaTrader
from src.macro_filter import SEVERITY_FLATTEN_ALL, SEVERITY_NORMAL, SEVERITY_NO_NEW_ENTRIES

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main autonomous trading bot loop.

    Scans a fixed set of instruments on configurable intervals, runs the
    assigned strategy, routes approved signals through the risk manager,
    and dispatches orders via AlpacaTrader.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────

    def __init__(self) -> None:

        # Alpaca trader (soft-fail so the engine can still report status)
        try:
            self.trader = AlpacaTrader()
        except Exception as exc:
            logger.error("Failed to initialise AlpacaTrader: %s", exc)
            self.trader = None

        # Risk manager fed from config
        self.risk_manager = RiskManager(
            max_risk_pct=config.MAX_RISK_PCT,
            max_position_pct=config.MAX_POSITION_PCT,
            max_notional=config.MAX_NOTIONAL,
            max_positions=config.MAX_POSITIONS,
        )

        # AI Parameter Tuner
        from src.ai_tuner import AITuner
        self.ai_tuner = AITuner()

        # ML Veto Filter
        from src.ml_model import get_model
        self.ml_model = get_model()
        # Kick off background training
        self.ml_model.start_training_async()

        # Macro Kill Switch
        from src.macro_filter import MacroFilter
        self.macro_filter = MacroFilter()

        # Strategy instances
        self.mean_rev = MeanReversionStrategy()
        self.momentum = MomentumBreakoutStrategy()
        self.trend = TrendFollowingStrategy()
        self.pullback = TrendPullbackStrategy()

        # Scan manifest — (ticker, strategy, interval)
        # SPY/QQQ run through BOTH mean_rev and pullback strategies.
        self.instruments = [
            {"ticker": "SPY", "strategy": self.mean_rev, "interval_seconds": 900, "last_scan": 0},
            {"ticker": "SPY", "strategy": self.pullback, "interval_seconds": 900, "last_scan": 0},
            {"ticker": "QQQ", "strategy": self.mean_rev, "interval_seconds": 900, "last_scan": 0},
            {"ticker": "QQQ", "strategy": self.pullback, "interval_seconds": 900, "last_scan": 0},
            {"ticker": "BTC-USD", "strategy": self.momentum, "interval_seconds": 3600, "last_scan": 0},
            {"ticker": "GLD", "strategy": self.trend, "interval_seconds": 14400, "last_scan": 0},
            {"ticker": "USO", "strategy": self.trend, "interval_seconds": 14400, "last_scan": 0},
        ]

        # Save base intervals for adaptive scanning
        for inst in self.instruments:
            inst["_base_interval"] = inst["interval_seconds"]



        # Session counters and uptime tracking
        self.signals_today = 0
        self.orders_today = 0
        self.cycle_count = 0
        self.start_time = time.time()
        self.running = False

        # Position tracking for trailing/time stops
        self._position_state: dict[str, dict] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the infinite scanning loop.

        Logs startup info, performs an initial sync of closed trades, then
        enters the main scan cycle.  Catches and recovers from per-cycle
        errors so the loop never dies on a single instrument failure.
        """
        logger.info("=" * 60)
        logger.info("MakeShiftTrades — Multi-Asset Trading Bot")
        logger.info("Instruments: %s", [i["ticker"] for i in self.instruments])
        logger.info(
            "DRY_RUN: %s | Risk: %.1f%% | Max positions: %d",
            config.DRY_RUN,
            config.MAX_RISK_PCT * 100,
            config.MAX_POSITIONS,
        )
        logger.info("=" * 60)

        if self.trader is None:
            logger.error("AlpacaTrader not available. Cannot run bot.")
            return

        # Initial sync of any pending closed trades
        try:
            self.trader.sync_closed_trades()
        except Exception as exc:
            logger.warning("Initial trade sync failed: %s", exc)

        self.running = True
        while self.running:
            try:
                self._scan_cycle()
            except KeyboardInterrupt:
                self.running = False
                raise
            except Exception as exc:
                logger.error("Scan cycle error: %s", exc)

            # Sleep in small increments to respond quickly to stop signal
            sleep_remaining = config.SCAN_INTERVAL
            while sleep_remaining > 0 and self.running:
                sleep_chunk = min(1.0, sleep_remaining)
                time.sleep(sleep_chunk)
                sleep_remaining -= sleep_chunk

    def stop(self) -> None:
        """Stop the running scanning loop."""
        logger.info("Stopping MakeShiftTrades trading engine...")
        self.running = False

    # ──────────────────────────────────────────────────────────────────────
    # Scan cycle (one pass over all instruments)
    # ──────────────────────────────────────────────────────────────────────

    def _scan_cycle(self) -> None:
        """Check each instrument, fetch data, run strategy, and execute."""
        self.cycle_count += 1
        now = time.time()

        # Let the AI tune the parameters before running the strategies
        self.ai_tuner.tune_parameters()

        # Adaptive scan intervals based on market regime
        if config.ADAPTIVE_SCAN_ENABLED:
            for instrument in self.instruments:
                strategy_name = instrument["strategy"].name
                regime = None
                if strategy_name == "mean_reversion":
                    regime = self.ai_tuner.current_regimes.get("Equity")
                elif strategy_name == "momentum_breakout":
                    regime = self.ai_tuner.current_regimes.get("Crypto")
                elif strategy_name == "trend_following":
                    regime = self.ai_tuner.current_regimes.get("Commodity")

                if regime and "Volatile" in regime:
                    instrument["interval_seconds"] = max(60, int(instrument.get("_base_interval", instrument["interval_seconds"]) * config.ADAPTIVE_SCAN_FAST_MULT))
                elif regime and "Chop" in regime:
                    instrument["interval_seconds"] = max(60, int(instrument.get("_base_interval", instrument["interval_seconds"]) * config.ADAPTIVE_SCAN_SLOW_MULT))
                else:
                    instrument["interval_seconds"] = instrument.get("_base_interval", instrument["interval_seconds"])

        # Keep risk manager in sync with user config (sector multipliers are now handled inside the risk manager per-trade)
        self.risk_manager.max_risk_pct = config.MAX_RISK_PCT

        # Macro Kill Switch — consume severity levels from check_event()
        if config.MACRO_FILTER_ENABLED:
            severity, event_name = self.macro_filter.check_event(now)
            if severity == SEVERITY_FLATTEN_ALL:
                logger.info(
                    "Macro Kill Switch: %s active (severity=FLATTEN_ALL). Flattening all positions.",
                    event_name,
                )
                # Close all active positions
                try:
                    positions = self.trader.get_positions()
                    for pos in positions:
                        sym = pos["symbol"]
                        logger.info("  Closing position %s due to macro flatten", sym)
                        try:
                            self.trader._client.close_position(sym)
                        except Exception as exc:
                            logger.error("Failed to close %s: %s", sym, exc)
                except Exception as exc:
                    logger.warning("Could not list positions for flatten: %s", exc)
                # Clear position state tracking
                self._position_state.clear()
                return
            elif severity == SEVERITY_NO_NEW_ENTRIES:
                logger.info(
                    "Macro Kill Switch: %s active (severity=NO_NEW_ENTRIES). Skipping new entries.",
                    event_name,
                )
                return

        self._cleanup_stale_orders()
        self._manage_positions()

        for instrument in self.instruments:
            elapsed = now - instrument.get("last_scan", 0)
            if elapsed < instrument["interval_seconds"]:
                continue

            ticker = instrument["ticker"]
            strategy = instrument["strategy"]

            try:
                df = fetch_ohlcv(
                    ticker,
                    period=strategy.period,
                    interval=strategy.timeframe,
                )
                if df is None or len(df) < 20:
                    logger.warning(
                        "Insufficient data for %s (%s)", ticker, strategy.timeframe
                    )
                    instrument["last_scan"] = now
                    continue

                signal = strategy.analyze(df, ticker)
                instrument["last_scan"] = now

                if signal is None:
                    logger.debug("No signal for %s", ticker)
                    continue

                self.signals_today += 1
                logger.info(
                    "[SIGNAL] %s %s | Conf: %.1f | %s",
                    signal.direction,
                    signal.ticker,
                    signal.confidence,
                    signal.reason,
                )

                self._execute_signal(signal, df)

            except Exception as exc:
                logger.error("Error scanning %s: %s", ticker, exc)
                instrument["last_scan"] = now

        # Periodic housekeeping
        if self.cycle_count % 10 == 0:
            try:
                if self.trader is not None:
                    self.trader.sync_closed_trades()
                    # Drain ML feedback from closed trades
                    if self.ml_model is not None:
                        self.trader.drain_ml_feedback_queue()
            except Exception as exc:
                logger.warning("Trade sync / ML feedback drain failed: %s", exc)

        if self.cycle_count % 5 == 0:
            self._log_status()

    # ──────────────────────────────────────────────────────────────────────
    # Execute a signal through the risk + trading pipeline
    # ──────────────────────────────────────────────────────────────────────

    def _execute_signal(self, signal: StrategySignal, df: pd.DataFrame = None) -> None:
        """Run approval pipeline and — if approved — place the order.

        In dry-run mode everything is logged but no order is submitted.
        """
        if self.trader is None:
            logger.error("AlpacaTrader is None, cannot execute signal")
            return

        account = self.trader.get_account()
        if not account.get("ok"):
            logger.error(
                "Cannot get account info: %s", account.get("error")
            )
            return

        # Allow overriding Alpaca's default $100k for realistic simulation
        equity = float(account.get("equity", 0))
        if config.VIRTUAL_EQUITY > 0:
            equity = config.VIRTUAL_EQUITY

        if equity <= 0:
            logger.error("Account equity is zero or negative")
            return

        # Fetch current live state
        positions = self.trader.get_positions()
        active_orders_res = self.trader.get_active_orders()
        active_orders = active_orders_res.get("orders", []) if active_orders_res.get("ok") else []

        # Market Hours Guard: Block non-crypto orders when market is closed to prevent 
        # overnight gap risk on queued market orders.
        is_crypto = signal.ticker in {"BTC-USD", "ETH-USD", "BTCUSD", "ETHUSD"}
        if not is_crypto:
            try:
                # NOTE: accessing private _client — no public get_clock() on AlpacaTrader
                clock = self.trader._client.get_clock()
                if not clock.is_open:
                    logger.info("[REJECTED] %s %s: Market is closed (guarding against overnight gap risk)", signal.direction, signal.ticker)
                    return
            except Exception as exc:
                logger.warning("Could not check market hours: %s", exc)

        # ── Signal Upgrade Intercept ───────────────────────────────────────
        # If we already have a position or pending entry orders for this ticker,
        # only proceed if the new signal's confidence is significantly higher.
        alpaca_ticker = self.trader.map_ticker(signal.ticker)
        state = self._position_state.setdefault(alpaca_ticker, {})
        existing_conf = state.get("confidence", 0.0)

        existing_pos = next((p for p in positions if p['symbol'] == alpaca_ticker), None)
        active_entry_orders = [o for o in active_orders if o.get('symbol') == alpaca_ticker and o.get('order_class') == 'bracket' and o.get('status') in ['new', 'accepted', 'pending_new']]

        for o in active_entry_orders:
            j_entry = self.trader._state.get("order_journal", {}).get(o.get('id', ''), {})
            existing_conf = max(existing_conf, j_entry.get("confidence", 0.0))

        if existing_pos or active_entry_orders:
            # Check price buffers for the upgrade
            existing_tp = state.get("take_profit", signal.take_profit)
            existing_sl = state.get("stop_loss", signal.stop_loss)

            tp_change_abs = abs(signal.take_profit - existing_tp)
            sl_change_abs = abs(signal.stop_loss - existing_sl)
            
            # Use ATR-based dynamic buffer (e.g. 25% of current ATR)
            atr_buffer = getattr(config, 'UPGRADE_BUFFER_ATR_FRACTION', 0.25) * signal.atr

            # Only upgrade if confidence is higher AND (TP or SL moved meaningfully compared to volatility)
            price_moved_enough = (tp_change_abs > atr_buffer) or (sl_change_abs > atr_buffer)

            if signal.confidence >= existing_conf + 2.0 and price_moved_enough:
                logger.info("[SIGNAL UPGRADE] %s: Conf %.1f > Old %.1f | TP moved: $%.2f, SL moved: $%.2f (Buffer: $%.2f)", 
                            signal.ticker, signal.confidence, existing_conf, tp_change_abs, sl_change_abs, atr_buffer)
                if active_entry_orders:
                    for o in active_entry_orders:
                        self.trader.cancel_order(str(o.get('id', '')))
                    time.sleep(1)
                    active_orders_res = self.trader.get_active_orders()
                    active_orders = active_orders_res.get("orders", []) if active_orders_res.get("ok") else []
                elif existing_pos:
                    self.trader.update_position_exits(alpaca_ticker, signal.take_profit, signal.stop_loss)
                    state['confidence'] = signal.confidence
                    state['take_profit'] = signal.take_profit
                    state['stop_loss'] = signal.stop_loss
                    return

        approved, notional, reason = self.risk_manager.approve(
            signal, equity, positions, active_orders
        )

        # --- SIGNAL TELEMETRY DUMP ---
        import json
        with open("overnight_signals.jsonl", "a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": signal.ticker,
                "direction": signal.direction,
                "strategy": signal.strategy_name,
                "confidence": signal.confidence,
                "approved": approved,
                "reason": reason
            }) + "\n")
        # -----------------------------

        if not approved:
            logger.info(
                "[REJECTED] %s %s: %s", signal.direction, signal.ticker, reason
            )
            return

        # ML Veto Filter
        if config.ML_VETO_ENABLED and self.ml_model is not None:
            try:
                ticker_for_ml = signal.ticker
                if df is not None and len(df) >= 40:
                    df_for_ml = df
                else:
                    df_for_ml = fetch_ohlcv(ticker_for_ml, period="5d", interval="15m")
                if df_for_ml is not None and len(df_for_ml) >= 40:
                    ml_result = self.ml_model.evaluate_signal(df_for_ml, signal.direction)
                    if ml_result.get("veto", False):
                        logger.info(
                            "[ML VETO] %s %s vetoed by ML (prob_up=%.3f, confidence=%.1f)",
                            signal.direction, signal.ticker,
                            ml_result.get("prob_up", 0.5),
                            ml_result.get("confidence", 0.0),
                        )
                        return
                    else:
                        logger.info(
                            "[ML OK] %s %s passed ML filter (prob_up=%.3f, confidence=%.1f)",
                            signal.direction, signal.ticker,
                            ml_result.get("prob_up", 0.5),
                            ml_result.get("confidence", 0.0),
                        )
            except Exception as exc:
                logger.warning("ML veto check failed (allowing trade): %s", exc)

        if config.DRY_RUN:
            logger.info(
                "[DRY RUN] Would %s %s $%.2f notional | SL:%.2f TP:%.2f | %s",
                signal.direction,
                signal.ticker,
                notional,
                signal.stop_loss,
                signal.take_profit,
                signal.reason,
            )
            self.orders_today += 1
            return

        result = self.trader.place_bot_order(
            ticker=signal.ticker,
            side=signal.direction,
            notional=notional,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            current_positions=positions,
            order_type=getattr(signal, 'order_type', 'MARKET'),
            metadata={
                "strategy": signal.strategy_name,
                "timeframe": signal.timeframe,
                "confidence": signal.confidence,
                "reason": signal.reason,
                "atr": signal.atr,
            },
        )

        if result.get("ok"):
            self.orders_today += 1
            self._position_state.setdefault(alpaca_ticker, {})['confidence'] = signal.confidence
            self._position_state[alpaca_ticker]['take_profit'] = signal.take_profit
            self._position_state[alpaca_ticker]['stop_loss'] = signal.stop_loss
            logger.info(
                "[ORDER PLACED] %s %s $%.2f | ID: %s",
                signal.direction,
                signal.ticker,
                notional,
                result.get("order_id"),
            )
        else:
            logger.error(
                "[ORDER FAILED] %s %s: %s",
                signal.direction,
                signal.ticker,
                result.get("error"),
            )

    # ──────────────────────────────────────────────────────────────────────
    # Position management (trailing stop / time stop)
    # ──────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────
    # TTL sweeper — cancel stale entry orders
    # ──────────────────────────────────────────────────────────────────────

    def _cleanup_stale_orders(self) -> None:
        if not self.trader: return
        res = self.trader.get_active_orders()
        if not res.get("ok"): return
        now = datetime.now(timezone.utc)
        for o in res.get("orders", []):
            if o.get("parent_id") is None:
                sub_str = o.get("submitted_at")
                if sub_str:
                    try:
                        # Handle potential 'Z' or offset formats
                        if sub_str.endswith('Z'):
                            sub_str = sub_str[:-1] + '+00:00'
                        sub = datetime.fromisoformat(sub_str)
                        if sub.tzinfo is None:
                            sub = sub.replace(tzinfo=timezone.utc)
                        age_h = (now - sub).total_seconds() / 3600.0
                        if age_h > config.ORDER_TTL_HOURS:
                            logger.info("[TTL SWEEP] Canceling stale entry order %s (Age: %.1fh)", o.get('symbol', 'unknown'), age_h)
                            self.trader.cancel_order(str(o.get('id', '')))
                    except Exception as e:
                        logger.error("Failed to parse submitted_at %s: %s", sub_str, e)

    def _manage_positions(self) -> None:
        """Monitor open positions and close based on trailing/stop loss or time stop."""
        if self.trader is None:
            return

        positions = self.trader.get_positions()
        current_syms = {pos['symbol'] for pos in positions}

        # Clean up symbols no longer in current positions
        for sym in list(self._position_state.keys()):
            if sym not in current_syms:
                del self._position_state[sym]

        for pos in positions:
            sym = pos['symbol']
            current_price = pos['current_price']
            unrealized_pl = pos['unrealized_pl']
            market_value = pos['market_value']
            side = pos['side']

            # Initialise tracking entry if new position or if missing tracking fields
            if sym not in self._position_state or 'open_ts' not in self._position_state[sym]:
                self._position_state.setdefault(sym, {}).update({
                    "open_ts": time.time(),
                    "highest": current_price,
                    "lowest": current_price,
                })

            state = self._position_state[sym]

            # Update highest/lowest
            if current_price > state['highest']:
                state['highest'] = current_price
            if current_price < state['lowest']:
                state['lowest'] = current_price

            # ── Trailing Stop ──────────────────────────────────────────
            trail_dist = current_price * 0.01 * config.TRAILING_STOP_PCT

            should_close = False
            if side == 'long' and current_price < state['highest'] - trail_dist:
                should_close = True
            elif side == 'short' and current_price > state['lowest'] + trail_dist:
                should_close = True

            if should_close:
                logger.info("Closing %s due to trailing stop", sym)
                try:
                    self.trader._client.close_position(sym)
                except Exception as e:
                    logger.error("Failed to close position %s: %s", sym, e)
                del self._position_state[sym]
                continue

            # ── Time Stop (per-asset-class) ────────────────────────────
            # Default to equity hours; override for crypto and commodity
            if sym in ("BTCUSD", "ETHUSD", "BCHUSD", "LTCUSD", "UNIUSD", "LINKUSD"):
                time_stop_hours = config.TIME_STOP_CRYPTO_HOURS
            elif sym in ("GLD", "USO"):
                time_stop_hours = config.TIME_STOP_COMMODITY_HOURS
            else:
                time_stop_hours = config.TIME_STOP_EQUITY_HOURS

            hours_open = (time.time() - state['open_ts']) / 3600.0
            if hours_open > time_stop_hours and (unrealized_pl / (market_value + 1e-9)) < 0.005:
                logger.info(
                    "Closing %s due to time stop (%.1f hours)",
                    sym, time_stop_hours,
                )
                try:
                    self.trader._client.close_position(sym)
                except Exception as e:
                    logger.error("Failed to close position %s: %s", sym, e)
                del self._position_state[sym]

    # ──────────────────────────────────────────────────────────────────────
    # Status logging
    # ──────────────────────────────────────────────────────────────────────

    def _log_status(self) -> None:
        """Log a summary of bot health, equity, and open positions."""
        if self.trader is None:
            logger.warning("AlpacaTrader is None, cannot log status")
            return

        try:
            account = self.trader.get_account()
            positions = self.trader.get_positions()
            uptime_min = (time.time() - self.start_time) / 60
            equity = float(account.get("equity", 0))

            logger.info(
                "[STATUS] Uptime: %.0fm | Cycles: %d | Signals: %d | "
                "Orders: %d | Equity: $%.2f | Open positions: %d",
                uptime_min,
                self.cycle_count,
                self.signals_today,
                self.orders_today,
                equity,
                len(positions),
            )

            # --- OVERNIGHT TELEMETRY DUMP ---
            import json
            telemetry_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "equity": equity,
                "cycles": self.cycle_count,
                "signals_today": self.signals_today,
                "orders_today": self.orders_today,
                "regimes": self.ai_tuner.current_regimes,
                "positions": [{"symbol": p['symbol'], "side": p['side'], "unrealized_pl": p['unrealized_pl']} for p in positions]
            }
            with open("overnight_equity_tracker.jsonl", "a") as f:
                f.write(json.dumps(telemetry_data) + "\n")
            # --------------------------------

            for pos in positions:
                logger.info(
                    "  Position: %s %s qty=%.2f P&L=$%.2f",
                    pos.get("side", "?").upper(),
                    pos.get("symbol", "?"),
                    pos.get("qty", 0),
                    pos.get("unrealized_pl", 0),
                )

            logger.info(
                "  Regimes: %s", self.ai_tuner.current_regimes
            )
            logger.info(
                "  ML Model: trained=%s, training=%s",
                getattr(self.ml_model, '_trained', False) if self.ml_model else False,
                getattr(self.ml_model, '_training', False) if self.ml_model else False,
            )
        except Exception as exc:
            logger.warning("Status log failed: %s", exc)
