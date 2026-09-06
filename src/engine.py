"""MakeShiftTrades — Autonomous Multi-Asset Trading Bot Engine
==============================================================

Orchestrates periodic strategy scans across instruments, executes signals
through the RiskManager → AlpacaTrader pipeline, and logs all activity.
"""

from __future__ import annotations

import logging
import time
import json
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
from src.macro_filter import MacroFilter
from src.database import init_db, log_trade, get_strategy_expectancy
from src.regime_classifier import RegimeClassifier
try:
    from src.trailing_stops import calculate_trailing_stop, get_strategy_trailing_logic
except ImportError:
    from trailing_stops import calculate_trailing_stop, get_strategy_trailing_logic

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

        init_db()

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
        self.macro_filter = MacroFilter()

        # Warn if macro filter is enabled but the calendar has no future events
        if config.MACRO_FILTER_ENABLED:
            all_events = MacroFilter.load_events()
            now_utc = datetime.now(timezone.utc)
            future_events = [e for e in all_events if e["event_time_utc"] > now_utc]
            if not future_events:
                logger.warning("=" * 60)
                logger.warning("MACRO FILTER WARNING: No future macro events found in calendar!")
                logger.warning("The kill switch will NOT block any entries until events are added.")
                logger.warning("Check macro_calendar.json or add upcoming event entries.")
                logger.warning("=" * 60)

        # HMM Regime Classifier (global market state detection)
        self.regime_classifier = RegimeClassifier(n_states=3)
        self.current_hmm_regime: str = "unknown"
        self._last_hmm_update: float = 0.0

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
            {"ticker": "PDBC", "strategy": self.trend, "interval_seconds": 14400, "last_scan": 0},
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
        # Cycle level cached DataFrames to avoid redundant fetches
        self._cycle_dfs: dict[str, pd.DataFrame] = {}

        # Initial HMM regime update at startup
        try:
            self._update_hmm_regime()
        except Exception as exc:
            logger.warning("Initial HMM regime update failed: %s", exc)

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
        self._cycle_dfs = {}
        self.cycle_count += 1
        now = time.time()

        # Refresh the HMM regime (rate-limited to once per day internally)
        self._update_hmm_regime()

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
                    ticker = instrument["ticker"]
                    if ticker == "GLD":
                        regime = self.ai_tuner.current_regimes.get("Gold")
                    elif ticker == "PDBC":
                        regime = self.ai_tuner.current_regimes.get("Broad Commodity")

                if regime and "Volatile" in regime:
                    instrument["interval_seconds"] = max(60, int(instrument.get("_base_interval", instrument["interval_seconds"]) * config.ADAPTIVE_SCAN_FAST_MULT))
                elif regime and "Chop" in regime:
                    instrument["interval_seconds"] = max(60, int(instrument.get("_base_interval", instrument["interval_seconds"]) * config.ADAPTIVE_SCAN_SLOW_MULT))
                else:
                    instrument["interval_seconds"] = instrument.get("_base_interval", instrument["interval_seconds"])

        # Keep risk manager in sync with user config (sector multipliers are now handled inside the risk manager per-trade)
        self.risk_manager.max_risk_pct = config.MAX_RISK_PCT

        # Macro Kill Switch — consume events from check_event()
        blocked_entry_tickers = set()
        if config.MACRO_FILTER_ENABLED:
            active_events = self.macro_filter.check_event(now)
            for event in active_events:
                event_name = event["event_name"]
                affected = event.get("affected_assets", [])
                actions = event.get("actions", [])
                
                # Check for flattening action
                should_flatten_intraday = "flatten_intraday_only" in actions
                
                # Close/flatten positions if required
                if should_flatten_intraday and self.trader is not None:
                    try:
                        positions = self.trader.get_positions()
                        for pos in positions:
                            sym = pos["symbol"]
                            normal_sym = sym.replace("-USD", "")
                            
                            # Check if the asset is affected by this event
                            is_affected = False
                            if "all" in affected:
                                is_affected = True
                            else:
                                normalized_affected = [a.replace("-USD", "") for a in affected]
                                if normal_sym in normalized_affected:
                                    is_affected = True
                                    
                            if is_affected:
                                # Retrieve position state to check if it's intraday
                                state = self._position_state.get(sym, {})
                                strat = state.get("strategy")
                                is_intra = False
                                if strat in ("mean_reversion", "trend_pullback", "momentum_breakout"):
                                    is_intra = True
                                elif normal_sym in ("SPY", "QQQ", "BTC"):
                                    is_intra = True
                                    
                                if is_intra:
                                    logger.info(
                                        "Macro Filter: Closing position %s due to %s event flatten action",
                                        sym,
                                        event_name,
                                    )
                                    try:
                                        self._close_and_log_position(
                                            sym,
                                            state,
                                            pos["current_price"],
                                            pos["unrealized_pl"],
                                            f"macro_{event_name.lower()}",
                                        )
                                        if sym in self._position_state:
                                            del self._position_state[sym]
                                    except Exception as exc:
                                        logger.error("Failed to close %s: %s", sym, exc)
                    except Exception as exc:
                        logger.warning("Could not process flatten for active event %s: %s", event_name, exc)
                
                # Check for "no new entries" action
                if "no_new_entries" in actions:
                    if "all" in affected:
                        blocked_entry_tickers.add("all")
                    else:
                        for asset in affected:
                            blocked_entry_tickers.add(asset)

        self._cleanup_stale_orders()
        self._manage_positions()

        cycle_signals = []

        for instrument in self.instruments:
            ticker = instrument["ticker"]
            if "all" in blocked_entry_tickers or ticker in blocked_entry_tickers or ticker.replace("-USD", "") in blocked_entry_tickers:
                logger.info("Macro Filter: skipping scan/entry for %s due to active macro event", ticker)
                continue

            elapsed = now - instrument.get("last_scan", 0)
            if elapsed < instrument["interval_seconds"]:
                continue
            strategy = instrument["strategy"]

            try:
                df = fetch_ohlcv(
                    ticker,
                    period=strategy.period,
                    interval=strategy.timeframe,
                )
                if df is not None:
                    self._cycle_dfs[ticker] = df
                    if "-USD" in ticker:
                        self._cycle_dfs[ticker.replace("-USD", "")] = df
                    else:
                        self._cycle_dfs[ticker + "-USD"] = df
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
                    
                # Strategy Routing based on Regime Profile
                if strategy.name == "trend_pullback":
                    routing = getattr(config, "ROUTING_TREND_PULLBACK", "ENABLED")
                    if routing == "DISABLED":
                        logger.debug("Signal rejected: trend_pullback disabled in current regime.")
                        if self.trader: self.trader.record_virtual_trade(signal, "Routing: trend_pullback disabled", 1.0, self._get_signal_regime(ticker))
                        continue
                    elif routing == "LONG_ONLY" and signal.direction != "BUY":
                        logger.debug("Signal rejected: trend_pullback restricted to long-only.")
                        if self.trader: self.trader.record_virtual_trade(signal, "Routing: long-only restriction", 1.0, self._get_signal_regime(ticker))
                        continue
                    elif routing == "SHORT_ONLY" and signal.direction != "SELL":
                        logger.debug("Signal rejected: trend_pullback restricted to short-only.")
                        if self.trader: self.trader.record_virtual_trade(signal, "Routing: short-only restriction", 1.0, self._get_signal_regime(ticker))
                        continue
                    elif routing == "REDUCED":
                        # Simulate reducing risk or confidence
                        signal.confidence *= 0.5
                
                elif strategy.name == "mean_reversion":
                    routing = getattr(config, "ROUTING_MEAN_REVERSION", "BOTH")
                    if routing == "DISABLED":
                        logger.debug("Signal rejected: mean_reversion disabled in current regime.")
                        if self.trader: self.trader.record_virtual_trade(signal, "Routing: mean_reversion disabled", 1.0, self._get_signal_regime(ticker))
                        continue
                    elif routing == "LONG_ONLY" and signal.direction != "BUY":
                        logger.debug("Signal rejected: mean_reversion restricted to long-only.")
                        if self.trader: self.trader.record_virtual_trade(signal, "Routing: long-only restriction", 1.0, self._get_signal_regime(ticker))
                        continue
                    elif routing == "SHORT_ONLY" and signal.direction != "SELL":
                        logger.debug("Signal rejected: mean_reversion restricted to short-only.")
                        if self.trader: self.trader.record_virtual_trade(signal, "Routing: short-only restriction", 1.0, self._get_signal_regime(ticker))
                        continue
                    elif routing == "LONG_ONLY_HIGH_QUALITY":
                        if signal.direction != "BUY":
                            logger.debug("Signal rejected: mean_reversion restricted to long-only.")
                            if self.trader: self.trader.record_virtual_trade(signal, "Routing: long-only restriction", 1.0, self._get_signal_regime(ticker))
                            continue
                        if signal.confidence < 70.0:
                            logger.debug("Signal rejected: mean_reversion requires high quality in this regime.")
                            if self.trader: self.trader.record_virtual_trade(signal, "Routing: high-quality requirement missed", 1.0, self._get_signal_regime(ticker))
                            continue
                    elif routing == "TINY_FAILED_EXTENSIONS_ONLY":
                        # Significantly reduce confidence/risk for this trade
                        signal.confidence *= 0.3
                    elif routing == "REDUCED":
                        signal.confidence *= 0.5

                # ── HMM Regime Gatekeeper ────────────────────────────────────
                # Crisis override: block trend-following & momentum-breakout
                if self.current_hmm_regime == "Bearish Volatile" and strategy.name in (
                    "trend_following",
                    "momentum_breakout",
                ):
                    hmm_reason = (
                        f"HMM Regime blocked: {self.current_hmm_regime} "
                        f"gate for {strategy.name}"
                    )
                    logger.info(
                        "[HMM GATE] %s %s rejected — %s",
                        signal.direction, signal.ticker, hmm_reason,
                    )
                    if self.trader:
                        self.trader.record_virtual_trade(
                            signal, hmm_reason, 1.0, self._get_signal_regime(ticker),
                        )
                    continue

                # Bullish Calm: tighten mean-reversion criteria
                if (
                    self.current_hmm_regime == "Bullish Calm"
                    and strategy.name == "mean_reversion"
                ):
                    if signal.confidence < 75.0 or signal.direction not in ("BUY",):
                        hmm_reason = (
                            f"HMM Regime blocked: {self.current_hmm_regime} "
                            f"tightens mean reversion (conf={signal.confidence:.1f})"
                        )
                        logger.info(
                            "[HMM GATE] %s %s rejected — %s",
                            signal.direction, signal.ticker, hmm_reason,
                        )
                        if self.trader:
                            self.trader.record_virtual_trade(
                                signal, hmm_reason, 1.0,
                                self._get_signal_regime(ticker),
                            )
                        continue
                    # Even allowed signals get a confidence reduction in Bullish Calm
                    signal.confidence *= 0.85
                # ── End HMM Gatekeeper ───────────────────────────────────────

                self.signals_today += 1
                logger.info(
                    "[SIGNAL] %s %s | Conf: %.1f | %s",
                    signal.direction,
                    signal.ticker,
                    signal.confidence,
                    signal.reason,
                )

                cycle_signals.append((signal, df))

            except Exception as exc:
                logger.error("Error scanning %s: %s", ticker, exc)
                instrument["last_scan"] = now

        # Rank and filter signals before execution
        # Groups by capital bucket (equity, crypto, commodity, general),
        # scores each on expected EV, regime fit, liquidity, ML agreement,
        # and picks the highest-ranked per bucket.
        if cycle_signals:
            ranked_signals = self._rank_and_filter_signals(cycle_signals)
            for signal, df in ranked_signals:
                try:
                    self._execute_signal(signal, df)
                except Exception as exc:
                    logger.error("Error executing ranked signal %s: %s", signal.ticker, exc)

        # Periodic housekeeping
        if self.cycle_count % 10 == 0:
            try:
                if self.trader is not None:
                    self.trader.sync_closed_trades()
                    # Drain ML feedback from closed trades and feed into veto tracker
                    if self.ml_model is not None:
                        feedback_records = self.trader.drain_ml_feedback_queue()
                        for fb in feedback_records:
                            # Log every feedback record to virtual_trades_outcomes.jsonl
                            try:
                                with open("virtual_trades_outcomes.jsonl", "a") as f:
                                    f.write(json.dumps(fb) + "\n")
                            except Exception as je:
                                logger.warning("Failed to write virtual trade outcome: %s", je)

                            ticker = fb.get("ticker", "")
                            fb_strategy = (fb.get("metadata") or {}).get("strategy") \
                                          or fb.get("signal_reason", "unknown")
                            regime = self._get_signal_regime(ticker)
                            reason = fb.get("result_reason", "unknown")
                            reject_reason = (fb.get("metadata") or {}).get("reject_reason", "")
                            if reject_reason.startswith("Strong ML disagreement"):
                                try:
                                    self.ml_model._track_veto_outcome(
                                        strategy=fb_strategy,
                                        regime=regime,
                                        ticker=ticker,
                                        sizing_multiplier=1.0,  # not available in feedback
                                        outcome=reason,
                                    )
                                except Exception as tx:
                                    logger.warning("Failed to track veto outcome: %s", tx)
            except Exception as exc:
                logger.warning("Trade sync / ML feedback drain failed: %s", exc)

        if self.cycle_count % 5 == 0:
            self._log_status()

    # ──────────────────────────────────────────────────────────────────────
    # HMM Regime Management
    # ──────────────────────────────────────────────────────────────────────

    def _update_hmm_regime(self) -> None:
        """Fetch recent SPY data, update the HMM classifier, and store the
        global market regime in ``self.current_hmm_regime``.

        Downloads ~1 year of daily SPY data via ``yfinance``, fits (or
        refits) the ``RegimeClassifier``, and records the latent state.

        Called at startup and at most once per day thereafter.
        """
        import yfinance as yf

        now = time.time()
        # Rate-limit: only re-fit / predict once per 24 hours
        if self._last_hmm_update > 0 and (now - self._last_hmm_update) < 86400:
            logger.debug("HMM regime update skipped — last update was <24h ago")
            return

        try:
            spy = yf.download(
                "SPY", period="1y", interval="1d", progress=False, auto_adjust=True
            )
            if spy is None or spy.empty:
                logger.warning("[HMM] Could not fetch SPY data for regime update")
                return

            # Fit on the latest data, then predict the current regime
            self.regime_classifier.fit(spy)
            label, state_id = self.regime_classifier.predict(spy)
            self.current_hmm_regime = label
            self._last_hmm_update = now
            logger.info(
                "[HMM] Regime updated: %s (state %d)", label, state_id
            )
        except Exception as exc:
            logger.warning("[HMM] Regime update failed: %s", exc)

    def _get_signal_regime(self, ticker: str) -> str:
        """Map a ticker to its current market regime.

        Incorporates the HMM global regime as an override:
          - If HMM detects ``"Bearish Volatile"`` it is returned immediately
            (crisis state takes precedence over per-sector AI regimes).
          - Otherwise delegates to the AI Tuner's sector regimes.
        """
        # HMM crisis override — Bearish Volatile trumps everything
        if self.current_hmm_regime == "Bearish Volatile":
            return self.current_hmm_regime

        # Regular regime mapping from AI Tuner
        if ticker in ("SPY", "QQQ"):
            return self.ai_tuner.current_regimes.get("Equity", "unknown")
        elif ticker in ("BTC-USD", "BTCUSD"):
            return self.ai_tuner.current_regimes.get("Crypto", "unknown")
        elif ticker == "GLD":
            return self.ai_tuner.current_regimes.get("Gold", "unknown")
        elif ticker == "PDBC":
            return self.ai_tuner.current_regimes.get("Broad Commodity", "unknown")
        return "unknown"


    @staticmethod
    def _get_default_time_stop_bars(strategy_name: str, ticker: str = "") -> int:
        """Return strategy-specific time_stop_bars based on codex.md item 13.

        Mean reversion  :  4–8 bars on 15m  → use 6
        Trend pullback  :  8–12 bars on 15m → use 10
        BTC breakout    :  6–12 bars on 1h  → use 8
        GLD trend       : 10–20 bars on 4h  → use 15
        PDBC trend      :  6–12 bars on 4h  → use 8
        """
        mapping: dict[str, int] = {
            "mean_reversion": 6,
            "trend_pullback": 10,
            "momentum_breakout": 8,
        }
        # Sub-differentiate trend_following by ticker
        if strategy_name == "trend_following":
            if ticker == "GLD":
                return 15
            elif ticker == "PDBC":
                return 8
            return 12  # conservative mid-range fallback
        return mapping.get(strategy_name, 10)


    # ──────────────────────────────────────────────────────────────────────
    # Signal Ranking Layer — score, bucket, select best per allocation pool
    # ──────────────────────────────────────────────────────────────────────

    def _rank_and_filter_signals(
        self,
        signals_with_data: list[tuple[StrategySignal, Optional[pd.DataFrame]]],
    ) -> list[tuple[StrategySignal, Optional[pd.DataFrame]]]:
        """Rank signals by composite score per risk bucket and return up to one per bucket.

        Risk buckets:
          "equity_beta"      — SPY, QQQ
          "crypto"           — BTC-USD, BTC
          "gold"             — GLD
          "broad_commodity"  — PDBC
          "general"          — anything else

        Each signal receives a composite score based on:
          · Expected EV    (historical R-multiple from the trade log)
          · Regime fit     (how well the strategy aligns with the current market regime)
          · Liquidity      (inverse of ATR/price — lower volatility = more liquid)
          · ML agreement   (sizing_multiplier produced by the ML model)
          · Slippage       (spread estimate penalty)

        The highest-scoring signal from each bucket is returned, allowing
        simultaneous execution of non-competing assets (e.g. SPY and GLD).
        """
        # 1. Ticker-to-bucket mapping — keys use the resolved canonical form
        BUCKET_MAP: dict[str, str] = {
            "SPY": "equity_beta",
            "QQQ": "equity_beta",
            "BTCUSD": "crypto",
            "GLD": "gold",
            "PDBC": "broad_commodity",
        }

        # 2. Initialise bucket storage
        scored_signals: dict[str, list[tuple[float, StrategySignal, Optional[pd.DataFrame]]]] = {
            "equity_beta": [],
            "crypto": [],
            "gold": [],
            "broad_commodity": [],
            "general": [],
        }

        for signal, df in signals_with_data:
            # ---- Constants for scoring ----
            LIQUIDITY_ATR_CEILING = 0.05
            SLIPPAGE_FRICTION = 0.1

            # 3. Compute composite score
            alpaca_ticker = self.risk_manager._resolve_ticker(signal.ticker)
            regime = self._get_signal_regime(signal.ticker)

            ev_r, sample_size = get_strategy_expectancy(
                signal.strategy_name, signal.direction,
                symbol=alpaca_ticker, regime=regime,
            )
            if sample_size < config.MIN_EXPECTANCY_SAMPLES:
                ev_r, sample_size = get_strategy_expectancy(
                    signal.strategy_name, signal.direction,
                )
            expected_ev = ev_r

            regime_fit = self._calc_regime_fit_score(regime, signal.strategy_name, signal.direction)

            atr_pct = (signal.atr / signal.entry) if signal.atr > 0 and signal.entry > 0 else 0.0
            liquidity_score = max(0.0, 1.0 - atr_pct / LIQUIDITY_ATR_CEILING)

            ml_agreement = 0.0
            if config.ML_VETO_ENABLED and self.ml_model is not None and df is not None and len(df) >= 40:
                try:
                    ml_result = self.ml_model.evaluate_signal(df, signal.direction)
                    sizing_multiplier = ml_result.get("sizing_multiplier", 1.0)
                    signal.sizing_multiplier = sizing_multiplier
                    ml_agreement = sizing_multiplier - 1.0
                except Exception as exc:
                    logger.warning("ML evaluate_signal failed: %s", exc)

            slippage_penalty = atr_pct * SLIPPAGE_FRICTION

            composite_score = expected_ev + regime_fit + liquidity_score + ml_agreement - slippage_penalty

            logger.info(
                "[RANK-SIGNAL] %s %s (%s) | EV: %+.3f | RegimeFit: %.3f | "
                "Liq: %.3f | ML: %+.3f | Slip: %.4f | Score: %+.4f",
                signal.direction, signal.ticker, signal.strategy_name,
                expected_ev, regime_fit, liquidity_score,
                ml_agreement, slippage_penalty, composite_score,
            )

            # 4. Append to the correct risk bucket
            bucket_key = BUCKET_MAP.get(alpaca_ticker, "general")
            scored_signals[bucket_key].append((composite_score, signal, df))

        # 5. Select the highest-scoring signal from each bucket
        result: list[tuple[StrategySignal, Optional[pd.DataFrame]]] = []
        for bucket_key in ("equity_beta", "crypto", "gold", "broad_commodity", "general"):
            bucket_entries = scored_signals.get(bucket_key, [])
            if bucket_entries:
                bucket_entries.sort(key=lambda x: x[0], reverse=True)
                best_score, best_signal, best_df = bucket_entries[0]
                logger.debug(
                    "[RANK-SELECT] Bucket %s: selected %s %s (score: %+.4f)",
                    bucket_key, best_signal.direction, best_signal.ticker, best_score,
                )
                result.append((best_signal, best_df))

        return result

    @staticmethod
    def _calc_regime_fit_score(regime: str, strategy_name: str, direction: str) -> float:
        """Return a 0–1 score indicating how well *strategy* fits *regime*.

        Regime labels come from the AI Tuner and are one of:
            Bullish Calm, Bullish Volatile, Range-Bound Calm,
            Bearish Chop, Bearish Volatile, or unknown.

        Direction-aware: for trend/momentum strategies, SELL signals invert
        the scoring to favor bearish regimes.
        """
        regime_lower = regime.lower().strip()

        fit_map = {}

        if strategy_name == "mean_reversion":
            # Mean reversion thrives in range-bound / choppy markets
            # and suffers in strong trending / volatile regimes.
            fit_map = {
                "range-bound calm": 1.0,
                "bearish chop": 0.8,
                "bullish calm": 0.6,
                "bullish volatile": 0.3,
                "bearish volatile": 0.1,
            }
        elif strategy_name in ("trend_pullback", "trend_following", "momentum_breakout"):
            if direction == "SELL":
                # SELL: favor bearish regimes, invert the bullish preference
                fit_map = {
                    "bearish volatile": 1.0,
                    "bearish chop": 0.8,
                    "range-bound calm": 0.5,
                    "bullish calm": 0.1,
                    "bullish volatile": 0.2,
                }
            else:
                # BUY (default): favor bullish regimes
                if strategy_name == "momentum_breakout":
                    fit_map = {
                        "bullish calm": 1.0,
                        "bullish volatile": 0.9,
                        "range-bound calm": 0.5,
                        "bearish chop": 0.2,
                        "bearish volatile": 0.0,
                    }
                else:
                    fit_map = {
                        "bullish calm": 1.0,
                        "bullish volatile": 0.8,
                        "range-bound calm": 0.6,
                        "bearish chop": 0.3,
                        "bearish volatile": 0.1,
                    }

        return fit_map.get(regime_lower, 0.5)

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

        # (Market hours guard removed to allow Extended Hours limit order entries)

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

        # Estimate live quote if available (placeholder until L1 data feed is wired)
        # Using a conservative spread estimate (0.5% of ATR or 1 tick)
        estimated_spread = max(0.01, signal.atr * 0.005)
        mock_quote = {"bid": signal.entry - estimated_spread/2, "ask": signal.entry + estimated_spread/2}

        # ML Veto Filter — use pre-computed sizing_multiplier from ranking phase
        ml_vetoed = False
        reason_veto = ""
        ml_multiplier = 1.0
        ml_action = "NEUTRAL"
        ml_action_reason = ""
        sizing_multiplier = getattr(signal, 'sizing_multiplier', 1.0)
        ml_regime = self._get_signal_regime(signal.ticker)

        if config.ML_VETO_ENABLED and self.ml_model is not None:
            try:
                # sizing_multiplier ∈ [0, ~2], centred at 1.0
                #   0      → strong disagreement (veto)
                #  < 1.0  → mild disagreement (reduce)
                #  > 1.0  → strong agreement (boost)
                #  = 1.0  → neutral
                if sizing_multiplier == 0.0:
                    ml_vetoed = True
                    ml_action = "VETO"
                    ml_action_reason = f"Strong ML disagreement: sizing_multiplier is 0.0"
                    reason_veto = ml_action_reason
                elif sizing_multiplier < 1.0:
                    ml_multiplier = sizing_multiplier
                    ml_action = "REDUCE"
                    ml_action_reason = f"Mild ML disagreement: sizing_multiplier={sizing_multiplier:.3f}"
                elif sizing_multiplier > 1.0:
                    ml_multiplier = sizing_multiplier
                    ml_action = "BOOST"
                    ml_action_reason = f"ML agreement: sizing_multiplier={sizing_multiplier:.3f}"
                else:
                    ml_action = "NEUTRAL"
                    ml_action_reason = "ML neutral: sizing_multiplier=1.0"
            except Exception as exc:
                logger.warning("ML veto check failed (allowing trade): %s", exc)

        # ── Helper to build a consistent ML action tracking log ────────
        def _build_ml_tracking_entry(action: str, action_reason: str,
                                      mult: float, notional_val: float = 0.0) -> dict:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": signal.ticker,
                "direction": signal.direction,
                "strategy": signal.strategy_name,
                "regime": ml_regime,
                "entry": signal.entry,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "confidence": signal.confidence,
                "notional": notional_val,
                "ml_multiplier": mult,
                "sizing_multiplier": sizing_multiplier,
                "action": action,
                "reason": action_reason,
            }

        if ml_vetoed:
            # Calculate hypothetical sizing for tracking virtual veto
            original_notional = self.risk_manager.calculate_position_size(
                signal, equity, signal_regime=self._get_signal_regime(signal.ticker)
            )
            logger.warning(
                "[ML VETO] %s %s vetoed by ML (sizing_multiplier=%.3f, direction=%s): %s",
                signal.direction, signal.ticker,
                sizing_multiplier, signal.direction,
                reason_veto
            )
            
            # Record virtual trade in order journal
            self.trader.record_virtual_trade(signal, reason_veto, original_notional, ml_regime)

            # Save virtual trade / veto log to vetoed_signals_tracker.jsonl
            try:
                entry = _build_ml_tracking_entry("VETO", reason_veto,
                                                  ml_multiplier, original_notional)
                with open("vetoed_signals_tracker.jsonl", "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                logger.exception("Failed to write virtual trade log: %s", e)
            return

        approved, notional, reason = self.risk_manager.approve(
            signal, equity, positions, active_orders,
            position_state=self._position_state,
            signal_regime=self._get_signal_regime(signal.ticker),
            df=df, quote=mock_quote,
            ml_multiplier=ml_multiplier,
        )

        # ML action tracking (notional already has ml_multiplier baked in from approve)
        if approved:
            if ml_multiplier < 1.0:
                ml_action = "REDUCE"
                ml_action_reason = f"ML reduce: sizing_multiplier={sizing_multiplier:.3f}"
            elif ml_multiplier > 1.0:
                ml_action = "BOOST"
                ml_action_reason = f"ML boost: sizing_multiplier={sizing_multiplier:.3f}"
            else:
                ml_action = "NEUTRAL"
                ml_action_reason = f"ML neutral: sizing_multiplier={sizing_multiplier:.3f}"

        # Log every ML action (VETO already logged above) to the tracking file
        if approved:
            try:
                track_entry = _build_ml_tracking_entry(
                    ml_action, ml_action_reason, ml_multiplier, notional
                )
                with open("vetoed_signals_tracker.jsonl", "a") as f:
                    f.write(json.dumps(track_entry) + "\n")
            except Exception as e:
                logger.exception("Failed to write ML action tracking log: %s", e)

        # --- SIGNAL TELEMETRY DUMP ---
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
            self.trader.record_virtual_trade(signal, reason, notional, self._get_signal_regime(signal.ticker))
            return

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
                "regime": self._get_signal_regime(signal.ticker),
            },
        )

        if result.get("ok"):
            self.orders_today += 1
            st = self._position_state.setdefault(alpaca_ticker, {})
            st['confidence'] = signal.confidence
            st['take_profit'] = signal.take_profit
            st['stop_loss'] = signal.stop_loss
            st['atr_at_entry'] = signal.atr
            st['entry_price'] = signal.entry
            st['notional'] = notional
            st['qty'] = notional / signal.entry
            st['side'] = signal.direction
            st['strategy'] = signal.strategy_name
            st['regime'] = self._get_signal_regime(signal.ticker)
            st['open_ts'] = time.time()
            st['highest'] = signal.entry
            st['lowest'] = signal.entry
            # time_stop_bars is already set strategy-specifically by each strategy's analyze()
            st['time_stop_bars'] = signal.time_stop_bars
            st['trailing_stop_logic'] = getattr(signal, 'trailing_stop_logic', 'default')
            st['timeframe'] = signal.timeframe
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

    def _close_and_log_position(
        self, sym: str, state: dict, current_price: float,
        unrealized_pl: float, exit_reason: str, qty_pct: Optional[str] = None,
    ) -> bool:
        """Calculate MFE/MAE, log trade, and close position (full or partial).

        Parameters
        ----------
        qty_pct : str, optional
            Percentage of the position to close (e.g. ``"50"``). When
            ``None`` the entire position is closed.
        """
        if self.trader is None:
            return False
        side = state.get('side', 'long')
        _entry = state.get('entry_price', current_price)

        # MFE / MAE — normalize side for various formats ('long', 'buy', 'short', 'sell')
        is_long = side.lower() in ('long', 'buy')
        if is_long:
            _mfe = state.get('highest', current_price) - _entry
            _mae = _entry - state.get('lowest', current_price)
        else:
            _mfe = _entry - state.get('lowest', current_price)
            _mae = state.get('highest', current_price) - _entry

        # Capture the order returned by Alpaca to get the actual fill price
        order = None
        try:
            if qty_pct is not None:
                order = self.trader._client.close_position(sym, percentage=qty_pct)
            else:
                order = self.trader._client.close_position(sym)
        except Exception as e:
            logger.error("Failed to close position %s: %s", sym, e)
            return False

        # Extract actual fill price from the returned Order object
        exit_price = current_price  # fallback
        order_id = None
        if order is not None:
            try:
                order_id = str(order.id) if getattr(order, 'id', None) else None
                fap = getattr(order, 'filled_avg_price', None)
                if fap is not None:
                    exit_price = float(fap)
                elif order_id:
                    # Short poll if fill hasn't settled yet
                    for _ in range(3):
                        time.sleep(0.3)
                        try:
                            refreshed = self.trader._client.get_order_by_id(order_id)
                            fap2 = getattr(refreshed, 'filled_avg_price', None)
                            if fap2 is not None:
                                exit_price = float(fap2)
                                break
                        except Exception:
                            pass
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Could not parse filled_avg_price for %s: %s — using current_price",
                    sym, exc,
                )

        # Mark order journal as DB-logged to avoid double-logging in sync_closed_trades
        if order_id:
            journal = self.trader._state.setdefault("order_journal", {})
            if order_id in journal:
                journal[order_id]["db_logged"] = True

        # Calculate exact fraction closed for accurate R-multiple logging
        closed_fraction = float(qty_pct) / 100.0 if qty_pct else 1.0
        logged_qty = state.get('qty', 0) * closed_fraction
        logged_pnl = unrealized_pl * closed_fraction

        # Log the trade only after successful close
        try:
            log_trade(
                symbol=sym,
                strategy=state.get('strategy', 'unknown'),
                direction=side,
                entry_price=_entry,
                exit_price=exit_price,
                stop_loss=state.get('stop_loss', 0.0),
                qty=logged_qty,
                pnl=logged_pnl,
                mfe=_mfe,
                mae=_mae,
                hold_hours=(time.time() - state.get('open_ts', time.time())) / 3600.0,
                exit_reason=exit_reason,
                regime=state.get('regime', 'unknown'),
            )
        except Exception as e:
            logger.error("Failed to log trade to DB for %s: %s", sym, e)

        return True

    def _manage_positions(self) -> None:
        """Monitor open positions and close based on trailing/stop loss or time stop."""
        if self.trader is None:
            return

        positions = self.trader.get_positions()
        current_syms = {pos['symbol'] for pos in positions}

        res = self.trader.get_active_orders()
        active_orders = res.get('orders', []) if res.get('ok') else []
        active_syms = {o.get('symbol') for o in active_orders}

        broker_exits = {}
        for o in active_orders:
            sym = o.get("symbol")
            if not sym: continue
            otype = o.get("type", "")
            if otype == "stop":
                broker_exits.setdefault(sym, {})["stop"] = True
            elif otype == "limit":
                broker_exits.setdefault(sym, {})["limit"] = True

        # Clean up symbols no longer in current positions
        for sym in list(self._position_state.keys()):
            if sym not in current_syms and sym not in active_syms:
                del self._position_state[sym]

        for pos in positions:
            sym = pos['symbol']
            current_price = pos['current_price']
            unrealized_pl = pos['unrealized_pl']
            market_value = pos['market_value']
            side = pos['side']

            # Initialise tracking entry if new position or if missing tracking fields
            if sym not in self._position_state or 'open_ts' not in self._position_state[sym]:
                _state = self._position_state.setdefault(sym, {})

                # Recover missing fields after restart using asset-specific fallback
                # logic from the risk manager (see _FALLBACK_RISK_MAP for ranges).
                if 'entry_price' not in _state:
                    _state['entry_price'] = pos.get('avg_entry_price', current_price)
                if 'qty' not in _state:
                    _state['qty'] = pos.get('qty', 0.0)
                if 'stop_loss' not in _state:
                    # Asset-specific fallback stop distance — uses the midpoints
                    # of the configured ranges (2-3% SPY/QQQ, 8-12% BTC, 3-5% GLD,
                    # 5-8% PDBC) from RiskManager._FALLBACK_RISK_MAP.
                    fallback_pct = self.risk_manager.get_fallback_risk_pct(sym)
                    _state['stop_loss'] = (
                        current_price * (1.0 - fallback_pct) if side == 'long'
                        else current_price * (1.0 + fallback_pct)
                    )
                if 'take_profit' not in _state:
                    # Use a symmetric 1:1 risk-reward as a conservative TP fallback
                    _state['take_profit'] = (
                        current_price + (current_price - _state['stop_loss']) if side == 'long'
                        else current_price - (_state['stop_loss'] - current_price)
                    )
                if 'side' not in _state:
                    _state['side'] = side
                if 'strategy' not in _state:
                    _state['strategy'] = 'unknown_restarted'
                if 'regime' not in _state:
                    _state['regime'] = 'unknown'
                if 'notional' not in _state:
                    _state['notional'] = _state.get('qty', 0.0) * _state.get('entry_price', current_price)

                entry = _state['entry_price']
                _state.update({
                    "open_ts": time.time(),
                    "highest": max(current_price, entry),
                    "lowest": min(current_price, entry),
                })

            state = self._position_state[sym]

            # If missing fields on state, populate them (for restart / legacy recovery):
            # Use strategy-specific defaults per codex.md item 13.
            if 'time_stop_bars' not in state:
                state['time_stop_bars'] = self._get_default_time_stop_bars(
                    state.get('strategy', 'unknown'), sym
                )
            if 'trailing_stop_logic' not in state:
                strat_name = state.get('strategy', 'unknown_restarted')
                state['trailing_stop_logic'] = get_strategy_trailing_logic(strat_name, sym)
            if 'timeframe' not in state:
                # Default timeframe based on asset class
                if "GLD" in sym or "PDBC" in sym:
                    state['timeframe'] = '4h'
                elif "BTC" in sym or "ETH" in sym:
                    state['timeframe'] = '1h'
                else:
                    state['timeframe'] = '15m'

            # Update highest/lowest
            if current_price > state['highest']:
                state['highest'] = current_price
            if current_price < state['lowest']:
                state['lowest'] = current_price

            # ── Dynamic Stop Loss Update (Trailing Stop Logic) ──
            trailing_logic = state.get('trailing_stop_logic', 'default')
            # Asset-class-aware timeframe fallback for trailing stop retrieval
            _tf_default = '15m'
            if "GLD" in sym or "PDBC" in sym:
                _tf_default = '4h'
            elif "BTC" in sym or "ETH" in sym:
                _tf_default = '1h'
            tf = state.get('timeframe', _tf_default)
            
            old_sl = state.get('stop_loss')
            
            # Reuse cached DF if possible, or fetch from Alpaca
            df = None
            if trailing_logic in ("sma20_or_ema", "donchian", "vwap"):
                df = self._cycle_dfs.get(sym)
                if df is None:
                    df = self._cycle_dfs.get(sym.replace("-USD", ""))
                if df is None:
                    try:
                        df = fetch_ohlcv(sym, period="5d", interval=tf)
                        if df is not None:
                            self._cycle_dfs[sym] = df
                    except Exception as e:
                        logger.warning("Failed to fetch OHLCV for %s: %s", sym, e)

            try:
                new_sl = calculate_trailing_stop(
                    logic_type=trailing_logic,
                    current_price=current_price,
                    current_sl=old_sl,
                    direction=side,
                    df=df,
                    atr=state.get('atr_at_entry', 0.0),
                    entry_price=state.get('entry_price', current_price),
                    tp_filled=state.get('tp_filled', False),
                    highest_price=state.get('highest'),
                    lowest_price=state.get('lowest'),
                    ticker=sym,
                )
                if new_sl is not None:
                    state['stop_loss'] = new_sl
            except Exception as e:
                logger.warning("Failed to calculate trailing stop for %s: %s", sym, e)

            # Sync with broker if stop loss changed and we have a trader instance
            if state.get('stop_loss') != old_sl and self.trader is not None:
                try:
                    tp_val = state.get('take_profit', current_price * 1.5 if side == 'long' else current_price * 0.5)
                    self.trader.update_position_exits(sym, take_profit=tp_val, stop_loss=state['stop_loss'])
                except Exception as exc:
                    logger.warning("Failed to sync trailing stop-loss update with Alpaca for %s: %s", sym, exc)

            # ── Synthetic Bracket Management (Crypto & Extended Hours) ──
            has_broker_sl = broker_exits.get(sym, {}).get("stop")
            has_broker_tp = broker_exits.get(sym, {}).get("limit")
            
            # 1. Synthetic Stop Loss
            if 'stop_loss' in state and not has_broker_sl:
                if (side == 'long' and current_price <= state['stop_loss']) or \
                   (side == 'short' and current_price >= state['stop_loss']):
                    logger.info("Closing position %s due to synthetic stop loss", sym)
                    if self._close_and_log_position(sym, state, current_price, unrealized_pl, 'synthetic_sl'):
                        del self._position_state[sym]
                    else:
                        logger.warning("Synthetic SL close failed; preserving state for retry")
                    continue

            # 2. Synthetic Take Profit
            if 'take_profit' in state and not state.get('tp_filled', False) and not has_broker_tp:
                if (side == 'long' and current_price >= state['take_profit']) or \
                   (side == 'short' and current_price <= state['take_profit']):
                    logger.info("Taking partial profit on %s due to synthetic take-profit limit", sym)
                    if self._close_and_log_position(sym, state, current_price, unrealized_pl, 'synthetic_tp', qty_pct="50"):
                        state['tp_filled'] = True
                        state['qty'] = state.get('qty', 0) * 0.5
                        # We keep tracking it as a runner now.

            # ── Time Stop (Strategy-specific bars converted to hours) ──
            time_stop_bars = state.get('time_stop_bars', 10)
            
            # Simple timeframe to hours parsing
            tf_lower = tf.lower()
            if tf_lower.endswith('m'):
                bar_hours = float(tf_lower[:-1]) / 60.0
            elif tf_lower.endswith('h'):
                bar_hours = float(tf_lower[:-1])
            elif tf_lower.endswith('d'):
                bar_hours = float(tf_lower[:-1]) * 24.0
            else:
                bar_hours = 0.25
                
            time_stop_hours = time_stop_bars * bar_hours
            if time_stop_hours < 0.25:
                time_stop_hours = 0.25
                
            hours_open = (time.time() - state['open_ts']) / 3600.0
            if hours_open > time_stop_hours:
                logger.info(
                    "Closing %s due to time stop (%.1f hours)",
                    sym, time_stop_hours,
                )
                if self._close_and_log_position(sym, state, current_price, unrealized_pl, 'time_stop'):
                    del self._position_state[sym]
                else:
                    logger.warning("Time stop close failed; preserving state for retry")

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

            # Fetch active orders to look up broker-level TP/SL
            active_orders_res = self.trader.get_active_orders()
            active_orders = active_orders_res.get("orders", []) if active_orders_res.get("ok") else []
            
            # Map symbol to its active stop and limit prices
            broker_exits = {}
            for o in active_orders:
                sym = o.get("symbol")
                if not sym: continue
                otype = o.get("type", "")
                if otype == "stop":
                    broker_exits.setdefault(sym, {})["stop"] = o.get("stop_price")
                elif otype == "limit":
                    broker_exits.setdefault(sym, {})["limit"] = o.get("limit_price")

            for pos in positions:
                sym = pos.get("symbol", "?")
                st = self._position_state.get(sym, {})
                
                # Use broker-level orders if present, otherwise fall back to internal tracker
                exits = broker_exits.get(sym, {})
                sl = exits.get("stop") if "stop" in exits else st.get("stop_loss")
                tp = exits.get("limit") if "limit" in exits else st.get("take_profit")
                
                sl_str = f"${sl:.2f}" if sl else "None"
                tp_str = f"${tp:.2f}" if tp else "None"

                logger.info(
                    "  Position: %s %s qty=%.2f P&L=$%.2f | SL: %s | TP: %s",
                    pos.get("side", "?").upper(),
                    sym,
                    pos.get("qty", 0),
                    pos.get("unrealized_pl", 0),
                    sl_str,
                    tp_str,
                )

            logger.info(
                "  Regimes: %s", self.ai_tuner.current_regimes
            )
            logger.info(
                "  ML Model: trained=%s, training=%s",
                getattr(self.ml_model, '_trained', False) if self.ml_model else False,
                getattr(self.ml_model, '_training', False) if self.ml_model else False,
            )
            # Log veto outcome stats summary
            if self.ml_model is not None:
                try:
                    veto_stats = self.ml_model.get_veto_stats()
                    for bucket_key, bucket_data in veto_stats.items():
                        if bucket_data:
                            summary_parts = []
                            for key, rec in sorted(bucket_data.items()):
                                summary_parts.append(
                                    f"{key}: {rec['total']}t {rec['wins']}w {rec['losses']}l "
                                    f"({rec['win_rate']*100:.0f}%)"
                                )
                            if summary_parts:
                                logger.info(
                                    "  ML Veto Stats [%s]: %s",
                                    bucket_key, " | ".join(summary_parts),
                                )
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Status log failed: %s", exc)

