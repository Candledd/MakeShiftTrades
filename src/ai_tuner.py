"""MakeShiftTrades — Market Regime AI Tuner
Analyzes recent benchmark data across Equity, Crypto, and Commodities
to classify sector-specific market regimes and select pre-approved parameter profiles.

The AI does NOT directly rewrite arbitrary config values. Instead it selects
from pre-approved parameter sets (REGIME_PROFILES in config.py) based on
quantifiable market features (SMA trend direction, ATR volatility).
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from charts.data import fetch_ohlcv
import config

logger = logging.getLogger(__name__)

class AITuner:
    """Sector-Specific Regime Classifier with Bounded Profile Selection.

    Analyzes quantifiable market features (trend direction via 20/50 SMA crossover,
    volatility via 14-period ATR% of close) to select a pre-approved regime profile
    for each sector from config.REGIME_PROFILES.

    The AI does NOT directly rewrite arbitrary strategy parameters or risk limits.
    All config changes are bounded to the pre-approved profiles defined in config.py.
    """

    def __init__(self):
        self.current_regimes = {
            "Equity": "Unknown",
            "Crypto": "Unknown",
            "Gold": "Unknown",
            "Broad Commodity": "Unknown"
        }
        self.last_tuned_time = 0
        self.tune_interval = 3600 * 4  # Retune every 4 hours

    def _analyze_asset_regime(self, ticker: str, volatile_threshold: float) -> Tuple[Optional[str], float]:
        """Analyzes a specific asset to determine its regime and current ATR.

        Uses quantifiable market features:
        - Trend direction: 20/50 SMA crossover
        - Volatility: 14-period ATR as % of close
        """
        try:
            df = fetch_ohlcv(ticker, period="6mo", interval="1d")
            if df is None or len(df) < 50:
                return None, 0.0

            df['sma20'] = df['Close'].rolling(20).mean()
            df['sma50'] = df['Close'].rolling(50).mean()
            current_close = df['Close'].iloc[-1]
            sma20 = df['sma20'].iloc[-1]
            sma50 = df['sma50'].iloc[-1]

            is_bull = sma20 > sma50 and current_close > sma50
            is_bear = sma20 < sma50 and current_close < sma50
            is_range = not is_bull and not is_bear

            df['tr'] = np.maximum(
                df['High'] - df['Low'],
                np.maximum(
                    abs(df['High'] - df['Close'].shift(1)),
                    abs(df['Low'] - df['Close'].shift(1))
                )
            )
            df['atr'] = df['tr'].rolling(14).mean()
            current_atr_pct = (df['atr'].iloc[-1] / current_close) * 100

            is_volatile = current_atr_pct > volatile_threshold

            if is_bull and not is_volatile:
                return "Bullish Calm", current_atr_pct
            elif is_bull and is_volatile:
                return "Bullish Volatile", current_atr_pct
            elif is_range and not is_volatile:
                return "Range-Bound Calm", current_atr_pct
            elif is_bear and not is_volatile:
                return "Bearish Chop", current_atr_pct
            else:
                return "Bearish Volatile", current_atr_pct

        except Exception as exc:
            logger.warning("AI Tuner failed to analyze %s: %s", ticker, exc)
            return None, 0.0

    def _apply_regime_profile(self, sector: str, regime_name: str) -> Dict[str, Any]:
        """Apply the pre-approved profile parameters for a given sector and regime.

        Looks up the profile in config.REGIME_PROFILES and applies only the
        pre-approved parameter overrides — no arbitrary config rewriting.
        Returns the applied parameter dict.
        """
        sector_profiles = config.REGIME_PROFILES.get(sector, {})
        profile = sector_profiles.get(regime_name, {})
        if not profile:
            logger.warning("No profile found for %s / %s", sector, regime_name)
            return {}

        with config.config_lock:
            for key, val in profile.items():
                setattr(config, key, val)

        logger.debug("Applied %s profile '%s': %s", sector, regime_name, profile)
        return profile

    def tune_parameters(self) -> Dict[str, Any]:
        """Analyze sectors using quantifiable market features and select pre-approved profiles.

        For each sector:
        1. Analyze market features (SMA trend + ATR % volatility)
        2. Select a regime from the pre-approved set in config.REGIME_PROFILES
        3. Apply only the bounded parameters from that profile

        Does NOT directly modify arbitrary strategy indicators or risk limits.
        Returns a dict of sector -> {regime, profile, atr_pct}.
        """
        now = time.time()
        if now - self.last_tuned_time < self.tune_interval and "Unknown" not in self.current_regimes.values():
            return {}

        results = {}
        log_msgs = []

        # 1. EQUITY SECTOR (SPY) -> impacts Mean Reversion
        eq_regime, eq_atr = self._analyze_asset_regime("SPY", volatile_threshold=1.5)
        if eq_regime:
            self.current_regimes["Equity"] = eq_regime
            profile = self._apply_regime_profile("Equity", eq_regime)
            results["Equity"] = {"regime": eq_regime, "profile": profile, "atr_pct": eq_atr}
            log_msgs.append(f"EQ: {eq_regime}")

        # 2. CRYPTO SECTOR (BTC-USD) -> impacts Momentum Breakout
        cr_regime, cr_atr = self._analyze_asset_regime("BTC-USD", volatile_threshold=4.0)
        if cr_regime:
            self.current_regimes["Crypto"] = cr_regime
            profile = self._apply_regime_profile("Crypto", cr_regime)
            results["Crypto"] = {"regime": cr_regime, "profile": profile, "atr_pct": cr_atr}
            log_msgs.append(f"CRYPTO: {cr_regime}")

        # 3. GOLD SECTOR (GLD) -> impacts Trend Following
        go_regime, go_atr = self._analyze_asset_regime("GLD", volatile_threshold=1.5)
        if go_regime:
            # Check macro USD proxy UUP
            uup_regime, _ = self._analyze_asset_regime("UUP", volatile_threshold=1.5)
            if uup_regime in ["Bullish Calm", "Bullish Volatile"]:
                # Penalize GLD regime
                old_regime = go_regime
                if go_regime == "Bullish Calm":
                    go_regime = "Range-Bound Calm"
                elif go_regime == "Bullish Volatile":
                    go_regime = "Bearish Chop"
                elif go_regime == "Range-Bound Calm":
                    go_regime = "Bearish Chop"
                elif go_regime == "Bearish Chop":
                    go_regime = "Bearish Volatile"
                logger.info(
                    "[AI TUNER] UUP (USD proxy) is strongly bullish (%s). Penalizing GLD regime: %s -> %s",
                    uup_regime, old_regime, go_regime
                )
            self.current_regimes["Gold"] = go_regime
            profile = self._apply_regime_profile("Gold", go_regime)
            results["Gold"] = {"regime": go_regime, "profile": profile, "atr_pct": go_atr}
            log_msgs.append(f"GOLD: {go_regime}")

        # 4. BROAD COMMODITY SECTOR (PDBC) -> impacts Trend Following
        bc_regime, bc_atr = self._analyze_asset_regime("PDBC", volatile_threshold=2.0)
        if bc_regime:
            # Check macro Oil proxy USO
            uso_regime, _ = self._analyze_asset_regime("USO", volatile_threshold=2.0)
            if uso_regime in ["Bullish Calm", "Bullish Volatile"]:
                # Boost PDBC regime
                old_regime = bc_regime
                if bc_regime == "Bearish Volatile":
                    bc_regime = "Bearish Chop"
                elif bc_regime == "Bearish Chop":
                    bc_regime = "Range-Bound Calm"
                elif bc_regime == "Range-Bound Calm":
                    bc_regime = "Bullish Calm"
                elif bc_regime == "Bullish Calm":
                    bc_regime = "Bullish Volatile"
                logger.info(
                    "[AI TUNER] USO (Oil proxy) is strongly bullish (%s). Boosting PDBC regime: %s -> %s",
                    uso_regime, old_regime, bc_regime
                )
            self.current_regimes["Broad Commodity"] = bc_regime
            profile = self._apply_regime_profile("Broad Commodity", bc_regime)
            results["Broad Commodity"] = {"regime": bc_regime, "profile": profile, "atr_pct": bc_atr}
            log_msgs.append(f"BROAD_COM: {bc_regime}")

        if results:
            self.last_tuned_time = now
            logger.info("[AI TUNER] Regimes updated -> %s", " | ".join(log_msgs))

        return results