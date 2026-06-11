"""MakeShiftTrades — Market Regime AI Tuner
Analyzes recent benchmark data across Equity, Crypto, and Commodities
to classify sector-specific market regimes and dynamically update strategy parameters.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from charts.data import fetch_ohlcv
import config

logger = logging.getLogger(__name__)

class AITuner:
    """Sector-Specific Parameter-Tuning AI Agent."""
    
    def __init__(self):
        self.current_regimes = {
            "Equity": "Unknown",
            "Crypto": "Unknown",
            "Commodity": "Unknown"
        }
        self.last_tuned_time = 0
        self.tune_interval = 3600 * 4  # Retune every 4 hours

    def _analyze_asset_regime(self, ticker: str, volatile_threshold: float) -> Tuple[str, float]:
        """Analyzes a specific asset to determine its regime and current ATR."""
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
            elif not is_bull and not is_volatile:
                return "Bearish Chop", current_atr_pct
            else:
                return "Bearish Volatile", current_atr_pct

        except Exception as exc:
            logger.warning("AI Tuner failed to analyze %s: %s", ticker, exc)
            return None, 0.0

    def tune_parameters(self) -> Dict[str, Any]:
        """Analyze sectors, determine regimes, and apply new configs."""
        now = time.time()
        if now - self.last_tuned_time < self.tune_interval and "Unknown" not in self.current_regimes.values():
            return {} 

        updates = {}
        log_msgs = []

        # 1. EQUITY SECTOR (SPY) -> impacts Mean Reversion
        eq_regime, eq_atr = self._analyze_asset_regime("SPY", volatile_threshold=1.5)
        if eq_regime:
            self.current_regimes["Equity"] = eq_regime
            if eq_regime == "Bullish Calm":
                updates["AI_RISK_MULTIPLIER_EQUITY"] = 1.0
                updates["MR_RSI_OVERSOLD"] = 40.0
            elif eq_regime == "Bullish Volatile":
                updates["AI_RISK_MULTIPLIER_EQUITY"] = 0.8
                updates["MR_RSI_OVERSOLD"] = 35.0
            elif eq_regime == "Bearish Chop":
                updates["AI_RISK_MULTIPLIER_EQUITY"] = 0.5
                updates["MR_RSI_OVERSOLD"] = 25.0
            else: # Bearish Volatile
                updates["AI_RISK_MULTIPLIER_EQUITY"] = 0.5
                updates["MR_RSI_OVERSOLD"] = 20.0
            log_msgs.append(f"EQ: {eq_regime}")

        # 2. CRYPTO SECTOR (BTC-USD) -> impacts Momentum Breakout
        cr_regime, cr_atr = self._analyze_asset_regime("BTC-USD", volatile_threshold=4.0)
        if cr_regime:
            self.current_regimes["Crypto"] = cr_regime
            if cr_regime == "Bullish Calm":
                updates["AI_RISK_MULTIPLIER_CRYPTO"] = 1.0
                updates["MB_ADX_THRESHOLD"] = 20.0
            elif cr_regime == "Bullish Volatile":
                updates["AI_RISK_MULTIPLIER_CRYPTO"] = 0.8
                updates["MB_ADX_THRESHOLD"] = 25.0
            elif cr_regime == "Bearish Chop":
                updates["AI_RISK_MULTIPLIER_CRYPTO"] = 0.5
                updates["MB_ADX_THRESHOLD"] = 25.0
            else: # Bearish Volatile
                updates["AI_RISK_MULTIPLIER_CRYPTO"] = 0.5
                updates["MB_ADX_THRESHOLD"] = 30.0
            log_msgs.append(f"CRYPTO: {cr_regime}")

        # 3. COMMODITY SECTOR (GLD) -> impacts Trend Following
        co_regime, co_atr = self._analyze_asset_regime("GLD", volatile_threshold=1.5)
        if co_regime:
            self.current_regimes["Commodity"] = co_regime
            if co_regime == "Bullish Calm":
                updates["AI_RISK_MULTIPLIER_COMMODITY"] = 1.0
                updates["TF_EMA_FAST"] = 20
            elif co_regime == "Bullish Volatile":
                updates["AI_RISK_MULTIPLIER_COMMODITY"] = 0.8
                updates["TF_EMA_FAST"] = 20
            elif co_regime == "Bearish Chop":
                updates["AI_RISK_MULTIPLIER_COMMODITY"] = 0.5
                updates["TF_EMA_FAST"] = 15
            else: # Bearish Volatile
                updates["AI_RISK_MULTIPLIER_COMMODITY"] = 0.5
                updates["TF_EMA_FAST"] = 10
            log_msgs.append(f"COM: {co_regime}")

        if updates:
            self.last_tuned_time = now
            for key, val in updates.items():
                setattr(config, key, val)
            
            logger.info("[AI TUNER] Regimes updated -> %s", " | ".join(log_msgs))

        return updates