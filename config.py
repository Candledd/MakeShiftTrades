import os
import threading
from dotenv import load_dotenv

# ── Global Config Lock for Thread Safety ───────────────────────────────────────
# Used by AITuner (and any other component) to serialize mutations to module-level
# config attributes, preventing race conditions when applying regime profiles.
config_lock = threading.Lock()

load_dotenv()

# ── Alpaca Paper Trading credentials ──────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# ── Risk Management ──────────────────────────────────────────────────────────
try:
    VIRTUAL_EQUITY = float(os.getenv("VIRTUAL_EQUITY", "0"))
except ValueError:
    VIRTUAL_EQUITY = 0.0

try:
    MAX_RISK_PCT = float(os.getenv("MAX_RISK_PCT", "0.005"))
except ValueError:
    raise ValueError("MAX_RISK_PCT must be a valid number")

try:
    MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "1.00"))
except ValueError:
    raise ValueError("MAX_POSITION_PCT must be a valid number")

try:
    MAX_NOTIONAL = float(os.getenv("MAX_NOTIONAL", "100000"))
except ValueError:
    raise ValueError("MAX_NOTIONAL must be a valid number")

try:
    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
except ValueError:
    raise ValueError("MAX_POSITIONS must be a valid integer")

try:
    MAX_OPEN_PORTFOLIO_RISK_PCT = float(os.getenv("MAX_OPEN_PORTFOLIO_RISK_PCT", "0.02"))
except ValueError:
    raise ValueError("MAX_OPEN_PORTFOLIO_RISK_PCT must be a valid number")

try:
    MAX_CLUSTER_RISK_PCT = float(os.getenv("MAX_CLUSTER_RISK_PCT", "0.01"))
except ValueError:
    raise ValueError("MAX_CLUSTER_RISK_PCT must be a valid number")

try:
    MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "-0.015"))
except ValueError:
    raise ValueError("MAX_DAILY_LOSS_PCT must be a valid number")

try:
    MAX_WEEKLY_LOSS_PCT = float(os.getenv("MAX_WEEKLY_LOSS_PCT", "-0.04"))
except ValueError:
    raise ValueError("MAX_WEEKLY_LOSS_PCT must be a valid number")

try:
    MIN_EXPECTANCY_R = float(os.getenv("MIN_EXPECTANCY_R", "0.05"))
except ValueError:
    raise ValueError("MIN_EXPECTANCY_R must be a valid number")

try:
    MIN_EXPECTANCY_SAMPLES = int(os.getenv("MIN_EXPECTANCY_SAMPLES", "5"))
except ValueError:
    raise ValueError("MIN_EXPECTANCY_SAMPLES must be a valid integer")

try:
    MAX_VIX_THRESHOLD = float(os.getenv("MAX_VIX_THRESHOLD", "30.0"))
except ValueError:
    raise ValueError("MAX_VIX_THRESHOLD must be a valid number")

# ── Tiered Risk Per Asset Class ──────────────────────────────────────────────
# These override MAX_RISK_PCT when a ticker matches the corresponding asset set.
try:
    RISK_TIER_EQUITY_PCT = float(os.getenv("RISK_TIER_EQUITY_PCT", "0.0025"))
except ValueError:
    raise ValueError("RISK_TIER_EQUITY_PCT must be a valid number")

try:
    RISK_TIER_CRYPTO_PCT = float(os.getenv("RISK_TIER_CRYPTO_PCT", "0.0050"))
except ValueError:
    raise ValueError("RISK_TIER_CRYPTO_PCT must be a valid number")

try:
    RISK_TIER_COMMODITY_PCT = float(os.getenv("RISK_TIER_COMMODITY_PCT", "0.0035"))
except ValueError:
    raise ValueError("RISK_TIER_COMMODITY_PCT must be a valid number")

# ── Spread / Liquidity Cap (ATR-based) ───────────────────────────────────────
# Trades where ATR/price exceeds this threshold are rejected (proxy for wide
# spreads or illiquid markets).
try:
    SPREAD_ATR_CAP_PCT = float(os.getenv("SPREAD_ATR_CAP_PCT", "0.05"))
except ValueError:
    raise ValueError("SPREAD_ATR_CAP_PCT must be a valid number")

# ── Gap / Slippage Buffer ────────────────────────────────────────────────────
# Notional is reduced by this fraction to cushion gap risk at open.
try:
    GAP_SLIPPAGE_BUFFER_PCT = float(os.getenv("GAP_SLIPPAGE_BUFFER_PCT", "0.001"))
except ValueError:
    raise ValueError("GAP_SLIPPAGE_BUFFER_PCT must be a valid number")

try:
    BACKTEST_SLIPPAGE_FRICTION_PCT = float(os.getenv("BACKTEST_SLIPPAGE_FRICTION_PCT", "0.0005"))
except ValueError:
    raise ValueError("BACKTEST_SLIPPAGE_FRICTION_PCT must be a valid number")

# ── Volatility Shock Adjustment ──────────────────────────────────────────────
# When ATR/price exceeds VOLATILITY_SHOCK_ATR_PCT, position size is multiplied
# by VOLATILITY_SHOCK_REDUCTION to account for sudden market volatility.
try:
    VOLATILITY_SHOCK_ATR_PCT = float(os.getenv("VOLATILITY_SHOCK_ATR_PCT", "0.03"))
except ValueError:
    raise ValueError("VOLATILITY_SHOCK_ATR_PCT must be a valid number")

try:
    VOLATILITY_SHOCK_REDUCTION = float(os.getenv("VOLATILITY_SHOCK_REDUCTION", "0.50"))
except ValueError:
    raise ValueError("VOLATILITY_SHOCK_REDUCTION must be a valid number")

# ── AI Regime Tuner — Sector Risk Multipliers ────────────────────────────────
# Used by risk_manager.py per-trade; tuned by AITuner via pre-approved profiles.
try:
    AI_RISK_MULTIPLIER_EQUITY = float(os.getenv("AI_RISK_MULTIPLIER_EQUITY", "1.0"))
except ValueError:
    raise ValueError("AI_RISK_MULTIPLIER_EQUITY must be a valid number")

try:
    AI_RISK_MULTIPLIER_CRYPTO = float(os.getenv("AI_RISK_MULTIPLIER_CRYPTO", "1.0"))
except ValueError:
    raise ValueError("AI_RISK_MULTIPLIER_CRYPTO must be a valid number")

try:
    AI_RISK_MULTIPLIER_COMMODITY = float(os.getenv("AI_RISK_MULTIPLIER_COMMODITY", "1.0"))
except ValueError:
    raise ValueError("AI_RISK_MULTIPLIER_COMMODITY must be a valid number")

# ── Bot Operation ─────────────────────────────────────────────────────────────
try:
    SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "60"))
except ValueError:
    raise ValueError("SCAN_INTERVAL must be a valid integer")

DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

try:
    ORDER_TTL_HOURS = float(os.getenv("ORDER_TTL_HOURS", "2.0"))
except ValueError:
    raise ValueError("ORDER_TTL_HOURS must be a valid number")

# ── Trend Pullback Strategy (SPY, QQQ — 15m) ───────────────────────────────
try:
    TP_BB_PERIOD = int(os.getenv("TP_BB_PERIOD", "10"))
except ValueError:
    raise ValueError("TP_BB_PERIOD must be a valid integer")

try:
    TP_BB_STD = float(os.getenv("TP_BB_STD", "2.2080"))
except ValueError:
    raise ValueError("TP_BB_STD must be a valid number")

try:
    TP_STOP_MULT = float(os.getenv("TP_STOP_MULT", "3.9597"))
except ValueError:
    raise ValueError("TP_STOP_MULT must be a valid number")

try:
    TP_PULLBACK_BUFFER = float(os.getenv("TP_PULLBACK_BUFFER", "1.0093"))
except ValueError:
    raise ValueError("TP_PULLBACK_BUFFER must be a valid number")

try:
    MR_TP_TARGET_MULT = float(os.getenv("MR_TP_TARGET_MULT", "2.3627"))
except ValueError:
    raise ValueError("MR_TP_TARGET_MULT must be a valid number")

try:
    MR_MIN_RR = float(os.getenv("MR_MIN_RR", "1.7599"))
except ValueError:
    raise ValueError("MR_MIN_RR must be a valid number")

try:
    TP_MIN_RR = float(os.getenv("TP_MIN_RR", "2.6153"))
except ValueError:
    raise ValueError("TP_MIN_RR must be a valid number")

try:
    TP_RSI_PERIOD = int(os.getenv("TP_RSI_PERIOD", "14"))
except ValueError:
    raise ValueError("TP_RSI_PERIOD must be a valid integer")

try:
    TP_VOL_SPIKE_MULT = float(os.getenv("TP_VOL_SPIKE_MULT", "0.5"))
except ValueError:
    raise ValueError("TP_VOL_SPIKE_MULT must be a valid number")

# ── Mean Reversion Strategy (SPY, QQQ — 15m) ─────────────────────────────────
try:
    MR_BB_PERIOD = int(os.getenv("MR_BB_PERIOD", "27"))
except ValueError:
    raise ValueError("MR_BB_PERIOD must be a valid integer")

try:
    MR_BB_STD = float(os.getenv("MR_BB_STD", "2.6080"))
except ValueError:
    raise ValueError("MR_BB_STD must be a valid number")

try:
    MR_STOP_MULT = float(os.getenv("MR_STOP_MULT", "1.8852"))
except ValueError:
    raise ValueError("MR_STOP_MULT must be a valid number")

try:
    MR_RSI_PERIOD = int(os.getenv("MR_RSI_PERIOD", "10"))
except ValueError:
    raise ValueError("MR_RSI_PERIOD must be a valid integer")

try:
    MR_RSI_OVERSOLD = float(os.getenv("MR_RSI_OVERSOLD", "39.9400"))
except ValueError:
    raise ValueError("MR_RSI_OVERSOLD must be a valid number")

try:
    MR_RSI_OVERBOUGHT = float(os.getenv("MR_RSI_OVERBOUGHT", "78.3060"))
except ValueError:
    raise ValueError("MR_RSI_OVERBOUGHT must be a valid number")

# ── Momentum Breakout Strategy (BTC — 1h) ────────────────────────────────────
try:
    MB_DONCHIAN_PERIOD = int(os.getenv("MB_DONCHIAN_PERIOD", "31"))
except ValueError:
    raise ValueError("MB_DONCHIAN_PERIOD must be a valid integer")

try:
    MB_ADX_THRESHOLD = float(os.getenv("MB_ADX_THRESHOLD", "27.9925"))
except ValueError:
    raise ValueError("MB_ADX_THRESHOLD must be a valid number")

try:
    MB_ATR_TARGET_MULT = float(os.getenv("MB_ATR_TARGET_MULT", "2.5"))
except ValueError:
    raise ValueError("MB_ATR_TARGET_MULT must be a valid number")

# ── Trend Following Strategy (GLD, PDBC — 4h) ─────────────────────────────────
try:
    TF_EMA_FAST = int(os.getenv("TF_EMA_FAST", "20"))
except ValueError:
    raise ValueError("TF_EMA_FAST must be a valid integer")

try:
    TF_EMA_SLOW = int(os.getenv("TF_EMA_SLOW", "50"))
except ValueError:
    raise ValueError("TF_EMA_SLOW must be a valid integer")

try:
    TF_ATR_TARGET_MULT = float(os.getenv("TF_ATR_TARGET_MULT", "3.0"))
except ValueError:
    raise ValueError("TF_ATR_TARGET_MULT must be a valid number")

# ── ML Veto Filter (autonomous engine) ────────────────────────────────────────
ML_VETO_ENABLED = os.getenv("ML_VETO_ENABLED", "true").lower() == "true"

# ── Cooldown / Anti-Whipsaw ───────────────────────────────────────────────────
try:
    SIGNAL_COOLDOWN_SECONDS = int(os.getenv("SIGNAL_COOLDOWN_SECONDS", "0"))
except ValueError:
    raise ValueError("SIGNAL_COOLDOWN_SECONDS must be a valid integer")

# ── Multi-Timeframe Confirmation ──────────────────────────────────────────────
MTF_CONFIRMATION_ENABLED = os.getenv("MTF_CONFIRMATION_ENABLED", "true").lower() == "true"

# ── Mean Reversion Enhancements ───────────────────────────────────────────────
MR_VWAP_ENABLED = os.getenv("MR_VWAP_ENABLED", "true").lower() == "true"

try:
    MR_STOCH_RSI_PERIOD = int(os.getenv("MR_STOCH_RSI_PERIOD", "14"))
except ValueError:
    raise ValueError("MR_STOCH_RSI_PERIOD must be a valid integer")

try:
    MR_STOCH_RSI_OVERSOLD = float(os.getenv("MR_STOCH_RSI_OVERSOLD", "20.0"))
except ValueError:
    raise ValueError("MR_STOCH_RSI_OVERSOLD must be a valid number")

try:
    MR_STOCH_RSI_OVERBOUGHT = float(os.getenv("MR_STOCH_RSI_OVERBOUGHT", "80.0"))
except ValueError:
    raise ValueError("MR_STOCH_RSI_OVERBOUGHT must be a valid number")

try:
    MR_VOL_SPIKE_MULT = float(os.getenv("MR_VOL_SPIKE_MULT", "0.0"))
except ValueError:
    raise ValueError("MR_VOL_SPIKE_MULT must be a valid number")

# ── Momentum Breakout Enhancements ────────────────────────────────────────────
try:
    MB_SQUEEZE_LOOKBACK = int(os.getenv("MB_SQUEEZE_LOOKBACK", "20"))
except ValueError:
    raise ValueError("MB_SQUEEZE_LOOKBACK must be a valid integer")

try:
    MB_SQUEEZE_BB_MULT = float(os.getenv("MB_SQUEEZE_BB_MULT", "2.0"))
except ValueError:
    raise ValueError("MB_SQUEEZE_BB_MULT must be a valid number")

try:
    MB_SQUEEZE_KC_MULT = float(os.getenv("MB_SQUEEZE_KC_MULT", "1.5"))
except ValueError:
    raise ValueError("MB_SQUEEZE_KC_MULT must be a valid number")

try:
    MB_FALSE_BREAKOUT_BARS = int(os.getenv("MB_FALSE_BREAKOUT_BARS", "3"))
except ValueError:
    raise ValueError("MB_FALSE_BREAKOUT_BARS must be a valid integer")

try:
    MB_MIN_VOLUME_RATIO = float(os.getenv("MB_MIN_VOLUME_RATIO", "0.1"))
except ValueError:
    raise ValueError("MB_MIN_VOLUME_RATIO must be a valid number")

try:
    MB_RSI_PERIOD = int(os.getenv("MB_RSI_PERIOD", "14"))
except ValueError:
    raise ValueError("MB_RSI_PERIOD must be a valid integer")

# ── Momentum Breakout — Compression / Expansion (two-stage model) ──────────────
try:
    MB_ATR_PERCENTILE_LOOKBACK = int(os.getenv("MB_ATR_PERCENTILE_LOOKBACK", "50"))
except ValueError:
    raise ValueError("MB_ATR_PERCENTILE_LOOKBACK must be a valid integer")

try:
    MB_COMPRESSION_THRESHOLD = float(os.getenv("MB_COMPRESSION_THRESHOLD", "0.0"))
except ValueError:
    raise ValueError("MB_COMPRESSION_THRESHOLD must be a valid number")

try:
    MB_EXPANSION_VOLUME_RATIO = float(os.getenv("MB_EXPANSION_VOLUME_RATIO", "0.1"))
except ValueError:
    raise ValueError("MB_EXPANSION_VOLUME_RATIO must be a valid number")

try:
    MB_PARTIAL_TP_RISK_MULT = float(os.getenv("MB_PARTIAL_TP_RISK_MULT", "1.5"))
except ValueError:
    raise ValueError("MB_PARTIAL_TP_RISK_MULT must be a valid number")

# ── Trend Following Enhancements ──────────────────────────────────────────────
try:
    TF_ADX_PERIOD = int(os.getenv("TF_ADX_PERIOD", "14"))
except ValueError:
    raise ValueError("TF_ADX_PERIOD must be a valid integer")

try:
    TF_ADX_MIN_STRENGTH = float(os.getenv("TF_ADX_MIN_STRENGTH", "25.0"))
except ValueError:
    raise ValueError("TF_ADX_MIN_STRENGTH must be a valid number")

TF_PULLBACK_ENABLED = os.getenv("TF_PULLBACK_ENABLED", "true").lower() == "true"

try:
    TF_RSI_PERIOD = int(os.getenv("TF_RSI_PERIOD", "14"))
except ValueError:
    raise ValueError("TF_RSI_PERIOD must be a valid integer")

try:
    TF_RSI_EXHAUSTION_HIGH = float(os.getenv("TF_RSI_EXHAUSTION_HIGH", "75.0"))
except ValueError:
    raise ValueError("TF_RSI_EXHAUSTION_HIGH must be a valid number")

try:
    TF_RSI_EXHAUSTION_LOW = float(os.getenv("TF_RSI_EXHAUSTION_LOW", "25.0"))
except ValueError:
    raise ValueError("TF_RSI_EXHAUSTION_LOW must be a valid number")

# ── Adaptive Scan Intervals ──────────────────────────────────────────────────
ADAPTIVE_SCAN_ENABLED = os.getenv("ADAPTIVE_SCAN_ENABLED", "true").lower() == "true"

try:
    ADAPTIVE_SCAN_FAST_MULT = float(os.getenv("ADAPTIVE_SCAN_FAST_MULT", "0.5"))
except ValueError:
    raise ValueError("ADAPTIVE_SCAN_FAST_MULT must be a valid number")

try:
    ADAPTIVE_SCAN_SLOW_MULT = float(os.getenv("ADAPTIVE_SCAN_SLOW_MULT", "2.0"))
except ValueError:
    raise ValueError("ADAPTIVE_SCAN_SLOW_MULT must be a valid number")

# ── Alpha Upgrades: Limit Orders, Trailing Stops, Time Stops ──────────────
USE_LIMIT_ORDERS_MR = os.getenv("USE_LIMIT_ORDERS_MR", "true").lower() == "true"

# Trailing stop as a percentage of current price (e.g., 3.0 = 3%)
try:
    TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", "3.0"))
except ValueError:
    raise ValueError("TRAILING_STOP_PCT must be a valid number")

# Per-asset-class time stop hours (triggers when a position is open too long with minimal profit)
try:
    TIME_STOP_EQUITY_HOURS = int(os.getenv("TIME_STOP_EQUITY_HOURS", "4"))
except ValueError:
    raise ValueError("TIME_STOP_EQUITY_HOURS must be a valid integer")

try:
    TIME_STOP_CRYPTO_HOURS = int(os.getenv("TIME_STOP_CRYPTO_HOURS", "10"))
except ValueError:
    raise ValueError("TIME_STOP_CRYPTO_HOURS must be a valid integer")

try:
    TIME_STOP_COMMODITY_HOURS = int(os.getenv("TIME_STOP_COMMODITY_HOURS", "12"))
except ValueError:
    raise ValueError("TIME_STOP_COMMODITY_HOURS must be a valid integer")

# ── Macro Kill Switch, Slippage Caps & Scale-Out ───────────────────────────────
MACRO_FILTER_ENABLED = os.getenv("MACRO_FILTER_ENABLED", "true").lower() == "true"

try:
    SLIPPAGE_CAP_PCT = float(os.getenv("SLIPPAGE_CAP_PCT", "0.0015"))
except ValueError:
    raise ValueError("SLIPPAGE_CAP_PCT must be a valid number")

SCALE_OUT_ENABLED = os.getenv("SCALE_OUT_ENABLED", "true").lower() == "true"

try:
    SCALE_OUT_RR_RATIO = float(os.getenv("SCALE_OUT_RR_RATIO", "1.5"))
except ValueError:
    raise ValueError("SCALE_OUT_RR_RATIO must be a valid number")

try:
    UPGRADE_BUFFER_ATR_FRACTION = float(os.getenv("UPGRADE_BUFFER_ATR_FRACTION", "0.25"))
except ValueError:
    raise ValueError("UPGRADE_BUFFER_ATR_FRACTION must be a valid number")

ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")

# ── Bounds Validation ─────────────────────────────────────────────────────────
if SIGNAL_COOLDOWN_SECONDS < 0:
    raise ValueError(f"SIGNAL_COOLDOWN_SECONDS must be >= 0, got {SIGNAL_COOLDOWN_SECONDS}")
if ADAPTIVE_SCAN_FAST_MULT <= 0:
    raise ValueError(f"ADAPTIVE_SCAN_FAST_MULT must be > 0, got {ADAPTIVE_SCAN_FAST_MULT}")
if ADAPTIVE_SCAN_SLOW_MULT <= 0:
    raise ValueError(f"ADAPTIVE_SCAN_SLOW_MULT must be > 0, got {ADAPTIVE_SCAN_SLOW_MULT}")
if MR_STOCH_RSI_PERIOD < 2:
    raise ValueError(f"MR_STOCH_RSI_PERIOD must be >= 2, got {MR_STOCH_RSI_PERIOD}")
if MB_SQUEEZE_LOOKBACK < 2:
    raise ValueError(f"MB_SQUEEZE_LOOKBACK must be >= 2, got {MB_SQUEEZE_LOOKBACK}")
if TF_ADX_PERIOD < 2:
    raise ValueError(f"TF_ADX_PERIOD must be >= 2, got {TF_ADX_PERIOD}")
if TF_RSI_PERIOD < 2:
    raise ValueError(f"TF_RSI_PERIOD must be >= 2, got {TF_RSI_PERIOD}")
if MB_FALSE_BREAKOUT_BARS < 1:
    raise ValueError(f"MB_FALSE_BREAKOUT_BARS must be >= 1, got {MB_FALSE_BREAKOUT_BARS}")
if MB_RSI_PERIOD < 2:
    raise ValueError(f"MB_RSI_PERIOD must be >= 2, got {MB_RSI_PERIOD}")
if MB_ATR_PERCENTILE_LOOKBACK < 10:
    raise ValueError(f"MB_ATR_PERCENTILE_LOOKBACK must be >= 10, got {MB_ATR_PERCENTILE_LOOKBACK}")
if MB_COMPRESSION_THRESHOLD < 0 or MB_COMPRESSION_THRESHOLD > 100:
    raise ValueError(f"MB_COMPRESSION_THRESHOLD must be between 0 and 100, got {MB_COMPRESSION_THRESHOLD}")
if MB_EXPANSION_VOLUME_RATIO <= 0:
    raise ValueError(f"MB_EXPANSION_VOLUME_RATIO must be > 0, got {MB_EXPANSION_VOLUME_RATIO}")
if MB_PARTIAL_TP_RISK_MULT <= 0:
    raise ValueError(f"MB_PARTIAL_TP_RISK_MULT must be > 0, got {MB_PARTIAL_TP_RISK_MULT}")
if TRAILING_STOP_PCT < 0.5:
    raise ValueError(f"TRAILING_STOP_PCT must be >= 0.5, got {TRAILING_STOP_PCT}")
if TIME_STOP_EQUITY_HOURS < 1:
    raise ValueError(f"TIME_STOP_EQUITY_HOURS must be >= 1, got {TIME_STOP_EQUITY_HOURS}")
if TIME_STOP_CRYPTO_HOURS < 1:
    raise ValueError(f"TIME_STOP_CRYPTO_HOURS must be >= 1, got {TIME_STOP_CRYPTO_HOURS}")
if TIME_STOP_COMMODITY_HOURS < 1:
    raise ValueError(f"TIME_STOP_COMMODITY_HOURS must be >= 1, got {TIME_STOP_COMMODITY_HOURS}")

# Ensure additional numeric thresholds are > 0 where division-by-zero is possible
for _name, _val, _label in [
    ("SCAN_INTERVAL", SCAN_INTERVAL, ">= 1"),
    ("MAX_POSITIONS", MAX_POSITIONS, ">= 1"),
    ("MAX_OPEN_PORTFOLIO_RISK_PCT", MAX_OPEN_PORTFOLIO_RISK_PCT, "> 0"),
    ("MAX_CLUSTER_RISK_PCT", MAX_CLUSTER_RISK_PCT, "> 0"),
    ("MAX_DAILY_LOSS_PCT", MAX_DAILY_LOSS_PCT, "<= 0"),
    ("MAX_WEEKLY_LOSS_PCT", MAX_WEEKLY_LOSS_PCT, "<= 0"),
    ("MIN_EXPECTANCY_R", MIN_EXPECTANCY_R, "> 0"),
    ("MIN_EXPECTANCY_SAMPLES", MIN_EXPECTANCY_SAMPLES, ">= 1"),
    ("MAX_RISK_PCT", MAX_RISK_PCT, "> 0"),
    ("MAX_POSITION_PCT", MAX_POSITION_PCT, "> 0"),
    ("MAX_NOTIONAL", MAX_NOTIONAL, "> 0"),
    ("RISK_TIER_EQUITY_PCT", RISK_TIER_EQUITY_PCT, "> 0"),
    ("RISK_TIER_CRYPTO_PCT", RISK_TIER_CRYPTO_PCT, "> 0"),
    ("RISK_TIER_COMMODITY_PCT", RISK_TIER_COMMODITY_PCT, "> 0"),
    ("SPREAD_ATR_CAP_PCT", SPREAD_ATR_CAP_PCT, "> 0"),
    ("GAP_SLIPPAGE_BUFFER_PCT", GAP_SLIPPAGE_BUFFER_PCT, ">= 0"),
    ("BACKTEST_SLIPPAGE_FRICTION_PCT", BACKTEST_SLIPPAGE_FRICTION_PCT, ">= 0"),
    ("VOLATILITY_SHOCK_ATR_PCT", VOLATILITY_SHOCK_ATR_PCT, "> 0"),
    ("VOLATILITY_SHOCK_REDUCTION", VOLATILITY_SHOCK_REDUCTION, "> 0"),
    ("TP_BB_PERIOD", TP_BB_PERIOD, ">= 2"),
    ("TP_BB_STD", TP_BB_STD, "> 0"),
    ("TP_RSI_PERIOD", TP_RSI_PERIOD, ">= 2"),
    ("TP_VOL_SPIKE_MULT", TP_VOL_SPIKE_MULT, "> 0"),
    ("MR_BB_PERIOD", MR_BB_PERIOD, ">= 2"),
    ("MR_BB_STD", MR_BB_STD, "> 0"),
    ("MR_RSI_PERIOD", MR_RSI_PERIOD, ">= 2"),
    ("MB_DONCHIAN_PERIOD", MB_DONCHIAN_PERIOD, ">= 2"),
    ("MB_ADX_THRESHOLD", MB_ADX_THRESHOLD, "> 0"),
    ("MB_ATR_TARGET_MULT", MB_ATR_TARGET_MULT, "> 0"),
    ("MB_SQUEEZE_BB_MULT", MB_SQUEEZE_BB_MULT, "> 0"),
    ("MB_SQUEEZE_KC_MULT", MB_SQUEEZE_KC_MULT, "> 0"),
    ("MB_MIN_VOLUME_RATIO", MB_MIN_VOLUME_RATIO, "> 0"),
    ("MB_ATR_PERCENTILE_LOOKBACK", MB_ATR_PERCENTILE_LOOKBACK, ">= 10"),
    ("MB_COMPRESSION_THRESHOLD", MB_COMPRESSION_THRESHOLD, ">= 0"),
    ("MB_EXPANSION_VOLUME_RATIO", MB_EXPANSION_VOLUME_RATIO, "> 0"),
    ("MB_PARTIAL_TP_RISK_MULT", MB_PARTIAL_TP_RISK_MULT, "> 0"),
    ("MR_VOL_SPIKE_MULT", MR_VOL_SPIKE_MULT, ">= 0"),
    ("TF_EMA_FAST", TF_EMA_FAST, ">= 2"),
    ("TF_EMA_SLOW", TF_EMA_SLOW, ">= 2"),
    ("TF_ATR_TARGET_MULT", TF_ATR_TARGET_MULT, "> 0"),
    ("TF_ADX_MIN_STRENGTH", TF_ADX_MIN_STRENGTH, "> 0"),
    ("ORDER_TTL_HOURS", ORDER_TTL_HOURS, "> 0"),
    ("AI_RISK_MULTIPLIER_EQUITY", AI_RISK_MULTIPLIER_EQUITY, "> 0"),
    ("AI_RISK_MULTIPLIER_CRYPTO", AI_RISK_MULTIPLIER_CRYPTO, "> 0"),
    ("AI_RISK_MULTIPLIER_COMMODITY", AI_RISK_MULTIPLIER_COMMODITY, "> 0"),
    ("MAX_VIX_THRESHOLD", MAX_VIX_THRESHOLD, "> 0"),
]:
    op, limit_str = _label.split(' ')
    limit = float(limit_str)
    if op == ">" and not (_val > limit):
        raise ValueError(f"{_name} must be {_label}, got {_val}")
    elif op == ">=" and not (_val >= limit):
        raise ValueError(f"{_name} must be {_label}, got {_val}")
    elif op == "<=" and not (_val <= limit):
        raise ValueError(f"{_name} must be {_label}, got {_val}")

# ── Pre-Approved Regime Profiles (bounded, no arbitrary AI rewriting) ─────────
# The AI tuner selects among these pre-approved parameter sets based on
# quantifiable market features (SMA trend direction + ATR volatility).
# Each profile maps a regime name → {config_key: value} overrides.
REGIME_PROFILES = {
    "Equity": {
        "Bullish Calm": {
            "AI_RISK_MULTIPLIER_EQUITY": 1.0,
            "MR_RSI_OVERSOLD": 40.0,
            "MR_RSI_OVERBOUGHT": 60.0,
        },
        "Bullish Volatile": {
            "AI_RISK_MULTIPLIER_EQUITY": 0.8,
            "MR_RSI_OVERSOLD": 35.0,
            "MR_RSI_OVERBOUGHT": 65.0,
        },
        "Bearish Chop": {
            "AI_RISK_MULTIPLIER_EQUITY": 0.5,
            "MR_RSI_OVERSOLD": 30.0,
            "MR_RSI_OVERBOUGHT": 70.0,
        },
        "Bearish Volatile": {
            "AI_RISK_MULTIPLIER_EQUITY": 0.5,
            "MR_RSI_OVERSOLD": 25.0,
            "MR_RSI_OVERBOUGHT": 75.0,
        },
    },
    "Crypto": {
        "Bullish Calm": {
            "AI_RISK_MULTIPLIER_CRYPTO": 1.0,
            "MB_ADX_THRESHOLD": 20.0,
        },
        "Bullish Volatile": {
            "AI_RISK_MULTIPLIER_CRYPTO": 0.8,
            "MB_ADX_THRESHOLD": 25.0,
        },
        "Bearish Chop": {
            "AI_RISK_MULTIPLIER_CRYPTO": 0.5,
            "MB_ADX_THRESHOLD": 25.0,
        },
        "Bearish Volatile": {
            "AI_RISK_MULTIPLIER_CRYPTO": 0.5,
            "MB_ADX_THRESHOLD": 30.0,
        },
    },
    "Commodity": {
        "Bullish Calm": {
            "AI_RISK_MULTIPLIER_COMMODITY": 1.0,
            "TF_EMA_FAST": 20,
        },
        "Bullish Volatile": {
            "AI_RISK_MULTIPLIER_COMMODITY": 0.8,
            "TF_EMA_FAST": 20,
        },
        "Bearish Chop": {
            "AI_RISK_MULTIPLIER_COMMODITY": 0.5,
            "TF_EMA_FAST": 15,
        },
        "Bearish Volatile": {
            "AI_RISK_MULTIPLIER_COMMODITY": 0.5,
            "TF_EMA_FAST": 10,
        },
    },
}
