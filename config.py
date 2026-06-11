import os
from dotenv import load_dotenv

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
    MAX_RISK_PCT = float(os.getenv("MAX_RISK_PCT", "0.05"))
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

# ── Mean Reversion Strategy (SPY, QQQ — 15m) ─────────────────────────────────
try:
    MR_BB_PERIOD = int(os.getenv("MR_BB_PERIOD", "20"))
except ValueError:
    raise ValueError("MR_BB_PERIOD must be a valid integer")

try:
    MR_BB_STD = float(os.getenv("MR_BB_STD", "2.0"))
except ValueError:
    raise ValueError("MR_BB_STD must be a valid number")

try:
    MR_RSI_PERIOD = int(os.getenv("MR_RSI_PERIOD", "14"))
except ValueError:
    raise ValueError("MR_RSI_PERIOD must be a valid integer")

try:
    MR_RSI_OVERSOLD = float(os.getenv("MR_RSI_OVERSOLD", "35"))
except ValueError:
    raise ValueError("MR_RSI_OVERSOLD must be a valid number")

try:
    MR_RSI_OVERBOUGHT = float(os.getenv("MR_RSI_OVERBOUGHT", "65"))
except ValueError:
    raise ValueError("MR_RSI_OVERBOUGHT must be a valid number")

# ── Momentum Breakout Strategy (BTC — 1h) ────────────────────────────────────
try:
    MB_DONCHIAN_PERIOD = int(os.getenv("MB_DONCHIAN_PERIOD", "20"))
except ValueError:
    raise ValueError("MB_DONCHIAN_PERIOD must be a valid integer")

try:
    MB_ADX_THRESHOLD = float(os.getenv("MB_ADX_THRESHOLD", "15.0"))
except ValueError:
    raise ValueError("MB_ADX_THRESHOLD must be a valid number")

try:
    MB_ATR_TARGET_MULT = float(os.getenv("MB_ATR_TARGET_MULT", "2.5"))
except ValueError:
    raise ValueError("MB_ATR_TARGET_MULT must be a valid number")

# ── Trend Following Strategy (GLD, USO — 4h) ─────────────────────────────────
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
    SIGNAL_COOLDOWN_SECONDS = int(os.getenv("SIGNAL_COOLDOWN_SECONDS", "1800"))
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
    MR_VOL_SPIKE_MULT = float(os.getenv("MR_VOL_SPIKE_MULT", "1.1"))
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
    MB_MIN_VOLUME_RATIO = float(os.getenv("MB_MIN_VOLUME_RATIO", "1.0"))
except ValueError:
    raise ValueError("MB_MIN_VOLUME_RATIO must be a valid number")

try:
    MB_RSI_PERIOD = int(os.getenv("MB_RSI_PERIOD", "14"))
except ValueError:
    raise ValueError("MB_RSI_PERIOD must be a valid integer")

# ── Trend Following Enhancements ──────────────────────────────────────────────
try:
    TF_ADX_PERIOD = int(os.getenv("TF_ADX_PERIOD", "14"))
except ValueError:
    raise ValueError("TF_ADX_PERIOD must be a valid integer")

try:
    TF_ADX_MIN_STRENGTH = float(os.getenv("TF_ADX_MIN_STRENGTH", "20.0"))
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

try:
    TRAILING_STOP_ATR_MULT = float(os.getenv("TRAILING_STOP_ATR_MULT", "3.0"))
except ValueError:
    raise ValueError("TRAILING_STOP_ATR_MULT must be a valid number")

try:
    TIME_STOP_HOURS = int(os.getenv("TIME_STOP_HOURS", "10"))
except ValueError:
    raise ValueError("TIME_STOP_HOURS must be a valid integer")

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
if TRAILING_STOP_ATR_MULT < 0.5:
    raise ValueError(f"TRAILING_STOP_ATR_MULT must be >= 0.5, got {TRAILING_STOP_ATR_MULT}")
if TIME_STOP_HOURS < 1:
    raise ValueError(f"TIME_STOP_HOURS must be >= 1, got {TIME_STOP_HOURS}")

# Ensure additional numeric thresholds are > 0 where division-by-zero is possible
for _name, _val, _label in [
    ("SCAN_INTERVAL", SCAN_INTERVAL, ">= 1"),
    ("MAX_POSITIONS", MAX_POSITIONS, ">= 1"),
    ("MAX_RISK_PCT", MAX_RISK_PCT, "> 0"),
    ("MAX_POSITION_PCT", MAX_POSITION_PCT, "> 0"),
    ("MAX_NOTIONAL", MAX_NOTIONAL, "> 0"),
    ("MR_BB_PERIOD", MR_BB_PERIOD, ">= 2"),
    ("MR_BB_STD", MR_BB_STD, "> 0"),
    ("MR_RSI_PERIOD", MR_RSI_PERIOD, ">= 2"),
    ("MB_DONCHIAN_PERIOD", MB_DONCHIAN_PERIOD, ">= 2"),
    ("MB_ADX_THRESHOLD", MB_ADX_THRESHOLD, "> 0"),
    ("MB_ATR_TARGET_MULT", MB_ATR_TARGET_MULT, "> 0"),
    ("MB_SQUEEZE_BB_MULT", MB_SQUEEZE_BB_MULT, "> 0"),
    ("MB_SQUEEZE_KC_MULT", MB_SQUEEZE_KC_MULT, "> 0"),
    ("MB_MIN_VOLUME_RATIO", MB_MIN_VOLUME_RATIO, "> 0"),
    ("MR_VOL_SPIKE_MULT", MR_VOL_SPIKE_MULT, "> 0"),
    ("TF_EMA_FAST", TF_EMA_FAST, ">= 2"),
    ("TF_EMA_SLOW", TF_EMA_SLOW, ">= 2"),
    ("TF_ATR_TARGET_MULT", TF_ATR_TARGET_MULT, "> 0"),
    ("TF_ADX_MIN_STRENGTH", TF_ADX_MIN_STRENGTH, "> 0"),
    ("ORDER_TTL_HOURS", ORDER_TTL_HOURS, "> 0"),
]:
    if _label.startswith(">= 1") and _val < 1:
        raise ValueError(f"{_name} must be {_label}, got {_val}")
    elif _label.startswith(">= 2") and _val < 2:
        raise ValueError(f"{_name} must be {_label}, got {_val}")
    elif _label.startswith("> 0") and _val <= 0:
        raise ValueError(f"{_name} must be {_label}, got {_val}")
