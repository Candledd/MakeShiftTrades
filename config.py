import os
import threading
import json
from dotenv import load_dotenv

# ── Global Config Lock for Thread Safety ───────────────────────────────────────
# Used by AITuner (and any other component) to serialize mutations to module-level
# config attributes, preventing race conditions when applying regime profiles.
config_lock = threading.Lock()

load_dotenv()

BEST_PARAMS = {}

# Load ML parameters
if os.path.exists("data/best_ml_params.json"):
    try:
        with open("data/best_ml_params.json", "r") as f:
            BEST_PARAMS.update(json.load(f))
    except Exception:
        pass
# Load Risk parameters (these will overwrite ML if there are duplicates, though they shouldn't overlap anymore)
if os.path.exists("data/best_risk_params.json"):
    try:
        with open("data/best_risk_params.json", "r") as f:
            BEST_PARAMS.update(json.load(f))
    except Exception:
        pass

def get_config_val(key, default):
    if key in BEST_PARAMS:
        return str(BEST_PARAMS[key])
    return os.getenv(key, default)

# ── Alpaca Paper Trading credentials ──────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# ── Risk Management ──────────────────────────────────────────────────────────
try:
    VIRTUAL_EQUITY = float(get_config_val("VIRTUAL_EQUITY", "0"))
except ValueError:
    VIRTUAL_EQUITY = 0.0

try:
    MAX_RISK_PCT = float(get_config_val("MAX_RISK_PCT", "0.04"))
except ValueError:
    raise ValueError("MAX_RISK_PCT must be a valid number")

try:
    MAX_POSITION_PCT = float(get_config_val("MAX_POSITION_PCT", "5.00"))
except ValueError:
    raise ValueError("MAX_POSITION_PCT must be a valid number")

try:
    MAX_NOTIONAL = float(get_config_val("MAX_NOTIONAL", "50000"))
except ValueError:
    raise ValueError("MAX_NOTIONAL must be a valid number")

try:
    MAX_POSITIONS = int(float(get_config_val("MAX_POSITIONS", "3")))
except ValueError:
    raise ValueError("MAX_POSITIONS must be a valid integer")

try:
    MAX_OPEN_PORTFOLIO_RISK_PCT = float(get_config_val("MAX_OPEN_PORTFOLIO_RISK_PCT", "0.10"))
except ValueError:
    raise ValueError("MAX_OPEN_PORTFOLIO_RISK_PCT must be a valid number")

try:
    MAX_CLUSTER_RISK_PCT = float(get_config_val("MAX_CLUSTER_RISK_PCT", "0.25"))
except ValueError:
    raise ValueError("MAX_CLUSTER_RISK_PCT must be a valid number")

try:
    MAX_DAILY_LOSS_PCT = float(get_config_val("MAX_DAILY_LOSS_PCT", "-0.05"))
except ValueError:
    raise ValueError("MAX_DAILY_LOSS_PCT must be a valid number")

try:
    MAX_WEEKLY_LOSS_PCT = float(get_config_val("MAX_WEEKLY_LOSS_PCT", "-0.12"))
except ValueError:
    raise ValueError("MAX_WEEKLY_LOSS_PCT must be a valid number")

try:
    MIN_EXPECTANCY_R = float(get_config_val("MIN_EXPECTANCY_R", "0.05"))
except ValueError:
    raise ValueError("MIN_EXPECTANCY_R must be a valid number")

try:
    MIN_EXPECTANCY_SAMPLES = int(float(get_config_val("MIN_EXPECTANCY_SAMPLES", "5")))
except ValueError:
    raise ValueError("MIN_EXPECTANCY_SAMPLES must be a valid integer")

try:
    EXPECTANCY_SOFT_BAND_1_MIN = int(float(get_config_val("EXPECTANCY_SOFT_BAND_1_MIN", "5")))
except ValueError:
    raise ValueError("EXPECTANCY_SOFT_BAND_1_MIN must be a valid integer")

try:
    EXPECTANCY_SOFT_BAND_1_MULT = float(get_config_val("EXPECTANCY_SOFT_BAND_1_MULT", "0.5"))
except ValueError:
    raise ValueError("EXPECTANCY_SOFT_BAND_1_MULT must be a valid number")

try:
    EXPECTANCY_SOFT_BAND_2_MIN = int(float(get_config_val("EXPECTANCY_SOFT_BAND_2_MIN", "20")))
except ValueError:
    raise ValueError("EXPECTANCY_SOFT_BAND_2_MIN must be a valid integer")

try:
    EXPECTANCY_SOFT_BAND_2_MULT = float(get_config_val("EXPECTANCY_SOFT_BAND_2_MULT", "0.25"))
except ValueError:
    raise ValueError("EXPECTANCY_SOFT_BAND_2_MULT must be a valid number")

try:
    EXPECTANCY_HARD_DISABLE_SAMPLES = int(float(get_config_val("EXPECTANCY_HARD_DISABLE_SAMPLES", "50")))
except ValueError:
    raise ValueError("EXPECTANCY_HARD_DISABLE_SAMPLES must be a valid integer")

try:
    MAX_VIX_THRESHOLD = float(get_config_val("MAX_VIX_THRESHOLD", "30.0"))
except ValueError:
    raise ValueError("MAX_VIX_THRESHOLD must be a valid number")

# ── Tiered Risk Per Asset Class ──────────────────────────────────────────────
# These override MAX_RISK_PCT when a ticker matches the corresponding asset set.
try:
    RISK_TIER_EQUITY_PCT = float(get_config_val("RISK_TIER_EQUITY_PCT", "0.04"))
except ValueError:
    raise ValueError("RISK_TIER_EQUITY_PCT must be a valid number")

try:
    RISK_TIER_CRYPTO_PCT = float(get_config_val("RISK_TIER_CRYPTO_PCT", "0.0050"))
except ValueError:
    raise ValueError("RISK_TIER_CRYPTO_PCT must be a valid number")

try:
    RISK_TIER_COMMODITY_PCT = float(get_config_val("RISK_TIER_COMMODITY_PCT", "0.0035"))
except ValueError:
    raise ValueError("RISK_TIER_COMMODITY_PCT must be a valid number")

# ── Spread / Liquidity Cap (ATR-based) ───────────────────────────────────────
# Trades where ATR/price exceeds this threshold are rejected (proxy for wide
# spreads or illiquid markets).
try:
    SPREAD_ATR_CAP_PCT = float(get_config_val("SPREAD_ATR_CAP_PCT", "0.05"))
except ValueError:
    raise ValueError("SPREAD_ATR_CAP_PCT must be a valid number")

# ── Gap / Slippage Buffer ────────────────────────────────────────────────────
# Notional is reduced by this fraction to cushion gap risk at open.
try:
    GAP_SLIPPAGE_BUFFER_PCT = float(get_config_val("GAP_SLIPPAGE_BUFFER_PCT", "0.001"))
except ValueError:
    raise ValueError("GAP_SLIPPAGE_BUFFER_PCT must be a valid number")

try:
    BACKTEST_SLIPPAGE_FRICTION_PCT = float(get_config_val("BACKTEST_SLIPPAGE_FRICTION_PCT", "0.0005"))
except ValueError:
    raise ValueError("BACKTEST_SLIPPAGE_FRICTION_PCT must be a valid number")

# ── Volatility Shock Adjustment ──────────────────────────────────────────────
# When ATR/price exceeds VOLATILITY_SHOCK_ATR_PCT, position size is multiplied
# by VOLATILITY_SHOCK_REDUCTION to account for sudden market volatility.
try:
    VOLATILITY_SHOCK_ATR_PCT = float(get_config_val("VOLATILITY_SHOCK_ATR_PCT", "0.03"))
except ValueError:
    raise ValueError("VOLATILITY_SHOCK_ATR_PCT must be a valid number")

try:
    VOLATILITY_SHOCK_REDUCTION = float(get_config_val("VOLATILITY_SHOCK_REDUCTION", "0.50"))
except ValueError:
    raise ValueError("VOLATILITY_SHOCK_REDUCTION must be a valid number")

# ── AI Regime Tuner — Sector Risk Multipliers ────────────────────────────────
# Used by risk_manager.py per-trade; tuned by AITuner via pre-approved profiles.
try:
    AI_RISK_MULTIPLIER_EQUITY = float(get_config_val("AI_RISK_MULTIPLIER_EQUITY", "1.0"))
except ValueError:
    raise ValueError("AI_RISK_MULTIPLIER_EQUITY must be a valid number")

try:
    AI_RISK_MULTIPLIER_CRYPTO = float(get_config_val("AI_RISK_MULTIPLIER_CRYPTO", "1.0"))
except ValueError:
    raise ValueError("AI_RISK_MULTIPLIER_CRYPTO must be a valid number")

try:
    AI_RISK_MULTIPLIER_GOLD = float(get_config_val("AI_RISK_MULTIPLIER_GOLD", "1.0"))
except ValueError:
    raise ValueError("AI_RISK_MULTIPLIER_GOLD must be a valid number")

try:
    AI_RISK_MULTIPLIER_BROAD_COMMODITY = float(get_config_val("AI_RISK_MULTIPLIER_BROAD_COMMODITY", "1.0"))
except ValueError:
    raise ValueError("AI_RISK_MULTIPLIER_BROAD_COMMODITY must be a valid number")

# ── Bot Operation ─────────────────────────────────────────────────────────────
try:
    SCAN_INTERVAL = int(float(get_config_val("SCAN_INTERVAL", "60")))
except ValueError:
    raise ValueError("SCAN_INTERVAL must be a valid integer")

DRY_RUN = get_config_val("DRY_RUN", "false").lower() == "true"
LOG_LEVEL = get_config_val("LOG_LEVEL", "INFO")

try:
    ORDER_TTL_HOURS = float(get_config_val("ORDER_TTL_HOURS", "2.0"))
except ValueError:
    raise ValueError("ORDER_TTL_HOURS must be a valid number")

# ── Trend Pullback Strategy (SPY, QQQ — 15m) ───────────────────────────────
try:
    TP_BB_PERIOD = int(float(get_config_val("TP_BB_PERIOD", "10")))
except ValueError:
    raise ValueError("TP_BB_PERIOD must be a valid integer")

try:
    TP_BB_STD = float(get_config_val("TP_BB_STD", "2.2080"))
except ValueError:
    raise ValueError("TP_BB_STD must be a valid number")

try:
    TP_STOP_MULT = float(get_config_val("TP_STOP_MULT", "1.5"))
except ValueError:
    raise ValueError("TP_STOP_MULT must be a valid number")

try:
    TP_PULLBACK_BUFFER = float(get_config_val("TP_PULLBACK_BUFFER", "1.0093"))
except ValueError:
    raise ValueError("TP_PULLBACK_BUFFER must be a valid number")

try:
    MR_TP_TARGET_MULT = float(get_config_val("MR_TP_TARGET_MULT", "2.3627"))
except ValueError:
    raise ValueError("MR_TP_TARGET_MULT must be a valid number")

try:
    MR_MIN_RR = float(get_config_val("MR_MIN_RR", "0.6"))
except ValueError:
    raise ValueError("MR_MIN_RR must be a valid number")

try:
    TP_MIN_RR = float(get_config_val("TP_MIN_RR", "1.2"))
except ValueError:
    raise ValueError("TP_MIN_RR must be a valid number")

try:
    TP_RSI_PERIOD = int(float(get_config_val("TP_RSI_PERIOD", "14")))
except ValueError:
    raise ValueError("TP_RSI_PERIOD must be a valid integer")

try:
    TP_VOL_SPIKE_MULT = float(get_config_val("TP_VOL_SPIKE_MULT", "0.5"))
except ValueError:
    raise ValueError("TP_VOL_SPIKE_MULT must be a valid number")

# ── VPIN (Volume-Synchronized Probability of Informed Trading) ─────────────
VPIN_WINDOW = 50
VPIN_MR_BLOCK_THRESHOLD = 0.75
VPIN_MB_BOOST_THRESHOLD = 0.70

# ── Volume Profile (Numba POC / Value Area) ──────────────────────────────────
try:
    VP_WINDOW = int(float(get_config_val("VP_WINDOW", "100")))
except ValueError:
    raise ValueError("VP_WINDOW must be a valid integer")

try:
    VP_NUM_BINS = int(float(get_config_val("VP_NUM_BINS", "24")))
except ValueError:
    raise ValueError("VP_NUM_BINS must be a valid integer")

try:
    VP_VA_THRESHOLD = float(get_config_val("VP_VA_THRESHOLD", "0.70"))
except ValueError:
    raise ValueError("VP_VA_THRESHOLD must be a valid number")

try:
    VP_POC_DISTANCE_THRESHOLD = float(get_config_val("VP_POC_DISTANCE_THRESHOLD", "0.003"))
except ValueError:
    raise ValueError("VP_POC_DISTANCE_THRESHOLD must be a valid number")

# ── Mean Reversion Strategy (SPY, QQQ — 15m) ─────────────────────────────────
try:
    MR_BB_PERIOD = int(float(get_config_val("MR_BB_PERIOD", "27")))
except ValueError:
    raise ValueError("MR_BB_PERIOD must be a valid integer")

try:
    MR_BB_STD = float(get_config_val("MR_BB_STD", "2"))
except ValueError:
    raise ValueError("MR_BB_STD must be a valid number")

try:
    MR_STOP_MULT = float(get_config_val("MR_STOP_MULT", "1.8852"))
except ValueError:
    raise ValueError("MR_STOP_MULT must be a valid number")

try:
    MR_RSI_PERIOD = int(float(get_config_val("MR_RSI_PERIOD", "10")))
except ValueError:
    raise ValueError("MR_RSI_PERIOD must be a valid integer")

try:
    MR_RSI_OVERSOLD = float(get_config_val("MR_RSI_OVERSOLD", "39.9400"))
except ValueError:
    raise ValueError("MR_RSI_OVERSOLD must be a valid number")

try:
    MR_RSI_OVERBOUGHT = float(get_config_val("MR_RSI_OVERBOUGHT", "78.3060"))
except ValueError:
    raise ValueError("MR_RSI_OVERBOUGHT must be a valid number")

# ── Momentum Breakout Strategy (BTC — 1h) ────────────────────────────────────
try:
    MB_DONCHIAN_PERIOD = int(float(get_config_val("MB_DONCHIAN_PERIOD", "48")))
except ValueError:
    raise ValueError("MB_DONCHIAN_PERIOD must be a valid integer")

try:
    MB_ADX_THRESHOLD = float(get_config_val("MB_ADX_THRESHOLD", "27.9925"))
except ValueError:
    raise ValueError("MB_ADX_THRESHOLD must be a valid number")

try:
    MB_ATR_TARGET_MULT = float(get_config_val("MB_ATR_TARGET_MULT", "2.5"))
except ValueError:
    raise ValueError("MB_ATR_TARGET_MULT must be a valid number")

# ── Trend Following Strategy (GLD, PDBC — 4h) ─────────────────────────────────
try:
    GLD_EMA_FAST = int(float(get_config_val("GLD_EMA_FAST", "20")))
except ValueError:
    raise ValueError("GLD_EMA_FAST must be a valid integer")

try:
    PDBC_EMA_FAST = int(float(get_config_val("PDBC_EMA_FAST", "20")))
except ValueError:
    raise ValueError("PDBC_EMA_FAST must be a valid integer")

try:
    TF_EMA_SLOW = int(float(get_config_val("TF_EMA_SLOW", "50")))
except ValueError:
    raise ValueError("TF_EMA_SLOW must be a valid integer")

try:
    TF_ATR_TARGET_MULT = float(get_config_val("TF_ATR_TARGET_MULT", "3.0"))
except ValueError:
    raise ValueError("TF_ATR_TARGET_MULT must be a valid number")

# ── Trend Following — GLD-specific ─────────────────────────────────────────
try:
    GLD_STOP_MULT = float(get_config_val("GLD_STOP_MULT", "3.0"))
except ValueError:
    raise ValueError("GLD_STOP_MULT must be a valid number")

GLD_TREND_FILTER = get_config_val("GLD_TREND_FILTER", "HTF")

GLD_PULLBACK_TRIGGER = get_config_val("GLD_PULLBACK_TRIGGER", "enabled")

# ── Trend Following — PDBC-specific ────────────────────────────────────────
try:
    PDBC_STOP_MULT = float(get_config_val("PDBC_STOP_MULT", "2.0"))
except ValueError:
    raise ValueError("PDBC_STOP_MULT must be a valid number")

try:
    PDBC_ADX_MIN = float(get_config_val("PDBC_ADX_MIN", "25.0"))
except ValueError:
    raise ValueError("PDBC_ADX_MIN must be a valid number")

try:
    PDBC_RANGE_EXPANSION_THRESHOLD = float(get_config_val("PDBC_RANGE_EXPANSION_THRESHOLD", "0.75"))
except ValueError:
    raise ValueError("PDBC_RANGE_EXPANSION_THRESHOLD must be a valid number")

# ── ML Veto Filter (autonomous engine) ────────────────────────────────────────
ML_VETO_ENABLED = get_config_val("ML_VETO_ENABLED", "true").lower() == "true"

try:
    ML_VETO_THRESHOLD = float(get_config_val("ML_VETO_THRESHOLD", "0.3"))
except ValueError:
    raise ValueError("ML_VETO_THRESHOLD must be a valid number")

try:
    ML_MILD_DISAGREEMENT_THRESHOLD = float(get_config_val("ML_MILD_DISAGREEMENT_THRESHOLD", "0.45"))
except ValueError:
    raise ValueError("ML_MILD_DISAGREEMENT_THRESHOLD must be a valid number")

try:
    ML_MILD_DISAGREEMENT_SCALING = float(get_config_val("ML_MILD_DISAGREEMENT_SCALING", "0.5"))
except ValueError:
    raise ValueError("ML_MILD_DISAGREEMENT_SCALING must be a valid number")

try:
    ML_AGREEMENT_THRESHOLD = float(get_config_val("ML_AGREEMENT_THRESHOLD", "0.6"))
except ValueError:
    raise ValueError("ML_AGREEMENT_THRESHOLD must be a valid number")

try:
    ML_AGREEMENT_BOOST = float(get_config_val("ML_AGREEMENT_BOOST", "1.2"))
except ValueError:
    raise ValueError("ML_AGREEMENT_BOOST must be a valid number")

# ── Cooldown / Anti-Whipsaw ───────────────────────────────────────────────────
try:
    SIGNAL_COOLDOWN_SECONDS = int(float(get_config_val("SIGNAL_COOLDOWN_SECONDS", "0")))
except ValueError:
    raise ValueError("SIGNAL_COOLDOWN_SECONDS must be a valid integer")

# ── Multi-Timeframe Confirmation ──────────────────────────────────────────────
MTF_CONFIRMATION_ENABLED = get_config_val("MTF_CONFIRMATION_ENABLED", "true").lower() == "true"

# ── Mean Reversion Enhancements ───────────────────────────────────────────────
MR_VWAP_ENABLED = get_config_val("MR_VWAP_ENABLED", "true").lower() == "true"

try:
    MR_STOCH_RSI_PERIOD = int(float(get_config_val("MR_STOCH_RSI_PERIOD", "14")))
except ValueError:
    raise ValueError("MR_STOCH_RSI_PERIOD must be a valid integer")

try:
    MR_STOCH_RSI_OVERSOLD = float(get_config_val("MR_STOCH_RSI_OVERSOLD", "20.0"))
except ValueError:
    raise ValueError("MR_STOCH_RSI_OVERSOLD must be a valid number")

try:
    MR_STOCH_RSI_OVERBOUGHT = float(get_config_val("MR_STOCH_RSI_OVERBOUGHT", "80.0"))
except ValueError:
    raise ValueError("MR_STOCH_RSI_OVERBOUGHT must be a valid number")

try:
    MR_VOL_SPIKE_MULT = float(get_config_val("MR_VOL_SPIKE_MULT", "0.0"))
except ValueError:
    raise ValueError("MR_VOL_SPIKE_MULT must be a valid number")

# ── Momentum Breakout Enhancements ────────────────────────────────────────────
try:
    MB_SQUEEZE_LOOKBACK = int(float(get_config_val("MB_SQUEEZE_LOOKBACK", "20")))
except ValueError:
    raise ValueError("MB_SQUEEZE_LOOKBACK must be a valid integer")

try:
    MB_SQUEEZE_BB_MULT = float(get_config_val("MB_SQUEEZE_BB_MULT", "2.0"))
except ValueError:
    raise ValueError("MB_SQUEEZE_BB_MULT must be a valid number")

try:
    MB_SQUEEZE_KC_MULT = float(get_config_val("MB_SQUEEZE_KC_MULT", "1.5"))
except ValueError:
    raise ValueError("MB_SQUEEZE_KC_MULT must be a valid number")

try:
    MB_MIN_VOLUME_RATIO = float(get_config_val("MB_MIN_VOLUME_RATIO", "0.1"))
except ValueError:
    raise ValueError("MB_MIN_VOLUME_RATIO must be a valid number")

try:
    MB_RSI_PERIOD = int(float(get_config_val("MB_RSI_PERIOD", "14")))
except ValueError:
    raise ValueError("MB_RSI_PERIOD must be a valid integer")

try:
    MB_BB_PERIOD = int(float(get_config_val("MB_BB_PERIOD", "20")))
except ValueError:
    raise ValueError("MB_BB_PERIOD must be a valid integer")

try:
    MB_RVOL_LOOKBACK = int(float(get_config_val("MB_RVOL_LOOKBACK", "21")))
except ValueError:
    raise ValueError("MB_RVOL_LOOKBACK must be a valid integer")

try:
    MB_HIGH_VOL_RATIO = float(get_config_val("MB_HIGH_VOL_RATIO", "2.5"))
except ValueError:
    raise ValueError("MB_HIGH_VOL_RATIO must be a valid number")

try:
    MB_MED_VOL_RATIO = float(get_config_val("MB_MED_VOL_RATIO", "2.0"))
except ValueError:
    raise ValueError("MB_MED_VOL_RATIO must be a valid number")

# ── Momentum Breakout — Compression / Expansion (two-stage model) ──────────────
try:
    MB_ATR_PERCENTILE_LOOKBACK = int(float(get_config_val("MB_ATR_PERCENTILE_LOOKBACK", "50")))
except ValueError:
    raise ValueError("MB_ATR_PERCENTILE_LOOKBACK must be a valid integer")

try:
    MB_COMPRESSION_THRESHOLD = float(get_config_val("MB_COMPRESSION_THRESHOLD", "55.0"))
except ValueError:
    raise ValueError("MB_COMPRESSION_THRESHOLD must be a valid number")

try:
    MB_EXPANSION_VOLUME_RATIO = float(get_config_val("MB_EXPANSION_VOLUME_RATIO", "0.1"))
except ValueError:
    raise ValueError("MB_EXPANSION_VOLUME_RATIO must be a valid number")

try:
    MB_PARTIAL_TP_RISK_MULT = float(get_config_val("MB_PARTIAL_TP_RISK_MULT", "1.5"))
except ValueError:
    raise ValueError("MB_PARTIAL_TP_RISK_MULT must be a valid number")

# ── Trend Following Enhancements ──────────────────────────────────────────────
try:
    TF_ADX_PERIOD = int(float(get_config_val("TF_ADX_PERIOD", "14")))
except ValueError:
    raise ValueError("TF_ADX_PERIOD must be a valid integer")

try:
    TF_ADX_MIN_STRENGTH = float(get_config_val("TF_ADX_MIN_STRENGTH", "25.0"))
except ValueError:
    raise ValueError("TF_ADX_MIN_STRENGTH must be a valid number")

TF_PULLBACK_ENABLED = get_config_val("TF_PULLBACK_ENABLED", "true").lower() == "true"

try:
    TF_RSI_PERIOD = int(float(get_config_val("TF_RSI_PERIOD", "14")))
except ValueError:
    raise ValueError("TF_RSI_PERIOD must be a valid integer")

try:
    TF_RSI_EXHAUSTION_HIGH = float(get_config_val("TF_RSI_EXHAUSTION_HIGH", "75.0"))
except ValueError:
    raise ValueError("TF_RSI_EXHAUSTION_HIGH must be a valid number")

try:
    TF_RSI_EXHAUSTION_LOW = float(get_config_val("TF_RSI_EXHAUSTION_LOW", "25.0"))
except ValueError:
    raise ValueError("TF_RSI_EXHAUSTION_LOW must be a valid number")

# ── Adaptive Scan Intervals ──────────────────────────────────────────────────
ADAPTIVE_SCAN_ENABLED = get_config_val("ADAPTIVE_SCAN_ENABLED", "true").lower() == "true"

try:
    ADAPTIVE_SCAN_FAST_MULT = float(get_config_val("ADAPTIVE_SCAN_FAST_MULT", "0.5"))
except ValueError:
    raise ValueError("ADAPTIVE_SCAN_FAST_MULT must be a valid number")

try:
    ADAPTIVE_SCAN_SLOW_MULT = float(get_config_val("ADAPTIVE_SCAN_SLOW_MULT", "2.0"))
except ValueError:
    raise ValueError("ADAPTIVE_SCAN_SLOW_MULT must be a valid number")

# ── Alpha Upgrades: Limit Orders, Trailing Stops, Time Stops ──────────────
USE_LIMIT_ORDERS_MR = get_config_val("USE_LIMIT_ORDERS_MR", "true").lower() == "true"

# ── Strategy-specific Trailing Stops ─────────────────────────────────────
# Each strategy uses its own trailing logic type and ATR multiplier.
# This replaces the old single TRAILING_STOP_PCT (percentage-based) with
# per-strategy algorithms aligned with codex.md item 14.
#
# Logic types: vwap, sma20_or_ema, donchian, chandelier, atr, breakeven_only

MR_TRAILING_STOP_LOGIC = get_config_val("MR_TRAILING_STOP_LOGIC", "vwap")
TP_TRAILING_STOP_LOGIC = get_config_val("TP_TRAILING_STOP_LOGIC", "sma20_or_ema")
MB_TRAILING_STOP_LOGIC = get_config_val("MB_TRAILING_STOP_LOGIC", "donchian")
TF_TRAILING_STOP_LOGIC = get_config_val("TF_TRAILING_STOP_LOGIC", "atr")

# Mean reversion: tight trail (exit at VWAP / SMA quickly, 0.5–1.0 ATR)
try:
    MR_TRAIL_ATR_MULT = float(get_config_val("MR_TRAIL_ATR_MULT", "0.8"))
except ValueError:
    raise ValueError("MR_TRAIL_ATR_MULT must be a valid number")

# Trend pullback: moderate trail below higher low / EMA (1.5–2.5 ATR)
try:
    TP_TRAIL_ATR_MULT = float(get_config_val("TP_TRAIL_ATR_MULT", "2.0"))
except ValueError:
    raise ValueError("TP_TRAIL_ATR_MULT must be a valid number")

# BTC momentum breakout: wider Donchian / chandelier (2.5–4.0 ATR)
try:
    MB_TRAIL_ATR_MULT = float(get_config_val("MB_TRAIL_ATR_MULT", "3.0"))
except ValueError:
    raise ValueError("MB_TRAIL_ATR_MULT must be a valid number")

# Trend following GLD: wider structure / ATR trail (3.0–5.0 ATR)
try:
    GLD_TRAIL_ATR_MULT = float(get_config_val("GLD_TRAIL_ATR_MULT", "3.5"))
except ValueError:
    raise ValueError("GLD_TRAIL_ATR_MULT must be a valid number")

# PDBC: event/volatility-aware ATR trail (2.0–3.5 ATR)
try:
    PDBC_TRAIL_ATR_MULT = float(get_config_val("PDBC_TRAIL_ATR_MULT", "2.5"))
except ValueError:
    raise ValueError("PDBC_TRAIL_ATR_MULT must be a valid number")

# PDBC event tightening factor: during macro events, multiply ATR distance
# by this factor (e.g., 0.5 = tighten to half the normal distance)
try:
    PDBC_EVENT_TIGHTEN_FACTOR = float(get_config_val("PDBC_EVENT_TIGHTEN_FACTOR", "0.5"))
except ValueError:
    raise ValueError("PDBC_EVENT_TIGHTEN_FACTOR must be a valid number")

# Per-asset-class time stop hours (triggers when a position is open too long with minimal profit)
try:
    TIME_STOP_EQUITY_HOURS = int(float(get_config_val("TIME_STOP_EQUITY_HOURS", "4")))
except ValueError:
    raise ValueError("TIME_STOP_EQUITY_HOURS must be a valid integer")

try:
    TIME_STOP_CRYPTO_HOURS = int(float(get_config_val("TIME_STOP_CRYPTO_HOURS", "10")))
except ValueError:
    raise ValueError("TIME_STOP_CRYPTO_HOURS must be a valid integer")

try:
    TIME_STOP_COMMODITY_HOURS = int(float(get_config_val("TIME_STOP_COMMODITY_HOURS", "12")))
except ValueError:
    raise ValueError("TIME_STOP_COMMODITY_HOURS must be a valid integer")

# ── Macro Kill Switch, Slippage Caps & Scale-Out ───────────────────────────────
MACRO_FILTER_ENABLED = get_config_val("MACRO_FILTER_ENABLED", "true").lower() == "true"

try:
    SLIPPAGE_CAP_PCT = float(get_config_val("SLIPPAGE_CAP_PCT", "0.0015"))
except ValueError:
    raise ValueError("SLIPPAGE_CAP_PCT must be a valid number")

SCALE_OUT_ENABLED = get_config_val("SCALE_OUT_ENABLED", "true").lower() == "true"

try:
    SCALE_OUT_RR_RATIO = float(get_config_val("SCALE_OUT_RR_RATIO", "1.5"))
except ValueError:
    raise ValueError("SCALE_OUT_RR_RATIO must be a valid number")

try:
    UPGRADE_BUFFER_ATR_FRACTION = float(get_config_val("UPGRADE_BUFFER_ATR_FRACTION", "0.25"))
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
if MR_TRAIL_ATR_MULT <= 0:
    raise ValueError(f"MR_TRAIL_ATR_MULT must be > 0, got {MR_TRAIL_ATR_MULT}")
if TP_TRAIL_ATR_MULT <= 0:
    raise ValueError(f"TP_TRAIL_ATR_MULT must be > 0, got {TP_TRAIL_ATR_MULT}")
if MB_TRAIL_ATR_MULT <= 0:
    raise ValueError(f"MB_TRAIL_ATR_MULT must be > 0, got {MB_TRAIL_ATR_MULT}")
if GLD_TRAIL_ATR_MULT <= 0:
    raise ValueError(f"GLD_TRAIL_ATR_MULT must be > 0, got {GLD_TRAIL_ATR_MULT}")
if PDBC_TRAIL_ATR_MULT <= 0:
    raise ValueError(f"PDBC_TRAIL_ATR_MULT must be > 0, got {PDBC_TRAIL_ATR_MULT}")
if PDBC_EVENT_TIGHTEN_FACTOR <= 0:
    raise ValueError(f"PDBC_EVENT_TIGHTEN_FACTOR must be > 0, got {PDBC_EVENT_TIGHTEN_FACTOR}")
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
    ("EXPECTANCY_SOFT_BAND_1_MIN", EXPECTANCY_SOFT_BAND_1_MIN, ">= 1"),
    ("EXPECTANCY_SOFT_BAND_1_MULT", EXPECTANCY_SOFT_BAND_1_MULT, "> 0"),
    ("EXPECTANCY_SOFT_BAND_2_MIN", EXPECTANCY_SOFT_BAND_2_MIN, ">= 1"),
    ("EXPECTANCY_SOFT_BAND_2_MULT", EXPECTANCY_SOFT_BAND_2_MULT, "> 0"),
    ("EXPECTANCY_HARD_DISABLE_SAMPLES", EXPECTANCY_HARD_DISABLE_SAMPLES, ">= 1"),
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
    ("MB_DONCHIAN_PERIOD", MB_DONCHIAN_PERIOD, ">= 10"),
    ("MB_ADX_THRESHOLD", MB_ADX_THRESHOLD, "> 0"),
    ("MB_ATR_TARGET_MULT", MB_ATR_TARGET_MULT, "> 0"),
    ("MB_SQUEEZE_BB_MULT", MB_SQUEEZE_BB_MULT, "> 0"),
    ("MB_SQUEEZE_KC_MULT", MB_SQUEEZE_KC_MULT, "> 0"),
    ("MB_MIN_VOLUME_RATIO", MB_MIN_VOLUME_RATIO, "> 0"),
    ("MB_BB_PERIOD", MB_BB_PERIOD, ">= 2"),
    ("MB_RVOL_LOOKBACK", MB_RVOL_LOOKBACK, ">= 5"),
    ("MB_HIGH_VOL_RATIO", MB_HIGH_VOL_RATIO, "> 0"),
    ("MB_MED_VOL_RATIO", MB_MED_VOL_RATIO, "> 0"),
    ("MB_ATR_PERCENTILE_LOOKBACK", MB_ATR_PERCENTILE_LOOKBACK, ">= 10"),
    ("MB_COMPRESSION_THRESHOLD", MB_COMPRESSION_THRESHOLD, ">= 0"),
    ("MB_EXPANSION_VOLUME_RATIO", MB_EXPANSION_VOLUME_RATIO, "> 0"),
    ("MB_PARTIAL_TP_RISK_MULT", MB_PARTIAL_TP_RISK_MULT, "> 0"),
    ("MR_VOL_SPIKE_MULT", MR_VOL_SPIKE_MULT, ">= 0"),
    ("GLD_EMA_FAST", GLD_EMA_FAST, ">= 2"),
    ("PDBC_EMA_FAST", PDBC_EMA_FAST, ">= 2"),
    ("TF_EMA_SLOW", TF_EMA_SLOW, ">= 2"),
    ("TF_ATR_TARGET_MULT", TF_ATR_TARGET_MULT, "> 0"),
    ("TF_ADX_MIN_STRENGTH", TF_ADX_MIN_STRENGTH, "> 0"),
    ("ORDER_TTL_HOURS", ORDER_TTL_HOURS, "> 0"),
    ("AI_RISK_MULTIPLIER_EQUITY", AI_RISK_MULTIPLIER_EQUITY, "> 0"),
    ("AI_RISK_MULTIPLIER_CRYPTO", AI_RISK_MULTIPLIER_CRYPTO, "> 0"),
    ("AI_RISK_MULTIPLIER_GOLD", AI_RISK_MULTIPLIER_GOLD, "> 0"),
    ("AI_RISK_MULTIPLIER_BROAD_COMMODITY", AI_RISK_MULTIPLIER_BROAD_COMMODITY, "> 0"),
    ("MAX_VIX_THRESHOLD", MAX_VIX_THRESHOLD, "> 0"),
    ("GLD_STOP_MULT", GLD_STOP_MULT, "> 0"),
    ("PDBC_STOP_MULT", PDBC_STOP_MULT, "> 0"),
    ("PDBC_ADX_MIN", PDBC_ADX_MIN, "> 0"),
    ("PDBC_RANGE_EXPANSION_THRESHOLD", PDBC_RANGE_EXPANSION_THRESHOLD, "> 0"),
    ("ML_VETO_THRESHOLD", ML_VETO_THRESHOLD, "> 0"),
    ("ML_MILD_DISAGREEMENT_THRESHOLD", ML_MILD_DISAGREEMENT_THRESHOLD, "> 0"),
    ("ML_MILD_DISAGREEMENT_SCALING", ML_MILD_DISAGREEMENT_SCALING, "> 0"),
    ("ML_AGREEMENT_THRESHOLD", ML_AGREEMENT_THRESHOLD, "> 0"),
    ("ML_AGREEMENT_BOOST", ML_AGREEMENT_BOOST, "> 0"),
    ("VP_WINDOW", VP_WINDOW, ">= 10"),
    ("VP_NUM_BINS", VP_NUM_BINS, ">= 4"),
    ("VP_VA_THRESHOLD", VP_VA_THRESHOLD, "> 0"),
    ("VP_VA_THRESHOLD", VP_VA_THRESHOLD, "< 1"),
    ("VP_POC_DISTANCE_THRESHOLD", VP_POC_DISTANCE_THRESHOLD, "> 0"),
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
            "ROUTING_TREND_PULLBACK": "ENABLED",
            "ROUTING_MEAN_REVERSION": "LONG_ONLY_HIGH_QUALITY",
        },
        "Range-Bound Calm": {
            "AI_RISK_MULTIPLIER_EQUITY": 1.0,
            "MR_RSI_OVERSOLD": 30.0,
            "MR_RSI_OVERBOUGHT": 70.0,
            "ROUTING_TREND_PULLBACK": "REDUCED",
            "ROUTING_MEAN_REVERSION": "BOTH",
        },
        "Bullish Volatile": {
            "AI_RISK_MULTIPLIER_EQUITY": 0.8,
            "MR_RSI_OVERSOLD": 35.0,
            "MR_RSI_OVERBOUGHT": 65.0,
            "ROUTING_TREND_PULLBACK": "ENABLED",
            "ROUTING_MEAN_REVERSION": "LONG_ONLY",
        },
        "Bearish Chop": {
            "AI_RISK_MULTIPLIER_EQUITY": 0.5,
            "MR_RSI_OVERSOLD": 30.0,
            "MR_RSI_OVERBOUGHT": 70.0,
            "ROUTING_TREND_PULLBACK": "DISABLED",
            "ROUTING_MEAN_REVERSION": "REDUCED",
        },
        "Bearish Volatile": {
            "AI_RISK_MULTIPLIER_EQUITY": 0.5,
            "MR_RSI_OVERSOLD": 25.0,
            "MR_RSI_OVERBOUGHT": 75.0,
            "ROUTING_TREND_PULLBACK": "SHORT_ONLY",
            "ROUTING_MEAN_REVERSION": "TINY_FAILED_EXTENSIONS_ONLY",
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
    "Gold": {
        "Bullish Calm": {
            "AI_RISK_MULTIPLIER_GOLD": 1.0,
            "GLD_EMA_FAST": 20,
        },
        "Bullish Volatile": {
            "AI_RISK_MULTIPLIER_GOLD": 0.8,
            "GLD_EMA_FAST": 20,
        },
        "Range-Bound Calm": {
            "AI_RISK_MULTIPLIER_GOLD": 1.0,
            "GLD_EMA_FAST": 15,
        },
        "Bearish Chop": {
            "AI_RISK_MULTIPLIER_GOLD": 0.5,
            "GLD_EMA_FAST": 15,
        },
        "Bearish Volatile": {
            "AI_RISK_MULTIPLIER_GOLD": 0.5,
            "GLD_EMA_FAST": 10,
        },
    },
    "Broad Commodity": {
        "Bullish Calm": {
            "AI_RISK_MULTIPLIER_BROAD_COMMODITY": 1.0,
            "PDBC_EMA_FAST": 20,
        },
        "Bullish Volatile": {
            "AI_RISK_MULTIPLIER_BROAD_COMMODITY": 0.8,
            "PDBC_EMA_FAST": 20,
        },
        "Range-Bound Calm": {
            "AI_RISK_MULTIPLIER_BROAD_COMMODITY": 1.0,
            "PDBC_EMA_FAST": 15,
        },
        "Bearish Chop": {
            "AI_RISK_MULTIPLIER_BROAD_COMMODITY": 0.5,
            "PDBC_EMA_FAST": 15,
        },
        "Bearish Volatile": {
            "AI_RISK_MULTIPLIER_BROAD_COMMODITY": 0.5,
            "PDBC_EMA_FAST": 10,
        },
    },
}
