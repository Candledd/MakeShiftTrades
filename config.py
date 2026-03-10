import os
from dotenv import load_dotenv

load_dotenv()

# ── Broker credentials ────────────────────────────────────────────────────────
API_KEY    = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

# ── Instrument ────────────────────────────────────────────────────────────────
SYMBOL = os.getenv("SYMBOL", "SPY")

# ── Risk management ───────────────────────────────────────────────────────────
try:
    MAX_TRADE_AMOUNT = float(os.getenv("MAX_TRADE_AMOUNT", "1000"))
except ValueError:
    raise ValueError("MAX_TRADE_AMOUNT must be a valid number")

DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

try:
    MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "3"))
except ValueError:
    raise ValueError("MAX_TRADES_PER_DAY must be a valid integer")

# ── SMC Strategy parameters ───────────────────────────────────────────────────
# Candle interval for signal generation: 1m, 2m, 5m, 15m, 30m, 60m …
INTERVAL = os.getenv("INTERVAL", "5m")

# Lookback window sent to yfinance: 1d, 5d, 1mo, 3mo …
PERIOD = os.getenv("PERIOD", "5d")

# Market-structure detection granularity: short | intermediate | long
MS_TERM = os.getenv("MS_TERM", "intermediate")

try:
    MIN_RR = float(os.getenv("MIN_RR", "2.0"))
except ValueError:
    raise ValueError("MIN_RR must be a valid number")

# Set to 'true' to require Order Block confluence before entering
REQUIRE_OB = os.getenv("REQUIRE_OB", "false").lower() == "true"

# Set to 'true' to require a liquidity-engulfing candle at the FVG zone
REQUIRE_ENGULFING = os.getenv("REQUIRE_ENGULFING", "false").lower() == "true"

# ── Session parameters ────────────────────────────────────────────────────────
try:
    SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
except ValueError:
    raise ValueError("SCAN_INTERVAL_SECONDS must be a valid integer")

try:
    MAX_SCANS = int(os.getenv("MAX_SCANS", "390"))
except ValueError:
    raise ValueError("MAX_SCANS must be a valid integer")

# ── Multi-Timeframe (MTF) parameters ─────────────────────────────────────────
# Primary timeframes cross-referenced with equal weight (1.0 each).
# 30m is supplementary only (weight 0.4) — it adds conviction to an
# already-strong signal but cannot override primary-TF disagreement.
MTF_MS_TERM = os.getenv("MTF_MS_TERM", MS_TERM)

try:
    MTF_MIN_RR = float(os.getenv("MTF_MIN_RR", str(MIN_RR)))
except ValueError:
    raise ValueError("MTF_MIN_RR must be a valid number")
