"""MakeShiftTrades — Multi-Asset Trading Bot

Strategies:
  - Mean Reversion on SPY, QQQ (15m candles)
  - Momentum Breakout on BTC-USD (1h candles)
  - Trend Following on GLD, USO (4h candles)

Usage:
    python main.py              # Run the bot
    python main.py --dry-run    # Log signals without executing
"""

import argparse
import logging
import sys

import config


def main():
    parser = argparse.ArgumentParser(description="MakeShiftTrades Trading Bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log signals without placing orders",
    )
    args = parser.parse_args()

    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    
    # File handler for overnight telemetry
    file_handler = logging.FileHandler("bot_overnight_telemetry.log", encoding="utf-8")
    file_handler.setFormatter(log_formatter)

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        handlers=[console_handler, file_handler]
    )
    
    logger = logging.getLogger(__name__)

    # Override dry run from CLI flag
    if args.dry_run:
        config.DRY_RUN = True

    logger.info(
        "Starting MakeShiftTrades bot | DRY_RUN=%s | Risk=%.1f%% | Max positions=%d",
        config.DRY_RUN,
        config.MAX_RISK_PCT * 100,
        config.MAX_POSITIONS,
    )

    from src.engine import TradingEngine

    engine = TradingEngine()

    try:
        engine.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C).")
        sys.exit(0)


if __name__ == "__main__":
    main()
