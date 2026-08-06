import os
import sys
import logging

# Ensure src is accessible
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Load env vars so config takes our changes
from dotenv import load_dotenv
load_dotenv()
import config

from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_pullback import TrendPullbackStrategy
import src.regime_classifier
src.regime_classifier.HMM_AVAILABLE = False

from src.backtester import run_stateful_backtest as _run_stateful_backtest

def run_stateful_backtest(strategies_to_test, days):
    return _run_stateful_backtest(strategies_to_test, days_to_test=days)

logging.basicConfig(level=logging.CRITICAL, format='%(message)s')
logging.getLogger("src.strategies").setLevel(logging.CRITICAL)
logging.getLogger("charts.data").setLevel(logging.CRITICAL)

def run_backtest():
    print("=" * 60)
    print("MakeShiftTrades - 1 Day Backtest (Scalping Profile)")
    print("=" * 60)
    print(f"Active Scalp Limits: BB_STD={config.MR_BB_STD}, RSI_OS={config.MR_RSI_OVERSOLD}, RSI_OB={config.MR_RSI_OVERBOUGHT}")
    print("-" * 60)
    
    strategies_to_test = [
        ("SPY", "15m", MeanReversionStrategy()),
        ("SPY", "15m", TrendPullbackStrategy()),
        ("QQQ", "15m", MeanReversionStrategy()),
        ("QQQ", "15m", TrendPullbackStrategy()),
    ]
    
    run_stateful_backtest(strategies_to_test, days=1)

if __name__ == "__main__":
    run_backtest()
