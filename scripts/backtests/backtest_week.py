import os
import sys
import logging

logging.basicConfig(level=logging.CRITICAL, format='%(message)s')

# Ensure src is accessible
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.trend_pullback import TrendPullbackStrategy
import src.regime_classifier
src.regime_classifier.HMM_AVAILABLE = False

from src.backtester import run_stateful_backtest as _run_stateful_backtest

def run_stateful_backtest(strategies_to_test, days):
    return _run_stateful_backtest(strategies_to_test, days_to_test=days)

def run_backtest():
    print("=" * 60)
    print("MakeShiftTrades - 1 Week Backtest")
    print("=" * 60)
    
    strategies_to_test = [
        ("SPY", "15m", MeanReversionStrategy()),
        ("SPY", "15m", TrendPullbackStrategy()),
        ("QQQ", "15m", MeanReversionStrategy()),
        ("QQQ", "15m", TrendPullbackStrategy()),
        ("GLD", "4h", TrendFollowingStrategy()),
        ("PDBC", "4h", TrendFollowingStrategy())
    ]
    
    run_stateful_backtest(strategies_to_test, days=7)

if __name__ == "__main__":
    run_backtest()
