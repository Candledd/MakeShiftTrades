import os
import sys
import json
import logging

# Ensure src is accessible
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from dotenv import load_dotenv
load_dotenv()
import config

from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum_breakout import MomentumBreakoutStrategy
from src.strategies.trend_pullback import TrendPullbackStrategy
import src.regime_classifier
src.regime_classifier.HMM_AVAILABLE = False

from src.backtester import run_stateful_backtest as _run_stateful_backtest

def run_stateful_backtest(strategies_to_test, days, end_date=None):
    return _run_stateful_backtest(strategies_to_test, days_to_test=days, end_date=end_date)

# Suppress debug logging during backtest to keep console clean
logging.basicConfig(level=logging.CRITICAL, format='%(message)s')
logging.getLogger("src.strategies").setLevel(logging.CRITICAL)
logging.getLogger("charts.data").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

def run_pnl_backtest(account_size=5000.0, risk_pct=1.0, months=6):
    if os.path.exists("data/best_ml_params.json") or os.path.exists("data/best_risk_params.json"):
        print("-> Loading optimized parameters from ML/Risk configs automatically via config.py...")
        # Since config.py loads them directly, we don't need to manually re-apply them here.
    else:
        print("-> Using default parameters from config.py...")

    print("=" * 70)
    print(f"{months}-Month Scalp Backtest (Simulated PnL)")
    print(f"Assuming ${account_size:,.0f} starting capital | {risk_pct}% Risk per Trade (${account_size * (risk_pct/100):.2f})")
    print("=" * 70)
    
    strategies_to_test = [
        ("SPY", "15m", MeanReversionStrategy()),
        ("QQQ", "15m", MeanReversionStrategy()),
        ("SPY", "15m", TrendPullbackStrategy()),
        ("QQQ", "15m", TrendPullbackStrategy()),
        ("BTC-USD", "1h", MomentumBreakoutStrategy())
    ]
    
    days = 30 * months
    run_stateful_backtest(strategies_to_test, days=days, end_date="2023-01-01")
    
if __name__ == "__main__":
    run_pnl_backtest()
