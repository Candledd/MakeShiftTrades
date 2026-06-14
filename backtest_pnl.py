import os
import sys
import pandas as pd
from datetime import timedelta
import logging

# Ensure src is accessible
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()
import config

from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum_breakout import MomentumBreakoutStrategy
from src.strategies.trend_pullback import TrendPullbackStrategy
from src.strategies.trend_following import TrendFollowingStrategy

from ml_optimizer import run_backtest_session
import ml_optimizer

# Suppress debug logging during backtest to keep console clean
logging.basicConfig(level=logging.CRITICAL, format='%(message)s')
logging.getLogger("src.strategies").setLevel(logging.CRITICAL)
logging.getLogger("charts.data").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

def run_pnl_backtest(account_size=5000.0, risk_pct=1.0, months=1):
    import json
    if os.path.exists("best_params.json"):
        print("-> Loading optimized parameters from best_params.json...")
        with open("best_params.json", "r") as f:
            best_params = json.load(f)
            for k, v in best_params.items():
                if hasattr(config, k):
                    orig_val = getattr(config, k)
                    if isinstance(orig_val, int):
                        setattr(config, k, int(float(v)))
                    else:
                        setattr(config, k, float(v))
            if "TF_EMA_FAST" in best_params:
                val = int(float(best_params["TF_EMA_FAST"]))
                config.GLD_EMA_FAST = val
                config.PDBC_EMA_FAST = val
    else:
        print("-> Using default parameters from config.py...")

    print("=" * 70)
    print(f"{months}-Month Scalp Backtest (Simulated PnL)")
    print(f"Assuming ${account_size:,.0f} starting capital | {risk_pct}% Risk per Trade (${account_size * (risk_pct/100):.2f})")
    print("=" * 70)
    
    configs_strat = [
        ("SPY", "15m", MeanReversionStrategy()),
        ("QQQ", "15m", MeanReversionStrategy()),
        ("SPY", "15m", TrendPullbackStrategy()),
        ("QQQ", "15m", TrendPullbackStrategy()),
        ("BTC-USD", "1h", MomentumBreakoutStrategy())
    ]
    
    ml_optimizer.load_data(months=months)
    
    total_pnl, total_signals, pnl_by_ticker, pnl_by_strategy, trades = run_backtest_session(
        configs_strat, is_oos_mode="IS", account_size=account_size, risk_pct=risk_pct
    )
    
    print("\n" + "="*50)
    print(f"6-MONTH RAW PNL BACKTEST (Account: ${account_size:,.2f})")
    print("="*50)
    print(f"Total Trades: {total_signals}")
    print(f"Total Net PnL: ${total_pnl:,.2f}")
    print("-"*50)
    print("PnL by Ticker:")
    for ticker, pnl in pnl_by_ticker.items():
        if pnl != 0.0 or ticker in ["SPY", "QQQ", "BTC-USD"]:
            print(f"  {ticker:<8}: ${pnl:,.2f}")
    print("-"*50)
    print("PnL by Strategy:")
    for strat, pnl in pnl_by_strategy.items():
        if pnl != 0.0 or strat in ["mean_reversion", "trend_pullback", "momentum_breakout"]:
            print(f"  {strat:<18}: ${pnl:,.2f}")
    print("="*50)
    
if __name__ == "__main__":
    run_pnl_backtest()
