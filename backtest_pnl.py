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
from charts.data import fetch_ohlcv
from src.backtester import simulate_trade

# Suppress debug logging during backtest to keep console clean
logging.basicConfig(level=logging.CRITICAL, format='%(message)s')
logging.getLogger("src.strategies").setLevel(logging.CRITICAL)
logging.getLogger("charts.data").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

def run_pnl_backtest(account_size=5000.0, risk_pct=1.0, months=6):
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
    
    configs = [
        ("SPY", "15m", MeanReversionStrategy()),
        ("QQQ", "15m", MeanReversionStrategy()),
        ("SPY", "15m", TrendPullbackStrategy()),
        ("QQQ", "15m", TrendPullbackStrategy()),
        ("BTC-USD", "1h", MomentumBreakoutStrategy()),
        ("GLD", "4h", TrendFollowingStrategy()),
        ("PDBC", "4h", TrendFollowingStrategy())
    ]
    
    total_signals = 0
    total_wins = 0
    total_losses = 0
    total_pnl = 0.0
    
    RISK_DOLLARS = account_size * (risk_pct / 100.0)
    
    for ticker, tf, strategy in configs:
        print(f"\nTesting {ticker} ({tf})...")
        # Fetch extra time to avoid warmup bias
        df = fetch_ohlcv(ticker, period=f"{months+1}mo", interval=tf)
        if df is None or len(df) < 200:
            print(f"  [!] Insufficient data for {ticker}")
            continue
            
        last_timestamp = df.index[-1]
        start_date = last_timestamp - timedelta(days=30 * months)
        test_start_idx = df.index.get_indexer([start_date], method='bfill')[0]
        
        ticker_signals = 0
        ticker_wins = 0
        ticker_losses = 0
        ticker_pnl = 0.0
        
        # Reset internal cooldowns ONCE per ticker
        strategy._last_signal_time = {} 
        
        for i in range(test_start_idx, len(df) - 1):
            window_df = df.iloc[:i+1]
            
            signal = strategy.analyze(window_df, ticker)
            if signal:
                outcome, exit_price, bars, pnl_per_share = simulate_trade(df, i, signal)
                if outcome == 'NO_FILL':
                    continue
                
                ticker_signals += 1
                
                # Calculate PnL mathematically based on standard Risk/Reward principles
                risk_per_share = abs(signal.entry - signal.stop_loss)
                if risk_per_share == 0: continue
                
                qty = RISK_DOLLARS / risk_per_share
                max_qty = account_size / signal.entry
                qty = min(qty, max_qty)
                
                trade_pnl = pnl_per_share * qty
                ticker_pnl += trade_pnl
                
                if outcome == 'TP' or outcome == 'TRAIL_SL' or trade_pnl > 0: 
                    ticker_wins += 1
                elif outcome == 'SL' or trade_pnl < 0: 
                    ticker_losses += 1
                    
        total_signals += ticker_signals
        total_wins += ticker_wins
        total_losses += ticker_losses
        total_pnl += ticker_pnl
        
        win_rate = (ticker_wins / (ticker_wins + ticker_losses) * 100) if (ticker_wins + ticker_losses) > 0 else 0
        print(f"  {ticker} Summary: {ticker_signals} signals | Win Rate: {win_rate:.1f}%")
        print(f"  {ticker} Net PnL: ${ticker_pnl:+.2f}")

    print("-" * 70)
    total_closed = total_wins + total_losses
    overall_win_rate = (total_wins / total_closed * 100) if total_closed > 0 else 0
    print(f"TOTAL SIGNALS : {total_signals}")
    print(f"OVERALL WIN RATE: {overall_win_rate:.1f}% ({total_wins}W {total_losses}L)")
    print(f"TOTAL NET PnL : ${total_pnl:+.2f}")
    print("=" * 70)

if __name__ == "__main__":
    run_pnl_backtest()
