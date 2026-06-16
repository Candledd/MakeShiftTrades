import os
import sys
import pandas as pd
from datetime import timedelta
import logging

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import config

from src.strategies.mean_reversion import MeanReversionStrategy
from charts.data import fetch_ohlcv

logging.basicConfig(level=logging.CRITICAL, format='%(message)s')
logging.getLogger("src.strategies").setLevel(logging.CRITICAL)

def simulate_trade(df, entry_idx, signal):
    future_df = df.iloc[entry_idx + 1:]
    for i, (_, row) in enumerate(future_df.iterrows()):
        high = row['High']
        low = row['Low']
        if signal.direction == 'BUY':
            if low <= signal.stop_loss: return 'SL', i + 1
            if high >= signal.take_profit: return 'TP', i + 1
        elif signal.direction == 'SELL':
            if high >= signal.stop_loss: return 'SL', i + 1
            if low <= signal.take_profit: return 'TP', i + 1
    return 'OPEN', len(future_df)

def run_optimization():
    print("=" * 60)
    print("Volume Ratio Optimization (Past 2 Weeks)")
    print("=" * 60)
    
    # We test on SPY and QQQ
    tickers = ["SPY", "QQQ"]
    data_dict = {}
    
    # Pre-fetch data
    for ticker in tickers:
        df = fetch_ohlcv(ticker, period="1mo", interval="15m")
        if df is not None and len(df) > 200:
            data_dict[ticker] = df
            
    if not data_dict:
        print("Failed to fetch data.")
        return

    # Test thresholds from 0.0 to 1.2
    thresholds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    
    print(f"{'Vol_Mult':<10} | {'Signals':<10} | {'Wins':<5} | {'Losses':<6} | {'Win Rate':<8}")
    print("-" * 55)

    for threshold in thresholds:
        config.MR_VOL_SPIKE_MULT = threshold
        
        total_signals = 0
        total_wins = 0
        total_losses = 0
        
        for ticker, df in data_dict.items():
            strategy = MeanReversionStrategy()
            
            last_timestamp = df.index[-1]
            two_weeks_ago = last_timestamp - timedelta(days=14)
            test_start_idx = df.index.get_indexer([two_weeks_ago], method='bfill')[0]
            
            for i in range(test_start_idx, len(df) - 1):
                window_df = df.iloc[:i+1]
                strategy._last_signal_time = {} # reset cooldown using proper attribute
                
                signal = strategy.analyze(window_df, ticker)
                if signal:
                    total_signals += 1
                    outcome, bars = simulate_trade(df, i, signal)
                    if outcome == 'TP': total_wins += 1
                    elif outcome == 'SL': total_losses += 1
                    
        win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
        print(f"{threshold:<10.1f} | {total_signals:<10} | {total_wins:<5} | {total_losses:<6} | {win_rate:.1f}%")

if __name__ == "__main__":
    run_optimization()
