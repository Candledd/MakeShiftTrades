import os
import sys
import pandas as pd
from datetime import timedelta
import logging

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()
import config

from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum_breakout import MomentumBreakoutStrategy
from src.strategies.trend_pullback import TrendPullbackStrategy
from charts.data import fetch_ohlcv

logging.basicConfig(level=logging.CRITICAL, format='%(message)s')

DATA_CACHE = {}
def load_data(months=3):
    for ticker, tf in [("SPY", "15m"), ("QQQ", "15m"), ("BTC-USD", "1h")]:
        df = fetch_ohlcv(ticker, period=f"{months+1}mo", interval=tf)
        if df is not None and len(df) >= 200:
            DATA_CACHE[(ticker, tf)] = df

def simulate_trade(df, entry_idx, signal):
    future_df = df.iloc[entry_idx + 1:]
    for i, (_, row) in enumerate(future_df.iterrows()):
        if signal.direction == 'BUY':
            if row['Low'] <= signal.stop_loss: return 'SL', signal.stop_loss
            if row['High'] >= signal.take_profit: return 'TP', signal.take_profit
        elif signal.direction == 'SELL':
            if row['High'] >= signal.stop_loss: return 'SL', signal.stop_loss
            if row['Low'] <= signal.take_profit: return 'TP', signal.take_profit
    return 'OPEN', future_df["Close"].iloc[-1] if len(future_df) > 0 else signal.entry

def test_params(params):
    # Apply params
    config.MB_ADX_THRESHOLD = params["MB_ADX_THRESHOLD"]
    config.MB_DONCHIAN_PERIOD = int(params["MB_DONCHIAN_PERIOD"])
    config.MB_FALSE_BREAKOUT_BARS = int(params["MB_FALSE_BREAKOUT_BARS"])
    config.MR_BB_PERIOD = int(params["MR_BB_PERIOD"])
    config.MR_BB_STD = params["MR_BB_STD"]
    config.MR_RSI_OVERBOUGHT = params["MR_RSI_OVERBOUGHT"]
    config.MR_RSI_OVERSOLD = params["MR_RSI_OVERSOLD"]
    config.MR_RSI_PERIOD = int(params["MR_RSI_PERIOD"])
    config.MR_STOP_MULT = params["MR_STOP_MULT"]
    config.TP_BB_PERIOD = int(params["TP_BB_PERIOD"])
    config.TP_BB_STD = params["TP_BB_STD"]
    config.TP_PULLBACK_BUFFER = params["TP_PULLBACK_BUFFER"]
    config.TP_STOP_MULT = params["TP_STOP_MULT"]
    
    # Add new R/R parameters gracefully (in case old json is used)
    config.MR_TP_TARGET_MULT = params.get("MR_TP_TARGET_MULT", getattr(config, 'MR_TP_TARGET_MULT', 1.0))
    config.MR_MIN_RR = params.get("MR_MIN_RR", getattr(config, 'MR_MIN_RR', 1.0))
    config.TP_MIN_RR = params.get("TP_MIN_RR", getattr(config, 'TP_MIN_RR', 1.2))

    strats = [
        ("SPY", "15m", MeanReversionStrategy()),
        ("QQQ", "15m", MeanReversionStrategy()),
        ("SPY", "15m", TrendPullbackStrategy()),
        ("QQQ", "15m", TrendPullbackStrategy()),
        ("BTC-USD", "1h", MomentumBreakoutStrategy())
    ]
    
    total_pnl = 0.0
    total_signals = 0
    RISK_DOLLARS = 50.0  # 1% of 5000
    
    for ticker, tf, strategy in strats:
        df = DATA_CACHE.get((ticker, tf))
        if df is None: continue
        
        start_date = df.index[-1] - timedelta(days=90)
        test_start_idx = df.index.get_indexer([start_date], method='bfill')[0]
        
        for i in range(test_start_idx, len(df) - 1):
            strategy._last_signal_time = {} 
            signal = strategy.analyze(df.iloc[:i+1], ticker)
            if signal:
                total_signals += 1
                outcome, exit_price = simulate_trade(df, i, signal)
                risk_per_share = abs(signal.entry - signal.stop_loss)
                if risk_per_share == 0: continue
                qty = RISK_DOLLARS / risk_per_share
                max_qty = 5000.0 / signal.entry
                qty = min(qty, max_qty)
                
                # Transaction friction penalty
                trade_pnl = (exit_price - signal.entry) * qty if signal.direction == 'BUY' else (signal.entry - exit_price) * qty
                trade_pnl -= (signal.entry * qty * 0.00025) + (exit_price * qty * 0.00025)
                
                total_pnl += trade_pnl
                
    return total_pnl, total_signals

if __name__ == "__main__":
    import json
    
    if not os.path.exists("surrogate_top_5.json"):
        print("Error: 'surrogate_top_5.json' not found. Please run 'python train_surrogate.py' first!")
        sys.exit(1)
        
    with open("surrogate_top_5.json", "r") as f:
        sets = json.load(f)
        
    print("Loading data for cache... (this takes a few seconds)")
    load_data()
    
    print(f"\n{'Set':<5} | {'Predicted PnL':<15} | {'Actual PnL':<15} | {'Trades':<10}")
    print("-" * 55)
    for i, p in enumerate(sets):
        predicted = p.get('Predicted_PnL', 0.0)
        actual_pnl, trades = test_params(p)
        print(f"#{i+1:<4} | ${predicted:<13.2f}  | ${actual_pnl:<13.2f}  | {trades:<10}")
        
    print("\n" + "="*55)
    while True:
        try:
            choice = input("Enter the Set # to save to best_params.json (1-5), or 0 to exit: ")
            choice_idx = int(choice)
            if choice_idx == 0:
                print("Exiting without saving.")
                break
            elif 1 <= choice_idx <= len(sets):
                selected_params = sets[choice_idx - 1]
                # Remove the prediction metadata before saving
                clean_params = {k: v for k, v in selected_params.items() if k != "Predicted_PnL"}
                
                with open("best_params.json", "w") as f:
                    json.dump(clean_params, f, indent=4)
                print(f"\nSUCCESS! Set #{choice_idx} has been permanently saved to best_params.json!")
                print("Your bot will now use these institutional-grade parameters.")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")