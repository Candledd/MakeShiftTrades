import os
import sys
import pandas as pd
from datetime import timedelta
import logging

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from dotenv import load_dotenv
load_dotenv()
import config

from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum_breakout import MomentumBreakoutStrategy
from src.strategies.trend_pullback import TrendPullbackStrategy
logging.basicConfig(level=logging.CRITICAL, format='%(message)s')

def simulate_trade(df, entry_idx, signal):
    future_df = df.iloc[entry_idx + 1:]
    max_bars = 16
    for i, (_, row) in enumerate(future_df.iterrows()):
        if i >= max_bars:
            return 'TIME_STOP', row['Close']
        if signal.direction == 'BUY':
            if row['Low'] <= signal.stop_loss: return 'SL', signal.stop_loss
            if row['High'] >= signal.take_profit: return 'TP', signal.take_profit
        elif signal.direction == 'SELL':
            if row['High'] >= signal.stop_loss: return 'SL', signal.stop_loss
            if row['Low'] <= signal.take_profit: return 'TP', signal.take_profit
    return 'OPEN', future_df["Close"].iloc[-1] if len(future_df) > 0 else signal.entry

from scripts.optimization.ml_optimizer import run_backtest_session
from scripts.optimization import ml_optimizer

def test_params(params):
    # Apply params
    # Apply Risk params
    config.MAX_POSITION_PCT = params.get("MAX_POSITION_PCT", getattr(config, 'MAX_POSITION_PCT', 5.0))
    config.MAX_RISK_PCT = params.get("MAX_RISK_PCT", getattr(config, 'MAX_RISK_PCT', 0.04))
    config.RISK_TIER_EQUITY_PCT = params.get("RISK_TIER_EQUITY_PCT", getattr(config, 'RISK_TIER_EQUITY_PCT', 0.04))

    config.MB_ADX_THRESHOLD = params.get("MB_ADX_THRESHOLD", config.MB_ADX_THRESHOLD)
    config.MB_DONCHIAN_PERIOD = int(params.get("MB_DONCHIAN_PERIOD", config.MB_DONCHIAN_PERIOD))
    config.MB_FALSE_BREAKOUT_BARS = int(params.get("MB_FALSE_BREAKOUT_BARS", getattr(config, "MB_FALSE_BREAKOUT_BARS", 5)))
    config.MR_BB_PERIOD = int(params.get("MR_BB_PERIOD", config.MR_BB_PERIOD))
    config.MR_BB_STD = params.get("MR_BB_STD", config.MR_BB_STD)
    config.MR_RSI_OVERBOUGHT = params.get("MR_RSI_OVERBOUGHT", config.MR_RSI_OVERBOUGHT)
    config.MR_RSI_OVERSOLD = params.get("MR_RSI_OVERSOLD", config.MR_RSI_OVERSOLD)
    config.MR_RSI_PERIOD = int(params.get("MR_RSI_PERIOD", config.MR_RSI_PERIOD))
    config.MR_STOP_MULT = params.get("MR_STOP_MULT", config.MR_STOP_MULT)
    config.TP_BB_PERIOD = int(params.get("TP_BB_PERIOD", config.TP_BB_PERIOD))
    config.TP_BB_STD = params.get("TP_BB_STD", config.TP_BB_STD)
    config.TP_PULLBACK_BUFFER = params.get("TP_PULLBACK_BUFFER", config.TP_PULLBACK_BUFFER)
    config.TP_STOP_MULT = params.get("TP_STOP_MULT", config.TP_STOP_MULT)
    
    # Add new R/R parameters gracefully
    config.MR_TP_TARGET_MULT = params.get("MR_TP_TARGET_MULT", getattr(config, 'MR_TP_TARGET_MULT', 1.0))
    config.MR_MIN_RR = params.get("MR_MIN_RR", getattr(config, 'MR_MIN_RR', 1.0))
    config.TP_MIN_RR = params.get("TP_MIN_RR", getattr(config, 'TP_MIN_RR', 1.2))

    strats = [
        ("SPY", "15m", MeanReversionStrategy()),
        ("QQQ", "15m", MeanReversionStrategy()),
        ("SPY", "15m", TrendPullbackStrategy()),
        ("QQQ", "15m", TrendPullbackStrategy())
    ]
    
    total_pnl, total_signals, pnl_by_ticker, pnl_by_strategy, trades = run_backtest_session(
        strats, is_oos_mode="IS", account_size=5000.0, risk_pct=1.0
    )
    return total_pnl, total_signals

if __name__ == "__main__":
    import json
    
    mode = input("Which version do you want to verify? (ml / risk): ").strip().lower()
    if mode == "ml":
        surrogate_file = "data/surrogate_top_5_ml.json"
        best_params_file = "data/best_ml_params.json"
    else:
        surrogate_file = "data/surrogate_top_5_risk.json"
        best_params_file = "data/best_risk_params.json"

    if not os.path.exists(surrogate_file):
        print(f"Error: '{surrogate_file}' not found. Please run 'python scripts/training/train_surrogate.py' first!")
        sys.exit(1)
        
    with open(surrogate_file, "r") as f:
        sets = json.load(f)
        
    print("Loading data for cache... (this takes a few seconds)")
    ml_optimizer.load_data(months=3)
    
    if os.path.exists(best_params_file):
        with open(best_params_file, "r") as f:
            current_params = json.load(f)
        actual_pnl, trades = test_params(current_params)
        print(f"{'CURRENT':<8} | {'N/A':<15} | ${actual_pnl:<13.2f}  | {trades:<10}")
        print("-" * 58)
        
    # 2. Test the top 5 AI predictions
    for i, p in enumerate(sets):
        predicted = p.get('Predicted_PnL', 0.0)
        actual_pnl, trades = test_params(p)
        print(f"#{i+1:<7} | ${predicted:<13.2f}  | ${actual_pnl:<13.2f}  | {trades:<10}")
        
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
                
                with open(best_params_file, "w") as f:
                    json.dump(clean_params, f, indent=4)
                print(f"\nSUCCESS! Set #{choice_idx} has been permanently saved to " + best_params_file + "!")
                print("Your bot will now use these institutional-grade parameters.")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")
