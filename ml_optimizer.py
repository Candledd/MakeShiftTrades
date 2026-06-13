import os
import sys
import pandas as pd
from datetime import timedelta
import logging
import json
import optuna

# Ensure src is accessible
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()
import config

from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum_breakout import MomentumBreakoutStrategy
from src.strategies.trend_pullback import TrendPullbackStrategy
from charts.data import fetch_ohlcv

# Set loggers
logging.basicConfig(level=logging.CRITICAL, format='%(message)s')
optuna.logging.set_verbosity(optuna.logging.INFO)
logging.getLogger("src.strategies").setLevel(logging.CRITICAL)
logging.getLogger("charts.data").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# Pre-fetch data to speed up Optuna
DATA_CACHE = {}
OOS_LOOKBACK_DAYS = 30

def load_data(months=3):
    configs = [
        ("SPY", "15m"),
        ("QQQ", "15m"),
        ("BTC-USD", "1h")
    ]
    for ticker, tf in configs:
        df = fetch_ohlcv(ticker, period=f"{months+1}mo", interval=tf)
        if df is not None and len(df) >= 200:
            DATA_CACHE[(ticker, tf)] = df

def simulate_trade(df, entry_idx, signal):
    future_df = df.iloc[entry_idx + 1:]
    for i, (_, row) in enumerate(future_df.iterrows()):
        high = row['High']
        low = row['Low']
        
        if signal.direction == 'BUY':
            if low <= signal.stop_loss: return 'SL', signal.stop_loss, i + 1
            if high >= signal.take_profit: return 'TP', signal.take_profit, i + 1
        elif signal.direction == 'SELL':
            if high >= signal.stop_loss: return 'SL', signal.stop_loss, i + 1
            if low <= signal.take_profit: return 'TP', signal.take_profit, i + 1
            
    return 'OPEN', future_df["Close"].iloc[-1] if len(future_df) > 0 else signal.entry, len(future_df)

def objective(trial):
    mr_bb_std = trial.suggest_float("MR_BB_STD", 1.5, 3.0, step=0.1)
    mr_rsi_oversold = trial.suggest_float("MR_RSI_OVERSOLD", 20.0, 40.0, step=1.0)
    mr_rsi_overbought = trial.suggest_float("MR_RSI_OVERBOUGHT", 60.0, 80.0, step=1.0)
    tp_bb_std = trial.suggest_float("TP_BB_STD", 1.5, 3.0, step=0.1)
    mb_donchian_period = trial.suggest_int("MB_DONCHIAN_PERIOD", 10, 40)
    
    # New missing parameters
    mr_stop_mult = trial.suggest_float("MR_STOP_MULT", 1.0, 4.0, step=0.1)
    tp_stop_mult = trial.suggest_float("TP_STOP_MULT", 1.0, 4.0, step=0.1)
    tp_pullback_buffer = trial.suggest_float("TP_PULLBACK_BUFFER", 1.000, 1.010, step=0.001)
    mb_adx_threshold = trial.suggest_float("MB_ADX_THRESHOLD", 15.0, 35.0, step=1.0)
    mr_bb_period = trial.suggest_int("MR_BB_PERIOD", 10, 40)
    mr_rsi_period = trial.suggest_int("MR_RSI_PERIOD", 5, 20)
    tp_bb_period = trial.suggest_int("TP_BB_PERIOD", 10, 40)
    mb_false_breakout_bars = trial.suggest_int("MB_FALSE_BREAKOUT_BARS", 1, 5)
    
    # R/R and Take Profit parameters
    mr_tp_target_mult = trial.suggest_float("MR_TP_TARGET_MULT", 0.5, 2.5, step=0.1)
    mr_min_rr = trial.suggest_float("MR_MIN_RR", 0.5, 2.5, step=0.1)
    tp_min_rr = trial.suggest_float("TP_MIN_RR", 0.8, 3.0, step=0.1)
    
    # Apply to config
    config.MR_BB_STD = mr_bb_std
    config.MR_RSI_OVERSOLD = mr_rsi_oversold
    config.MR_RSI_OVERBOUGHT = mr_rsi_overbought
    config.TP_BB_STD = tp_bb_std
    config.MB_DONCHIAN_PERIOD = mb_donchian_period
    
    config.MR_STOP_MULT = mr_stop_mult
    config.TP_STOP_MULT = tp_stop_mult
    config.TP_PULLBACK_BUFFER = tp_pullback_buffer
    config.MB_ADX_THRESHOLD = mb_adx_threshold
    config.MR_BB_PERIOD = mr_bb_period
    config.MR_RSI_PERIOD = mr_rsi_period
    config.TP_BB_PERIOD = tp_bb_period
    config.MB_FALSE_BREAKOUT_BARS = mb_false_breakout_bars
    
    config.MR_TP_TARGET_MULT = mr_tp_target_mult
    config.MR_MIN_RR = mr_min_rr
    config.TP_MIN_RR = tp_min_rr
    
    # Instantiate strategies after config update
    configs_strat = [
        ("SPY", "15m", MeanReversionStrategy()),
        ("QQQ", "15m", MeanReversionStrategy()),
        ("SPY", "15m", TrendPullbackStrategy()),
        ("QQQ", "15m", TrendPullbackStrategy()),
        ("BTC-USD", "1h", MomentumBreakoutStrategy())
    ]
    
    account_size = 5000.0
    risk_pct = 1.0
    RISK_DOLLARS = account_size * (risk_pct / 100.0)
    
    total_signals = 0
    total_pnl = 0.0
    months = 3
    
    # Track individual PnL
    pnl_by_ticker = {"SPY": 0.0, "QQQ": 0.0, "BTC-USD": 0.0}
    
    for ticker, tf, strategy in configs_strat:
        df = DATA_CACHE.get((ticker, tf))
        if df is None:
            continue
            
        last_timestamp = df.index[-1]
        start_date = last_timestamp - timedelta(days=30 * months)
        is_start_idx = df.index.get_indexer([start_date], method='bfill')[0]
        
        # Out-Of-Sample split point (last 30 days)
        oos_start_date = last_timestamp - timedelta(days=OOS_LOOKBACK_DAYS)
        oos_start_idx = df.index.get_indexer([oos_start_date], method='bfill')[0]
        
        # Reset internal cooldowns ONCE per ticker
        strategy._last_signal_time = {} 
        
        # Optimize on In-Sample (IS) data
        for i in range(is_start_idx, oos_start_idx):
            window_df = df.iloc[:i+1]
            
            signal = strategy.analyze(window_df, ticker)
            if signal:
                total_signals += 1
                outcome, exit_price, bars = simulate_trade(df, i, signal)
                
                risk_per_share = abs(signal.entry - signal.stop_loss)
                if risk_per_share == 0: continue
                qty = RISK_DOLLARS / risk_per_share
                
                # Prevent infinite margin leverage (Max 1x account size per trade)
                max_qty = account_size / signal.entry
                qty = min(qty, max_qty)
                
                if signal.direction == 'BUY':
                    trade_pnl = (exit_price - signal.entry) * qty
                else:
                    trade_pnl = (signal.entry - exit_price) * qty
                
                # Deduct transaction cost (0.05% slippage/commission per leg, i.e., 0.05% of entry + 0.05% of exit)
                trade_pnl -= (signal.entry * qty * (config.BACKTEST_SLIPPAGE_FRICTION_PCT / 2.0)) + (exit_price * qty * (config.BACKTEST_SLIPPAGE_FRICTION_PCT / 2.0))
                
                total_pnl += trade_pnl
                pnl_by_ticker[ticker] += trade_pnl
                
    # Penalize low frequency heavily
    if total_signals < 20:
        total_pnl -= 1000 * (20 - total_signals)
        
    # Store individual metrics in the trial so we can review them
    trial.set_user_attr("SPY_PnL", pnl_by_ticker["SPY"])
    trial.set_user_attr("QQQ_PnL", pnl_by_ticker["QQQ"])
    trial.set_user_attr("BTC_PnL", pnl_by_ticker["BTC-USD"])
    trial.set_user_attr("Total_Trades", total_signals)
    
    return total_pnl

def evaluate_oos_params(best_params):
    # Apply to config
    config.MR_BB_STD = best_params.get("MR_BB_STD", config.MR_BB_STD)
    config.MR_RSI_OVERSOLD = best_params.get("MR_RSI_OVERSOLD", config.MR_RSI_OVERSOLD)
    config.MR_RSI_OVERBOUGHT = best_params.get("MR_RSI_OVERBOUGHT", config.MR_RSI_OVERBOUGHT)
    config.TP_BB_STD = best_params.get("TP_BB_STD", config.TP_BB_STD)
    config.MB_DONCHIAN_PERIOD = best_params.get("MB_DONCHIAN_PERIOD", config.MB_DONCHIAN_PERIOD)
    
    config.MR_STOP_MULT = best_params.get("MR_STOP_MULT", config.MR_STOP_MULT)
    config.TP_STOP_MULT = best_params.get("TP_STOP_MULT", config.TP_STOP_MULT)
    config.TP_PULLBACK_BUFFER = best_params.get("TP_PULLBACK_BUFFER", config.TP_PULLBACK_BUFFER)
    config.MB_ADX_THRESHOLD = best_params.get("MB_ADX_THRESHOLD", config.MB_ADX_THRESHOLD)
    config.MR_BB_PERIOD = best_params.get("MR_BB_PERIOD", config.MR_BB_PERIOD)
    config.MR_RSI_PERIOD = best_params.get("MR_RSI_PERIOD", config.MR_RSI_PERIOD)
    config.TP_BB_PERIOD = best_params.get("TP_BB_PERIOD", config.TP_BB_PERIOD)
    config.MB_FALSE_BREAKOUT_BARS = best_params.get("MB_FALSE_BREAKOUT_BARS", config.MB_FALSE_BREAKOUT_BARS)
    
    configs_strat = [
        ("SPY", "15m", MeanReversionStrategy()),
        ("QQQ", "15m", MeanReversionStrategy()),
        ("SPY", "15m", TrendPullbackStrategy()),
        ("QQQ", "15m", TrendPullbackStrategy()),
        ("BTC-USD", "1h", MomentumBreakoutStrategy())
    ]
    
    account_size = 5000.0
    risk_pct = 1.0
    RISK_DOLLARS = account_size * (risk_pct / 100.0)
    
    total_signals = 0
    total_pnl = 0.0
    months = 3
    
    pnl_by_ticker = {"SPY": 0.0, "QQQ": 0.0, "BTC-USD": 0.0}
    
    for ticker, tf, strategy in configs_strat:
        df = DATA_CACHE.get((ticker, tf))
        if df is None:
            continue
            
        last_timestamp = df.index[-1]
        oos_start_date = last_timestamp - timedelta(days=OOS_LOOKBACK_DAYS)
        oos_start_idx = df.index.get_indexer([oos_start_date], method='bfill')[0]
        
        # Reset internal cooldowns ONCE per ticker
        strategy._last_signal_time = {} 
        
        for i in range(oos_start_idx, len(df) - 1):
            window_df = df.iloc[:i+1]
            
            signal = strategy.analyze(window_df, ticker)
            if signal:
                total_signals += 1
                outcome, exit_price, bars = simulate_trade(df, i, signal)
                
                risk_per_share = abs(signal.entry - signal.stop_loss)
                if risk_per_share == 0: continue
                qty = RISK_DOLLARS / risk_per_share
                
                # Prevent infinite margin leverage (Max 1x account size per trade)
                max_qty = account_size / signal.entry
                qty = min(qty, max_qty)
                
                if signal.direction == 'BUY':
                    trade_pnl = (exit_price - signal.entry) * qty
                else:
                    trade_pnl = (signal.entry - exit_price) * qty
                
                # Deduct transaction cost
                trade_pnl -= (signal.entry * qty * (config.BACKTEST_SLIPPAGE_FRICTION_PCT / 2.0)) + (exit_price * qty * (config.BACKTEST_SLIPPAGE_FRICTION_PCT / 2.0))
                
                total_pnl += trade_pnl
                pnl_by_ticker[ticker] += trade_pnl
                
    return total_pnl, total_signals, pnl_by_ticker

if __name__ == "__main__":
    print("Loading data for cache...")
    load_data(months=3)
    print("Data loaded. Starting Optuna study...")
    
    # Use SQLite database to permanently save study history so it learns across multiple runs
    # Added timeout=60 to prevent "database is locked" errors when running across multiple terminals
    study = optuna.create_study(
        study_name="makeshift_trades_v4",
        storage="sqlite:///optuna_study.db?timeout=60",
        direction="maximize",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=1400)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    
    print("=" * 50)
    print("Best params:", best_params)
    print("Best Score (Penalized PnL):", best_trial.value)
    print("SPY PnL:", best_trial.user_attrs.get("SPY_PnL"))
    print("QQQ PnL:", best_trial.user_attrs.get("QQQ_PnL"))
    print("BTC PnL:", best_trial.user_attrs.get("BTC_PnL"))
    print("Total Trades:", best_trial.user_attrs.get("Total_Trades"))
    
    # Evaluate on OOS data
    print("Evaluating best parameters on Out-Of-Sample (OOS) blind data...")
    oos_pnl, oos_trades, oos_pnl_by_ticker = evaluate_oos_params(best_params)
    print("OOS Net PnL:", oos_pnl)
    print("OOS Total Trades:", oos_trades)
    print("OOS SPY PnL:", oos_pnl_by_ticker["SPY"])
    print("OOS QQQ PnL:", oos_pnl_by_ticker["QQQ"])
    print("OOS BTC PnL:", oos_pnl_by_ticker["BTC-USD"])
    print("=" * 50)
    
    try:
        with open("best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)
            
        # Export all trials (including historical ones from past runs) to CSV
        df_trials = study.trials_dataframe()
        df_trials.to_csv("optuna_trials_data.csv", index=False)
            
        print("Saved best parameters to best_params.json")
        print(f"Saved ALL {len(df_trials)} historical trial data to optuna_trials_data.csv for your analysis.")
    except PermissionError:
        print("Another terminal is currently saving the CSV/JSON files. Skipping file export in this terminal to prevent crashes.")
        print("Don't worry - all trial data is safely stored in the SQLite database!")
        