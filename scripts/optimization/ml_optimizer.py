import os
import sys
import pandas as pd
from datetime import timedelta
import logging
import json
import optuna

# Ensure src is accessible
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from dotenv import load_dotenv
load_dotenv()
import config

from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum_breakout import MomentumBreakoutStrategy
from src.strategies.trend_pullback import TrendPullbackStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from charts.data import fetch_ohlcv
from src.backtester import simulate_trade
from src.risk_manager import RiskManager

# Set loggers
logging.basicConfig(level=logging.CRITICAL, format='%(message)s')
optuna.logging.set_verbosity(optuna.logging.INFO)
logging.getLogger("src.strategies").setLevel(logging.CRITICAL)
logging.getLogger("charts.data").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# Pre-fetch data to speed up Optuna
DATA_CACHE = {}
OOS_LOOKBACK_DAYS = 30

def load_data(months=6):
    configs = [
        ("SPY", "15m"),
        ("QQQ", "15m")
        # ("GLD", "4h"),
        # ("PDBC", "4h")
    ]
    for ticker, tf in configs:
        df = fetch_ohlcv(ticker, period=f"{months+1}mo", interval=tf)
        if df is not None and len(df) >= 200:
            DATA_CACHE[(ticker, tf)] = df

def run_backtest_session(configs_strat, is_oos_mode="IS", account_size=5000.0, risk_pct=1.0):
    total_signals = 0
    total_pnl = 0.0
    months = 12
    
    pnl_by_ticker = {"SPY": 0.0, "QQQ": 0.0, "GLD": 0.0, "PDBC": 0.0}
    pnl_by_strategy = {"mean_reversion": 0.0, "trend_pullback": 0.0, "momentum_breakout": 0.0, "trend_following": 0.0}
    
    trades = []
    all_signals = []
    
    for ticker, tf, strategy in configs_strat:
        df = DATA_CACHE.get((ticker, tf))
        if df is None:
            continue
            
        last_timestamp = df.index[-1]
        strategy._last_signal_time = {} 
        
        if is_oos_mode == "IS":
            start_date = last_timestamp - timedelta(days=30 * months)
            start_idx = df.index.get_indexer([start_date], method='bfill')[0]
            oos_start_date = last_timestamp - timedelta(days=OOS_LOOKBACK_DAYS)
            end_idx = df.index.get_indexer([oos_start_date], method='bfill')[0]
        else: # OOS
            oos_start_date = last_timestamp - timedelta(days=OOS_LOOKBACK_DAYS)
            start_idx = df.index.get_indexer([oos_start_date], method='bfill')[0]
            end_idx = len(df) - 1
            
        for i in range(start_idx, end_idx):
            window_df = df.iloc[:i+1]
            signal = strategy.analyze(window_df, ticker)
            if signal:
                all_signals.append({
                    'time': df.index[i],
                    'ticker': ticker,
                    'strategy': strategy.name,
                    'signal': signal,
                    'df': df,
                    'entry_idx': i
                })
                
    all_signals.sort(key=lambda x: x['time'])
    
    active_trades = []
    
    risk_manager = RiskManager(
        max_risk_pct=getattr(config, 'MAX_RISK_PCT', 0.01),
        max_position_pct=getattr(config, 'MAX_POSITION_PCT', 5.0) / 100.0,
        max_notional=50000.0,
        max_positions=5,
    )
    
    for sig_data in all_signals:
        signal_time = sig_data['time']
        ticker = sig_data['ticker']
        strategy_name = sig_data['strategy']
        signal = sig_data['signal']
        df = sig_data['df']
        entry_idx = sig_data['entry_idx']
        
        # Process exits to maintain active list
        active_trades = [t for t in active_trades if t['exit_time'] > signal_time]

        # Build RiskManager-format position list and position_state
        current_positions = [
            {
                "symbol": t["ticker"],
                "side": t["side"],
                "market_value": t["notional"],
            }
            for t in active_trades
        ]
        position_state = {
            t["ticker"]: {
                "entry_price": t["entry_price"],
                "stop_loss": t["stop_loss"],
                "qty": t["qty"]
            }
            for t in active_trades
        }

        # Call RiskManager.approve() — handles all gates (VIX, ML veto, macro, expectancy, heat)
        approved, notional, reason = risk_manager.approve(
            signal=signal,
            account_equity=account_size,
            current_positions=current_positions,
            active_orders=[],
            position_state=position_state,
        )
        if not approved:
            continue

        # Simulate trade execution
        outcome, exit_price, bars, pnl_per_share = simulate_trade(df, entry_idx, signal)
        if outcome == 'NO_FILL':
            continue
            
        exit_time = df.index[entry_idx + bars - 1] if (entry_idx + bars - 1) < len(df) else df.index[-1]
        
        if signal.entry <= 0:
            continue
        
        qty = notional / signal.entry
        trade_pnl = pnl_per_share * qty
        
        total_signals += 1
        side = "long" if signal.direction == "BUY" else "short"
        active_trades.append({
            'ticker': ticker,
            'symbol': ticker,
            'side': side,
            'exit_time': exit_time,
            'notional': notional,
            'entry_price': signal.entry,
            'stop_loss': signal.stop_loss,
            'qty': qty,
            'risk_pct': risk_pct,
            'strategy_name': strategy_name,
            'direction': signal.direction,
        })
        
        trades.append({'exit_time': exit_time, 'pnl': trade_pnl, 'ticker': ticker})
        total_pnl += trade_pnl
        pnl_by_ticker[ticker] += trade_pnl
        pnl_by_strategy[strategy_name] += trade_pnl
        
    return total_pnl, total_signals, pnl_by_ticker, pnl_by_strategy, trades

def objective(trial):
    # Mean Reversion parameters
    mr_bb_std = trial.suggest_float("MR_BB_STD", 1.5, 3.0, step=0.1)
    mr_bb_period = trial.suggest_int("MR_BB_PERIOD", 10, 40)
    mr_rsi_period = trial.suggest_int("MR_RSI_PERIOD", 5, 20)
    mr_rsi_oversold = trial.suggest_float("MR_RSI_OVERSOLD", 20.0, 40.0, step=1.0)
    mr_rsi_overbought = trial.suggest_float("MR_RSI_OVERBOUGHT", 60.0, 80.0, step=1.0)
    mr_vol_spike_mult = trial.suggest_float("MR_VOL_SPIKE_MULT", 0.0, 2.0, step=0.1)
    mr_vwap_dev_pct = trial.suggest_float("MR_VWAP_DEV_PCT", 0.001, 0.010, step=0.001)

    # Trend Pullback parameters
    tp_bb_std = trial.suggest_float("TP_BB_STD", 1.5, 3.0, step=0.1)
    tp_bb_period = trial.suggest_int("TP_BB_PERIOD", 10, 40)
    tp_rsi_period = trial.suggest_int("TP_RSI_PERIOD", 5, 20)
    
    # Apply parameters to config
    config.MR_BB_STD = mr_bb_std
    config.MR_BB_PERIOD = mr_bb_period
    config.MR_RSI_PERIOD = mr_rsi_period
    config.MR_RSI_OVERSOLD = mr_rsi_oversold
    config.MR_RSI_OVERBOUGHT = mr_rsi_overbought
    config.MR_VOL_SPIKE_MULT = mr_vol_spike_mult
    config.MR_VWAP_DEV_PCT = mr_vwap_dev_pct
    
    config.TP_BB_STD = tp_bb_std
    config.TP_BB_PERIOD = tp_bb_period
    config.TP_RSI_PERIOD = tp_rsi_period

    # Pre-instantiate strategies to save time
    configs_strat = [
        ("SPY", "15m", MeanReversionStrategy()),
        ("QQQ", "15m", MeanReversionStrategy()),
        ("SPY", "15m", TrendPullbackStrategy()),
        ("QQQ", "15m", TrendPullbackStrategy())
        # ("GLD", "4h", TrendFollowingStrategy()),
        # ("PDBC", "4h", TrendFollowingStrategy())
    ]
    
    account_size = 5000.0
    risk_pct = 1.0
    RISK_DOLLARS = account_size * (risk_pct / 100.0)
    
    total_pnl, total_signals, pnl_by_ticker, pnl_by_strategy, trades = run_backtest_session(
        configs_strat, is_oos_mode="IS", account_size=account_size, risk_pct=risk_pct
    )
    
    # Evaluate robust score
    if total_signals >= 20 and trades:
        trades_df = pd.DataFrame(trades).sort_values('exit_time')
        
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.0
        profit_factor_bonus = min(profit_factor, 5.0)
        
        running_pnl = trades_df['pnl'].cumsum()
        peak = running_pnl.cummax()
        drawdown = peak - running_pnl
        max_drawdown = drawdown.max()
        
        net_R = total_pnl / RISK_DOLLARS
        max_drawdown_R = max_drawdown / RISK_DOLLARS
        
        total_abs_strat_pnl = sum(abs(v) for v in pnl_by_strategy.values())
        strategy_concentration = max(abs(v) for v in pnl_by_strategy.values()) / total_abs_strat_pnl if total_abs_strat_pnl > 0 else 1.0
        
        total_abs_ticker_pnl = sum(abs(v) for v in pnl_by_ticker.values())
        ticker_concentration = max(abs(v) for v in pnl_by_ticker.values()) / total_abs_ticker_pnl if total_abs_ticker_pnl > 0 else 1.0
        
        max_trade_pnl = trades_df['pnl'].max()
        max_trade_concentration = max_trade_pnl / total_pnl if total_pnl > 0 else 1.0
        parameter_instability_penalty = ticker_concentration + max_trade_concentration
        
        # ML Optimizer score — optimized for SIGNAL QUALITY / STRATEGY EDGE.
        # Tunes strategy params (BB period, RSI period, Donchian, etc.)
        # that determine which signals fire and how often.
        # Profit factor + trade count rewarded because more + better signals
        # = more data for the ML model. Drawdown penalty exists but is secondary
        # since strategy params don't directly control sizing.
        score = (
            (net_R * 1.0)                            # Pure profit
            + (profit_factor_bonus * 0.5)            # Signal quality bonus
            + (0.08 * total_signals)                 # More trades = better signal diversity
            - (1.0 * max_drawdown_R)                 # Drawdown matters, secondary concern
            - 0.5 * abs(strategy_concentration)      # Want all strategies contributing
            - 0.25 * parameter_instability_penalty   # Single-trade concentration penalty
            - (2.0 * max(0, max_drawdown_R - 8))     # Tail penalty above 8R ($400 DD)
        )

        # ── Commodity validation: GLD and PDBC ─────────────────────────
        # '''
        # Hard validation: reject (return -inf) if trade count <= 10
        # or profit factor < 1.2 for either commodity.
        # for com_label in ["GLD", "PDBC"]:
        #     com_trades_list = [t for t in trades if t.get('ticker') == com_label]
        #     com_count = len(com_trades_list)
        #     if com_count > 0:
        #         com_gp = sum(t['pnl'] for t in com_trades_list if t['pnl'] > 0)
        #         com_gl = abs(sum(t['pnl'] for t in com_trades_list if t['pnl'] < 0))
        #         com_pf = com_gp / com_gl if com_gl > 0 else 999.0
        #         if com_count <= 10:
        #             return float("-inf")
        #         if com_pf < 1.2:
        #             return float("-inf")
        # '''
    else:
        score = -1000.0 - (20 - total_signals) * 10
        
    trial.set_user_attr("SPY_PnL", pnl_by_ticker["SPY"])
    trial.set_user_attr("QQQ_PnL", pnl_by_ticker["QQQ"])
    # trial.set_user_attr("GLD_PnL", pnl_by_ticker.get("GLD", 0.0))
    # trial.set_user_attr("PDBC_PnL", pnl_by_ticker.get("PDBC", 0.0))
    trial.set_user_attr("MR_PnL", pnl_by_strategy["mean_reversion"])
    trial.set_user_attr("TP_PnL", pnl_by_strategy["trend_pullback"])
    trial.set_user_attr("MB_PnL", pnl_by_strategy["momentum_breakout"])
    trial.set_user_attr("TF_PnL", pnl_by_strategy["trend_following"])
    trial.set_user_attr("Total_Trades", total_signals)
    trial.set_user_attr("Score", score)
    
    return score

def evaluate_oos_params(best_params):
    config.MR_BB_STD = best_params.get("MR_BB_STD", config.MR_BB_STD)
    # config.MR_RSI_OVERSOLD = best_params.get("MR_RSI_OVERSOLD", config.MR_RSI_OVERSOLD)
    # config.MR_RSI_OVERBOUGHT = best_params.get("MR_RSI_OVERBOUGHT", config.MR_RSI_OVERBOUGHT)
    config.TP_BB_STD = best_params.get("TP_BB_STD", config.TP_BB_STD)
    config.MB_DONCHIAN_PERIOD = best_params.get("MB_DONCHIAN_PERIOD", config.MB_DONCHIAN_PERIOD)
    
    # config.TP_PULLBACK_BUFFER = best_params.get("TP_PULLBACK_BUFFER", config.TP_PULLBACK_BUFFER)
    # config.MB_ADX_THRESHOLD = best_params.get("MB_ADX_THRESHOLD", config.MB_ADX_THRESHOLD)
    config.MR_BB_PERIOD = best_params.get("MR_BB_PERIOD", config.MR_BB_PERIOD)
    config.MR_RSI_PERIOD = best_params.get("MR_RSI_PERIOD", config.MR_RSI_PERIOD)
    config.TP_BB_PERIOD = best_params.get("TP_BB_PERIOD", config.TP_BB_PERIOD)
    config.MB_COMPRESSION_THRESHOLD = best_params.get("MB_COMPRESSION_THRESHOLD", config.MB_COMPRESSION_THRESHOLD)
    # config.MR_TP_TARGET_MULT = best_params.get("MR_TP_TARGET_MULT", config.MR_TP_TARGET_MULT)

    # if "TF_EMA_FAST" in best_params:
    #     config.GLD_EMA_FAST = int(best_params["TF_EMA_FAST"])
    #     config.PDBC_EMA_FAST = int(best_params["TF_EMA_FAST"])
    # config.TF_EMA_SLOW = int(best_params.get("TF_EMA_SLOW", config.TF_EMA_SLOW))
    # config.TF_ATR_TARGET_MULT = float(best_params.get("TF_ATR_TARGET_MULT", config.TF_ATR_TARGET_MULT))

    # Apply GLD parameters
    # if "GLD_STOP_MULT" in best_params:
    #     config.GLD_STOP_MULT = float(best_params["GLD_STOP_MULT"])
    # if "GLD_TREND_FILTER" in best_params:
    #     config.GLD_TREND_FILTER = best_params["GLD_TREND_FILTER"]
    # if "GLD_PULLBACK_TRIGGER" in best_params:
    #     config.GLD_PULLBACK_TRIGGER = best_params["GLD_PULLBACK_TRIGGER"]

    # Apply PDBC parameters
    # if "PDBC_STOP_MULT" in best_params:
    #     config.PDBC_STOP_MULT = float(best_params["PDBC_STOP_MULT"])
    # if "PDBC_ADX_MIN" in best_params:
    #     config.PDBC_ADX_MIN = float(best_params["PDBC_ADX_MIN"])
    # if "PDBC_RANGE_EXPANSION_THRESHOLD" in best_params:
    #     config.PDBC_RANGE_EXPANSION_THRESHOLD = float(best_params["PDBC_RANGE_EXPANSION_THRESHOLD"])
    
    configs_strat = [
        ("SPY", "15m", MeanReversionStrategy()),
        ("QQQ", "15m", MeanReversionStrategy()),
        ("SPY", "15m", TrendPullbackStrategy()),
        ("QQQ", "15m", TrendPullbackStrategy())
        # ("GLD", "4h", TrendFollowingStrategy()),
        # ("PDBC", "4h", TrendFollowingStrategy())
    ]
    
    account_size = 5000.0
    risk_pct = 1.0
    
    total_pnl, total_signals, pnl_by_ticker, pnl_by_strategy, trades = run_backtest_session(
        configs_strat, is_oos_mode="OOS", account_size=account_size, risk_pct=risk_pct
    )
    return total_pnl, total_signals, pnl_by_ticker

if __name__ == "__main__":
    print("Loading data for cache...")
    load_data(months=6)
    print("Data loaded. Starting Optuna study...")
    
    study = optuna.create_study(
        study_name="makeshift_trades_6mo_v6",
        storage="sqlite:///data/optuna_study.db?timeout=60",
        direction="maximize",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=5000)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    
    print("=" * 50)
    print("Best params:", best_params)
    print("Best Score (Penalized PnL):", best_trial.value)
    print("SPY PnL:", best_trial.user_attrs.get("SPY_PnL"))
    print("QQQ PnL:", best_trial.user_attrs.get("QQQ_PnL"))
    # print("GLD PnL:", best_trial.user_attrs.get("GLD_PnL"))
    # print("PDBC PnL:", best_trial.user_attrs.get("PDBC_PnL"))
    print("Total Trades:", best_trial.user_attrs.get("Total_Trades"))
    
    print("Evaluating best parameters on Out-Of-Sample (OOS) blind data...")
    oos_pnl, oos_trades, oos_pnl_by_ticker = evaluate_oos_params(best_params)
    print("OOS Net PnL:", oos_pnl)
    print("OOS Total Trades:", oos_trades)
    print("OOS SPY PnL:", oos_pnl_by_ticker["SPY"])
    print("OOS QQQ PnL:", oos_pnl_by_ticker["QQQ"])
    # print("OOS GLD PnL:", oos_pnl_by_ticker.get("GLD", 0.0))
    # print("OOS PDBC PnL:", oos_pnl_by_ticker.get("PDBC", 0.0))
    print("=" * 50)
    
    try:
        with open("data/best_ml_params.json", "w") as f:
            json.dump(best_params, f, indent=4)
            
        df_trials = study.trials_dataframe()
        df_trials.to_csv("data/optuna_trials_ml.csv", index=False)
            
        print("Saved best parameters to data/best_ml_params.json")
        print(f"Saved ALL {len(df_trials)} historical trial data to optuna_trials_data.csv for your analysis.")
    except PermissionError:
        print("Another terminal is currently saving the CSV/JSON files. Skipping file export in this terminal to prevent crashes.")
        print("Don't worry - all trial data is safely stored in the SQLite database!")
