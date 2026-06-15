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
from src.strategies.trend_following import TrendFollowingStrategy
from charts.data import fetch_ohlcv
from src.backtester import simulate_trade

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
        ("QQQ", "15m"),
        ("BTC-USD", "1h")
        # ("GLD", "4h"),
        # ("PDBC", "4h")
    ]
    for ticker, tf in configs:
        df = fetch_ohlcv(ticker, period=f"{months+1}mo", interval=tf)
        if df is not None and len(df) >= 200:
            DATA_CACHE[(ticker, tf)] = df

def run_backtest_session(configs_strat, is_oos_mode="IS", account_size=5000.0, risk_pct=1.0):
    RISK_DOLLARS = account_size * (risk_pct / 100.0)
    total_signals = 0
    total_pnl = 0.0
    months = 6
    
    pnl_by_ticker = {"SPY": 0.0, "QQQ": 0.0, "BTC-USD": 0.0, "GLD": 0.0, "PDBC": 0.0}
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
    expectancy_history = {}
    
    for sig_data in all_signals:
        signal_time = sig_data['time']
        ticker = sig_data['ticker']
        strategy_name = sig_data['strategy']
        signal = sig_data['signal']
        df = sig_data['df']
        entry_idx = sig_data['entry_idx']
        
        # Process exits to maintain active list & expectancy stats
        new_active = []
        for t in active_trades:
            if t['exit_time'] <= signal_time:
                gate_key = (t['strategy_name'], t['direction'])
                if gate_key not in expectancy_history:
                    expectancy_history[gate_key] = []
                expectancy_history[gate_key].append(t['r_multiple'])
            else:
                new_active.append(t)
        active_trades = new_active

        # VIX Filter Gate
        vix = float(os.getenv("MOCK_VIX_LEVEL", "20.0"))
        if vix > getattr(config, 'MAX_VIX_THRESHOLD', 30.0) and strategy_name == "mean_reversion":
            continue
            
        # ML Veto Gate
        signal_risk_dollars = RISK_DOLLARS
        if getattr(config, 'ML_VETO_ENABLED', True):
            p = getattr(signal, 'confidence', 100) / 100.0
            if p < 0.3:
                # Strong disagreement -> veto
                continue
            elif 0.3 <= p < 0.45:
                # Mild disagreement -> reduce position size
                signal_risk_dollars *= 0.5
            elif p > 0.6:
                # Agreement -> slight size boost
                signal_risk_dollars *= 1.2

        # Macro Filter Gate
        if getattr(config, 'MACRO_FILTER_ENABLED', True):
            from src.macro_filter import MacroFilter
            timestamp_now = signal_time.timestamp()
            active_events = MacroFilter.check_event(timestamp_now)
            should_skip = False
            for event in active_events:
                affected = event.get("affected_assets", [])
                actions = event.get("actions", [])
                normal_ticker = signal.ticker.replace("-USD", "")
                normalized_affected = [a.replace("-USD", "") for a in affected]
                if "all" in normalized_affected or normal_ticker in normalized_affected:
                    if "no_new_entries" in actions or "flatten_intraday_only" in actions:
                        should_skip = True
                        break
            if should_skip:
                continue
                
        # Expectancy Gate
        gate_key = (strategy_name, signal.direction)
        past_r = expectancy_history.get(gate_key, [])
        min_samples = getattr(config, 'MIN_EXPECTANCY_SAMPLES', 5)
        min_expectancy = getattr(config, 'MIN_EXPECTANCY_R', 0.05)
        if len(past_r) >= min_samples:
            avg_r = sum(past_r) / len(past_r)
            if avg_r < min_expectancy:
                continue

        # Portfolio Heat check
        current_heat = sum(t['risk_pct'] for t in active_trades)
        if current_heat + risk_pct > getattr(config, 'MAX_OPEN_PORTFOLIO_RISK_PCT', 0.02) * 100:
            continue
            
        # SPY/QQQ Contention Guard
        if ticker in ["SPY", "QQQ"]:
            counterpart = "QQQ" if ticker == "SPY" else "SPY"
            if any(t['ticker'] == counterpart for t in active_trades):
                continue
                
        # Simulate trade execution
        outcome, exit_price, bars, pnl_per_share = simulate_trade(df, entry_idx, signal)
        if outcome == 'NO_FILL':
            continue
            
        exit_time = df.index[entry_idx + bars - 1] if (entry_idx + bars - 1) < len(df) else df.index[-1]
        
        risk_per_share = abs(signal.entry - signal.stop_loss)
        if risk_per_share == 0:
            continue
            
        qty = signal_risk_dollars / risk_per_share
        max_qty = (account_size * getattr(config, 'MAX_POSITION_PCT', 1.0)) / signal.entry
        qty = min(qty, max_qty)
        
        trade_pnl = pnl_per_share * qty
        r_multiple = trade_pnl / signal_risk_dollars
        
        total_signals += 1
        active_trades.append({
            'ticker': ticker,
            'exit_time': exit_time,
            'risk_pct': risk_pct,
            'strategy_name': strategy_name,
            'direction': signal.direction,
            'r_multiple': r_multiple
        })
        
        trades.append({'exit_time': exit_time, 'pnl': trade_pnl, 'ticker': ticker})
        total_pnl += trade_pnl
        pnl_by_ticker[ticker] += trade_pnl
        pnl_by_strategy[strategy_name] += trade_pnl
        
    return total_pnl, total_signals, pnl_by_ticker, pnl_by_strategy, trades

def objective(trial):
    mr_bb_std = trial.suggest_float("MR_BB_STD", 1.5, 3.0, step=0.1)
    # mr_rsi_oversold = trial.suggest_float("MR_RSI_OVERSOLD", 20.0, 40.0, step=1.0)
    # mr_rsi_overbought = trial.suggest_float("MR_RSI_OVERBOUGHT", 60.0, 80.0, step=1.0)
    tp_bb_std = trial.suggest_float("TP_BB_STD", 1.5, 3.0, step=0.1)
    mb_donchian_period = trial.suggest_int("MB_DONCHIAN_PERIOD", 36, 72)
    
    mr_stop_mult = trial.suggest_float("MR_STOP_MULT", 1.0, 4.0, step=0.1)
    tp_stop_mult = trial.suggest_float("TP_STOP_MULT", 1.0, 4.0, step=0.1)
    # tp_pullback_buffer = trial.suggest_float("TP_PULLBACK_BUFFER", 1.000, 1.050, step=0.001)
    # mb_adx_threshold = trial.suggest_float("MB_ADX_THRESHOLD", 15.0, 35.0, step=1.0)
    mr_bb_period = trial.suggest_int("MR_BB_PERIOD", 10, 40)
    mr_rsi_period = trial.suggest_int("MR_RSI_PERIOD", 5, 20)
    tp_bb_period = trial.suggest_int("TP_BB_PERIOD", 10, 40)
    mb_compression_threshold = trial.suggest_int("MB_COMPRESSION_THRESHOLD", 50, 65, step=5)
    
    # mr_tp_target_mult = trial.suggest_float("MR_TP_TARGET_MULT", 0.5, 2.5, step=0.1)
    mr_min_rr = trial.suggest_float("MR_MIN_RR", 0.5, 2.5, step=0.1)
    tp_min_rr = trial.suggest_float("TP_MIN_RR", 0.8, 3.0, step=0.1)

    # tf_ema_fast = trial.suggest_int("TF_EMA_FAST", 10, 25)
    # tf_ema_slow = trial.suggest_int("TF_EMA_SLOW", 40, 60)
    # tf_atr_target_mult = trial.suggest_float("TF_ATR_TARGET_MULT", 2.0, 5.0, step=0.1)
    
    # Apply parameters to config
    config.MR_BB_STD = mr_bb_std
    # config.MR_RSI_OVERSOLD = mr_rsi_oversold
    # config.MR_RSI_OVERBOUGHT = mr_rsi_overbought
    config.TP_BB_STD = tp_bb_std
    config.MB_DONCHIAN_PERIOD = mb_donchian_period
    
    config.MR_STOP_MULT = mr_stop_mult
    config.TP_STOP_MULT = tp_stop_mult
    # config.TP_PULLBACK_BUFFER = tp_pullback_buffer
    # config.MB_ADX_THRESHOLD = mb_adx_threshold
    config.MR_BB_PERIOD = mr_bb_period
    config.MR_RSI_PERIOD = mr_rsi_period
    config.TP_BB_PERIOD = tp_bb_period
    config.MB_COMPRESSION_THRESHOLD = mb_compression_threshold
    
    # config.MR_TP_TARGET_MULT = mr_tp_target_mult
    config.MR_MIN_RR = mr_min_rr
    config.TP_MIN_RR = tp_min_rr

    # config.GLD_EMA_FAST = tf_ema_fast
    # config.PDBC_EMA_FAST = tf_ema_fast
    # config.TF_EMA_SLOW = tf_ema_slow
    # config.TF_ATR_TARGET_MULT = tf_atr_target_mult

    # ⛏️ GLD-specific tuning parameters ⛏️
    # gld_stop_mult = trial.suggest_float("GLD_STOP_MULT", 2.0, 4.0, step=0.1)
    # gld_trend_filter = trial.suggest_categorical("GLD_TREND_FILTER", ["HTF", "ADX", "BOTH"])
    # gld_pullback_trigger = trial.suggest_categorical("GLD_PULLBACK_TRIGGER", ["enabled", "disabled"])

    # 🛢️ PDBC-specific tuning parameters 🛢️
    # pdbc_stop_mult = trial.suggest_float("PDBC_STOP_MULT", 2.0, 4.0, step=0.1)
    # pdbc_adx_min = trial.suggest_float("PDBC_ADX_MIN", 20.0, 30.0, step=1.0)
    # pdbc_range_expansion_threshold = trial.suggest_float(
    #     "PDBC_RANGE_EXPANSION_THRESHOLD", 0.5, 1.0, step=0.05
    # )

    # Apply GLD parameters to config
    # config.GLD_STOP_MULT = gld_stop_mult
    # config.GLD_TREND_FILTER = gld_trend_filter
    # config.GLD_PULLBACK_TRIGGER = gld_pullback_trigger

    # Apply PDBC parameters to config
    # config.PDBC_STOP_MULT = pdbc_stop_mult
    # config.PDBC_ADX_MIN = pdbc_adx_min
    # config.PDBC_RANGE_EXPANSION_THRESHOLD = pdbc_range_expansion_threshold

    # Pre-instantiate strategies to save time
    configs_strat = [
        ("SPY", "15m", MeanReversionStrategy()),
        ("QQQ", "15m", MeanReversionStrategy()),
        ("SPY", "15m", TrendPullbackStrategy()),
        ("QQQ", "15m", TrendPullbackStrategy()),
        ("BTC-USD", "1h", MomentumBreakoutStrategy())
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
        
        # Classic Quant Institute profile: We EXPECT a 30-40% winrate with massive runners.
        # Stop penalizing drawdown so aggressively, and heavily reward sheer R-multiple generation.
        # Add a small bonus for taking more trades (Law of Large Numbers).
        
        score = (
            (net_R * 1.5)                            # Massively prioritize pure alpha/profit
            - (0.5 * max_drawdown_R)                 # Accept drawdowns as the cost of doing business
            + (0.1 * total_signals)                  # Reward the bot for taking more shots (active trading)
            - 0.5 * abs(strategy_concentration)      # Keep diversification penalty
            - 0.25 * parameter_instability_penalty   # Keep single-trade concentration penalty
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
    trial.set_user_attr("BTC_PnL", pnl_by_ticker["BTC-USD"])
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
    
    config.MR_STOP_MULT = best_params.get("MR_STOP_MULT", config.MR_STOP_MULT)
    config.TP_STOP_MULT = best_params.get("TP_STOP_MULT", config.TP_STOP_MULT)
    # config.TP_PULLBACK_BUFFER = best_params.get("TP_PULLBACK_BUFFER", config.TP_PULLBACK_BUFFER)
    # config.MB_ADX_THRESHOLD = best_params.get("MB_ADX_THRESHOLD", config.MB_ADX_THRESHOLD)
    config.MR_BB_PERIOD = best_params.get("MR_BB_PERIOD", config.MR_BB_PERIOD)
    config.MR_RSI_PERIOD = best_params.get("MR_RSI_PERIOD", config.MR_RSI_PERIOD)
    config.TP_BB_PERIOD = best_params.get("TP_BB_PERIOD", config.TP_BB_PERIOD)
    config.MB_COMPRESSION_THRESHOLD = best_params.get("MB_COMPRESSION_THRESHOLD", config.MB_COMPRESSION_THRESHOLD)
    # config.MR_TP_TARGET_MULT = best_params.get("MR_TP_TARGET_MULT", config.MR_TP_TARGET_MULT)
    config.MR_MIN_RR = best_params.get("MR_MIN_RR", config.MR_MIN_RR)
    config.TP_MIN_RR = best_params.get("TP_MIN_RR", config.TP_MIN_RR)

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
        ("QQQ", "15m", TrendPullbackStrategy()),
        ("BTC-USD", "1h", MomentumBreakoutStrategy())
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
        study_name="makeshift_trades_6mo_v5",
        storage="sqlite:///optuna_study.db?timeout=60",
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
    print("BTC PnL:", best_trial.user_attrs.get("BTC_PnL"))
    # print("GLD PnL:", best_trial.user_attrs.get("GLD_PnL"))
    # print("PDBC PnL:", best_trial.user_attrs.get("PDBC_PnL"))
    print("Total Trades:", best_trial.user_attrs.get("Total_Trades"))
    
    print("Evaluating best parameters on Out-Of-Sample (OOS) blind data...")
    oos_pnl, oos_trades, oos_pnl_by_ticker = evaluate_oos_params(best_params)
    print("OOS Net PnL:", oos_pnl)
    print("OOS Total Trades:", oos_trades)
    print("OOS SPY PnL:", oos_pnl_by_ticker["SPY"])
    print("OOS QQQ PnL:", oos_pnl_by_ticker["QQQ"])
    print("OOS BTC PnL:", oos_pnl_by_ticker["BTC-USD"])
    # print("OOS GLD PnL:", oos_pnl_by_ticker.get("GLD", 0.0))
    # print("OOS PDBC PnL:", oos_pnl_by_ticker.get("PDBC", 0.0))
    print("=" * 50)
    
    try:
        with open("best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)
            
        df_trials = study.trials_dataframe()
        df_trials.to_csv("optuna_trials_data.csv", index=False)
            
        print("Saved best parameters to best_params.json")
        print(f"Saved ALL {len(df_trials)} historical trial data to optuna_trials_data.csv for your analysis.")
    except PermissionError:
        print("Another terminal is currently saving the CSV/JSON files. Skipping file export in this terminal to prevent crashes.")
        print("Don't worry - all trial data is safely stored in the SQLite database!")
