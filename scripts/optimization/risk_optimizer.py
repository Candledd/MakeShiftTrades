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
    months = 6
    
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
    # --- Risk & Leverage Parameters ---
    max_position_pct = trial.suggest_float("MAX_POSITION_PCT", 0.5, 5.0, step=0.511)
    max_risk_pct = trial.suggest_float("MAX_RISK_PCT", 0.01, 0.06, step=0.01)
    risk_tier_equity = trial.suggest_float("RISK_TIER_EQUITY_PCT", 0.01, 0.06, step=0.01)
    
    # --- Strategy-specific Risk Limits ---
    tp_stop_mult = trial.suggest_float("TP_STOP_MULT", 1.0, 3.0, step=0.1)
    tp_min_rr = trial.suggest_float("TP_MIN_RR", 1.0, 2.0, step=0.1)
    mr_stop_mult = trial.suggest_float("MR_STOP_MULT", 1.0, 3.0, step=0.1)
    # Mean Reversion inherently has low R/R (scalping local means). Forcing it > 1.0 breaks it.
    mr_min_rr = trial.suggest_float("MR_MIN_RR", 0.3, 1.0, step=0.1)
    
    # Apply parameters to config
    config.MAX_POSITION_PCT = max_position_pct
    config.MAX_RISK_PCT = max_risk_pct
    config.RISK_TIER_EQUITY_PCT = risk_tier_equity
    
    config.TP_STOP_MULT = tp_stop_mult
    config.TP_MIN_RR = tp_min_rr
    config.MR_STOP_MULT = mr_stop_mult
    config.MR_MIN_RR = mr_min_rr
    
    # Portfolio-level safety constraints — keep tight to prevent runaway drawdown.
    # These act as hard guards during backtest, not just score penalties.
    config.MAX_OPEN_PORTFOLIO_RISK_PCT = 0.10   # Max 6 concurrent trades (heat = risk_pct each)
    config.MAX_CLUSTER_RISK_PCT = 0.50
    config.MAX_DAILY_LOSS_PCT = -0.05           # Halt new entries after -5% daily
    config.MAX_WEEKLY_LOSS_PCT = -0.12          # Halt new entries after -12% weekly

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
    if total_signals >= 50 and trades:
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
        
        # ── Consistency & trade quality metrics ─────────────────────────
        # Equity curve R² (linearity): higher = more consistent returns
        trade_idx = pd.Series(range(len(trades_df)))
        equity_r = trade_idx.corr(running_pnl)
        equity_r_squared = equity_r ** 2 if pd.notna(equity_r) else 0.0
        
        # Downside deviation (Sortino-style): penalizes frequent small losses
        r_values = trades_df['pnl'] / RISK_DOLLARS
        negative_r = r_values[r_values < 0]
        downside_deviation_R = negative_r.std() if len(negative_r) > 1 else 0.0
        
        # Average win / loss ratio (payoff structure)
        win_trades = trades_df[trades_df['pnl'] > 0]
        loss_trades = trades_df[trades_df['pnl'] < 0]
        avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0.0
        avg_loss = abs(loss_trades['pnl'].mean()) if len(loss_trades) > 0 else 0.0
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 3.0
        
        # RISK Optimizer score — optimized for DRAWDOWN CONTROL / RISK EFFICIENCY.
        # Tunes risk params (position size, stop mult, min RR, equity tiers) that
        # DIRECTLY control drawdown. Penalties here must be aggressive because this
        # optimizer's entire purpose is containing drawdown while preserving profit.
        # A high-profit / high-drawdown param set should score WORSE than
        # moderate-profit / low-drawdown here.
        score = (
            # Core risk-adjusted return
            (net_R * 0.8)                            # Profit discounted — not the main goal
            - (2.5 * max_drawdown_R)                 # Drawdown penalized 2.5x — this is the job
            - (4.0 * max(0, max_drawdown_R - 3))     # Tail penalty starts at just 3R ($150 DD)
            # Consistency
            + (0.5 * equity_r_squared)               # Equity curve linearity bonus
            - (0.3 * downside_deviation_R)           # Sortino-style penalty for frequent small losses
            # Trade quality
            + (0.02 * (total_signals ** 0.5))        # Trade bonus with diminishing returns
            + (0.3 * min(avg_win_loss_ratio, 3.0))   # Payoff ratio bonus (capped)
            # Diversification
            - 1.0 * abs(strategy_concentration)      # Heavy diversification requirement
            - 0.5 * parameter_instability_penalty    # Heavy single-trade concentration penalty
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
        score = -1000.0 - (50 - total_signals) * 10
        
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
    config.MAX_POSITION_PCT = best_params.get("MAX_POSITION_PCT", config.MAX_POSITION_PCT)
    config.MAX_RISK_PCT = best_params.get("MAX_RISK_PCT", config.MAX_RISK_PCT)
    config.RISK_TIER_EQUITY_PCT = best_params.get("RISK_TIER_EQUITY_PCT", config.RISK_TIER_EQUITY_PCT)
    
    config.TP_STOP_MULT = best_params.get("TP_STOP_MULT", config.TP_STOP_MULT)
    config.TP_MIN_RR = best_params.get("TP_MIN_RR", config.TP_MIN_RR)
    config.MR_STOP_MULT = best_params.get("MR_STOP_MULT", config.MR_STOP_MULT)
    config.MR_MIN_RR = best_params.get("MR_MIN_RR", config.MR_MIN_RR)
    
    # Apply same portfolio-level constraints as in objective
    config.MAX_OPEN_PORTFOLIO_RISK_PCT = 0.06
    config.MAX_CLUSTER_RISK_PCT = 0.50
    config.MAX_DAILY_LOSS_PCT = -0.05
    config.MAX_WEEKLY_LOSS_PCT = -0.12

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
        study_name="makeshift_trades_risk_v1",
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
        with open("data/best_risk_params.json", "w") as f:
            json.dump(best_params, f, indent=4)
            
        df_trials = study.trials_dataframe()
        df_trials.to_csv("data/optuna_trials_risk.csv", index=False)
            
        print("Saved best parameters to data/best_risk_params.json")
        print(f"Saved ALL {len(df_trials)} historical trial data to optuna_trials_data.csv for your analysis.")
    except PermissionError:
        print("Another terminal is currently saving the CSV/JSON files. Skipping file export in this terminal to prevent crashes.")
        print("Don't worry - all trial data is safely stored in the SQLite database!")
