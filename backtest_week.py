import os
import sys
import pandas as pd
from datetime import timedelta
import logging
logging.basicConfig(level=logging.CRITICAL, format='%(message)s')

# Ensure src is accessible
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum_breakout import MomentumBreakoutStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.trend_pullback import TrendPullbackStrategy
from charts.data import fetch_ohlcv
from src.backtester import simulate_trade

def run_backtest():
    print("=" * 60)
    print("MakeShiftTrades - 1 Week Backtest")
    print("=" * 60)
    
    strategies_to_test = [
        ("SPY", "15m", MeanReversionStrategy()),
        ("SPY", "15m", TrendPullbackStrategy()),
        ("QQQ", "15m", MeanReversionStrategy()),
        ("QQQ", "15m", TrendPullbackStrategy()),
        ("BTC-USD", "1h", MomentumBreakoutStrategy()),
        ("GLD", "4h", TrendFollowingStrategy()),
        ("PDBC", "4h", TrendFollowingStrategy())
    ]
    
    for ticker, tf, strategy in strategies_to_test:
        print(f"\nTesting {strategy.name} on {ticker} ({tf})...")
        try:
            fetch_period = "6mo" if tf == "4h" else "1mo"
            df = fetch_ohlcv(ticker, period=fetch_period, interval=tf)
            if df is None or len(df) < 200:
                print(f"  Insufficient data. Got {len(df) if df is not None else 'None'} bars.")
                continue
                
            # Filter for the last 7 days of trading
            last_timestamp = df.index[-1]
            one_week_ago = last_timestamp - timedelta(days=7)
            
            # Find the index where the last week starts
            test_start_idx = df.index.get_indexer([one_week_ago], method='bfill')[0]
            
            signals_generated = 0
            wins = 0
            losses = 0
            open_trades = 0
            
            for i in range(test_start_idx, len(df) - 1): # -1 because we need at least 1 future bar
                # Slice data up to the current bar to prevent lookahead bias
                window_df = df.iloc[:i+1]
                
                # Suppress the strategy internal logging to avoid spam
                import logging
                logging.getLogger("src.strategies").setLevel(logging.CRITICAL)
                logging.getLogger("charts.data").setLevel(logging.CRITICAL)
                signal = strategy.analyze(window_df, ticker)
                
                if signal:
                    outcome, exit_price, bars, pnl_per_share = simulate_trade(df, i, signal)
                    if outcome == 'NO_FILL':
                        print(f"  [{df.index[i]}] SIGNAL: {signal.direction} | Outcome: NO_FILL")
                        continue
                        
                    signals_generated += 1
                    if outcome in ('TP', 'TRAIL_SL') or pnl_per_share > 0:
                        wins += 1
                    elif outcome == 'SL' or pnl_per_share < 0:
                        losses += 1
                    else:
                        open_trades += 1
                        
                    print(f"  [{df.index[i]}] SIGNAL: {signal.direction} | R/R: {abs(signal.take_profit-signal.entry)/abs(signal.stop_loss-signal.entry):.2f} | Outcome: {outcome} ({bars} bars) | PnL/share: {pnl_per_share:+.4f}")
            
            print(f"  Summary: {signals_generated} signals | {wins} Wins, {losses} Losses, {open_trades} Open")
            if (wins + losses) > 0:
                win_rate = (wins / (wins + losses)) * 100
                print(f"  Win Rate (Closed Trades): {win_rate:.1f}%")
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    run_backtest()
