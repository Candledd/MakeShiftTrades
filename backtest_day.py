import os
import sys
import pandas as pd
from datetime import timedelta
import logging

# Ensure src is accessible
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load env vars so config takes our changes
from dotenv import load_dotenv
load_dotenv()
import config

from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum_breakout import MomentumBreakoutStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.trend_pullback import TrendPullbackStrategy
from charts.data import fetch_ohlcv
from src.backtester import simulate_trade

logging.basicConfig(level=logging.CRITICAL, format='%(message)s')
logging.getLogger("src.strategies").setLevel(logging.CRITICAL)
logging.getLogger("charts.data").setLevel(logging.CRITICAL)

def run_backtest():
    print("=" * 60)
    print("MakeShiftTrades - 1 Day Backtest (Scalping Profile)")
    print("=" * 60)
    print(f"Active Scalp Limits: BB_STD={config.MR_BB_STD}, RSI_OS={config.MR_RSI_OVERSOLD}, RSI_OB={config.MR_RSI_OVERBOUGHT}")
    print("-" * 60)
    
    strategies_to_test = [
        ("SPY", "15m", MeanReversionStrategy()),
        ("SPY", "15m", TrendPullbackStrategy()),
        ("QQQ", "15m", MeanReversionStrategy()),
        ("QQQ", "15m", TrendPullbackStrategy()),
        ("BTC-USD", "1h", MomentumBreakoutStrategy()),
    ]
    
    for ticker, tf, strategy in strategies_to_test:
        print(f"\nTesting {strategy.name} on {ticker} ({tf})...")
        try:
            # Fetch 1 month to ensure enough data for indicators
            df = fetch_ohlcv(ticker, period="1mo", interval=tf)
            if df is None or len(df) < 200:
                print(f"  Insufficient data (Got {len(df) if df is not None else 0} bars).")
                continue
                
            # Filter for the last 2 calendar days
            last_timestamp = df.index[-1]
            one_day_ago = last_timestamp - timedelta(days=2)
            
            test_start_idx = df.index.get_indexer([one_day_ago], method='bfill')[0]
            
            signals_generated = 0
            wins = 0
            losses = 0
            open_trades = 0
            
            vol_rejects = 0
            rr_rejects = 0

            class CounterHandler(logging.Handler):
                def emit(self, record):
                    nonlocal vol_rejects, rr_rejects
                    msg = self.format(record)
                    if "volume too low" in msg:
                        vol_rejects += 1
                    elif "bad R/R" in msg or "poor R/R" in msg:
                        rr_rejects += 1

            handler = CounterHandler()
            log = logging.getLogger("src.strategies")
            log.setLevel(logging.DEBUG)
            log.addHandler(handler)

            for i in range(test_start_idx, len(df) - 1):
                window_df = df.iloc[:i+1]
                
                # Reset cooldowns so backtest can show all potential signals
                strategy._last_signal_time = {}
                
                signal = strategy.analyze(window_df, ticker)
                
                if signal:
                    outcome, exit_price, bars, pnl_per_share = simulate_trade(df, i, signal)
                    if outcome == 'NO_FILL':
                        print(f"  [{df.index[i].strftime('%Y-%m-%d %H:%M')}] SIGNAL: {signal.direction} | Outcome: NO_FILL")
                        continue
                        
                    signals_generated += 1
                    if outcome in ('TP', 'TRAIL_SL') or pnl_per_share > 0:
                        wins += 1
                    elif outcome == 'SL' or pnl_per_share < 0:
                        losses += 1
                    else:
                        open_trades += 1
                        
                    print(f"  [{df.index[i].strftime('%Y-%m-%d %H:%M')}] SIGNAL: {signal.direction} | R/R: {abs(signal.take_profit-signal.entry)/abs(signal.stop_loss-signal.entry):.2f} | Outcome: {outcome} ({bars} bars) | PnL/share: {pnl_per_share:+.4f}")
            
            print(f"  Summary: {signals_generated} signals | {wins} Wins, {losses} Losses, {open_trades} Open")
            print(f"  Rejects: {vol_rejects} due to Volume | {rr_rejects} due to R/R")
            log.removeHandler(handler)
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    run_backtest()
