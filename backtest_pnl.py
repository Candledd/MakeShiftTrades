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
from charts.data import fetch_ohlcv

# Suppress debug logging during backtest to keep console clean
logging.basicConfig(level=logging.CRITICAL, format='%(message)s')
logging.getLogger("src.strategies").setLevel(logging.CRITICAL)
logging.getLogger("charts.data").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

def simulate_trade(df, entry_idx, signal):
    """Simulate future price action to determine outcome (TP, SL, OPEN)."""
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
            
    # If the trade is still open at the end of the data, use the last close price
    return 'OPEN', future_df["Close"].iloc[-1] if len(future_df) > 0 else signal.entry, len(future_df)

def run_pnl_backtest(account_size=5000.0, risk_pct=1.0, months=1):
    print("=" * 70)
    print(f"{months}-Month Scalp Backtest (Simulated PnL)")
    print(f"Assuming ${account_size:,.0f} starting capital | {risk_pct}% Risk per Trade (${account_size * (risk_pct/100):.2f})")
    print("=" * 70)
    
    configs = [
        ("SPY", "15m", MeanReversionStrategy()),
        ("QQQ", "15m", MeanReversionStrategy()),
        ("BTC-USD", "1h", MomentumBreakoutStrategy())
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
        
        for i in range(test_start_idx, len(df) - 1):
            window_df = df.iloc[:i+1]
            
            # Reset internal cooldowns to allow testing all valid setups over time
            strategy._last_signal_time = {} 
            
            signal = strategy.analyze(window_df, ticker)
            if signal:
                ticker_signals += 1
                outcome, exit_price, bars = simulate_trade(df, i, signal)
                
                # Calculate PnL mathematically based on standard Risk/Reward principles
                risk_per_share = abs(signal.entry - signal.stop_loss)
                if risk_per_share == 0: continue
                
                qty = RISK_DOLLARS / risk_per_share
                
                if signal.direction == 'BUY':
                    trade_pnl = (exit_price - signal.entry) * qty
                else:
                    trade_pnl = (signal.entry - exit_price) * qty
                
                ticker_pnl += trade_pnl
                
                if outcome == 'TP': 
                    ticker_wins += 1
                elif outcome == 'SL': 
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
