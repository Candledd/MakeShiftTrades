import config
import pandas as pd
try:
    from src.utils import parse_timeframe_to_hours
    from src.trailing_stops import calculate_trailing_stop
except ImportError:
    from utils import parse_timeframe_to_hours
    from trailing_stops import calculate_trailing_stop


import math

def simulate_trade(df, entry_idx, signal):
    future_df = df.iloc[entry_idx + 1:]
    
    # 1. Limit fill simulation
    if len(future_df) == 0:
        return 'NO_FILL', signal.entry, 0, 0.0
    
    tick_size = 0.01 if "BTC" not in signal.ticker.upper() and "ETH" not in signal.ticker.upper() else 0.5
    
    offset = 0
    if getattr(signal, 'order_type', 'MARKET') != 'MARKET':
        # LIMIT order: look ahead up to ttl_bars
        ttl_hours = getattr(config, 'ORDER_TTL_HOURS', 2.0)
        tf_hours = parse_timeframe_to_hours(getattr(signal, 'timeframe', '15m'))
        ttl_bars = max(1, math.ceil(ttl_hours / tf_hours))
        
        fill_offset = -1
        for j in range(min(ttl_bars, len(future_df))):
            row = future_df.iloc[j]
            if signal.direction == 'BUY':
                if row['Low'] <= signal.entry + tick_size:
                    fill_offset = j
                    break
            else:
                if row['High'] >= signal.entry - tick_size:
                    fill_offset = j
                    break
        
        if fill_offset == -1:
            return 'NO_FILL', signal.entry, ttl_bars, 0.0
        
        offset = fill_offset
        
        # Slice future_df starting from the fill bar
        future_df = future_df.iloc[offset:]
    
    # Calculate max_bars from signal's time_stop_bars
    max_bars = getattr(signal, 'time_stop_bars', 10)
    
    qty_left = 1.0
    partial_taken = False
    current_sl = signal.stop_loss
    net_pnl_per_share = 0.0
    
    slippage_pct = getattr(config, 'BACKTEST_SLIPPAGE_FRICTION_PCT', 0.0005)
    
    # Precompute trailing indicators if needed
    trailing_logic = getattr(signal, 'trailing_stop_logic', 'default')
    
    for i, (_, row) in enumerate(future_df.iterrows()):
        curr_idx = entry_idx + 1 + offset + i
        # Enforce Time Stop
        if i >= max_bars:
            # Exit remaining quantity at current Close
            exit_price = row['Close']
            exit_slippage = exit_price * (slippage_pct / 2.0)
            entry_slippage = signal.entry * (slippage_pct / 2.0)
            if signal.direction == 'BUY':
                leg_pnl = (exit_price - signal.entry - (entry_slippage + exit_slippage)) * qty_left
            else:
                leg_pnl = (signal.entry - exit_price - (entry_slippage + exit_slippage)) * qty_left
            net_pnl_per_share += leg_pnl
            return 'TIME_STOP', exit_price, offset + i + 1, net_pnl_per_share
            
        high = row['High']
        low = row['Low']
        close = row['Close']
        
        if signal.direction == 'BUY':
            if low <= current_sl:
                # Stop Loss hit
                exit_price = current_sl
                exit_slippage = exit_price * (slippage_pct / 2.0)
                entry_slippage = signal.entry * (slippage_pct / 2.0)
                leg_pnl = (exit_price - signal.entry - (entry_slippage + exit_slippage)) * qty_left
                net_pnl_per_share += leg_pnl
                return 'SL' if not partial_taken else 'TRAIL_SL', exit_price, offset + i + 1, net_pnl_per_share
                
            if not partial_taken and high >= signal.take_profit:
                # Partial TP exit (0.5 qty)
                exit_price = signal.take_profit
                exit_slippage = exit_price * (slippage_pct / 2.0)
                entry_slippage = signal.entry * (slippage_pct / 2.0)
                partial_qty = qty_left * 0.5
                leg_pnl = (exit_price - signal.entry - (entry_slippage + exit_slippage)) * partial_qty
                net_pnl_per_share += leg_pnl
                qty_left -= partial_qty
                partial_taken = True
                
            if partial_taken:
                current_sl = calculate_trailing_stop(
                    logic_type=trailing_logic,
                    current_price=close,
                    current_sl=current_sl,
                    direction='BUY',
                    df=df.iloc[max(0, curr_idx + 1 - 100):curr_idx + 1],
                    atr=getattr(signal, 'atr', 0.0),
                    entry_price=signal.entry,
                    tp_filled=partial_taken
                )
                
        elif signal.direction == 'SELL':
            if high >= current_sl:
                # Stop Loss hit
                exit_price = current_sl
                exit_slippage = exit_price * (slippage_pct / 2.0)
                entry_slippage = signal.entry * (slippage_pct / 2.0)
                leg_pnl = (signal.entry - exit_price - (entry_slippage + exit_slippage)) * qty_left
                net_pnl_per_share += leg_pnl
                return 'SL' if not partial_taken else 'TRAIL_SL', exit_price, offset + i + 1, net_pnl_per_share
                
            if not partial_taken and low <= signal.take_profit:
                # Partial TP exit (0.5 qty)
                exit_price = signal.take_profit
                exit_slippage = exit_price * (slippage_pct / 2.0)
                entry_slippage = signal.entry * (slippage_pct / 2.0)
                partial_qty = qty_left * 0.5
                leg_pnl = (signal.entry - exit_price - (entry_slippage + exit_slippage)) * partial_qty
                net_pnl_per_share += leg_pnl
                qty_left -= partial_qty
                partial_taken = True
                
            if partial_taken:
                current_sl = calculate_trailing_stop(
                    logic_type=trailing_logic,
                    current_price=close,
                    current_sl=current_sl,
                    direction='SELL',
                    df=df.iloc[max(0, curr_idx + 1 - 100):curr_idx + 1],
                    atr=getattr(signal, 'atr', 0.0),
                    entry_price=signal.entry,
                    tp_filled=partial_taken
                )
                
    # EOD or end of data close
    exit_price = future_df["Close"].iloc[-1] if len(future_df) > 0 else signal.entry
    exit_slippage = exit_price * (slippage_pct / 2.0)
    entry_slippage = signal.entry * (slippage_pct / 2.0)
    if signal.direction == 'BUY':
        leg_pnl = (exit_price - signal.entry - (entry_slippage + exit_slippage)) * qty_left
    else:
        leg_pnl = (signal.entry - exit_price - (entry_slippage + exit_slippage)) * qty_left
    net_pnl_per_share += leg_pnl
    return 'OPEN', exit_price, offset + len(future_df), net_pnl_per_share
