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
                
            # Always calculate trailing stops (not just after partial TP)
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
                
            # Always calculate trailing stops (not just after partial TP)
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


def run_stateful_backtest(strategies_to_test: list, days_to_test: int, starting_equity: float = 5000.0):
    """Run a stateful, event-driven backtest across multiple strategies/tickers.

    Parameters
    ----------
    strategies_to_test : list of (ticker, timeframe, strategy) tuples
        Example: [("SPY", "15m", MeanReversionStrategy()), ...]
    days_to_test : int
        Number of calendar days to simulate over the most recent data.
    starting_equity : float
        Starting account equity for position sizing.

    Returns
    -------
    None — prints a detailed summary to stdout.
    """
    import logging
    from datetime import timedelta, timezone
    from datetime import datetime

    from charts.data import fetch_ohlcv
    from src.regime_classifier import RegimeClassifier
    from src.risk_manager import RiskManager
    from src.backtester import simulate_trade

    # Suppress lower-level logs to keep output clean
    logging.basicConfig(level=logging.CRITICAL, format="%(message)s")

    # ════════════════════════════════════════════════════════════════════
    # 1. Initialize RegimeClassifier and fit on SPY daily data (2y lookback)
    # ════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("MakeShiftTrades - Stateful Event-Driven Backtest")
    print("=" * 60)
    print(f"Days to test:      {days_to_test}")
    print(f"Starting equity:   ${starting_equity:>10,.2f}")
    print()

    print("RegimeClassifier: fitting on SPY (2y/1d)...", end=" ")
    regime_classifier = RegimeClassifier(n_states=3)
    spy_df = fetch_ohlcv("SPY", period="2y", interval="1d")
    if spy_df is not None and not spy_df.empty:
        last_spy_ts = spy_df.index[-1]
        start_timestamp = last_spy_ts - timedelta(days=days_to_test)
        # Fit RegimeClassifier ONLY on data where index < start_timestamp (prior to backtest start)
        fit_df = spy_df[spy_df.index < start_timestamp]
        if len(fit_df) > 50:
            regime_classifier.fit(fit_df)
            print("fitted.")
        else:
            print("failed fitting (too little data prior to backtest start).")
    else:
        print("failed (no SPY data).")

    # ════════════════════════════════════════════════════════════════════
    # 2. Pre-fetch historical data for all unique (ticker, timeframe) with 1y horizon
    # ════════════════════════════════════════════════════════════════════
    ticker_data: dict[tuple[str, str], tuple[pd.DataFrame, int]] = {}

    for ticker, tf, strategy in strategies_to_test:
        key = (ticker, tf)
        if key in ticker_data:
            continue

        print(f"  Fetching {ticker} ({tf}, period=1y)...", end=" ")
        df = fetch_ohlcv(ticker, period="1y", interval=tf)
        if df is None or len(df) < 50:
            print("SKIPPED (insufficient data)")
            continue

        # Identify the index corresponding to *now - days_to_test*
        last_ts = df.index[-1]
        test_start = last_ts - timedelta(days=days_to_test)
        test_start_idx = df.index.get_indexer([test_start], method="bfill")[0]

        ticker_data[key] = (df, test_start_idx)
        print(f"{len(df)} bars [{df.index[0]} -> {df.index[-1]}] "
              f"| test window: {len(df) - test_start_idx} bars")

    if not ticker_data:
        print("\nNo data available. Exiting.")
        return

    # ════════════════════════════════════════════════════════════════════
    # 3. Generate all raw signals by walking bars in the test window
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print("GENERATING SIGNALS")
    print("-" * 60)

    all_signals: list[dict] = []

    for ticker, tf, strategy in strategies_to_test:
        key = (ticker, tf)
        if key not in ticker_data:
            continue
        df, test_start_idx = ticker_data[key]

        print(f"  {strategy.name:20s} on {ticker:8s} ({tf:3s})...", end=" ")
        signals_count = 0

        # Silence verbose loggers during analysis
        logging.getLogger("src.strategies").setLevel(logging.CRITICAL)
        logging.getLogger("charts.data").setLevel(logging.CRITICAL)

        for i in range(test_start_idx, len(df) - 1):
            window_df = df.iloc[: i + 1]
            signal = strategy.analyze(window_df, ticker)
            if signal is not None:
                all_signals.append({
                    "time": df.index[i],          # bar timestamp
                    "signal": signal,
                    "ticker": ticker,
                    "strategy_name": strategy.name,
                    "df": df,
                    "entry_idx": i,
                })
                signals_count += 1

        print(f"{signals_count} raw signals")

    # ════════════════════════════════════════════════════════════════════
    # 4. Sort all signals chronologically, prioritizing higher confidence
    # ════════════════════════════════════════════════════════════════════
    all_signals.sort(key=lambda x: (x["time"], -x["signal"].confidence))
    print(f"\n  Total raw signals (pre-risk): {len(all_signals)}")

    if not all_signals:
        print("No signals generated. Exiting.")
        return

    # ════════════════════════════════════════════════════════════════════
    # 5. Initialize RiskManager and simulation state
    # ════════════════════════════════════════════════════════════════════
    risk_manager = RiskManager(
        max_risk_pct=0.04,
        max_position_pct=5.00,
        max_notional=50000.0,
        max_positions=2,
    )

    account_equity = starting_equity
    active_trades: list[dict] = []
    peak_equity = starting_equity
    max_drawdown = 0.0
    regime_cache = {}

    total_closed_pnl = 0.0
    wins = 0
    losses = 0
    strategy_stats = {}
    strategy_stats = {}

    # ════════════════════════════════════════════════════════════════════
    # 6. Iterate through signals chronologically (event loop)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print("EXECUTING SIGNALS (stateful simulation)")
    print("-" * 60)

    for entry in all_signals:
        signal_time = entry["time"]
        signal = entry["signal"]
        ticker = entry["ticker"]
        df = entry["df"]
        entry_idx = entry["entry_idx"]

        # ── 6a. Remove positions whose exit_time <= current_signal_time ──
        remaining: list[dict] = []
        for pos in active_trades:
            exit_time = pos.get("exit_time")
            if exit_time is not None and exit_time <= signal_time:
                # Trade is closed — settle PnL
                trade_pnl = pos.get("pnl", 0.0)
                total_closed_pnl += trade_pnl
                account_equity += trade_pnl
                if trade_pnl > 0:
                    wins += 1
                else:
                    losses += 1

                strat = pos.get("strategy_name", "unknown")
                if strat not in strategy_stats:
                    strategy_stats[strat] = {"wins": 0, "losses": 0, "pnl": 0.0}
                strategy_stats[strat]["pnl"] += trade_pnl
                if trade_pnl > 0:
                    strategy_stats[strat]["wins"] += 1
                else:
                    strategy_stats[strat]["losses"] += 1
                # Update peak equity and max drawdown when trades close
                peak_equity = max(peak_equity, account_equity)
                max_drawdown = max(max_drawdown, (peak_equity - account_equity) / peak_equity)
            else:
                remaining.append(pos)
        active_trades = remaining

        # ── 6b. Build RiskManager-format position list and detailed position_state ──
        current_positions = [
            {
                "symbol": t["symbol"],
                "side": t["side"],
                "market_value": t["notional"],
            }
            for t in active_trades
        ]

        position_state = {
            t["symbol"]: {
                "entry_price": t["entry_price"],
                "stop_loss": t["stop_loss"],
                "qty": t["qty"]
            }
            for t in active_trades
        }

        # ── 6c. Determine HMM regime dynamically with lookahead elimination (cached by date) ──
        signal_date = signal_time.date()
        if signal_date in regime_cache:
            current_regime, regime_id = regime_cache[signal_date]
        else:
            if spy_df is not None and not spy_df.empty:
                # Ensure signal_time and spy_df index have compatible timezone awareness.
                if signal_time.tzinfo is not None and spy_df.index.tz is None:
                    spy_df_filtered = spy_df.tz_localize('UTC')
                    spy_df_filtered = spy_df_filtered[spy_df_filtered.index <= signal_time]
                elif signal_time.tzinfo is None and spy_df.index.tz is not None:
                    spy_df_filtered = spy_df.tz_convert(None)
                    spy_df_filtered = spy_df_filtered[spy_df_filtered.index <= signal_time]
                else:
                    spy_df_filtered = spy_df[spy_df.index <= signal_time]

                current_regime, regime_id = regime_classifier.predict(spy_df_filtered)
            else:
                current_regime, regime_id = "unknown", -1
            regime_cache[signal_date] = (current_regime, regime_id)

        # ── 6d. Call RiskManager.approve() with position_state and current_positions ──
        approved, notional, reason = risk_manager.approve(
            signal=signal,
            account_equity=account_equity,
            current_positions=current_positions,
            active_orders=[],
            position_state=position_state,
            signal_regime=current_regime,
        )

        if not approved:
            continue

        # ── 6e. Simulate the trade ──
        outcome, exit_price, bars, pnl_per_share = simulate_trade(df, entry_idx, signal)

        if outcome == "NO_FILL":
            continue

        # ── 6f. Calculate exit timestamp from bars ──
        exit_bar_idx = min(entry_idx + bars, len(df) - 1)
        exit_time = df.index[exit_bar_idx]

        # ── 6g. Convert per-share PnL to notional-based PnL ──
        shares = notional / signal.entry if signal.entry > 0 else 0
        trade_pnl = pnl_per_share * shares

        # ── 6h. Add the new position to tracking with rich state ──
        side = "long" if signal.direction == "BUY" else "short"
        active_trades.append({
            "symbol": ticker,
            "side": side,
            "exit_time": exit_time,
            "pnl": trade_pnl,
            "notional": notional,
            "entry_price": signal.entry,
            "stop_loss": signal.stop_loss,
            "qty": shares,
            "entry_time": signal_time,
            "outcome": outcome,
            "bars": bars, "strategy_name": signal.strategy_name,
        })

        # ── 6i. Print one-line trade log ──
        try:
            rr = abs(signal.take_profit - signal.entry) / abs(signal.stop_loss - signal.entry)
        except ZeroDivisionError:
            rr = 0.0
        print(
            f"  [{signal_time}] {signal.direction:4s} {ticker:8s} "
            f"({signal.strategy_name:18s}) | "
            f"R/R: {rr:.2f} | {outcome:8s} ({bars:2d} bars) | "
            f"PnL: ${trade_pnl:+.2f}"
        )

    # ── 7. Close any remaining open positions ──
    for pos in active_trades:
        trade_pnl = pos.get("pnl", 0.0)
        total_closed_pnl += trade_pnl
        account_equity += trade_pnl
        if trade_pnl > 0:
            wins += 1
        else:
            losses += 1
        # Update peak equity and max drawdown when trades close at end of backtest
        peak_equity = max(peak_equity, account_equity)
        max_drawdown = max(max_drawdown, (peak_equity - account_equity) / peak_equity)

    # ════════════════════════════════════════════════════════════════════
    # 8. Print detailed summary
    # ════════════════════════════════════════════════════════════════════
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    max_drawdown_pct = max_drawdown * 100

    print()
    print("=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"  Days Tested:         {days_to_test}")
    print(f"  Starting Equity:     ${starting_equity:>10,.2f}")
    print(f"  Final Equity:        ${account_equity:>10,.2f}")
    print(f"  Total PnL:           ${total_closed_pnl:>+10,.2f}")
    print(f"  Peak Equity:         ${peak_equity:>10,.2f}")
    print(f"  Max Drawdown:        {max_drawdown_pct:>9.2f}%")
    print(f"  Total Trades Executed: {total_trades}")
    print(f"  Wins / Losses:       {wins} / {losses}")
    print(f"  Win Rate:            {win_rate:>9.1f}%")
    print("-" * 60)
    print("  Strategy Breakdown:")
    for strat, stats in strategy_stats.items():
        s_w = stats["wins"]
        s_l = stats["losses"]
        s_tot = s_w + s_l
        s_wr = (s_w / s_tot * 100) if s_tot > 0 else 0.0
        s_pnl = stats["pnl"]
        print(f"    - {strat:18s}: {s_tot:3d} trades | Win Rate: {s_wr:>5.1f}% | PnL: ${s_pnl:>+8,.2f}")
    print("=" * 60)
