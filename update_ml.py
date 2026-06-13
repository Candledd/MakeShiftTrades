import re

def main():
    with open("ml_optimizer.py", "r") as f:
        content = f.read()

    # Track individual PnL
    # pnl_by_ticker = {"SPY": 0.0, "QQQ": 0.0, "BTC-USD": 0.0}
    content = content.replace(
        'pnl_by_ticker = {"SPY": 0.0, "QQQ": 0.0, "BTC-USD": 0.0}',
        'pnl_by_ticker = {"SPY": 0.0, "QQQ": 0.0, "BTC-USD": 0.0}\n    pnl_by_strategy = {"mean_reversion": 0.0, "trend_pullback": 0.0, "momentum_breakout": 0.0}'
    )

    # pnl_by_ticker[ticker] += trade_pnl
    content = content.replace(
        'pnl_by_ticker[ticker] += trade_pnl',
        'pnl_by_ticker[ticker] += trade_pnl\n                pnl_by_strategy[strategy.name] += trade_pnl'
    )

    # trial.set_user_attr
    content = content.replace(
        'trial.set_user_attr("BTC_PnL", pnl_by_ticker["BTC-USD"])',
        'trial.set_user_attr("BTC_PnL", pnl_by_ticker["BTC-USD"])\n    trial.set_user_attr("MR_PnL", pnl_by_strategy["mean_reversion"])\n    trial.set_user_attr("TP_PnL", pnl_by_strategy["trend_pullback"])\n    trial.set_user_attr("MB_PnL", pnl_by_strategy["momentum_breakout"])'
    )

    with open("ml_optimizer.py", "w") as f:
        f.write(content)

if __name__ == "__main__":
    main()