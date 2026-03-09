from src.trader import Trader
import config


def main():
    print("MakeShiftTrades - Daytrading Bot")
    print(f"  Symbol     : {config.SYMBOL}")
    print(f"  Max amount : ${config.MAX_TRADE_AMOUNT}")
    print(f"  Dry run    : {config.DRY_RUN}")

    trader = Trader(
        api_key=config.API_KEY,
        api_secret=config.API_SECRET,
        symbol=config.SYMBOL,
        max_trade_amount=config.MAX_TRADE_AMOUNT,
        dry_run=config.DRY_RUN,
    )
    try:
        trader.run()
    except NotImplementedError as exc:
        print(f"\n[ERROR] {exc}")


if __name__ == "__main__":
    main()
