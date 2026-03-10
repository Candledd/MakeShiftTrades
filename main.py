import config
from src.trader import Trader


def main():
    trader = Trader(
        api_key=config.API_KEY,
        api_secret=config.API_SECRET,
        symbol=config.SYMBOL,
        max_trade_amount=config.MAX_TRADE_AMOUNT,
        dry_run=config.DRY_RUN,
        interval=config.INTERVAL,
        period=config.PERIOD,
        ms_term=config.MS_TERM,
        min_rr=config.MIN_RR,
        require_ob=config.REQUIRE_OB,
        require_engulfing=config.REQUIRE_ENGULFING,
    )
    trader.run(
        scan_interval_seconds=config.SCAN_INTERVAL_SECONDS,
        max_scans=config.MAX_SCANS,
        max_trades_per_day=config.MAX_TRADES_PER_DAY,
    )


if __name__ == "__main__":
    main()
