from src.utils import log


class Trader:
    def __init__(self, api_key, api_secret, symbol, max_trade_amount, dry_run=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.max_trade_amount = max_trade_amount
        self.dry_run = dry_run
        

    def get_price(self) -> float:
        """Fetch the current price for the symbol. Replace with a real API call."""
        raise NotImplementedError(
            "get_price() must be implemented. "
            "Integrate with your data provider or broker API."
        )

    def should_buy(self, price):
        """Return True if the bot should place a buy order. Implement your strategy here."""
        return False

    def should_sell(self, price):
        """Return True if the bot should place a sell order. Implement your strategy here."""
        return False

    def place_order(self, side, price):
        if self.dry_run:
            log(f"[DRY RUN] Would place {side} order for {self.symbol} at ${price:.2f}")
        else:
            log(f"Placing {side} order for {self.symbol} at ${price:.2f}")
            # TODO: integrate with your broker's API

    def run(self):
        log("Trader started.")
        price = self.get_price()
        log(f"Current price of {self.symbol}: ${price:.2f}")

        if self.should_buy(price):
            self.place_order("BUY", price)
        elif self.should_sell(price):
            self.place_order("SELL", price)
        else:
            log("No trade signal. Holding.")
