import time
from typing import Literal, Optional

import yfinance as yf

from src.strategy import SMCStrategy, TradeSignal
from src.utils import log

# ─────────────────────────────────────────────────────────────────────────────
# Position tracking
# ─────────────────────────────────────────────────────────────────────────────

class _Position:
    """Tracks a single open paper/live position."""

    def __init__(
        self,
        direction: Literal["BUY", "SELL"],
        entry: float,
        stop_loss: float,
        take_profit: float,
        shares: int,
        symbol: str,
    ) -> None:
        self.direction   = direction
        self.entry       = entry
        self.stop_loss   = stop_loss
        self.take_profit = take_profit
        self.shares      = shares
        self.symbol      = symbol

    def is_stopped_out(self, price: float) -> bool:
        if self.direction == "BUY":
            return price <= self.stop_loss
        return price >= self.stop_loss

    def is_target_hit(self, price: float) -> bool:
        if self.direction == "BUY":
            return price >= self.take_profit
        return price <= self.take_profit

    def pnl(self, price: float) -> float:
        mult = 1 if self.direction == "BUY" else -1
        return mult * (price - self.entry) * self.shares


# ─────────────────────────────────────────────────────────────────────────────
# Trader
# ─────────────────────────────────────────────────────────────────────────────

class Trader:
    """ICT / SMC day-trading bot.

    Uses the SMCStrategy for signal generation and yfinance for real-time
    (delayed) price data.  Set dry_run=False and implement _execute_order()
    to connect to a live broker (e.g. Alpaca, IBKR, TD Ameritrade).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        max_trade_amount: float,
        dry_run: bool = True,
        # Strategy knobs — can be tuned without touching strategy.py
        interval: str = "5m",
        period: str = "5d",
        ms_term: str = "intermediate",
        min_rr: float = 2.0,
        require_ob: bool = False,
        require_engulfing: bool = False,
    ) -> None:
        self.api_key          = api_key
        self.api_secret       = api_secret
        self.symbol           = symbol
        self.max_trade_amount = max_trade_amount
        self.dry_run          = dry_run

        self.strategy = SMCStrategy(
            symbol=symbol,
            interval=interval,
            period=period,
            ms_term=ms_term,
            min_rr=min_rr,
            require_ob=require_ob,
            require_engulfing=require_engulfing,
        )

        self._position: Optional[_Position] = None
        self._trades_today: int = 0

    # ------------------------------------------------------------------ #
    # Price data                                                           #
    # ------------------------------------------------------------------ #

    def get_price(self) -> float:
        """Fetch the latest 1-min close from yfinance (≈15-min delayed)."""
        ticker = yf.Ticker(self.symbol)
        hist = ticker.history(period="1d", interval="1m")
        if hist.empty:
            raise ValueError(f"Could not fetch price for '{self.symbol}'.")
        return float(hist["Close"].iloc[-1])

    # ------------------------------------------------------------------ #
    # Order execution                                                      #
    # ------------------------------------------------------------------ #

    def _calc_shares(self, entry: float, stop_loss: float) -> int:
        """Size position so total risk ≤ max_trade_amount."""
        risk_per_share = abs(entry - stop_loss)
        if risk_per_share == 0:
            return 1
        return max(1, int(self.max_trade_amount / risk_per_share))

    def place_order(
        self,
        signal: TradeSignal,
    ) -> None:
        shares = self._calc_shares(signal.entry, signal.stop_loss)

        if self.dry_run:
            log(
                f"[DRY RUN] {signal.direction} {shares} share(s) of {self.symbol} "
                f"@ ${signal.entry:.2f} | SL: ${signal.stop_loss:.2f} | "
                f"TP: ${signal.take_profit:.2f} | R/R: {signal.risk_reward:.2f} | "
                f"Confidence: {signal.confidence.upper()}"
            )
            log(f"          Reason: {signal.reason}")
        else:
            self._execute_order(signal, shares)

        self._position = _Position(
            direction=signal.direction,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            shares=shares,
            symbol=self.symbol,
        )
        self._trades_today += 1

    def _execute_order(self, signal: TradeSignal, shares: int) -> None:
        """Override this method to connect to a live broker API.

        Example integrations:
          - Alpaca : import alpaca_trade_api as tradeapi
          - IBKR   : from ib_insync import IB, MarketOrder
          - Robinhood: import robin_stocks.robinhood as rh
        """
        raise NotImplementedError(
            "Live order execution is not implemented.  "
            "Override _execute_order() or use dry_run=True."
        )

    def close_position(self, reason: str, current_price: float) -> None:
        if self._position is None:
            return
        pnl = self._position.pnl(current_price)
        log(
            f"[CLOSE] {reason} | {self._position.direction} "
            f"{self._position.shares} share(s) @ ${current_price:.2f} | "
            f"P&L: ${pnl:+.2f}"
        )
        self._position = None

    # ------------------------------------------------------------------ #
    # Position monitoring                                                  #
    # ------------------------------------------------------------------ #

    def _check_position(self, price: float) -> None:
        """Check SL/TP on the open position."""
        if self._position is None:
            return
        if self._position.is_target_hit(price):
            self.close_position("TARGET HIT", price)
        elif self._position.is_stopped_out(price):
            self.close_position("STOP HIT  ", price)

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    def run(
        self,
        scan_interval_seconds: int = 60,
        max_scans: int = 390,       # ~6.5 hours at 1-min cadence
        max_trades_per_day: int = 3,
    ) -> None:
        """Scan for ICT/SMC trade setups on a fixed interval.

        Parameters
        ----------
        scan_interval_seconds : int
            Seconds to sleep between scans (default 60).
        max_scans : int
            Total number of scans before the session ends (default 390).
        max_trades_per_day : int
            Cap on trades placed in one session (default 3).
        """
        log("=" * 60)
        log("MakeShiftTrades — ICT SMC Day Trading Bot")
        log(self.strategy.describe())
        log(f"  Dry run  : {self.dry_run}")
        log("=" * 60)

        for scan in range(1, max_scans + 1):
            log(f"─── Scan #{scan}/{max_scans} ───")

            try:
                price = self.get_price()
                log(f"Current price of {self.symbol}: ${price:.2f}")

                # First check if an open position has been resolved
                self._check_position(price)

                # Do not enter new trades if position is open or limit reached
                if self._position is not None:
                    log(
                        f"Position open ({self._position.direction} @ "
                        f"${self._position.entry:.2f}). Monitoring."
                    )
                elif self._trades_today >= max_trades_per_day:
                    log(f"Daily trade limit ({max_trades_per_day}) reached. Not scanning.")
                else:
                    # Run full SMC analysis
                    signal = self.strategy.analyze()
                    if signal:
                        log(f"SIGNAL DETECTED: {signal.direction} | {signal.reason}")
                        self.place_order(signal)
                    else:
                        log("No valid SMC setup found. Holding.")

            except Exception as exc:
                log(f"[ERROR] {exc}")

            if scan < max_scans:
                log(f"Sleeping {scan_interval_seconds}s …")
                time.sleep(scan_interval_seconds)

        log("Trading session complete.")
        log(f"Total trades placed this session: {self._trades_today}")
