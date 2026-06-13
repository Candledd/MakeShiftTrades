import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config

_log = logging.getLogger(__name__)

# ── Global Alpaca data clients ────────────────────────────────────────────
_stock_client = StockHistoricalDataClient(
    api_key=config.ALPACA_API_KEY,
    secret_key=config.ALPACA_SECRET_KEY,
)
_crypto_client = CryptoHistoricalDataClient()

# ── Interval → Alpaca TimeFrame mapping ──────────────────────────────────
_INTERVAL_TO_TIMEFRAME: dict[str, TimeFrame] = {
    "1m":  TimeFrame(amount=1, unit=TimeFrameUnit.Minute),
    "15m": TimeFrame(amount=15, unit=TimeFrameUnit.Minute),
    "1h":  TimeFrame(amount=1, unit=TimeFrameUnit.Hour),
    "4h":  TimeFrame(amount=4, unit=TimeFrameUnit.Hour),
    "1d":  TimeFrame(amount=1, unit=TimeFrameUnit.Day),
}

# ── Period → days delta ──────────────────────────────────────────────────
_PERIOD_TO_DAYS: dict[str, int] = {
    "5d":   5,
    "1mo":  30,
    "3mo":  90,
    "6mo":  180,
    "1y":   365,
}

_CRYPTO_SUFFIXES: tuple[str, ...] = ("-USD", "-USDT", "-USDC")


def _is_crypto(ticker: str) -> bool:
    """Return ``True`` if *ticker* is a crypto symbol (ends in ``-USD`` etc.)."""
    return ticker.upper().endswith(_CRYPTO_SUFFIXES)


def _alpaca_symbol(ticker: str) -> str:
    """Convert a ticker to Alpaca's expected symbol format.

    Crypto symbols (e.g. ``BTC-USD``) use ``BTC/USD`` with a slash separator.
    Equity/commodity symbols are passed through unchanged.
    """
    if _is_crypto(ticker):
        return ticker.replace("-", "/")
    return ticker


def _fix_flat_ohlcv(df: pd.DataFrame, ticker: str = "", interval: str = "") -> pd.DataFrame:
    """Yahoo Finance returns Open=High=Low=Close for 1m crypto bars because
    it only stores last-trade snapshots, not proper OHLC tick aggregations.
    When ≥80% of bars are detected flat, synthesise Open/High/Low from
    consecutive Close prices: Open[i] = Close[i-1], which is the standard
    convention for tick-level data and matches what the 3m resampled bars
    already produce naturally.
    """
    if len(df) < 2:
        return df
    flat = (
        (df["Open"] == df["Close"]) &
        (df["High"] == df["Close"]) &
        (df["Low"]  == df["Close"])
    )
    if flat.mean() < 0.80:
        return df
    _log.info(
        "Synthesising OHLC for %s %s (%.0f%% flat bars — Yahoo last-price data)",
        ticker, interval, flat.mean() * 100,
    )
    closes = df["Close"].to_numpy(dtype=float)
    opens  = np.empty_like(closes)
    opens[0]  = closes[0]      # first bar has no prior close
    opens[1:] = closes[:-1]    # all others: open = previous close
    highs = np.maximum(opens, closes)
    lows  = np.minimum(opens, closes)
    df = df.copy()
    df["Open"]  = opens
    df["High"]  = highs
    df["Low"]   = lows
    return df



def fetch_ohlcv(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV bars using the Alpaca market data API.

    Parameters
    ----------
    ticker : str
        Symbol to fetch (e.g. ``"SPY"``, ``"BTC-USD"``, ``"GLD"``).
    period : str
        Lookback period. Supported: ``"5d"``, ``"1mo"``, ``"3mo"``.
        Unrecognised periods default to 90 days.
    interval : str
        Bar interval. Supported: ``"1m"``, ``"15m"``, ``"1h"``, ``"4h"``, ``"1d"``.

    Returns
    -------
    pd.DataFrame
        Columns ``Open``, ``High``, ``Low``, ``Close``, ``Volume`` with a
        timezone-aware ``DatetimeIndex``.

    Raises
    ------
    ValueError
        If *interval* is unsupported or no data is returned.
    """
    if interval not in _INTERVAL_TO_TIMEFRAME:
        raise ValueError(
            f"Unsupported interval '{interval}'. "
            f"Supported intervals: {list(_INTERVAL_TO_TIMEFRAME.keys())}"
        )

    timeframe = _INTERVAL_TO_TIMEFRAME[interval]

    # Parse period string dynamically if possible
    days = 90
    if period in _PERIOD_TO_DAYS:
        days = _PERIOD_TO_DAYS[period]
    elif period.endswith("mo"):
        try:
            days = int(period[:-2]) * 30
        except ValueError:
            pass
    elif period.endswith("y"):
        try:
            days = int(period[:-1]) * 365
        except ValueError:
            pass

    start = datetime.now(timezone.utc) - timedelta(days=days)

    symbol = _alpaca_symbol(ticker)

    # ── Route to the correct request/client ────────────────────────────────
    if _is_crypto(ticker):
        req = CryptoBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=start,
        )
        bars = _crypto_client.get_crypto_bars(req)
    else:
        feed = getattr(config, "ALPACA_DATA_FEED", "iex")
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=start,
            feed=feed,
        )
        bars = _stock_client.get_stock_bars(req)

    df = bars.df

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' from Alpaca.")

    # ── Flatten the index if it's a MultiIndex ─────────────────────────────
    # Alpaca returns a (symbol, timestamp) MultiIndex for all bar types when
    # the request result contains multiple symbols; for single-symbol requests
    # it may still produce a MultiIndex (notably for crypto).  Drop the symbol
    # level so we get a flat DatetimeIndex.
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level="symbol", drop=True)

    # ── Rename lowercase Alpaca columns → uppercase strategy schema ────────
    # Alpaca columns: open, high, low, close, volume, trade_count, vwap, …
    rename = {"open": "Open", "high": "High", "low": "Low",
              "close": "Close", "volume": "Volume"}
    df = df.rename(columns=rename)
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # ── Normalise index ────────────────────────────────────────────────────
    df.index = pd.to_datetime(df.index)

    # ── Safety net: synthesise OHLC from close if bars are abnormally flat ─
    df = _fix_flat_ohlcv(df, ticker, interval)

    return df
