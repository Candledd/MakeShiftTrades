import logging
import numpy as np
import pandas as pd
import yfinance as yf

_log = logging.getLogger(__name__)

# Maximum periods allowed by Yahoo Finance for each interval
_MAX_PERIOD_DAYS: dict[str, int] = {
    "1m":  7,
    "2m":  60,
    "3m":  7,   # synthetic — resampled from 1m
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "1h":  730,
    "90m": 60,
    "1d":  3650,
    "5d":  3650,
    "1wk": 3650,
    "1mo": 3650,
    "3mo": 3650,
}

# Intervals that don't exist in yfinance and must be built by resampling.
# Maps synthetic_interval → (base_interval, pandas_resample_rule)
_RESAMPLE_MAP: dict[str, tuple[str, str]] = {
    "3m": ("1m", "3min"),
}

_PERIOD_DAYS: dict[str, int] = {
    "1d":   1,
    "5d":   5,
    "7d":   7,
    "1mo":  30,
    "3mo":  90,
    "6mo":  180,
    "1y":   365,
    "2y":   730,
    "5y":   1825,
    "10y":  3650,
    "max":  99999,
}

_DAY_TO_PERIOD: list[tuple[int, str]] = sorted(
    [(v, k) for k, v in _PERIOD_DAYS.items()], key=lambda t: t[0]
)


def _cap_period(period: str, interval: str) -> str:
    """Return a capped period string that Yahoo Finance will accept."""
    max_days = _MAX_PERIOD_DAYS.get(interval, 3650)
    req_days = _PERIOD_DAYS.get(period, 365)
    if req_days <= max_days:
        return period
    # Find the largest period that fits within max_days
    result = "1d"
    for days, p in _DAY_TO_PERIOD:
        if days <= max_days:
            result = p
    return result


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


def _fetch_raw(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download and normalise a single yfinance-supported interval.

    Uses Ticker.history() instead of yf.download() to guarantee a flat column
    structure regardless of yfinance version. yf.download() with a single
    crypto ticker can return a MultiIndex where level-0 is the ticker symbol
    (not the price field), causing all OHLC columns to resolve to the same
    closing price and rendering flat doji candles with no wicks.
    """
    period = _cap_period(period, interval)
    df: pd.DataFrame = yf.Ticker(ticker).history(
        period=period,
        interval=interval,
        auto_adjust=True,
    )
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    # history() returns flat columns; defensive MultiIndex guard for edge cases.
    # yfinance can return MultiIndex with either (field, ticker) or (ticker, field)
    # ordering depending on the interval/version, so check which level holds the
    # OHLCV field names rather than blindly taking level 0.
    if isinstance(df.columns, pd.MultiIndex):
        _ohlcv = {"Open", "High", "Low", "Close", "Volume"}
        _lvl0 = set(df.columns.get_level_values(0))
        if _ohlcv.issubset(_lvl0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return _fix_flat_ohlcv(df, ticker, interval)


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an OHLCV DataFrame to *rule* (e.g. '3min')."""
    resampled = df.resample(rule).agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}
    ).dropna()
    return resampled


def fetch_ohlcv(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV bars, handling synthetic intervals via resampling.

    • Automatically caps the period to what Yahoo Finance allows.
    • '3m' is synthesised by downloading 1m data and resampling.

    Returns a DataFrame with columns Open, High, Low, Close, Volume and a
    DatetimeIndex.  Raises ValueError if no data is returned.
    """
    if interval in _RESAMPLE_MAP:
        base_interval, rule = _RESAMPLE_MAP[interval]
        df_raw = _fetch_raw(ticker, period, base_interval)
        return _resample_ohlcv(df_raw, rule)

    return _fetch_raw(ticker, period, interval)
