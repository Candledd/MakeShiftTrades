import pandas as pd
import yfinance as yf

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


def _fetch_raw(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download and normalise a single yfinance-supported interval."""
    period = _cap_period(period, interval)
    df: pd.DataFrame = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    # yfinance ≥ 0.2 may return MultiIndex columns when downloading one symbol
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


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
