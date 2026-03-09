"""Liquidity Heatmap — pivot-based liquidity levels.

Translates the Nephew_Sam_ Liquidity Heatmap Pine Script.

Pivot highs/lows at multiple lookback windows represent where resting
orders (stop-losses) tend to cluster.  Each window maps to a "strength"
tier used for styling opacity in the renderer.
"""

from __future__ import annotations

import pandas as pd


# (label, left_bars, right_bars, strength 1-5)
_CONFIGS: list[tuple[str, int, int, int]] = [
    ("ST",   5,  5,  1),
    ("ID",   7,  7,  2),
    ("SW",  10, 10,  3),
    ("IT",  14, 14,  4),
    ("HTF", 20, 20,  5),
]


def _pivot_highs(series: pd.Series, left: int, right: int) -> dict:
    vals = series.to_numpy()
    n = len(vals)
    out: dict = {}
    for i in range(left, n - right):
        window_max = float("-inf")
        for k in range(i - left, i + right + 1):
            if vals[k] > window_max:
                window_max = vals[k]
        if vals[i] == window_max:
            out[series.index[i]] = float(vals[i])
    return out


def _pivot_lows(series: pd.Series, left: int, right: int) -> dict:
    vals = series.to_numpy()
    n = len(vals)
    out: dict = {}
    for i in range(left, n - right):
        window_min = float("inf")
        for k in range(i - left, i + right + 1):
            if vals[k] < window_min:
                window_min = vals[k]
        if vals[i] == window_min:
            out[series.index[i]] = float(vals[i])
    return out


def detect_liquidity_levels(df: pd.DataFrame) -> list[dict]:
    """Return a list of liquidity level dicts.

    Keys: date, price, dir ('high'|'low'), strength (1-5), label
    """
    levels: list[dict] = []
    for label, left, right, strength in _CONFIGS:
        if len(df) < left + right + 1:
            continue
        ph = _pivot_highs(df["High"], left, right)
        pl = _pivot_lows(df["Low"],  left, right)
        for date, price in ph.items():
            levels.append(
                {"date": date, "price": price, "dir": "high",
                 "strength": strength, "label": label}
            )
        for date, price in pl.items():
            levels.append(
                {"date": date, "price": price, "dir": "low",
                 "strength": strength, "label": label}
            )
    return levels
