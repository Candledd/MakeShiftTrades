"""Key price levels for ICT / SMC day trading.

Provides:
  - Previous Day High / Low / Close      (PDH, PDL, PDC)
  - Previous Week High / Low             (PWH, PWL)
  - VWAP  — Volume Weighted Avg Price, daily reset
  - ICT Session Kill Zones               (Asia, London, NY Open, PM)
  - Equilibrium / Premium-Discount zone  (50% of latest swing range)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_unix(ts) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        return int(t.timestamp())
    return int(t.tz_localize("UTC").timestamp())


def _get_utc_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _is_intraday(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    diff = (df.index[1] - df.index[0]).total_seconds()
    return diff < 86400


# ── Previous Day / Week High-Low ──────────────────────────────────────────────

def detect_key_levels(df: pd.DataFrame) -> list[dict]:
    """Return PDH, PDL, PDC, PWH, PWL as {price, label, type} dicts."""
    if len(df) < 2:
        return []

    idx_utc = _get_utc_index(df)
    dates   = pd.Series([t.date() for t in idx_utc], index=df.index)
    unique_dates = sorted(dates.unique())

    levels: list[dict] = []

    # ── Previous Day ──────────────────────────────────────────────────────
    if len(unique_dates) >= 2:
        prev_d    = unique_dates[-2]
        prev_mask = dates == prev_d
        pday      = df[prev_mask]
        if not pday.empty:
            levels += [
                {"price": float(pday["High"].max()),    "label": "PDH", "type": "pdh"},
                {"price": float(pday["Low"].min()),     "label": "PDL", "type": "pdl"},
                {"price": float(pday["Close"].iloc[-1]),"label": "PDC", "type": "pdc"},
            ]

    # ── Previous Week ──────────────────────────────────────────────────────
    try:
        weeks = pd.Series(
            [f"{t.isocalendar()[0]}-{t.isocalendar()[1]:02d}" for t in idx_utc],
            index=df.index,
        )
        unique_weeks = sorted(weeks.unique())
        if len(unique_weeks) >= 2:
            prev_w    = unique_weeks[-2]
            prev_mask = weeks == prev_w
            pweek     = df[prev_mask]
            if not pweek.empty:
                levels += [
                    {"price": float(pweek["High"].max()), "label": "PWH", "type": "pwh"},
                    {"price": float(pweek["Low"].min()),  "label": "PWL", "type": "pwl"},
                ]
    except Exception:
        pass

    return levels


# ── VWAP ──────────────────────────────────────────────────────────────────────

def detect_vwap(df: pd.DataFrame) -> list[dict]:
    """Return per-bar VWAP {time, value} with daily reset."""
    if df.empty:
        return []

    idx_utc = _get_utc_index(df)
    dates   = pd.Series([t.date() for t in idx_utc], index=df.index)
    tp      = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_vol  = tp * df["Volume"]
    vwap    = pd.Series(np.nan, index=df.index, dtype=float)

    for d in dates.unique():
        mask      = dates == d
        cum_tpvol = tp_vol[mask].cumsum()
        cum_vol   = df["Volume"][mask].replace(0, np.nan).cumsum()
        vwap[mask] = cum_tpvol / cum_vol

    return [
        {"time": _to_unix(ts), "value": round(float(v), 4)}
        for ts, v in vwap.items()
        if not np.isnan(v)
    ]


# ── Session Kill Zones ────────────────────────────────────────────────────────

# (name, start_hour_utc_inclusive, end_hour_utc_exclusive, fill_colour)
_SESSION_DEFS: list[tuple[str, int, int, str]] = [
    ("Asia KZ",    0,  4, "rgba(59,130,246,0.10)"),
    ("London KZ",  7, 10, "rgba(255,165,0,0.10)"),
    ("NY KZ",     13, 16, "rgba(8,153,129,0.10)"),
    ("PM KZ",     18, 21, "rgba(167,139,250,0.09)"),
]


def detect_sessions(df: pd.DataFrame) -> list[dict]:
    """Return ICT kill-zone boxes as {name, start_time, end_time, color}."""
    if not _is_intraday(df) or df.empty:
        return []

    idx_utc = _get_utc_index(df)
    bars    = list(zip(df.index, idx_utc))   # (original_ts, utc_ts)
    dates   = sorted({t.date() for t in idx_utc})

    sessions: list[dict] = []
    for d in dates:
        for name, sh, eh, color in _SESSION_DEFS:
            day_bars = [
                orig for orig, utc in bars
                if utc.date() == d and sh <= utc.hour < eh
            ]
            if not day_bars:
                continue
            sessions.append({
                "name":       name,
                "start_time": _to_unix(day_bars[0]),
                "end_time":   _to_unix(day_bars[-1]),
                "color":      color,
            })

    return sessions


# ── Equilibrium / Premium-Discount zone ──────────────────────────────────────

def detect_equilibrium(df: pd.DataFrame) -> Optional[dict]:
    """Return premium/discount zone based on the most recent swing H/L pair.

    Midpoint (50%) = equilibrium.
    Above 75%      = premium zone (sell bias).
    Below 25%      = discount zone (buy bias).
    """
    from charts.indicators.price_action import detect_swing_points

    if len(df) < 30:
        return None

    swings = detect_swing_points(df, term="long")
    if swings.empty:
        return None

    highs = swings[swings["type"] == "high"].tail(2)
    lows  = swings[swings["type"] == "low"].tail(2)
    if highs.empty or lows.empty:
        return None

    sh = float(highs["price"].max())
    sl = float(lows["price"].min())
    if sh <= sl:
        return None

    rng = sh - sl
    return {
        "high":     round(sh,              4),
        "low":      round(sl,              4),
        "eq":       round(sl + rng * 0.50, 4),
        "premium":  round(sl + rng * 0.75, 4),
        "discount": round(sl + rng * 0.25, 4),
    }
