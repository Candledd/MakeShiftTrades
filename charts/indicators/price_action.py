"""Price Action — ICT / SMC concepts.

Translates key ideas from the LuxAlgo Pure Price Action ICT Tools script:

  • Swing Highs / Lows  (pivot-based)
  • Market Structure    (BOS = Break of Structure, CHoCH = Change of Character)
  • Order Blocks        (last opposing candle before a confirmed BOS/CHoCH)
"""

from __future__ import annotations

import pandas as pd


# Detection window sizes for each "term" setting
_TERM_PARAMS: dict[str, tuple[int, int]] = {
    "short":        (3, 3),
    "intermediate": (5, 5),
    "long":         (10, 10),
}


# ──────────────────────────────────────────────
# Internal pivot helpers
# ──────────────────────────────────────────────

def _pivot_highs(series: pd.Series, left: int, right: int) -> dict[int, float]:
    vals = series.to_numpy()
    n = len(vals)
    out: dict[int, float] = {}
    for i in range(left, n - right):
        if vals[i] == max(vals[i - left: i + right + 1]):
            out[i] = float(vals[i])
    return out


def _pivot_lows(series: pd.Series, left: int, right: int) -> dict[int, float]:
    vals = series.to_numpy()
    n = len(vals)
    out: dict[int, float] = {}
    for i in range(left, n - right):
        if vals[i] == min(vals[i - left: i + right + 1]):
            out[i] = float(vals[i])
    return out


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def detect_swing_points(
    df: pd.DataFrame, term: str = "intermediate"
) -> pd.DataFrame:
    """Return swing highs and lows as a DataFrame.

    Columns: date, price, type ('high'|'low')
    """
    left, right = _TERM_PARAMS.get(term, (5, 5))
    sh = _pivot_highs(df["High"], left, right)
    sl = _pivot_lows(df["Low"],  left, right)

    rows: list[dict] = []
    for idx, price in sh.items():
        rows.append({"date": df.index[idx], "price": price, "type": "high"})
    for idx, price in sl.items():
        rows.append({"date": df.index[idx], "price": price, "type": "low"})

    if not rows:
        return pd.DataFrame(columns=["date", "price", "type"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def detect_market_structure(
    df: pd.DataFrame, term: str = "intermediate"
) -> list[dict]:
    """Detect BOS and CHoCH events.

    Returns a list of dicts with keys:
        date, price, type, label ('BOS'|'CHoCH'), color
    """
    left, right = _TERM_PARAMS.get(term, (5, 5))
    sh_dict = _pivot_highs(df["High"], left, right)
    sl_dict = _pivot_lows(df["Low"],  left, right)

    closes = df["Close"].to_numpy()
    n = len(df)

    events: list[dict] = []
    trend: str | None = None
    last_sh: float | None = None
    last_sl: float | None = None

    for i in range(n):
        # Register pivots that are now fully confirmed (right-bars old)
        confirm_idx = i - right
        if confirm_idx >= 0:
            if confirm_idx in sh_dict:
                last_sh = sh_dict[confirm_idx]
            if confirm_idx in sl_dict:
                last_sl = sl_dict[confirm_idx]

        c = closes[i]

        if last_sh is not None and c > last_sh:
            label = "CHoCH" if trend == "down" else "BOS"
            events.append(
                {
                    "date":  df.index[i],
                    "price": last_sh,
                    "type":  f"{label}_bull",
                    "label": label,
                    "color": "#089981",
                }
            )
            trend   = "up"
            last_sh = None

        if last_sl is not None and c < last_sl:
            label = "CHoCH" if trend == "up" else "BOS"
            events.append(
                {
                    "date":  df.index[i],
                    "price": last_sl,
                    "type":  f"{label}_bear",
                    "label": label,
                    "color": "#f23645",
                }
            )
            trend   = "down"
            last_sl = None

    return events


def detect_order_blocks(
    df: pd.DataFrame, term: str = "long", n_last: int = 3
) -> list[dict]:
    """Identify Order Blocks (the last opposing candle before BOS / CHoCH).

    Bullish OB = last bearish candle before a bullish structure break.
    Bearish OB = last bullish candle before a bearish structure break.

    Returns the `n_last` most-recent OBs of each polarity.
    """
    ms_events = detect_market_structure(df, term)
    obs: list[dict] = []
    seen_dates: set = set()

    for event in ms_events:
        event_idx = df.index.get_loc(event["date"])
        look_back = min(30, event_idx)

        if "bull" in event["type"]:
            for j in range(event_idx - 1, event_idx - look_back, -1):
                o, c = df["Open"].iloc[j], df["Close"].iloc[j]
                if c < o:   # bearish candle → bullish OB
                    d = df.index[j]
                    if d not in seen_dates:
                        seen_dates.add(d)
                        obs.append(
                            {
                                "type":   "bullish",
                                "date":   d,
                                "top":    max(o, c),
                                "bottom": min(o, c),
                                "active": True,
                            }
                        )
                    break

        elif "bear" in event["type"]:
            for j in range(event_idx - 1, event_idx - look_back, -1):
                o, c = df["Open"].iloc[j], df["Close"].iloc[j]
                if c > o:   # bullish candle → bearish OB
                    d = df.index[j]
                    if d not in seen_dates:
                        seen_dates.add(d)
                        obs.append(
                            {
                                "type":   "bearish",
                                "date":   d,
                                "top":    max(o, c),
                                "bottom": min(o, c),
                                "active": True,
                            }
                        )
                    break

    bull_obs = [o for o in obs if o["type"] == "bullish"][-n_last:]
    bear_obs = [o for o in obs if o["type"] == "bearish"][-n_last:]
    return bull_obs + bear_obs
