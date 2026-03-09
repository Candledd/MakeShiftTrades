"""Liquidity Engulfing candle-pattern detection.

Matches the Nephew_Sam_ Pine Script logic:

  Bullish engulfing:
    • Current candle is bullish  (close > open)
    • Current low  < previous low   (sweeps liquidity below)
    • Current close > previous open (closes above the prior candle's open)

  Bearish engulfing:
    • Current candle is bearish  (close < open)
    • Current high > previous high  (sweeps liquidity above)
    • Current close < previous open (closes below the prior candle's open)
"""

import pandas as pd


def detect_engulfing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with columns: date, type ('bullish'|'bearish'), price.

    `price` is the sweep extreme (low for bullish, high for bearish).
    """
    opens  = df["Open"].to_numpy()
    highs  = df["High"].to_numpy()
    lows   = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()

    rows: list[dict] = []
    for i in range(1, len(df)):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        po, ph, pl  = opens[i - 1], highs[i - 1], lows[i - 1]

        if c > o and l < pl and c > po:
            rows.append({"date": df.index[i], "type": "bullish", "price": l})
        elif c < o and h > ph and c < po:
            rows.append({"date": df.index[i], "type": "bearish", "price": h})

    if not rows:
        return pd.DataFrame(columns=["date", "type", "price"])
    return pd.DataFrame(rows)
