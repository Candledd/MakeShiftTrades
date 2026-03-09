"""Fair Value Gap (FVG) and Inversion FVG (IFVG) detection.

Logic (identical to the TradingFinder / ICT convention):

  Three consecutive candles: [i-2]  [i-1 driver]  [i]

  Bullish FVG at bar i:  low[i]  > high[i-2]
      Zone top    = low[i]
      Zone bottom = high[i-2]

  Bearish FVG at bar i:  high[i] < low[i-2]
      Zone top    = low[i-2]
      Zone bottom = high[i]

  Mitigation:  price closes beyond the zone mid-point → FVG becomes an IFVG
  (it now acts in the opposite direction).
"""

import pandas as pd


def detect_fvg(df: pd.DataFrame, validity_bars: int = 200) -> pd.DataFrame:
    """Return a DataFrame of FVG zones with columns:

    date, type ('bullish'|'bearish'), top, bottom, mid,
    origin_idx, active, ifvg, end_date
    """
    n = len(df)
    rows: list[dict] = []

    highs = df["High"].to_numpy()
    lows  = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()

    for i in range(2, n):
        lo_curr  = lows[i]
        hi_curr  = highs[i]
        hi_prev2 = highs[i - 2]
        lo_prev2 = lows[i - 2]

        if lo_curr > hi_prev2:
            rows.append(
                dict(
                    date=df.index[i],
                    type="bullish",
                    top=lo_curr,
                    bottom=hi_prev2,
                    mid=(lo_curr + hi_prev2) / 2,
                    origin_idx=i,
                    active=True,
                    ifvg=False,
                    end_date=df.index[min(i + validity_bars, n - 1)],
                )
            )
        elif hi_curr < lo_prev2:
            rows.append(
                dict(
                    date=df.index[i],
                    type="bearish",
                    top=lo_prev2,
                    bottom=hi_curr,
                    mid=(lo_prev2 + hi_curr) / 2,
                    origin_idx=i,
                    active=True,
                    ifvg=False,
                    end_date=df.index[min(i + validity_bars, n - 1)],
                )
            )

    if not rows:
        return pd.DataFrame(
            columns=["date", "type", "top", "bottom", "mid",
                     "origin_idx", "active", "ifvg", "end_date"]
        )

    fvg_df = pd.DataFrame(rows)

    # Mitigation pass — price closes past mid-point → zone becomes IFVG
    for idx in range(len(fvg_df)):
        row = fvg_df.iloc[idx]
        i0    = int(row["origin_idx"])
        limit = min(i0 + validity_bars, n)
        mid   = row["mid"]

        for j in range(i0 + 1, limit):
            c = closes[j]
            if row["type"] == "bullish" and c <= mid:
                fvg_df.iat[idx, fvg_df.columns.get_loc("active")]   = False
                fvg_df.iat[idx, fvg_df.columns.get_loc("ifvg")]     = True
                fvg_df.iat[idx, fvg_df.columns.get_loc("end_date")] = df.index[j]
                break
            elif row["type"] == "bearish" and c >= mid:
                fvg_df.iat[idx, fvg_df.columns.get_loc("active")]   = False
                fvg_df.iat[idx, fvg_df.columns.get_loc("ifvg")]     = True
                fvg_df.iat[idx, fvg_df.columns.get_loc("end_date")] = df.index[j]
                break

    return fvg_df
