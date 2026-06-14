"""Numba-compiled indicators for high-performance computation.

Provides:
  - calc_volume_profile: ``@njit`` function computing a rolling Volume Profile,
    returning the Point of Control (POC) and Value Area (VA) boundaries.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit


@njit
def calc_volume_profile(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    num_bins: int = 24,
    va_threshold: float = 0.70,
) -> Tuple[float, float, float, float]:
    """Compute a rolling Volume Profile for the input price/volume arrays.

    The function bins prices into *num_bins* dynamically spaced ticks over the
    observed price range, distributes each bar's volume proportionally across
    the bins its high–low range spans, and returns the Point of Control (POC)
    — the price level with the highest traded volume — along with the Value
    Area boundaries (the narrowest price range containing *va_threshold* of
    total volume, expanding outward from the POC bin).

    Parameters
    ----------
    high : np.ndarray
        1-D array of high prices (length ``N``).
    low : np.ndarray
        1-D array of low prices (length ``N``).
    close : np.ndarray
        1-D array of close prices (length ``N``).
    volume : np.ndarray
        1-D array of volume values (length ``N``).
    num_bins : int
        Number of price bins to divide the observed range into (default 24).
    va_threshold : float
        Fraction of total volume that defines the Value Area (default 0.70,
        i.e. 70 %).

    Returns
    -------
    poc_price : float
        Point of Control — the mid-price of the bin with the highest
        aggregate traded volume.
    va_high : float
        Value Area High — the upper boundary of the Value Area.
    va_low : float
        Value Area Low — the lower boundary of the Value Area.
    poc_volume : float
        The total volume accumulated in the Point of Control bin.
    """
    n = len(close)
    if n < 2:
        if n > 0:
            mid = (high[0] + low[0]) / 2.0
            return mid, mid, mid, 0.0
        return 0.0, 0.0, 0.0, 0.0

    # ------------------------------------------------------------------
    # 1. Determine the full price range over the window
    # ------------------------------------------------------------------
    min_price = np.min(low)
    max_price = np.max(high)
    price_range = max_price - min_price

    # Degenerate case — all prices are identical
    if price_range < 1e-10:
        mid_price = (min_price + max_price) / 2.0
        total_vol = np.sum(volume)
        return mid_price, mid_price, mid_price, total_vol

    # ------------------------------------------------------------------
    # 2. Build bin boundaries (dynamically spaced ticks)
    # ------------------------------------------------------------------
    bin_size = price_range / num_bins

    # ------------------------------------------------------------------
    # 3. Aggregate volume per bin
    #
    #     For each bar, distribute its volume proportionally across all
    #     bins that the bar's price range (high – low) spans.
    # ------------------------------------------------------------------
    bin_volume = np.zeros(num_bins, dtype=np.float64)

    for i in range(n):
        bar_low = low[i]
        bar_high = high[i]
        bar_vol = volume[i]

        if bar_high <= bar_low or bar_vol <= 0.0:
            continue

        # Determine which bins this bar touches
        start_bin = int((bar_low - min_price) / bin_size)
        end_bin = int((bar_high - min_price) / bin_size)

        # Clamp to valid [0, num_bins - 1] range
        if start_bin < 0:
            start_bin = 0
        if end_bin >= num_bins:
            end_bin = num_bins - 1
        if start_bin > end_bin:
            start_bin = end_bin

        bar_range = bar_high - bar_low

        if start_bin == end_bin:
            # Bar fits entirely into a single bin
            bin_volume[start_bin] += bar_vol
        else:
            # Distribute volume proportionally across spanned bins
            for b in range(start_bin, end_bin + 1):
                bin_low = min_price + b * bin_size
                bin_high = bin_low + bin_size
                overlap_low = max(bar_low, bin_low)
                overlap_high = min(bar_high, bin_high)
                overlap = overlap_high - overlap_low
                if overlap > 0.0:
                    bin_volume[b] += bar_vol * (overlap / bar_range)

    # ------------------------------------------------------------------
    # 4. Locate the Point of Control — bin with the highest volume
    # ------------------------------------------------------------------
    total_vol = np.sum(bin_volume)
    if total_vol <= 0.0:
        return close[-1], close[-1], close[-1], 0.0

    poc_bin = 0
    max_vol = bin_volume[0]
    for b in range(1, num_bins):
        if bin_volume[b] > max_vol:
            max_vol = bin_volume[b]
            poc_bin = b

    poc_price = min_price + (poc_bin + 0.5) * bin_size
    poc_volume = max_vol

    # ------------------------------------------------------------------
    # 5. Compute Value Area (VA)
    #
    #     Expand outward from the POC bin, adding the next-higher-volume
    #     adjacent bin at each step, until *va_threshold* of total volume
    #     is included.
    # ------------------------------------------------------------------
    target_vol = total_vol * va_threshold
    cum_vol = bin_volume[poc_bin]
    va_low_idx = poc_bin
    va_high_idx = poc_bin

    while cum_vol < target_vol:
        left_avail = va_low_idx > 0
        right_avail = va_high_idx < num_bins - 1

        if not left_avail and not right_avail:
            break

        # Expand toward the adjacent bin with higher volume
        if left_avail and (
            not right_avail
            or bin_volume[va_low_idx - 1] >= bin_volume[va_high_idx + 1]
        ):
            va_low_idx -= 1
            cum_vol += bin_volume[va_low_idx]
        elif right_avail:
            va_high_idx += 1
            cum_vol += bin_volume[va_high_idx]
        else:
            break

    va_low_price = min_price + va_low_idx * bin_size
    va_high_price = min_price + (va_high_idx + 1) * bin_size

    return poc_price, va_high_price, va_low_price, poc_volume
