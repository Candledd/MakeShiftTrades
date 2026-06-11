"""Macro Kill Switch — Filters out high-risk event windows.

Blocks trading during known macro-economic news releases where
volatility spikes can wreak havoc on intraday positions:

- FOMC (Federal Open Market Committee) approximation:
  Every Wednesday 18:00–18:30 UTC.
- NFP (Non-Farm Payrolls) approximation:
  First Friday of every month 12:30–13:00 UTC.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


class MacroFilter:
    """Schedule-based macro-event red-flag window detector."""

    @staticmethod
    def is_red_flag_window(timestamp_now: float) -> bool:
        """Return True if *timestamp_now* falls inside a known high-risk window.

        Parameters
        ----------
        timestamp_now : float
            Unix/epoch timestamp (seconds since 1970-01-01).

        Returns
        -------
        bool
            True when the market is inside a macro-event red-flag window.
        """
        dt = datetime.fromtimestamp(timestamp_now, tz=timezone.utc)

        # ── FOMC approximation: Wednesday 18:00–18:30 UTC ──────────────
        if dt.weekday() == 2:  # Wednesday
            hour_min = dt.hour * 60 + dt.minute
            if 18 * 60 <= hour_min < 18 * 60 + 30:
                logger.debug("MacroFilter: FOMC window (Wed 18:00-18:30 UTC)")
                return True

        # ── NFP approximation: first Friday of month 12:30–13:00 UTC ──
        if dt.weekday() == 4 and 1 <= dt.day <= 7:  # Friday in first 7 days
            hour_min = dt.hour * 60 + dt.minute
            if 12 * 60 + 30 <= hour_min < 13 * 60:
                logger.debug("MacroFilter: NFP window (1st Fri 12:30-13:00 UTC)")
                return True

        return False
