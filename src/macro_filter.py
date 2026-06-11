"""Macro Kill Switch — Filters out high-risk event windows.

Blocks trading during known macro-economic news releases where
volatility spikes can wreak havoc on intraday positions:

- FOMC (Federal Open Market Committee) approximation:
  Every Wednesday ~18:00 UTC. Pre-event window: 120 min. Severity: FLATTEN_ALL.
- NFP (Non-Farm Payrolls) approximation:
  First Friday of every month ~12:30 UTC. Pre-event window: 60 min. Severity: FLATTEN_ALL.
- CPI (Consumer Price Index) approximation:
  Monthly ~12:30 UTC (release typically around 13th). Pre-event window: 60 min. Severity: NO_NEW_ENTRIES.

Each event defines its own look-ahead window (minutes *before* the scheduled release)
and severity level so that the engine can distinguish between "block new entries only"
and "flatten all existing positions".
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Tuple

logger = logging.getLogger(__name__)

# ── Event Severity Constants ─────────────────────────────────────────────
# These are returned by :meth:`MacroFilter.check_event` as the first element
# of the result tuple.  Consumers use them to decide how to respond.

SEVERITY_NORMAL = "normal"
"""No event is active — trade freely."""

SEVERITY_NO_NEW_ENTRIES = "no_new_entries"
"""Mild event — block new entry orders but keep existing positions open."""

SEVERITY_FLATTEN_ALL = "flatten_all"
"""Severe event — block new entries AND close all existing positions."""


class MacroFilter:
    """Schedule-based macro-event red-flag window detector.

    Usage
    -----
    >>> filter = MacroFilter()
    >>> severity, event = filter.check_event(time.time())
    >>> if severity == SEVERITY_FLATTEN_ALL:
    ...     # flatten positions + pause scans
    ... elif severity == SEVERITY_NO_NEW_ENTRIES:
    ...     # pause scans only
    """

    @staticmethod
    def check_event(timestamp_now: float) -> Tuple[str, str]:
        """Check if *timestamp_now* falls inside (or ahead of) a known event window.

        Parameters
        ----------
        timestamp_now : float
            Unix/epoch timestamp (seconds since 1970-01-01).

        Returns
        -------
        Tuple[str, str]
            ``(severity, event_name)`` where *severity* is one of
            ``SEVERITY_NORMAL``, ``SEVERITY_NO_NEW_ENTRIES``,
            or ``SEVERITY_FLATTEN_ALL``, and *event_name* is a short
            human-readable label (e.g. ``"FOMC"``, ``"NFP"``, ``"CPI"``)
            or ``""`` when no event is active.

        Events are evaluated in descending severity order so that the most
        restrictive action (flatten) takes precedence when windows overlap.
        """
        dt = datetime.fromtimestamp(timestamp_now, tz=timezone.utc)

        # ── Highest severity first ──────────────────────────────────────

        # FOMC — Wed 18:00 UTC, 120 min pre-event window
        result = MacroFilter._check_fomc(dt)
        if result[0] != SEVERITY_NORMAL:
            return result

        # NFP — 1st Fri 12:30 UTC, 60 min pre-event window
        result = MacroFilter._check_nfp(dt)
        if result[0] != SEVERITY_NORMAL:
            return result

        # ── Medium severity ─────────────────────────────────────────────

        # CPI — ~13th of month 12:30 UTC, 60 min pre-event window
        result = MacroFilter._check_cpi(dt)
        if result[0] != SEVERITY_NORMAL:
            return result

        return (SEVERITY_NORMAL, "")

    @staticmethod
    def is_red_flag_window(timestamp_now: float) -> bool:
        """Return True if *timestamp_now* falls inside a known high-risk window.

        Legacy convenience method.  Prefer :meth:`check_event` for finer-grained
        severity information so that the engine can distinguish between
        "block new entries only" and "flatten all positions".

        Parameters
        ----------
        timestamp_now : float
            Unix/epoch timestamp (seconds since 1970-01-01).

        Returns
        -------
        bool
            True when the market is inside a macro-event red-flag window.
        """
        severity, _ = MacroFilter.check_event(timestamp_now)
        return severity != SEVERITY_NORMAL

    # ──────────────────────────────────────────────────────────────────────
    # Private event checkers  (one per event, return (severity, event_name))
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_fomc(dt: datetime) -> Tuple[str, str]:
        """FOMC decision day — Wednesday ~18:00 UTC.

        Window: 120 minutes *before* the event + 30 minutes *after*
        (total protected window 16:00–18:30 UTC).
        Severity: FLATTEN_ALL (unexpected policy shifts can trigger sharp
        reversals; existing positions should be flattened).
        """
        if dt.weekday() != 2:  # Not Wednesday
            return (SEVERITY_NORMAL, "")

        event_start_min = 18 * 60               # 18:00 UTC
        pre_window_min = 120                     # 120 minutes ahead
        event_duration_min = 30                  # 30-minute release window
        window_start_min = event_start_min - pre_window_min  # 16:00 UTC

        now_min = dt.hour * 60 + dt.minute
        if window_start_min <= now_min < event_start_min + event_duration_min:
            logger.debug("MacroFilter: FOMC window (Wed ~18:00 UTC, severity=flatten_all)")
            return (SEVERITY_FLATTEN_ALL, "FOMC")

        return (SEVERITY_NORMAL, "")

    @staticmethod
    def _check_nfp(dt: datetime) -> Tuple[str, str]:
        """NFP release — first Friday of month, 12:30 UTC (8:30 AM ET).

        Window: 60 minutes *before* + 30 minutes *after*
        (total protected window 11:30–13:00 UTC).
        Severity: FLATTEN_ALL (the single most market-moving monthly release;
        spreads widen dramatically and false breakouts are common).
        """
        if dt.weekday() != 4 or not (1 <= dt.day <= 7):  # First Friday
            return (SEVERITY_NORMAL, "")

        event_start_min = 12 * 60 + 30           # 12:30 UTC
        pre_window_min = 60                      # 60 minutes ahead
        event_duration_min = 30                  # 30-minute release window
        window_start_min = event_start_min - pre_window_min  # 11:30 UTC

        now_min = dt.hour * 60 + dt.minute
        if window_start_min <= now_min < event_start_min + event_duration_min:
            logger.debug("MacroFilter: NFP window (1st Fri 12:30 UTC, severity=flatten_all)")
            return (SEVERITY_FLATTEN_ALL, "NFP")

        return (SEVERITY_NORMAL, "")

    @staticmethod
    def _check_cpi(dt: datetime) -> Tuple[str, str]:
        """CPI release — US Bureau of Labor Statistics, ~12:30 UTC (8:30 AM ET).

        Approximated as any weekday between the 10th and the 16th of the month
        (the actual release date varies but typically falls in that window).
        Window: 60 minutes *before* + 30 minutes *after*
        (total protected window 11:30–13:00 UTC).
        Severity: NO_NEW_ENTRIES (CPI is important but rarely triggers the
        kind of violent reversals that justify forced flattening).
        """
        # Must be a weekday AND between the 10th and 16th of the month
        if dt.weekday() >= 5 or not (10 <= dt.day <= 16):
            return (SEVERITY_NORMAL, "")

        event_start_min = 12 * 60 + 30           # 12:30 UTC
        pre_window_min = 60                      # 60 minutes ahead
        event_duration_min = 30                  # 30-minute release window
        window_start_min = event_start_min - pre_window_min  # 11:30 UTC

        now_min = dt.hour * 60 + dt.minute
        if window_start_min <= now_min < event_start_min + event_duration_min:
            logger.debug("MacroFilter: CPI window (~13th 12:30 UTC, severity=no_new_entries)")
            return (SEVERITY_NO_NEW_ENTRIES, "CPI")

        return (SEVERITY_NORMAL, "")
