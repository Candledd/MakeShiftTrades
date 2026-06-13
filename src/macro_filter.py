"""Macro Kill Switch — Filters out high-risk event windows.

Blocks trading during known macro-economic news releases where
volatility spikes can wreak havoc on intraday positions.

Uses an explicit event list with event_time_utc, affected_assets,
pre_window, post_window, and actions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

# ── Event Severity Constants (kept for backward compatibility) ─────────────
SEVERITY_NORMAL = "normal"
SEVERITY_NO_NEW_ENTRIES = "no_new_entries"
SEVERITY_FLATTEN_ALL = "flatten_all"

# Mock explicit event list using realistic schemas
MOCK_EVENTS: List[Dict[str, Any]] = [
    # FOMC Events
    {
        "event_name": "FOMC",
        "event_time_utc": datetime(2026, 6, 17, 18, 0, tzinfo=timezone.utc),
        "affected_assets": ["all"],
        "pre_window": 120,
        "post_window": 30,
        "actions": ["no_new_entries", "flatten_intraday_only"]
    },
    {
        "event_name": "FOMC",
        "event_time_utc": datetime(2026, 7, 29, 18, 0, tzinfo=timezone.utc),
        "affected_assets": ["all"],
        "pre_window": 120,
        "post_window": 30,
        "actions": ["no_new_entries", "flatten_intraday_only"]
    },
    # NFP Events
    {
        "event_name": "NFP",
        "event_time_utc": datetime(2026, 6, 5, 12, 30, tzinfo=timezone.utc),
        "affected_assets": ["SPY", "QQQ", "BTC-USD"],
        "pre_window": 60,
        "post_window": 30,
        "actions": ["no_new_entries", "flatten_intraday_only"]
    },
    {
        "event_name": "NFP",
        "event_time_utc": datetime(2026, 7, 3, 12, 30, tzinfo=timezone.utc),
        "affected_assets": ["SPY", "QQQ", "BTC-USD"],
        "pre_window": 60,
        "post_window": 30,
        "actions": ["no_new_entries", "flatten_intraday_only"]
    },
    # CPI Events
    {
        "event_name": "CPI",
        "event_time_utc": datetime(2026, 6, 12, 12, 30, tzinfo=timezone.utc),
        "affected_assets": ["SPY", "QQQ", "BTC-USD", "GLD", "PDBC"],
        "pre_window": 60,
        "post_window": 30,
        "actions": ["no_new_entries"]
    },
    {
        "event_name": "CPI",
        "event_time_utc": datetime(2026, 7, 14, 12, 30, tzinfo=timezone.utc),
        "affected_assets": ["SPY", "QQQ", "BTC-USD", "GLD", "PDBC"],
        "pre_window": 60,
        "post_window": 30,
        "actions": ["no_new_entries"]
    },
    # EIA / OPEC Events
    {
        "event_name": "EIA",
        "event_time_utc": datetime(2026, 6, 10, 14, 30, tzinfo=timezone.utc),
        "affected_assets": ["PDBC"],
        "pre_window": 30,
        "post_window": 30,
        "actions": ["no_new_entries"]
    },
    {
        "event_name": "OPEC",
        "event_time_utc": datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc),
        "affected_assets": ["PDBC"],
        "pre_window": 60,
        "post_window": 120,
        "actions": ["no_new_entries"]
    }
]


class MacroFilter:
    """Schedule-based macro-event red-flag window detector using explicit event lists."""

    @staticmethod
    def check_event(dt_or_ts: float | datetime) -> List[Dict[str, Any]]:
        """Check if the given datetime or timestamp falls inside any active events.

        Parameters
        ----------
        dt_or_ts : float | datetime
            Unix/epoch timestamp (seconds since 1970-01-01) or a datetime object.

        Returns
        -------
        List[Dict[str, Any]]
            A list of active event dictionaries with their specific rules,
            or an empty list if no event is active.
        """
        if isinstance(dt_or_ts, (int, float)):
            dt = datetime.fromtimestamp(dt_or_ts, tz=timezone.utc)
        elif isinstance(dt_or_ts, datetime):
            if dt_or_ts.tzinfo is None:
                dt = dt_or_ts.replace(tzinfo=timezone.utc)
            else:
                dt = dt_or_ts.astimezone(timezone.utc)
        else:
            raise TypeError("dt_or_ts must be a float timestamp or a datetime object")

        active_events = []
        for event in MOCK_EVENTS:
            event_time = event["event_time_utc"]
            pre_win = timedelta(minutes=event["pre_window"])
            post_win = timedelta(minutes=event["post_window"])
            
            start_time = event_time - pre_win
            end_time = event_time + post_win
            
            if start_time <= dt < end_time:
                active_events.append(event)
                
        return active_events

    @staticmethod
    def is_red_flag_window(timestamp_now: float) -> bool:
        """Return True if *timestamp_now* falls inside a known high-risk window."""
        return len(MacroFilter.check_event(timestamp_now)) > 0
