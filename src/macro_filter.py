"""Macro Kill Switch — Filters out high-risk event windows.

Blocks trading during known macro-economic news releases where
volatility spikes can wreak havoc on intraday positions.

Uses an explicit event list with event_time_utc, affected_assets,
pre_window, post_window, and actions.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)


class MacroFilter:
    """Schedule-based macro-event red-flag window detector using explicit event lists."""

    _events: List[Dict[str, Any]] = []
    _last_mtime: float = 0.0
    _last_check_time: float = 0.0
    _CHECK_INTERVAL: float = 5.0
    _lock = threading.Lock()
    _calendar_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "macro_calendar.json")

    @classmethod
    def _load_events(cls) -> List[Dict[str, Any]]:
        """Load events from macro_calendar.json, caching by mtime for hot-reloading.

        If the file has been modified since the last load (or this is the first
        call), re-read and parse the JSON.  On any error (missing file, invalid
        JSON, missing keys, etc.) the error is logged and an empty list is
        returned.
        """
        with cls._lock:
            now = time.time()
            if now - cls._last_check_time < cls._CHECK_INTERVAL and cls._events:
                return cls._events
            cls._last_check_time = now

            try:
                current_mtime = os.path.getmtime(cls._calendar_path)
                if current_mtime == cls._last_mtime and cls._events:
                    return cls._events

                with open(cls._calendar_path, "r") as f:
                    raw_events: List[Dict[str, Any]] = json.load(f)

                parsed = []
                for event in raw_events:
                    try:
                        dt = datetime.fromisoformat(event["time_utc"].replace("Z", "+00:00"))
                        event_time_utc = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                        parsed.append({
                            "event_name": event["name"],
                            "event_time_utc": event_time_utc,
                            "affected_assets": event.get("affected_assets", []),
                            "pre_window_td": timedelta(minutes=int(event["pre_window"])),
                            "post_window_td": timedelta(minutes=int(event["post_window"])),
                            "actions": event.get("actions", [])
                        })
                    except Exception as item_err:
                        logger.warning("Skipping invalid event %s: %s", event, item_err)

                cls._events = parsed
                cls._last_mtime = current_mtime
                return cls._events
            except Exception as e:
                logger.error("Failed to load macro calendar from %s: %s", cls._calendar_path, e)
                cls._events = []
                cls._last_mtime = 0.0
                return []

    @classmethod
    def check_event(cls, dt_or_ts: float | datetime) -> List[Dict[str, Any]]:
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
        for event in cls._load_events():
            event_time = event["event_time_utc"]
            pre_win_td = event["pre_window_td"]
            post_win_td = event["post_window_td"]

            start_time = event_time - pre_win_td
            end_time = event_time + post_win_td

            if start_time <= dt < end_time:
                active_events.append(event)

        return active_events

    @staticmethod
    def is_red_flag_window(timestamp_now: float) -> bool:
        """Return True if *timestamp_now* falls inside a known high-risk window."""
        return len(MacroFilter.check_event(timestamp_now)) > 0
