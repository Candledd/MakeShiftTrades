"""
SQLite database layer for trade logging and expectancy tracking.

Provides persistence for completed trades, strategy-level expectancy
calculations, and time-windowed PnL aggregation used by the Risk Manager
to enforce portfolio-level heat, daily/weekly loss limits, and EV gates.
"""

import os
import sqlite3
from datetime import datetime, timedelta, timezone

# ── DB Path ────────────────────────────────────────────────────────────────────
# Database file lives next to the project root (one level up from src/).
_db_path: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "makeshift.db",
)

# Persistent connection pool (single connection reused across calls).
_conn: sqlite3.Connection | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_connection() -> sqlite3.Connection:
    """Return the persistent SQLite connection, creating it on first call."""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(_db_path, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
    return _conn


# ═══════════════════════════════════════════════════════════════════════════════
# Schema setup
# ═══════════════════════════════════════════════════════════════════════════════

def init_db() -> None:
    """Create the ``trade_log`` table if it does not already exist."""
    conn = _get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT    NOT NULL,
            strategy        TEXT    NOT NULL,
            direction       TEXT    NOT NULL,
            entry_price     REAL    NOT NULL,
            exit_price      REAL    NOT NULL,
            stop_loss       REAL    NOT NULL,
            qty             REAL    NOT NULL,
            pnl             REAL    NOT NULL,
            r_multiple      REAL    NOT NULL,
            mfe             REAL    NOT NULL,
            mae             REAL    NOT NULL,
            hold_hours      REAL    NOT NULL,
            exit_reason     TEXT    NOT NULL,
            regime          TEXT    DEFAULT 'unknown',
            timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_strategy_dir ON trade_log (strategy, direction)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_timestamp ON trade_log (timestamp)"
    )
    # Migrate existing databases — silently skip if column already exists.
    try:
        conn.execute("ALTER TABLE trade_log ADD COLUMN regime TEXT DEFAULT 'unknown'")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()


# ═══════════════════════════════════════════════════════════════════════════════
# Write helpers
# ═══════════════════════════════════════════════════════════════════════════════

def log_trade(
    symbol: str,
    strategy: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    stop_loss: float,
    qty: float,
    pnl: float,
    mfe: float,
    mae: float,
    hold_hours: float,
    exit_reason: str,
    regime: str = 'unknown',
) -> None:
    """Insert a completed trade into ``trade_log`` with its R-multiple.

    ``pnl`` must be expressed in absolute dollar terms (positive = gain).
    ``r_multiple`` is calculated internally as ``pnl / risk_dollars`` where
    ``risk_dollars = abs(entry_price - stop_loss) * qty``.
    """
    risk_per_unit = abs(entry_price - stop_loss)
    risk_dollars = risk_per_unit * qty
    r_multiple = pnl / risk_dollars if risk_dollars > 0 else 0.0

    conn = _get_connection()
    conn.execute(
        """
        INSERT INTO trade_log
            (symbol, strategy, direction, entry_price, exit_price,
             stop_loss, qty, pnl, r_multiple, mfe, mae,
             hold_hours, exit_reason, regime)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            symbol,
            strategy,
            direction,
            entry_price,
            exit_price,
            stop_loss,
            qty,
            pnl,
            r_multiple,
            mfe,
            mae,
            hold_hours,
            exit_reason,
            regime,
        ),
    )
    conn.commit()


# ═══════════════════════════════════════════════════════════════════════════════
# Read helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_strategy_expectancy(
    strategy: str,
    direction: str,
    symbol: str | None = None,
    regime: str | None = None,
) -> tuple[float, int]:
    """Return ``(ev_r, sample_size)`` for a given strategy + direction.

    Optionally further filter by ``symbol`` and/or ``regime``.
    ``ev_r`` is the arithmetic mean of ``r_multiple`` across all logged
    trades matching the filter.  Returns ``(0.0, 0)`` when no trades exist.
    """
    conn = _get_connection()

    query = """
        SELECT AVG(r_multiple) AS avg_r, COUNT(*) AS cnt
        FROM trade_log
        WHERE strategy = ? AND direction = ?
    """
    params: list = [strategy, direction]

    if symbol is not None:
        query += " AND symbol = ?"
        params.append(symbol)
    if regime is not None:
        query += " AND regime = ?"
        params.append(regime)

    row = conn.execute(query, params).fetchone()

    if row["cnt"] == 0:
        return 0.0, 0
    return row["avg_r"], row["cnt"]


def get_realized_pnl(days: int) -> float:
    """Return the sum of realised PnL over the last ``days`` calendar days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    conn = _get_connection()
    row = conn.execute(
        """
        SELECT COALESCE(SUM(pnl), 0.0) AS total
        FROM trade_log
        WHERE timestamp >= ?
        """,
        (cutoff,),
    ).fetchone()

    return row["total"]

# Automatically initialize the DB on import
init_db()
