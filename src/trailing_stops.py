"""
Strategy-specific trailing stop logic engine.

Provides trailing stop calculation for each strategy type, replacing the
old single TRAILING_STOP_PCT percentage multiplier with per-strategy
algorithms (VWAP, SMA/EMA, Donchian, chandelier, ATR) and configurable
ATR multipliers aligned with codex.md item 14.

Strategy → trailing logic mapping (config-overridable):

    mean_reversion    → vwap           (tight trailing at VWAP after TP)
    trend_pullback    → sma20_or_ema   (trail below SMA20 / EMA50)
    momentum_breakout → donchian       (20-bar Donchian channel)
    trend_following   → atr            (ATR chandelier, wider for GLD)
    PDBC              → tighter_atr    (ATR trail + event-risk tightening)
"""

from __future__ import annotations

import time
import pandas as pd
import numpy as np
import config
from src.macro_filter import MacroFilter


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _compute_vwap(df: pd.DataFrame) -> float | None:
    """Compute the latest Close-based VWAP value.

    Uses the same calculation as the strategy modules (Close × Volume)
    for consistency.
    """
    if df is None or len(df) < 1:
        return None
    vwap_series = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    val = vwap_series.iloc[-1]
    return float(val) if pd.notna(val) else None


def _get_atr_multiplier(logic_type: str, ticker: str = "") -> float:
    """Return the ATR multiplier for *logic_type*, with ticker-specific overrides.

    Ticker-level overrides (GLD, PDBC) take precedence over the generic
    logic-type defaults so we can satisfy recommendations from codex.md:
    - GLD: wider hold window
    - PDBC: tighter trail + event-risk awareness
    """
    # ── Ticker-specific overrides ──────────────────────────────────────────
    if "GLD" in ticker.upper():
        return getattr(config, "GLD_TRAIL_ATR_MULT", 3.5)
    if "PDBC" in ticker.upper():
        return getattr(config, "PDBC_TRAIL_ATR_MULT", 2.5)

    # ── Logic-type defaults (fallback when ticker is not recognised) ───────
    mult_map = {
        "vwap":         getattr(config, "MR_TRAIL_ATR_MULT", 0.8),
        "sma20_or_ema": getattr(config, "TP_TRAIL_ATR_MULT", 2.0),
        "donchian":     getattr(config, "MB_TRAIL_ATR_MULT", 3.0),
        "chandelier":   getattr(config, "MB_TRAIL_ATR_MULT", 3.0),
        "atr":          getattr(config, "GLD_TRAIL_ATR_MULT", 3.5),
        "tighter_atr":  getattr(config, "PDBC_TRAIL_ATR_MULT", 2.5),
    }
    return mult_map.get(logic_type, 2.0)


def get_strategy_trailing_logic(strategy_name: str, ticker: str = "") -> str:
    """Map *strategy_name* to the appropriate trailing logic type.

    Respects config-level overrides (``{STRATEGY}_TRAILING_STOP_LOGIC``)
    so the AI tuner can swap logic per-regime.  Falls back to sensible
    defaults aligned with codex.md item 14.
    """
    import config as cfg

    # Config-based override (e.g. MR_TRAILING_STOP_LOGIC)
    config_key = f"{strategy_name.upper()}_TRAILING_STOP_LOGIC"
    override = getattr(cfg, config_key, None)
    if override:
        return override

    # Hard-coded defaults
    mapping = {
        "mean_reversion":    "vwap",
        "trend_pullback":    "sma20_or_ema",
        "momentum_breakout": "donchian",
        "trend_following":   "atr",
    }
    return mapping.get(strategy_name, "default")


# ── Main trailing-stop dispatcher ───────────────────────────────────────────────

def calculate_trailing_stop(
    logic_type: str,
    current_price: float,
    current_sl: float | None,
    direction: str,
    df: pd.DataFrame | None,
    atr: float,
    entry_price: float | None = None,
    tp_filled: bool = False,
    highest_price: float | None = None,
    lowest_price: float | None = None,
    ticker: str = "",
) -> float | None:
    """Compute the new stop-loss price for the given trailing *logic_type*.

    Parameters
    ----------
    logic_type:
        One of ``breakeven_only``, ``vwap``, ``sma20_or_ema``, ``donchian``,
        ``chandelier``, ``atr``, ``tighter_atr``, or a custom value.
    current_price:
        Latest market price.
    current_sl:
        Current stop-loss value (may be *None* on first call).
    direction:
        ``"long"`` / ``"buy"`` or ``"short"`` / ``"sell"``.
    df:
        OHLCV DataFrame (required by ``vwap``, ``sma20_or_ema``, ``donchian``).
    atr:
        ATR value at entry (used by ATR-based logic types).
    entry_price:
        Position entry price (used by ``breakeven_only``).
    tp_filled:
        Whether a partial take-profit has already filled.
    highest_price / lowest_price:
        Extreme prices since entry (used by chandelier / ATR trailing).
    ticker:
        Ticker symbol for ticker-specific overrides (PDBC event tightening).
    """
    is_long = str(direction).lower() in ("long", "buy")

    if current_sl is None:
        return None

    ref_high = highest_price if highest_price is not None else current_price
    ref_low = lowest_price if lowest_price is not None else current_price

    # ── breakeven_only (mean reversion current default) ───────────────────
    if logic_type == "breakeven_only":
        if tp_filled and entry_price is not None:
            return max(current_sl, entry_price) if is_long else min(current_sl, entry_price)
        return current_sl

    # ── vwap (mean reversion recommended — codex.md item 14) ──────────────
    elif logic_type == "vwap":
        if not tp_filled:
            # Do not trail until partial TP is taken (let the mean reversion play out)
            return current_sl
        vwap_val = _compute_vwap(df)
        if vwap_val is not None:
            if is_long:
                return max(current_sl, vwap_val)
            else:
                return min(current_sl, vwap_val)
        return current_sl

    # ── sma20_or_ema (trend pullback) ─────────────────────────────────────
    elif logic_type == "sma20_or_ema":
        if df is not None and len(df) >= 20:
            close = df["Close"]
            sma20_series = close.rolling(20).mean()
            if len(sma20_series) > 0 and pd.notna(sma20_series.iloc[-1]):
                val_sma20 = float(sma20_series.iloc[-1])
                ema50_series = close.ewm(span=50, adjust=False).mean()
                val_ema50 = (
                    float(ema50_series.iloc[-1])
                    if len(ema50_series) > 0 and pd.notna(ema50_series.iloc[-1])
                    else val_sma20
                )

                trail_ref = val_sma20
                if is_long:
                    if val_ema50 > trail_ref:
                        trail_ref = val_ema50
                    return max(current_sl, trail_ref)
                else:
                    if val_ema50 < trail_ref:
                        trail_ref = val_ema50
                    return min(current_sl, trail_ref)
        return current_sl

    # ── donchian (BTC momentum breakout) ──────────────────────────────────
    elif logic_type == "donchian":
        if df is not None and len(df) >= 21:
            low = df["Low"]
            high = df["High"]
            donchian_lower_series = low.shift(1).rolling(20).min()
            donchian_upper_series = high.shift(1).rolling(20).max()

            if is_long:
                if len(donchian_lower_series) > 0 and pd.notna(donchian_lower_series.iloc[-1]):
                    return max(current_sl, float(donchian_lower_series.iloc[-1]))
            else:
                if len(donchian_upper_series) > 0 and pd.notna(donchian_upper_series.iloc[-1]):
                    return min(current_sl, float(donchian_upper_series.iloc[-1]))
        return current_sl

    # ── chandelier (alternative for BTC / GLD — codex.md item 14) ─────────
    elif logic_type == "chandelier":
        atr_val = atr if atr is not None else 0.0
        if atr_val > 0:
            mult = _get_atr_multiplier(logic_type, ticker)
            if is_long:
                return max(current_sl, ref_high - mult * atr_val)
            else:
                return min(current_sl, ref_low + mult * atr_val)
        return current_sl

    # ── atr / tighter_atr (GLD, PDBC trend following) ─────────────────────
    elif logic_type in ("atr", "tighter_atr"):
        atr_val = atr if atr is not None else 0.0
        if atr_val > 0:
            mult = _get_atr_multiplier(logic_type, ticker)

            # PDBC event-risk tightening (codex.md item 14): tighten the
            # trailing distance during macro events so the position is
            # less exposed to event-driven gap risk.
            # Only applied when there is an active event affecting the ticker.
            if "PDBC" in ticker.upper():
                active_events = MacroFilter.check_event(time.time())
                # Normalize ticker for comparison (remove "-USD" suffix)
                normal_ticker = ticker.upper().replace("-USD", "")
                has_active_event = False
                for event in active_events:
                    affected = event.get("affected_assets", [])
                    normalized_affected = [a.upper().replace("-USD", "") for a in affected]
                    if "ALL" in normalized_affected or normal_ticker in normalized_affected:
                        has_active_event = True
                        break
                if has_active_event:
                    event_factor = getattr(config, "PDBC_EVENT_TIGHTEN_FACTOR", 0.5)
                    mult = mult * event_factor

            trail_dist = atr_val * mult
            if is_long:
                return max(current_sl, ref_high - trail_dist)
            else:
                return min(current_sl, ref_low + trail_dist)
        return current_sl

    # ── Default fallback (unknown logic types) ────────────────────────────
    else:
        # Use ATR-based trailing with a configurable multiplier instead of
        # the old percentage-based TRAILING_STOP_PCT.
        atr_val = atr if atr is not None else 0.0
        if atr_val > 0:
            mult = _get_atr_multiplier(logic_type, ticker)
            trail_dist = atr_val * mult
            if is_long:
                return max(current_sl, ref_high - trail_dist)
            else:
                return min(current_sl, ref_low + trail_dist)
        return current_sl
