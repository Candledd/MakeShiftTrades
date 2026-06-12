"""Alpaca Paper Trading Integration
===================================

Handles authentication, order validation, bracket order placement,
confidence-based position sizing, and order management.

Authentication uses environment variables ALPACA_API_KEY and
ALPACA_SECRET_KEY — never hard-coded values.

Architecture
------------
- AlpacaTrader  : singleton-style class, lazy-initialised in server.py
- validate_order: standalone "double-check" function called before every
                  dispatch — verifies side/entry/SL/TP consistency and R/R
- TICKER_MAP     : yfinance → Alpaca equity proxies (futures not supported
                   on Alpaca paper; mapped to equivalent ETFs)
- $5,000 hard cap enforced per-trade; grows with realised profits tracked
  in trades_state.json
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

import config
from src.database import log_trade

logger = logging.getLogger(__name__)

# ── Tick → Alpaca equity proxy ───────────────────────────────────────────────
# Alpaca paper trading supports equities and crypto only; futures are
# represented by their closest ETF proxy.
TICKER_MAP: Dict[str, str] = {
    "NQ=F":    "QQQ",     # NASDAQ-100 futures  → Invesco QQQ ETF
    "ES=F":    "SPY",     # S&P 500 futures     → SPDR S&P 500 ETF
    "YM=F":    "DIA",     # Dow Jones futures   → SPDR DJIA ETF
    "RTY=F":   "IWM",     # Russell 2000 futures→ iShares Russell 2000 ETF
    "GC=F":    "GLD",     # Gold futures        → SPDR Gold Shares ETF
    "CL=F":    "PDBC",     # Crude oil/Commodities futures -> PDBC
    "BTC-USD": "BTCUSD",  # Bitcoin             → BTC/USD crypto pair
    "ETH-USD": "ETHUSD",  # Ethereum            → ETH/USD crypto pair
}

# Alpaca symbols that are crypto pairs (trade 24/7 — use GTC not DAY)
CRYPTO_SYMBOLS: set = {"BTCUSD", "ETHUSD", "BCHUSD", "LTCUSD", "UNIUSD", "LINKUSD"}

# Starting hard cash limit (grows with realised profits).
INITIAL_CASH_LIMIT: float = 5_000.0

# Minimum R/R the double-check validator enforces.
MIN_RR_HARD: float = 1.0

# Maximum entry-vs-current-price deviation (%) before rejecting the order.
MAX_ENTRY_DEVIATION_PCT: float = 2.0

# Minimum distance (in price units) that TP/SL must sit away from the Alpaca
# market fill price (base_price).  Alpaca enforces >= 0.01; we use 0.02 to
# give a small buffer against last-millisecond price drift.
MIN_BRACKET_BUFFER: float = 0.02


# ─────────────────────────────────────────────────────────────────────────────
# Double-Check Validator (standalone, used before every order dispatch)
# ─────────────────────────────────────────────────────────────────────────────

def validate_order(
    side: str,
    entry: float,
    stop_loss: float,
    take_profit: float,
    current_price: float,
    min_rr: float = MIN_RR_HARD,
) -> Tuple[bool, str]:
    """Secondary validation of order parameters against strategy levels.

    Checks:
    1. Side is "BUY" or "SELL".
    2. Stop-loss is correctly placed relative to entry.
    3. Take-profit is correctly placed relative to entry.
    4. Calculated R/R >= min_rr.
    5. Entry price is within MAX_ENTRY_DEVIATION_PCT of the current price
       (stale signal guard).

    Returns
    -------
    (True, "OK") if all checks pass, else (False, <reason>).
    """
    side = side.upper()
    if side not in ("BUY", "SELL"):
        return False, f"Unknown order side: {side!r}. Must be 'BUY' or 'SELL'."

    if side == "BUY":
        if stop_loss >= entry:
            return False, (
                f"BUY stop_loss ${stop_loss:.4f} must be strictly below "
                f"entry ${entry:.4f}."
            )
        if take_profit <= entry:
            return False, (
                f"BUY take_profit ${take_profit:.4f} must be strictly above "
                f"entry ${entry:.4f}."
            )
    else:  # SELL
        if stop_loss <= entry:
            return False, (
                f"SELL stop_loss ${stop_loss:.4f} must be strictly above "
                f"entry ${entry:.4f}."
            )
        if take_profit >= entry:
            return False, (
                f"SELL take_profit ${take_profit:.4f} must be strictly below "
                f"entry ${entry:.4f}."
            )

    risk   = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    if risk == 0:
        return False, "Risk (|entry - stop_loss|) is zero — cannot size position."
    rr = reward / risk
    if rr < min_rr:
        return False, (
            f"R/R {rr:.2f}:1 is below the minimum {min_rr:.2f}:1 requirement."
        )

    if current_price > 0:
        deviation_pct = abs(entry - current_price) / current_price * 100.0
        if deviation_pct > MAX_ENTRY_DEVIATION_PCT:
            return False, (
                f"Entry ${entry:.4f} is {deviation_pct:.1f}% away from current "
                f"price ${current_price:.4f} (max {MAX_ENTRY_DEVIATION_PCT}%). "
                "Signal may be stale."
            )

    return True, "OK"


# ─────────────────────────────────────────────────────────────────────────────
# Confidence → risk fraction helper
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_risk_fraction(confidence: float) -> float:
    """Map signal confidence (0–100) to fraction of available capital to risk.

    Tiers:
      ≥ 90 % → 4 % of available capital
      ≥ 80 % → 3 %
      ≥ 70 % → 2 %
      < 70 % → 1 %
    """
    if confidence >= 90.0:
        return 0.04
    elif confidence >= 80.0:
        return 0.03
    elif confidence >= 70.0:
        return 0.02
    return 0.01


# ─────────────────────────────────────────────────────────────────────────────
# AlpacaTrader
# ─────────────────────────────────────────────────────────────────────────────

class AlpacaTrader:
    """Manages paper trading via the Alpaca Paper Trading API.

    Instantiate once and share via the Flask app context.  Credentials are
    read exclusively from environment variables; the constructor raises
    EnvironmentError if they are absent.
    """

    STATE_FILE: Path = Path(__file__).parent.parent / "trades_state.json"

    def __init__(self) -> None:
        api_key    = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set as environment "
                "variables (e.g. in .env).  Do not hard-code credentials."
            )

        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,
        )
        self._state: Dict[str, Any] = self._load_state()
        logger.info("AlpacaTrader initialised (paper=True).")

    # ── State persistence ──────────────────────────────────────────────────────

    def _load_state(self) -> Dict[str, Any]:
        """Load persistent trade state from disk.  Returns a safe default on error."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                # Ensure expected keys exist (forward-compatibility)
                data.setdefault("cash_limit",    INITIAL_CASH_LIMIT)
                data.setdefault("active_orders", {})
                data.setdefault("paper_enabled", {})
                data.setdefault("total_pnl",     0.0)
                data.setdefault("order_journal", {})
                data.setdefault("ml_feedback_queue", [])
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read trades_state.json: %s", exc)

        return {
            "cash_limit":    INITIAL_CASH_LIMIT,
            "active_orders": {},   # {alpaca_symbol: order_id}
            "paper_enabled": {},   # {original_ticker: bool}
            "total_pnl":     0.0,
            "order_journal": {},   # {order_id: {...}}
            "ml_feedback_queue": [],
        }

    def _save_state(self) -> None:
        try:
            with open(self.STATE_FILE, "w", encoding="utf-8") as fh:
                json.dump(self._state, fh, indent=2)
        except OSError as exc:
            logger.error("Could not write trades_state.json: %s", exc)

    # ── Ticker mapping ─────────────────────────────────────────────────────────

    def map_ticker(self, ticker: str) -> str:
        """Return the Alpaca-compatible symbol for a given (possibly futures) ticker."""
        return TICKER_MAP.get(ticker.upper(), ticker.upper())

    # ── Toggle state ───────────────────────────────────────────────────────────

    def set_enabled(self, ticker: str, enabled: bool) -> None:
        """Enable or disable paper trading for a specific dashboard ticker."""
        self._state.setdefault("paper_enabled", {})[ticker] = enabled
        self._save_state()
        logger.info("Paper trading %s for %s.", "ENABLED" if enabled else "DISABLED", ticker)

    def is_enabled(self, ticker: str) -> bool:
        return bool(self._state.get("paper_enabled", {}).get(ticker, False))

    def get_all_enabled(self) -> Dict[str, bool]:
        return dict(self._state.get("paper_enabled", {}))

    # ── Account info ───────────────────────────────────────────────────────────

    def get_account(self) -> Dict[str, Any]:
        """Return account summary including our enforced cash_limit."""
        try:
            acct = self._client.get_account()
            return {
                "ok":           True,
                "status":       str(acct.status),
                "equity":       float(acct.equity),
                "buying_power": float(acct.buying_power),
                "cash":         float(acct.cash),
                "cash_limit":   self._state.get("cash_limit", INITIAL_CASH_LIMIT),
                "total_pnl":    round(self._state.get("total_pnl", 0.0), 2),
            }
        except Exception as exc:
            logger.error("get_account failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    # ── Active orders ──────────────────────────────────────────────────────────

    def get_active_orders(self) -> Dict[str, Any]:
        """Return all open / pending orders from Alpaca."""
        try:
            orders = self._client.get_orders(
                filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
            result = []
            for o in orders:
                result.append({
                    "id":               str(o.id),
                    "symbol":           str(o.symbol),
                    "side":             str(o.side.value) if hasattr(o.side, "value") else str(o.side),
                    "type":             str(o.order_type.value) if hasattr(o.order_type, "value") else str(getattr(o, "order_type", "")),
                    "qty":              str(o.qty),
                    "status":           str(o.status.value) if hasattr(o.status, "value") else str(o.status),
                    "submitted_at":     str(o.submitted_at),
                    "limit_price":      float(o.limit_price) if getattr(o, "limit_price", None) else None,
                    "stop_price":       float(o.stop_price) if getattr(o, "stop_price", None) else None,
                    "filled_avg_price": float(o.filled_avg_price) if getattr(o, "filled_avg_price", None) else None,
                    "order_class":      str(o.order_class.value) if hasattr(o.order_class, "value") else str(o.order_class),
                    "parent_id":        str(o.parent_id) if getattr(o, "parent_id", None) else None,
                })
            return {"ok": True, "orders": result}
        except Exception as exc:
            logger.error("get_active_orders failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    # ── Get positions ──────────────────────────────────────────────────────────

    def get_positions(self) -> List[Dict[str, Any]]:
        """Return all open positions with symbol, side, qty, market_value, unrealized_pl."""
        try:
            positions = self._client.get_all_positions()
            result = []
            for pos in positions:
                side_raw = getattr(pos, 'side', '')
                side_str = str(getattr(side_raw, 'value', side_raw)).strip().lower()
                result.append({
                    'symbol': str(pos.symbol),
                    'side': side_str,
                    'qty': float(pos.qty),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl) if pos.unrealized_pl else 0.0,
                    'current_price': float(pos.current_price) if pos.current_price else 0.0,
                    'avg_entry_price': float(pos.avg_entry_price) if pos.avg_entry_price else 0.0,
                })
            return result
        except Exception as exc:
            logger.error("get_positions failed: %s", exc)
            return []

    # ── Cancel order ───────────────────────────────────────────────────────────

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order by its Alpaca UUID."""
        try:
            self._client.cancel_order_by_id(order_id)
            # Clean up local state
            for sym, oid in list(self._state.get("active_orders", {}).items()):
                if oid == order_id:
                    del self._state["active_orders"][sym]
            self._save_state()
            logger.info("Cancelled order %s.", order_id)
            return {"ok": True, "cancelled": order_id}
        except Exception as exc:
            logger.error("cancel_order(%s) failed: %s", order_id, exc)
            return {"ok": False, "error": str(exc)}

    # ── Position sizing ────────────────────────────────────────────────────────

    def calc_notional(
        self,
        entry: float,
        stop_loss: float,
        confidence: float,
        available_capital: float,
    ) -> float:
        """Calculate the dollar notional to deploy based on confidence.

        Algorithm
        ---------
        risk_amount  = available_capital × risk_fraction(confidence)
        risk_per_$   = |entry − stop_loss| / entry          (fraction of price at risk)
        notional     = risk_amount / risk_per_$              ($ amount that risks exactly risk_amount)
        notional     = min(notional, available_capital)      (never exceed available capital)
        notional     = round to 2 dp, min $1.00
        """
        risk_per_share = abs(entry - stop_loss)
        if risk_per_share == 0 or entry <= 0:
            return 0.0

        risk_fraction = _confidence_risk_fraction(confidence)
        risk_amount   = available_capital * risk_fraction
        notional      = (risk_amount / risk_per_share) * entry
        notional      = min(notional, available_capital)
        return max(1.0, round(notional, 2))

    # ── Preflight conflict guards ─────────────────────────────────────────────

    def _list_open_orders_for_symbol(self, symbol: str) -> List[Any]:
        """Return currently open Alpaca orders for a specific symbol."""
        orders = self._client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        return [o for o in orders if str(getattr(o, "symbol", "")).upper() == symbol.upper()]

    def _position_side_for_symbol(self, symbol: str) -> Optional[str]:
        """Return 'LONG'/'SHORT' if a live position exists for symbol, else None.

        Raises on API failure — callers must catch to avoid treating an error
        as "no position exists" and allowing a conflicting order through.
        """
        positions = self._client.get_all_positions()
        for pos in positions:
            if str(getattr(pos, "symbol", "")).upper() != symbol.upper():
                continue
            raw_side = getattr(pos, "side", "")
            side_str = str(getattr(raw_side, "value", raw_side)).strip().upper()
            if side_str in {"LONG", "SHORT"}:
                return side_str
        return None

    def _preflight_symbol_conflict(self, symbol: str, side: str,
                                current_positions: Optional[List[Dict]] = None) -> Tuple[bool, str]:
        """Block non-entry bracket attempts before sending to Alpaca.

        Alpaca rejects bracket orders with `bracket orders must be entry orders`
        when there is an active order chain or existing position on the symbol.
        This guard turns those into deterministic, actionable responses.

        If *current_positions* is provided it is used instead of calling the
        Alpaca API to look up existing positions on the symbol.
        """
        desired = side.upper()

        try:
            open_orders = self._list_open_orders_for_symbol(symbol)
        except Exception as exc:
            logger.warning(
                "Open-orders check failed for %s: %s — blocking as precaution.", symbol, exc
            )
            return False, f"Could not verify open orders for {symbol}: {exc}. Try again shortly."

        if open_orders:
            o = open_orders[0]
            oid = str(getattr(o, "id", ""))
            o_side = str(getattr(getattr(o, "side", ""), "value", getattr(o, "side", ""))).upper()
            o_status = str(getattr(getattr(o, "status", ""), "value", getattr(o, "status", ""))).upper()
            return False, (
                f"Existing open order on {symbol} ({o_side}/{o_status}, id={oid[:8]}...). "
                "Skip new entry until it fills/closes or cancel it first."
            )

        # Determine position side from provided list or via API
        pos_side: Optional[str] = None
        if current_positions is not None:
            for pos in current_positions:
                if str(pos.get("symbol", "")).upper() == symbol.upper():
                    side_raw = str(pos.get("side", "")).strip().upper()
                    if side_raw in ("LONG", "SHORT"):
                        pos_side = side_raw
                    break
        else:
            try:
                pos_side = self._position_side_for_symbol(symbol)
            except Exception as exc:
                logger.warning(
                    "Position check failed for %s: %s — blocking as precaution.", symbol, exc
                )
                return False, f"Could not verify position state for {symbol}: {exc}. Try again shortly."

        if pos_side is None:
            return True, "OK"

        if (desired == "BUY" and pos_side == "LONG") or (desired == "SELL" and pos_side == "SHORT"):
            return False, f"Existing {pos_side} position on {symbol}. Duplicate same-side entry blocked."

        return False, (
            f"Existing {pos_side} position on {symbol} conflicts with {desired} bracket entry. "
            "Flatten the position first."
        )

    # ── Place bracket order ────────────────────────────────────────────────────

    def place_bracket_order(
        self,
        ticker: str,
        side: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        current_price: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Validate and submit a bracket order (entry + limit TP + stop SL).

        The double-check validator runs first.  If it passes, account capital
        is checked against the $5,000 hard limit.  Position size is then
        computed from confidence before the order is dispatched.

        All failures and successes are logged to the application logger.

        Returns a dict with key "ok" plus context on success or error details.
        """
        alpaca_ticker = self.map_ticker(ticker)

        # ── Step 0: Preflight conflict guard ──────────────────────────────────
        pre_ok, pre_msg = self._preflight_symbol_conflict(alpaca_ticker, side)
        if not pre_ok:
            logger.warning("[PRECHECK BLOCKED] %s/%s — %s", ticker, alpaca_ticker, pre_msg)
            return {
                "ok": False,
                "error": pre_msg,
                "error_code": "preflight_conflict",
            }

        # ── Step 1: Double-check validation ───────────────────────────────────
        valid, reason = validate_order(
            side=side,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=current_price,
        )
        if not valid:
            msg = f"[VALIDATION FAILED] {ticker}/{alpaca_ticker} — {reason}"
            logger.warning(msg)
            return {"ok": False, "error": f"Validation failed: {reason}"}

        # ── Step 2: Capital check against $5 k hard limit ─────────────────────
        cash_limit = float(self._state.get("cash_limit", INITIAL_CASH_LIMIT))
        try:
            acct = self._client.get_account()
            # Crypto orders use cash (not marginable buying power).
            # Equity orders can use the full buying_power (which may include margin).
            is_crypto_order = alpaca_ticker.upper() in CRYPTO_SYMBOLS
            raw_bp = float(acct.cash) if is_crypto_order else float(acct.buying_power)
            available_bp = min(raw_bp, cash_limit)
        except Exception as exc:
            logger.error("Account fetch failed before order placement: %s", exc)
            return {"ok": False, "error": f"Account fetch failed: {exc}"}

        if available_bp < 1.0:
            if alpaca_ticker.upper() in CRYPTO_SYMBOLS:
                msg = (
                    f"Insufficient crypto USD balance: available ${available_bp:.2f}. "
                    "Alpaca crypto trading uses a separate USD wallet — check that your "
                    "paper account has crypto enabled and USD funded in the Alpaca dashboard."
                )
            else:
                msg = (
                    f"Insufficient capital: available ${available_bp:.2f} "
                    f"(hard limit ${cash_limit:.2f})."
                )
            logger.error("[REJECTED] %s", msg)
            return {"ok": False, "error": msg}

        # ── Step 3: Position sizing (dollar notional) ──────────────────────────
        notional = self.calc_notional(entry, stop_loss, confidence, available_bp)
        if notional < 1.0:
            return {
                "ok": False,
                "error": "Position size resolved to $0 — stop distance too small or capital too low.",
            }

        # ── Step 3½: Clamp TP/SL to be valid relative to current_price ───────────
        # Alpaca validates bracket legs against the actual fill price (base_price),
        # NOT our signal entry.  If the market has moved even a cent, our TP can
        # end up on the wrong side and the whole order is rejected.
        # We snap each leg to be at least MIN_BRACKET_BUFFER away from
        # current_price in the direction the strategy requires.
        # (Skipped for crypto — broker rejects bracket orders; simple orders used.)
        ref    = current_price if current_price > 0 else entry
        is_buy = side.upper() == "BUY"
        if alpaca_ticker.upper() not in CRYPTO_SYMBOLS:
            if is_buy:
                tp_min = round(ref + MIN_BRACKET_BUFFER, 2)
                sl_max = round(ref - MIN_BRACKET_BUFFER, 2)
                if take_profit < tp_min:
                    logger.warning(
                        "[BRACKET CLAMP] BUY TP %.4f < ref+buf %.4f — clamping.",
                        take_profit, tp_min,
                    )
                    take_profit = tp_min
                if stop_loss > sl_max:
                    logger.warning(
                        "[BRACKET CLAMP] BUY SL %.4f > ref-buf %.4f — clamping.",
                        stop_loss, sl_max,
                    )
                    stop_loss = sl_max
            else:  # SELL
                tp_max = round(ref - MIN_BRACKET_BUFFER, 2)
                sl_min = round(ref + MIN_BRACKET_BUFFER, 2)
                if take_profit > tp_max:
                    logger.warning(
                        "[BRACKET CLAMP] SELL TP %.4f > ref-buf %.4f — clamping.",
                        take_profit, tp_max,
                    )
                    take_profit = tp_max
                if stop_loss < sl_min:
                    logger.warning(
                        "[BRACKET CLAMP] SELL SL %.4f < ref+buf %.4f — clamping.",
                        stop_loss, sl_min,
                    )
                    stop_loss = sl_min

        # ── Step 4: Submit order ────────────────────────────────────────────
        import json as _json
        import re as _re

        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        is_crypto  = alpaca_ticker.upper() in CRYPTO_SYMBOLS

        def _build_req(use_qty: bool = False) -> MarketOrderRequest:
            # We use GTC for everything so that Stop-Loss and Take-Profit legs 
            # survive overnight. (The market hours guard prevents entry market
            # orders from queueing overnight).
            tif = TimeInForce.GTC
            if is_crypto:
                # Alpaca does not support OTO/OCO/Bracket orders for crypto.
                # Submit a simple market order; SL/TP are tracked in the journal
                # but not enforced at the broker level for crypto positions.
                common: dict = dict(
                    symbol=alpaca_ticker,
                    side=order_side,
                    time_in_force=tif,
                    order_class=OrderClass.SIMPLE,
                )
            else:
                common = dict(
                    symbol=alpaca_ticker,
                    side=order_side,
                    time_in_force=tif,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
                    stop_loss=StopLossRequest(stop_price=round(stop_loss, 2)),
                )
            if use_qty:
                ref_px = ref if ref > 0 else max(entry, 0.01)
                qty = max(1, int(notional / ref_px))
                return MarketOrderRequest(qty=qty, **common)
            return MarketOrderRequest(notional=notional, **common)

        try:
            order = self._client.submit_order(_build_req())
        except Exception as first_exc:
            err_str = str(first_exc)

            # Alpaca limitation: bracket orders cannot use fractional notional.
            # Retry once with whole-share quantity sizing.
            if "fractional orders must be simple orders" in err_str.lower():
                try:
                    order = self._client.submit_order(_build_req(use_qty=True))
                except Exception as qty_exc:
                    err_str = str(qty_exc)
                    logger.error(
                        "[ORDER ERROR after qty retry] %s (%s): %s",
                        ticker, alpaca_ticker, err_str,
                    )
                    return {"ok": False, "error": f"Order submission error: {err_str}"}
                else:
                    err_str = ""

            # Alpaca rejects bracket/OTO/OCO for crypto — retry as simple order.
            elif "does not support complex orders" in err_str.lower() or (
                is_crypto and ("bracket" in err_str.lower() or "oto" in err_str.lower())
            ):
                logger.warning(
                    "[CRYPTO SIMPLE RETRY] %s — broker rejected complex order; retrying as simple.",
                    alpaca_ticker,
                )
                from alpaca.trading.requests import MarketOrderRequest as _MOR
                tif = TimeInForce.GTC
                simple_req: MarketOrderRequest
                try:
                    if notional >= 1.0:
                        simple_req = _MOR(
                            symbol=alpaca_ticker,
                            side=order_side,
                            time_in_force=tif,
                            order_class=OrderClass.SIMPLE,
                            notional=notional,
                        )
                    else:
                        qty = max(1, int(notional / max(ref, 0.01)))
                        simple_req = _MOR(
                            symbol=alpaca_ticker,
                            side=order_side,
                            time_in_force=tif,
                            order_class=OrderClass.SIMPLE,
                            qty=qty,
                        )
                    order = self._client.submit_order(simple_req)
                    err_str = ""
                except Exception as simple_exc:
                    err_str = str(simple_exc)
                    logger.error(
                        "[ORDER ERROR after simple retry] %s (%s): %s",
                        ticker, alpaca_ticker, err_str,
                    )
                    return {"ok": False, "error": f"Order submission error: {err_str}"}

            # ── Alpaca bracket-leg rejection (error 42210000) ──────────────────
            # Alpaca validates SL/TP against its own fill quote (base_price), which
            # can differ from our yfinance reference by a few cents.  On this specific
            # error we parse base_price from the payload, re-clamp both legs to be
            # strictly valid, and retry exactly once.
            if err_str and ("42210000" in err_str or (
                "base_price" in err_str and "stop_price" in err_str
            )):
                bp: Optional[float] = None
                try:
                    # SDK usually stringifies the raw JSON error body.
                    err_json = _json.loads(err_str)
                    bp = float(err_json["base_price"])
                except (ValueError, KeyError, TypeError):
                    m = _re.search(r'"base_price"\s*:\s*"?([\d.]+)"?', err_str)
                    if m:
                        bp = float(m.group(1))

                if bp is not None and bp > 0:
                    logger.warning(
                        "[BRACKET RETRY] Alpaca base_price=%.4f differs from "
                        "ref=%.4f — re-clamping SL/TP to base_price and retrying.",
                        bp, ref,
                    )
                    if is_buy:
                        stop_loss   = round(bp - MIN_BRACKET_BUFFER, 2)
                        take_profit = max(take_profit, round(bp + MIN_BRACKET_BUFFER, 2))
                    else:
                        stop_loss   = round(bp + MIN_BRACKET_BUFFER, 2)
                        take_profit = min(take_profit, round(bp - MIN_BRACKET_BUFFER, 2))
                    try:
                        order = self._client.submit_order(_build_req())
                    except Exception as retry_exc:
                        err_str = str(retry_exc)
                        logger.error(
                            "[ORDER ERROR after retry] %s (%s): %s",
                            ticker, alpaca_ticker, err_str,
                        )
                        return {"ok": False, "error": f"Order submission error: {err_str}"}
                else:
                    logger.error("[ORDER ERROR] %s (%s): %s", ticker, alpaca_ticker, err_str)
                    return {"ok": False, "error": f"Order submission error: {err_str}"}

            # Rate-limit: 429
            elif err_str and ("429" in err_str or "rate limit" in err_str.lower()):
                logger.warning("[RATE LIMITED] Alpaca API — will retry in 2 s. %s", err_str)
                time.sleep(2)
                return {"ok": False, "error": f"Rate limited by Alpaca API: {err_str}"}

            # Insufficient buying power / crypto USD wallet empty
            elif err_str and ("insufficient" in err_str.lower() or "buying power" in err_str.lower()):
                logger.error("[BUYING POWER] %s → %s", ticker, err_str)
                hint = (
                    " (Alpaca crypto uses a separate USD wallet — verify it is funded "
                    "in the Alpaca dashboard.)"
                ) if alpaca_ticker.upper() in CRYPTO_SYMBOLS else ""
                return {"ok": False, "error": f"Insufficient buying power: {err_str}{hint}"}

            elif err_str:
                logger.error("[ORDER ERROR] %s (%s): %s", ticker, alpaca_ticker, err_str)
                return {"ok": False, "error": f"Order submission error: {err_str}"}

        order_id = str(order.id)

        # Persist active order mapping
        self._state.setdefault("active_orders", {})[alpaca_ticker] = order_id
        self._state.setdefault("order_journal", {})[order_id] = {
            "order_id": order_id,
            "ticker": ticker,
            "symbol": alpaca_ticker,
            "side": side.upper(),
            "entry": float(entry),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "confidence": float(confidence),
            "notional": float(notional),
            "submitted_ts": time.time(),
            "settled": False,
            "metadata": metadata or {},
        }
        self._save_state()

        logger.info(
            "[ORDER PLACED] %s %s $%.2f notional | %s→%s | Entry:%.4f SL:%.4f TP:%.4f | ID:%s",
            side.upper(), alpaca_ticker, notional,
            ticker, alpaca_ticker,
            entry, stop_loss, take_profit, order_id,
        )
        return {
            "ok":              True,
            "order_id":        order_id,
            "symbol":          alpaca_ticker,
            "original_ticker": ticker,
            "side":            side.upper(),
            "notional":        notional,
            "entry":           entry,
            "stop_loss":       stop_loss,
            "take_profit":     take_profit,
            "confidence":      confidence,
            "risk_fraction":   _confidence_risk_fraction(confidence),
        }

    # ── Place bot order (pre-computed sizing) ──────────────────────────────────

    def place_bot_order(
        self,
        ticker: str,
        side: str,
        notional: float,
        entry: float,
        stop_loss: float,
        take_profit: float,
        current_positions: list[dict],
        metadata: Optional[dict] = None,
        order_type: str = "MARKET",
    ) -> Dict[str, Any]:
        """Entry point for autonomous engine orders."""
        alpaca_ticker = self.map_ticker(ticker)

        # 1. Validation check
        pre_ok, pre_msg = validate_order(
            side=side,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=entry,
            min_rr=MIN_RR_HARD,
        )
        if not pre_ok:
            logger.warning("[BOT PRECHECK BLOCKED] %s/%s — %s", ticker, alpaca_ticker, pre_msg)
            return {"ok": False, "error": pre_msg, "error_code": "preflight_conflict"}

        if notional < 1.0:
            return {"ok": False, "error": "Notional too small (< $1.00)"}

        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        is_crypto = alpaca_ticker.upper() in CRYPTO_SYMBOLS
        is_limit = order_type.upper() == "LIMIT"
        
        # We strictly use GTC (Good Till Canceled) for all autonomous bot orders
        # This ensures the stop-loss and take-profit legs survive overnight and 
        # over the weekend to protect open swing/trend positions. 
        # (The Market Hours guard in engine.py prevents entry market orders from queuing overnight)
        tif = TimeInForce.GTC
        
        is_ext_hours = False
        if not is_crypto and is_limit:
            try:
                clock = self._client.get_clock()
                is_ext_hours = not clock.is_open
            except Exception:
                pass

        # ── Scale-out: split into two bracket orders (half size each) ──────────
        if config.SCALE_OUT_ENABLED and not is_crypto and not is_ext_hours:
            try:
                qty = max(1, int(notional / max(entry, 0.01)))
                half_qty = max(1, qty // 2)
                risk_per_share = abs(entry - stop_loss)
                tp1 = entry + (risk_per_share * config.SCALE_OUT_RR_RATIO) if order_side == OrderSide.BUY else entry - (risk_per_share * config.SCALE_OUT_RR_RATIO)
                
                # Build req1 (Half size, TP1)
                req_class = LimitOrderRequest if is_limit else MarketOrderRequest
                kwargs = {
                    "symbol": alpaca_ticker, "side": order_side, "time_in_force": tif,
                    "order_class": OrderClass.BRACKET,
                    "take_profit": TakeProfitRequest(limit_price=round(tp1, 2)),
                    "stop_loss": StopLossRequest(stop_price=round(stop_loss, 2)),
                    "qty": half_qty
                }
                if is_limit: kwargs["limit_price"] = round(entry, 2)
                req1 = req_class(**kwargs)
                
                # Build req2 (Remaining size, Original TP)
                kwargs2 = kwargs.copy()
                kwargs2["take_profit"] = TakeProfitRequest(limit_price=round(take_profit, 2))
                kwargs2["qty"] = qty - half_qty
                req2 = req_class(**kwargs2)
                
                order1 = self._client.submit_order(req1)
                order2 = self._client.submit_order(req2)
                
                # We return the first order_id for tracking, journal both optionally or just order1.
                return {"ok": True, "order_id": str(order1.id), "symbol": alpaca_ticker, "side": side.upper()}
            except Exception as exc:
                logger.error("Scale-out order failed: %s. Falling back to single order.", exc)
                # Fall through to original logic

        try:
            if is_limit:
                # Limit orders must use qty (LimitOrderRequest does not support notional)
                qty = max(1, int(notional / max(entry, 0.01)))
                if is_crypto:
                    # Crypto: simple limit order (no bracket support)
                    req = LimitOrderRequest(
                        symbol=alpaca_ticker,
                        side=order_side,
                        time_in_force=tif,
                        order_class=OrderClass.SIMPLE,
                        limit_price=round(entry, 2),
                        qty=qty,
                    )
                elif is_ext_hours:
                    # Extended Hours: bracket not supported, use simple limit with extended_hours=True
                    req = LimitOrderRequest(
                        symbol=alpaca_ticker,
                        side=order_side,
                        time_in_force=tif,
                        order_class=OrderClass.SIMPLE,
                        limit_price=round(entry, 2),
                        qty=qty,
                        extended_hours=True,
                    )
                else:
                    # Equity: bracket limit order with SL/TP
                    req = LimitOrderRequest(
                        symbol=alpaca_ticker,
                        side=order_side,
                        time_in_force=tif,
                        order_class=OrderClass.BRACKET,
                        limit_price=round(entry, 2),
                        take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
                        stop_loss=StopLossRequest(stop_price=round(stop_loss, 2)),
                        qty=qty,
                    )
            elif is_crypto:
                # Crypto: simple market order (no bracket support)
                req = MarketOrderRequest(
                    symbol=alpaca_ticker,
                    side=order_side,
                    time_in_force=tif,
                    order_class=OrderClass.SIMPLE,
                    notional=round(notional, 2),
                )
            else:
                # Equity: bracket order with SL/TP
                req = MarketOrderRequest(
                    symbol=alpaca_ticker,
                    side=order_side,
                    time_in_force=tif,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
                    stop_loss=StopLossRequest(stop_price=round(stop_loss, 2)),
                    notional=round(notional, 2),
                )

            order = self._client.submit_order(req)
            order_id = str(order.id)

            # Record in journal
            meta = metadata or {}
            self._state.setdefault("order_journal", {})[order_id] = {
                "order_id": order_id,
                "ticker": ticker,
                "symbol": alpaca_ticker,
                "side": side.upper(),
                "entry": float(entry),
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "confidence": float(meta.get("confidence", 0.0)),
                "submitted_ts": time.time(),
                "settled": False,
                "metadata": meta,
            }
            self._save_state()

            return {
                "ok": True,
                "order_id": order_id,
                "symbol": alpaca_ticker,
                "side": side.upper(),
            }

        except Exception as exc:
            err_str = str(exc)
            # Fractional/Complex bracket fallback
            if not is_crypto and ("fractional" in err_str.lower() or "complex" in err_str.lower()):
                try:
                    qty = max(1, int(notional / max(entry, 0.01)))
                    if is_limit:
                        fb_req = LimitOrderRequest(
                            symbol=alpaca_ticker,
                            side=order_side,
                            time_in_force=tif,
                            order_class=OrderClass.BRACKET,
                            limit_price=round(entry, 2),
                            take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
                            stop_loss=StopLossRequest(stop_price=round(stop_loss, 2)),
                            qty=qty,
                        )
                    else:
                        fb_req = MarketOrderRequest(
                            symbol=alpaca_ticker,
                            side=order_side,
                            time_in_force=tif,
                            order_class=OrderClass.BRACKET,
                            take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
                            stop_loss=StopLossRequest(stop_price=round(stop_loss, 2)),
                            qty=qty,
                        )
                    fb_order = self._client.submit_order(fb_req)
                    order_id = str(fb_order.id)
                    
                    meta = metadata or {}
                    self._state.setdefault("order_journal", {})[order_id] = {
                        "order_id": order_id,
                        "ticker": ticker,
                        "symbol": alpaca_ticker,
                        "side": side.upper(),
                        "entry": float(entry),
                        "stop_loss": float(stop_loss),
                        "take_profit": float(take_profit),
                        "confidence": float(meta.get("confidence", 0.0)),
                        "submitted_ts": time.time(),
                        "settled": False,
                        "metadata": meta,
                    }
                    self._save_state()
                    return {"ok": True, "order_id": order_id, "symbol": alpaca_ticker, "side": side.upper()}
                except Exception as fb_exc:
                    logger.error("[BOT ORDER FAILED] %s fallback: %s", ticker, fb_exc)
                    return {"ok": False, "error": str(fb_exc)}

            logger.error("[BOT ORDER FAILED] %s: %s", ticker, err_str)
            return {"ok": False, "error": err_str}


    # ── Scale-out / Breakeven management ──────────────────────────────────────────

    def move_stop_to_breakeven(self, symbol: str, entry_price: float) -> None:
        """Move the stop-loss leg of an open bracket order to breakeven entry price.

        Called by the engine after a scale-out TP1 fill reduces position size,
        so the remaining position is risk-free.
        """
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        open_orders = self._client.get_orders(req)
        for o in open_orders:
            if str(o.order_type).lower() == "stop":
                try:
                    rep = ReplaceOrderRequest(stop_price=round(entry_price, 2))
                    self._client.replace_order_by_id(o.id, rep)
                except Exception as e:
                    logger.error("Failed to move stop for %s: %s", symbol, e)

    def update_position_exits(self, symbol: str, take_profit: float, stop_loss: float) -> None:
        """Update the TP and SL legs of an active position."""
        from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        open_orders = self._client.get_orders(req)
        for o in open_orders:
            try:
                o_type = o.order_type.value if hasattr(o.order_type, 'value') else str(o.order_type).lower()
                if "stop" in o_type:
                    rep = ReplaceOrderRequest(stop_price=round(stop_loss, 2))
                    self._client.replace_order_by_id(str(o.id), rep)
                    logger.info("Updated Stop Loss for %s to %.2f", symbol, stop_loss)
                elif "limit" in o_type:
                    rep = ReplaceOrderRequest(limit_price=round(take_profit, 2))
                    self._client.replace_order_by_id(str(o.id), rep)
                    logger.info("Updated Take Profit for %s to %.2f", symbol, take_profit)
            except Exception as e:
                logger.error("Failed to update exit leg for %s: %s", symbol, e)

    def _safe_status(self, obj: Any, fallback: str = "") -> str:
        """Return normalized lowercase status string from Alpaca object fields."""
        if obj is None:
            return fallback
        raw = getattr(obj, "value", obj)
        return str(raw).strip().lower()

    def _finalize_reason_from_order(self, order: Any, parent_side: str) -> tuple[bool, str, Optional[float]]:
        """Infer whether trade worked and why from a terminal bracket/simple order state."""
        status = self._safe_status(getattr(order, "status", ""))

        if status in {"rejected", "canceled", "cancelled", "expired"}:
            return False, f"order_{status}", None

        legs = getattr(order, "legs", None) or []
        for leg in legs:
            leg_status = self._safe_status(getattr(leg, "status", ""))
            if leg_status != "filled":
                continue

            leg_type = self._safe_status(getattr(leg, "order_type", ""))
            exit_px = None
            try:
                if getattr(leg, "filled_avg_price", None) is not None:
                    exit_px = float(leg.filled_avg_price)
            except (TypeError, ValueError):
                exit_px = None

            if "limit" in leg_type:
                return True, "take_profit_hit", exit_px
            if "stop" in leg_type:
                return False, "stop_loss_hit", exit_px

        if status == "filled":
            # For simple market orders (e.g. crypto) there are no bracket legs.
            # Extract fill price directly from the order object.
            fill_px: Optional[float] = None
            try:
                if getattr(order, "filled_avg_price", None) is not None:
                    fill_px = float(order.filled_avg_price)
            except (TypeError, ValueError):
                fill_px = None
            return True, "filled_simple", fill_px

        return False, f"terminal_{status or 'unknown'}", None

    def sync_closed_trades(self) -> Dict[str, Any]:
        """Poll Alpaca for terminal orders, derive outcomes, and queue ML feedback."""
        journals = self._state.get("order_journal", {})
        if not journals:
            return {"ok": True, "processed": 0, "queued": 0}

        processed = 0
        queued = 0

        for order_id, rec in list(journals.items()):
            if rec.get("settled"):
                continue

            try:
                order = self._client.get_order_by_id(order_id)
            except Exception as exc:
                logger.warning("sync_closed_trades: get_order_by_id(%s) failed: %s", order_id, exc)
                continue

            status = self._safe_status(getattr(order, "status", ""))
            terminal_states = {
                "filled", "canceled", "cancelled", "rejected", "expired", "done_for_day"
            }
            if status not in terminal_states:
                continue

            worked, reason, exit_price = self._finalize_reason_from_order(order, rec.get("side", "BUY"))
            entry_price = float(rec.get("entry", 0.0) or 0.0)
            side = str(rec.get("side", "BUY")).upper()

            # ── NEW: Bracket entry-fill guard ────────────────────────────
            # Bracket orders enter "filled" when the entry fills, but
            # the trade isn't complete until a TP/SL leg fills.
            # Don't settle prematurely — wait for the leg to fill.
            order_class = getattr(order, "order_class", None)
            is_bracket = order_class is not None and "bracket" in str(order_class).lower()
            if is_bracket and reason == "filled_simple":
                continue

            # Skip DB logging if _close_and_log_position already handled it
            skip_db_log = rec.get("db_logged", False)

            # ── NEW: Log bracket completions to trade_log DB ─────────────
            if not skip_db_log and is_bracket and reason in ("take_profit_hit", "stop_loss_hit"):
                _exit_price = exit_price or 0.0
                _qty = max(float(getattr(order, "filled_qty", 0) or 0), 0)
                _pnl = 0.0
                notional_val = float(rec.get("notional", 0.0) or 0.0)
                if _exit_price > 0 and entry_price > 0:
                    signed = (_exit_price - entry_price) / entry_price
                    if side == "SELL":
                        signed *= -1.0
                    _pnl = notional_val * signed

                stop_loss = float(rec.get("stop_loss", 0.0) or 0.0)
                take_profit = float(rec.get("take_profit", 0.0) or 0.0)

                if reason == "take_profit_hit":
                    _mfe = abs(take_profit - entry_price)
                    _mae = 0.0
                else:
                    _mfe = 0.0
                    _mae = abs(entry_price - stop_loss)

                sub_ts = float(rec.get("submitted_ts", time.time()))
                _hold = max((time.time() - sub_ts) / 3600.0, 0.0)
                meta = rec.get("metadata") or {}

                try:
                    log_trade(
                        symbol=rec.get("symbol", ""),
                        strategy=meta.get("strategy", "unknown"),
                        direction=side,
                        entry_price=entry_price,
                        exit_price=_exit_price,
                        stop_loss=stop_loss,
                        qty=_qty if _qty > 0 else max(1, int(notional_val / max(entry_price, 0.01))),
                        pnl=_pnl,
                        mfe=_mfe,
                        mae=_mae,
                        hold_hours=_hold,
                        exit_reason=reason,
                        regime=meta.get("regime", "unknown"),
                    )
                except Exception as e:
                    logger.error("Failed to log bracket fill to DB for %s: %s", rec.get("symbol"), e)

            pnl_pct = 0.0
            if exit_price and entry_price > 0:
                signed = (exit_price - entry_price) / entry_price
                if side == "SELL":
                    signed *= -1.0
                pnl_pct = float(signed * 100.0)

            feedback = {
                "order_id": order_id,
                "ticker": rec.get("ticker"),
                "symbol": rec.get("symbol"),
                "side": side,
                "worked": bool(worked),
                "result_reason": reason,
                "entry": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "confidence": float(rec.get("confidence", 0.0) or 0.0),
                "submitted_ts": float(rec.get("submitted_ts", time.time())),
                "closed_ts": time.time(),
                "feature_row": (rec.get("metadata") or {}).get("feature_row"),
                "signal_reason": (rec.get("metadata") or {}).get("signal_reason"),
            }

            self._state.setdefault("ml_feedback_queue", []).append(feedback)
            rec["settled"] = True
            rec["result_reason"] = reason
            rec["worked"] = bool(worked)
            rec["closed_ts"] = time.time()

            # Record dollar PnL
            if pnl_pct != 0.0:
                notional = float(rec.get("notional", 0.0))
                if notional > 0:
                    dollar_pnl = notional * (pnl_pct / 100.0)
                    self.record_realised_pnl(dollar_pnl)

            sym = str(rec.get("symbol", ""))
            if sym and self._state.get("active_orders", {}).get(sym) == order_id:
                del self._state["active_orders"][sym]

            processed += 1
            queued += 1

        # ── Prune settled journals (memory-leak guard) ─────────────────────────
        # Keep only the 200 most recently closed settled records to prevent
        # unbounded growth of the order_journal dict.
        settled: list[tuple[str, float]] = []
        for key, rec in journals.items():
            if rec.get("settled") and rec.get("closed_ts"):
                settled.append((key, float(rec["closed_ts"])))
        if len(settled) > 200:
            settled.sort(key=lambda x: x[1], reverse=True)
            keep = {k for k, _ in settled[:200]}
            for key, _ in settled[200:]:
                if key not in keep:
                    del journals[key]
            logger.info(
                "Pruned %d settled journal entries (kept 200 most recent).",
                len(settled) - 200,
            )

        self._save_state()
        return {"ok": True, "processed": processed, "queued": queued}

    def drain_ml_feedback_queue(self) -> list[Dict[str, Any]]:
        """Return and clear queued trade-outcome feedback records."""
        queue = self._state.get("ml_feedback_queue", [])
        if not queue:
            return []
        self._state["ml_feedback_queue"] = []
        self._save_state()
        return list(queue)

    def get_recent_outcomes(self, limit: int = 50) -> Dict[str, Any]:
        """Return recent settled paper-trade outcomes with reason labels."""
        journal = self._state.get("order_journal", {})
        rows: list[Dict[str, Any]] = []
        for _, rec in journal.items():
            if not rec.get("settled"):
                continue
            rows.append({
                "order_id": rec.get("order_id"),
                "ticker": rec.get("ticker"),
                "symbol": rec.get("symbol"),
                "side": rec.get("side"),
                "worked": bool(rec.get("worked", False)),
                "reason": rec.get("result_reason", "unknown"),
                "entry": rec.get("entry"),
                "stop_loss": rec.get("stop_loss"),
                "take_profit": rec.get("take_profit"),
                "confidence": rec.get("confidence"),
                "submitted_ts": rec.get("submitted_ts"),
                "closed_ts": rec.get("closed_ts"),
                "signal_reason": (rec.get("metadata") or {}).get("signal_reason"),
            })

        rows.sort(key=lambda x: float(x.get("closed_ts") or 0.0), reverse=True)
        return {"ok": True, "count": len(rows), "outcomes": rows[:max(1, int(limit))]}

    # ── Profit callback (call after a position closes to grow cap) ─────────────

    def record_realised_pnl(self, pnl: float) -> None:
        """Update the running P&L and grow the cash limit on profitable trades."""
        self._state["total_pnl"] = round(
            self._state.get("total_pnl", 0.0) + pnl, 2
        )
        if pnl > 0:
            self._state["cash_limit"] = round(
                self._state.get("cash_limit", INITIAL_CASH_LIMIT) + pnl, 2
            )
        self._save_state()
        logger.info(
            "Realised P&L: %+.2f | Total P&L: %.2f | Cash limit: %.2f",
            pnl,
            self._state["total_pnl"],
            self._state["cash_limit"],
        )
