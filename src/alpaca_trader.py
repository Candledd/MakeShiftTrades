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
from typing import Any, Dict, Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

logger = logging.getLogger(__name__)

# ── Tick → Alpaca equity proxy ───────────────────────────────────────────────
# Alpaca paper trading supports equities and crypto only; futures are
# represented by their closest ETF proxy.
TICKER_MAP: Dict[str, str] = {
    "NQ=F":  "QQQ",   # NASDAQ-100 futures  → Invesco QQQ ETF
    "ES=F":  "SPY",   # S&P 500 futures     → SPDR S&P 500 ETF
    "YM=F":  "DIA",   # Dow Jones futures   → SPDR DJIA ETF
    "RTY=F": "IWM",   # Russell 2000 futures→ iShares Russell 2000 ETF
    "GC=F":  "GLD",   # Gold futures        → SPDR Gold Shares ETF
    "CL=F":  "USO",   # Crude oil futures   → United States Oil ETF
}

# Starting hard cash limit (grows with realised profits).
INITIAL_CASH_LIMIT: float = 5_000.0

# Minimum R/R the double-check validator enforces.
MIN_RR_HARD: float = 1.5

# Maximum entry-vs-current-price deviation (%) before rejecting the order.
MAX_ENTRY_DEVIATION_PCT: float = 2.0


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
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read trades_state.json: %s", exc)

        return {
            "cash_limit":    INITIAL_CASH_LIMIT,
            "active_orders": {},   # {alpaca_symbol: order_id}
            "paper_enabled": {},   # {original_ticker: bool}
            "total_pnl":     0.0,
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
                    "qty":              str(o.qty),
                    "status":           str(o.status.value) if hasattr(o.status, "value") else str(o.status),
                    "submitted_at":     str(o.submitted_at),
                    "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                    "order_class":      str(o.order_class.value) if hasattr(o.order_class, "value") else str(o.order_class),
                })
            return {"ok": True, "orders": result}
        except Exception as exc:
            logger.error("get_active_orders failed: %s", exc)
            return {"ok": False, "error": str(exc)}

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

    def calc_qty(
        self,
        entry: float,
        stop_loss: float,
        confidence: float,
        available_capital: float,
    ) -> int:
        """Calculate share quantity from confidence and capital constraints.

        Algorithm
        ---------
        risk_amount   = available_capital × risk_fraction(confidence)
        qty_by_risk   = floor(risk_amount / |entry − stop_loss|)
        qty_by_capital= floor(available_capital / entry)
        qty           = max(1, min(qty_by_risk, qty_by_capital))
        """
        risk_per_share = abs(entry - stop_loss)
        if risk_per_share == 0 or entry <= 0:
            return 0

        risk_fraction  = _confidence_risk_fraction(confidence)
        risk_amount    = available_capital * risk_fraction
        qty_by_risk    = int(risk_amount / risk_per_share)
        qty_by_capital = int(available_capital / entry)

        qty = min(qty_by_risk, qty_by_capital)
        return max(1, qty)

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
    ) -> Dict[str, Any]:
        """Validate and submit a bracket order (entry + limit TP + stop SL).

        The double-check validator runs first.  If it passes, account capital
        is checked against the $5,000 hard limit.  Position size is then
        computed from confidence before the order is dispatched.

        All failures and successes are logged to the application logger.

        Returns a dict with key "ok" plus context on success or error details.
        """
        alpaca_ticker = self.map_ticker(ticker)

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
            # Effective buying power is the lesser of Alpaca's buying_power
            # and our self-imposed hard limit.
            available_bp = min(float(acct.buying_power), cash_limit)
        except Exception as exc:
            logger.error("Account fetch failed before order placement: %s", exc)
            return {"ok": False, "error": f"Account fetch failed: {exc}"}

        if available_bp < entry:
            msg = (
                f"Insufficient capital: available ${available_bp:.2f} "
                f"(hard limit ${cash_limit:.2f}), entry ${entry:.2f}."
            )
            logger.error("[REJECTED] %s", msg)
            return {"ok": False, "error": msg}

        # ── Step 3: Position sizing ────────────────────────────────────────────
        qty = self.calc_qty(entry, stop_loss, confidence, available_bp)
        if qty <= 0:
            return {
                "ok": False,
                "error": "Position size resolved to 0 shares — stop distance too small.",
            }

        # ── Step 4: Submit bracket order ──────────────────────────────────────
        try:
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            order_req  = MarketOrderRequest(
                symbol=alpaca_ticker,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
                stop_loss=StopLossRequest(stop_price=round(stop_loss, 2)),
            )
            order    = self._client.submit_order(order_req)
            order_id = str(order.id)

            # Persist active order mapping
            self._state.setdefault("active_orders", {})[alpaca_ticker] = order_id
            self._save_state()

            logger.info(
                "[ORDER PLACED] %s %s x%d | %s→%s | Entry:%.4f SL:%.4f TP:%.4f | ID:%s",
                side.upper(), alpaca_ticker, qty,
                ticker, alpaca_ticker,
                entry, stop_loss, take_profit, order_id,
            )
            return {
                "ok":              True,
                "order_id":        order_id,
                "symbol":          alpaca_ticker,
                "original_ticker": ticker,
                "side":            side.upper(),
                "qty":             qty,
                "entry":           entry,
                "stop_loss":       stop_loss,
                "take_profit":     take_profit,
                "confidence":      confidence,
                "risk_fraction":   _confidence_risk_fraction(confidence),
            }

        except Exception as exc:
            err_str = str(exc)

            # Rate-limit: 429
            if "429" in err_str or "rate limit" in err_str.lower():
                logger.warning("[RATE LIMITED] Alpaca API — will retry in 2 s. %s", err_str)
                time.sleep(2)
                return {"ok": False, "error": f"Rate limited by Alpaca API: {err_str}"}

            # Insufficient buying power
            if "insufficient" in err_str.lower() or "buying power" in err_str.lower():
                logger.error("[BUYING POWER] %s → %s", ticker, err_str)
                return {"ok": False, "error": f"Insufficient buying power: {err_str}"}

            logger.error("[ORDER ERROR] %s (%s): %s", ticker, alpaca_ticker, err_str)
            return {"ok": False, "error": f"Order submission error: {err_str}"}

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
