"""MakeShiftTrades — Flask + HTML chart server

Run with:
    python server.py

Then open  http://localhost:5000  in your browser.
"""
import logging
import threading
import traceback

import pandas as pd
import plotly.io as pio
from flask import Flask, Response, jsonify, render_template, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

from charts.data import fetch_ohlcv
from charts.renderer import build_chart
from charts.indicators.fvg import detect_fvg
from charts.indicators.engulfing import detect_engulfing
from charts.indicators.liquidity import detect_liquidity_levels
from charts.indicators.price_action import (
    detect_swing_points,
    detect_market_structure,
    detect_order_blocks,
)
from charts.indicators.levels import (
    detect_key_levels,
    detect_vwap,
    detect_sessions,
    detect_equilibrium,
)

app = Flask(__name__)

TICKERS = ["NQ=F", "ES=F", "YM=F", "RTY=F", "SPY", "QQQ", "AAPL", "TSLA", "GC=F", "CL=F"]

# ── Background ML training (fires once at server start) ────────────────────────
from src.ml_model import get_model as _get_ml_model


def _bg_train() -> None:
    logging.getLogger(__name__).info("Background ML training started…")
    try:
        _get_ml_model().fit()
    except Exception as _exc:
        logging.getLogger(__name__).error("ML training failed: %s", _exc)


threading.Thread(target=_bg_train, daemon=True).start()


def _to_ts(ts) -> int:
    """Convert a pandas Timestamp to UTC Unix seconds."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        return int(t.timestamp())
    return int(t.tz_localize("UTC").timestamp())


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", tickers=TICKERS)


# ── Candles API ───────────────────────────────────────────────────────────────

@app.route("/api/candles")
def api_candles():
    ticker   = request.args.get("ticker",   "SPY").strip().upper()
    interval = request.args.get("interval", "5m")
    period   = request.args.get("period",   "5d")

    try:
        df = fetch_ohlcv(ticker, period=period, interval=interval)
        candles = [
            {
                "time":   _to_ts(ts),
                "open":   round(float(row["Open"]),   4),
                "high":   round(float(row["High"]),   4),
                "low":    round(float(row["Low"]),    4),
                "close":  round(float(row["Close"]),  4),
                "volume": float(row["Volume"]),
            }
            for ts, row in df.iterrows()
        ]
        return jsonify({"ok": True, "candles": candles})

    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Server error: {exc}"}), 500


# ── Indicators API ────────────────────────────────────────────────────────────

@app.route("/api/indicators")
def api_indicators():
    ticker   = request.args.get("ticker",     "SPY").strip().upper()
    interval = request.args.get("interval",   "5m")
    period   = request.args.get("period",     "5d")
    raw_ind  = request.args.get("indicators", "fvg,engulfing,liquidity,ob,ms,swings")
    active   = {i.strip() for i in raw_ind.split(",") if i.strip()}

    try:
        df      = fetch_ohlcv(ticker, period=period, interval=interval)
        last_ts = _to_ts(df.index[-1])
        result: dict = {}

        if "fvg" in active:
            fvg_df = detect_fvg(df)
            result["fvg"] = [
                {
                    "type":       row["type"],
                    "ifvg":       bool(row["ifvg"]),
                    "top":        float(row["top"]),
                    "bottom":     float(row["bottom"]),
                    "start_time": _to_ts(row["date"]),
                    "end_time":   last_ts if row["active"] else _to_ts(row["end_date"]),
                }
                for _, row in fvg_df.tail(60).iterrows()
            ]

        if "engulfing" in active:
            eng_df = detect_engulfing(df)
            result["engulfing"] = [
                {
                    "time":  _to_ts(row["date"]),
                    "type":  row["type"],
                    "price": float(row["price"]),
                }
                for _, row in eng_df.iterrows()
            ]

        if "liquidity" in active:
            levels = detect_liquidity_levels(df)
            seen: set = set()
            liq: list = []
            for lv in levels:
                if lv["strength"] < 3:
                    continue
                key = round(lv["price"], 1)
                if key in seen:
                    continue
                seen.add(key)
                liq.append({
                    "price":      float(lv["price"]),
                    "dir":        lv["dir"],
                    "strength":   lv["strength"],
                    "start_time": _to_ts(lv["date"]),
                })
            result["liquidity"] = liq

        if "ob" in active:
            obs = detect_order_blocks(df)
            result["ob"] = [
                {
                    "type":       ob["type"],
                    "top":        float(ob["top"]),
                    "bottom":     float(ob["bottom"]),
                    "start_time": _to_ts(ob["date"]),
                    "end_time":   last_ts,
                }
                for ob in obs
            ]

        if "ms" in active:
            ms_events = detect_market_structure(df)
            result["ms"] = [
                {
                    "time":  _to_ts(ev["date"]),
                    "label": ev["label"],
                    "price": float(ev["price"]),
                    "color": ev["color"],
                }
                for ev in ms_events
            ]

        if "swings" in active:
            sw_df = detect_swing_points(df)
            result["swings"] = [
                {
                    "time":  _to_ts(row["date"]),
                    "type":  row["type"],
                    "price": float(row["price"]),
                }
                for _, row in sw_df.iterrows()
            ]

        if "key_levels" in active:
            result["key_levels"] = detect_key_levels(df)

        if "vwap" in active:
            result["vwap"] = detect_vwap(df)

        if "sessions" in active:
            result["sessions"] = detect_sessions(df)

        if "equilibrium" in active:
            result["equilibrium"] = detect_equilibrium(df)

        return jsonify({"ok": True, **result})

    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Server error: {exc}"}), 500


# ── Legacy chart endpoint (kept for compatibility) ────────────────────────────

@app.route("/api/chart")
def api_chart():
    ticker    = request.args.get("ticker",     "SPY").strip().upper()
    interval  = request.args.get("interval",   "5m")
    period    = request.args.get("period",     "1mo")
    raw_ind   = request.args.get("indicators", "fvg,engulfing,liquidity,ob,ms,swings")
    indicators = [i.strip() for i in raw_ind.split(",") if i.strip()]

    try:
        df  = fetch_ohlcv(ticker, period=period, interval=interval)
        fig = build_chart(df, ticker, indicators)
        fig_json = pio.to_json(fig)
        payload  = '{"ok":true,"figure":' + fig_json + "}"
        return Response(payload, mimetype="application/json")

    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Server error: {exc}"}), 500


# ── Signal API ─────────────────────────────────────────────────────────────────

@app.route("/api/signal")
def api_signal():
    """
    Returns the combined SMC + ML trade signal for the requested ticker.

    Query params: ticker, interval, period
    Response JSON keys:
      ok, ticker, signal, confidence, alignment,
      ml  : { signal, confidence, probabilities, trained },
      smc : { signal, entry, stop_loss, take_profit, risk_reward,
              confidence, reason, smc_score }
    """
    ticker   = request.args.get("ticker",   "SPY").strip().upper()
    interval = request.args.get("interval", "5m")
    period   = request.args.get("period",   "5d")

    try:
        df = fetch_ohlcv(ticker, period=period, interval=interval)

        # ── SMC signal ──────────────────────────────────────────────
        from src.strategy import SMCStrategy
        strategy     = SMCStrategy(ticker, interval=interval, period=period)
        strict_sig   = strategy.analyze(df)
        smc_signal   = strict_sig or strategy.find_setup(df)
        price_at_zone = strict_sig is not None

        smc_dir = smc_signal.direction if smc_signal else None

        # ── Enhanced SMC scoring (0-6) ──────────────────────────────
        smc_score = 0
        if smc_signal:
            smc_score = 2
            if smc_signal.raw_data.get("has_ob"):
                smc_score += 1
            if smc_signal.raw_data.get("has_engulf"):
                smc_score += 1
            if price_at_zone:
                smc_score += 1
            try:
                eq = detect_equilibrium(df)
                if eq:
                    cur = float(df["Close"].iloc[-1])
                    if smc_dir == "BUY"  and cur < eq["eq"]:
                        smc_score += 1
                    elif smc_dir == "SELL" and cur > eq["eq"]:
                        smc_score += 1
            except Exception:
                pass

        smc_block = {
            "signal":        smc_dir,
            "entry":         smc_signal.entry       if smc_signal else None,
            "stop_loss":     smc_signal.stop_loss   if smc_signal else None,
            "take_profit":   smc_signal.take_profit if smc_signal else None,
            "risk_reward":   smc_signal.risk_reward if smc_signal else None,
            "confidence":    smc_signal.confidence  if smc_signal else None,
            "reason":        smc_signal.reason      if smc_signal else None,
            "smc_score":     smc_score,
            "price_at_zone": price_at_zone,
        }

        # ── ML signal ───────────────────────────────────────────────
        ml_result = _get_ml_model().predict(df)
        ml_dir    = ml_result["signal"]
        ml_conf   = ml_result["confidence"]

        # ── Combine SMC + ML ─────────────────────────────────────────
        score_boost = smc_score / 6

        if price_at_zone and smc_dir and smc_dir == ml_dir:
            final_signal = smc_dir
            final_conf   = min(100.0, ml_conf + 25.0 * score_boost)
            alignment    = "aligned"
        elif price_at_zone and smc_dir:
            final_signal = smc_dir
            final_conf   = min(100.0, ml_conf + 12.0 * score_boost)
            alignment    = "smc_only"
        elif smc_dir and smc_dir == ml_dir:
            final_signal = smc_dir
            final_conf   = ml_conf * 0.90
            alignment    = "aligned"
        elif smc_dir and ml_dir != "HOLD" and smc_dir != ml_dir:
            final_signal = ml_dir
            final_conf   = ml_conf * 0.70
            alignment    = "disagreement"
        elif smc_dir:
            final_signal = smc_dir
            final_conf   = min(100.0, ml_conf * 0.80 + 5.0 * score_boost)
            alignment    = "smc_only"
        else:
            final_signal = ml_dir
            final_conf   = ml_conf
            alignment    = "ml_only"

        return jsonify({
            "ok":         True,
            "ticker":     ticker,
            "signal":     final_signal,
            "confidence": round(final_conf, 1),
            "alignment":  alignment,
            "ml":         ml_result,
            "smc":        smc_block,
        })

    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Server error: {exc}"}), 500


# ── Multi-Timeframe Consensus API ─────────────────────────────────────────────

@app.route("/api/mtf_signal")
def api_mtf_signal():
    """
    Multi-timeframe consensus signal.

    Query params: ticker, interval
    Response JSON keys:
      ok, ticker, consensus, consensus_score, long_pct, short_pct,
      entry, stop_loss, take_profit, risk_reward, entry_tf, target_tf,
      timeframes : { "1m": {...}, "3m": {...}, "5m": {...}, "15m": {...} }
    """
    ticker   = request.args.get("ticker",   "SPY").strip().upper()
    interval = request.args.get("interval", "1m")

    try:
        from src.mtf import MultiTimeframeAnalysis
        import config as _cfg
        mtf    = MultiTimeframeAnalysis(ticker, active_interval=interval, ms_term=_cfg.MTF_MS_TERM, min_rr=_cfg.MTF_MIN_RR)
        result = mtf.analyze()

        return jsonify({
            "ok":              True,
            "ticker":          ticker,
            "consensus":       result.consensus,
            "consensus_score": result.consensus_score,
            "long_pct":        result.long_pct,
            "short_pct":       result.short_pct,
            "entry":           result.entry,
            "stop_loss":       result.stop_loss,
            "take_profit":     result.take_profit,
            "risk_reward":     result.risk_reward,
            "entry_tf":        result.entry_tf,
            "target_tf":       result.target_tf,
            "timeframes":      result.timeframes,
        })

    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Server error: {exc}"}), 500


# ── Alpaca Paper Trading API ──────────────────────────────────────────────────
#
# Lazy-initialised singleton so the server starts even if ALPACA_* env vars
# are absent (the routes return a clear error in that case).
# ─────────────────────────────────────────────────────────────────────────────

_alpaca_trader = None
_alpaca_init_error: str = ""


def _get_alpaca():
    """Return the AlpacaTrader singleton, initialising on first call."""
    global _alpaca_trader, _alpaca_init_error
    if _alpaca_trader is not None:
        return _alpaca_trader
    try:
        from src.alpaca_trader import AlpacaTrader
        _alpaca_trader = AlpacaTrader()
        _alpaca_init_error = ""
    except Exception as exc:
        _alpaca_init_error = str(exc)
        logging.getLogger(__name__).error("AlpacaTrader init failed: %s", exc)
        _alpaca_trader = None
    return _alpaca_trader


@app.route("/api/paper/account")
def api_paper_account():
    """Return Alpaca paper account info + our enforced cash limit."""
    trader = _get_alpaca()
    if trader is None:
        return jsonify({"ok": False, "error": _alpaca_init_error or "AlpacaTrader unavailable"}), 503
    return jsonify(trader.get_account())


@app.route("/api/paper/orders")
def api_paper_orders():
    """Return all currently open / pending bracket orders."""
    trader = _get_alpaca()
    if trader is None:
        return jsonify({"ok": False, "error": _alpaca_init_error or "AlpacaTrader unavailable"}), 503
    return jsonify(trader.get_active_orders())


@app.route("/api/paper/cancel/<order_id>", methods=["DELETE"])
def api_paper_cancel(order_id: str):
    """Cancel an open order by its Alpaca UUID."""
    trader = _get_alpaca()
    if trader is None:
        return jsonify({"ok": False, "error": _alpaca_init_error or "AlpacaTrader unavailable"}), 503
    return jsonify(trader.cancel_order(order_id))


@app.route("/api/paper/toggle", methods=["POST"])
def api_paper_toggle():
    """Enable or disable paper trading for a dashboard ticker.

    Body JSON: { "ticker": "NQ=F", "enabled": true }
    Response:  { "ok": true, "ticker": "NQ=F", "enabled": true }
    """
    trader = _get_alpaca()
    if trader is None:
        return jsonify({"ok": False, "error": _alpaca_init_error or "AlpacaTrader unavailable"}), 503
    data = request.get_json(silent=True) or {}
    ticker  = str(data.get("ticker", "")).strip().upper()
    enabled = bool(data.get("enabled", False))
    if not ticker:
        return jsonify({"ok": False, "error": "Missing 'ticker' field."}), 400
    trader.set_enabled(ticker, enabled)
    return jsonify({"ok": True, "ticker": ticker, "enabled": enabled})


@app.route("/api/paper/status")
def api_paper_status():
    """Return the enabled/disabled state for every ticker plus connection health.

    Query param:  ticker  (optional, single ticker check)
    Response: {
      "ok": true,
      "connected": true,
      "enabled": { "NQ=F": true, "SPY": false, ... },
      "ticker_enabled": true    # present only if ?ticker= was provided
    }
    """
    trader = _get_alpaca()
    if trader is None:
        return jsonify({
            "ok": False,
            "connected": False,
            "error": _alpaca_init_error or "AlpacaTrader unavailable",
            "enabled": {},
        })

    # Test connectivity with a lightweight account call
    acct = trader.get_account()
    connected = acct.get("ok", False)

    result: dict = {
        "ok":        True,
        "connected": connected,
        "enabled":   trader.get_all_enabled(),
    }

    ticker = request.args.get("ticker", "").strip().upper()
    if ticker:
        result["ticker_enabled"] = trader.is_enabled(ticker)

    return jsonify(result)


@app.route("/api/paper/execute", methods=["POST"])
def api_paper_execute():
    """Execute a paper bracket order for a given ticker.

    Body JSON (all required):
    {
      "ticker":      "NQ=F",
      "side":        "BUY",          // "BUY" or "SELL"
      "entry":       19500.25,
      "stop_loss":   19480.00,
      "take_profit": 19550.00,
      "confidence":  78.5            // 0–100 from the signal API
    }

    The endpoint:
    1. Checks that paper trading is enabled for the ticker.
    2. Fetches current price (latest close from yfinance).
    3. Runs the double-check validator.
    4. Sizes the position by confidence × available capital (≤ $5 k hard limit).
    5. Submits the bracket order to Alpaca Paper API.

    All failures are logged and returned to the caller with "ok": false.
    """
    trader = _get_alpaca()
    if trader is None:
        return jsonify({"ok": False, "error": _alpaca_init_error or "AlpacaTrader unavailable"}), 503

    data = request.get_json(silent=True) or {}

    ticker     = str(data.get("ticker",     "")).strip().upper()
    side       = str(data.get("side",       "")).strip().upper()
    confidence = float(data.get("confidence", 0))

    # Parse numeric order levels
    try:
        entry       = float(data["entry"])
        stop_loss   = float(data["stop_loss"])
        take_profit = float(data["take_profit"])
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": f"Missing/invalid numeric field: {exc}"}), 400

    if not ticker or not side:
        return jsonify({"ok": False, "error": "Fields 'ticker' and 'side' are required."}), 400

    # Guard: paper trading must be explicitly enabled for this ticker
    if not trader.is_enabled(ticker):
        return jsonify({
            "ok":    False,
            "error": f"Paper trading is not enabled for {ticker}. "
                     "Toggle it ON in the dashboard first.",
        }), 409

    # Guard: minimum confidence threshold (60 %)
    import config as _cfg
    min_conf = getattr(_cfg, "PAPER_MIN_CONFIDENCE", 60.0)
    if confidence < min_conf:
        return jsonify({
            "ok":    False,
            "error": f"Signal confidence {confidence:.1f}% is below the minimum "
                     f"{min_conf:.1f}% required for execution.",
        }), 422

    # Fetch latest price for stale-entry guard inside validate_order
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="1d", interval="1m")
        current_price = float(hist["Close"].iloc[-1]) if not hist.empty else entry
    except Exception as price_exc:
        logging.getLogger(__name__).warning(
            "Could not fetch current price for %s (using entry as fallback): %s",
            ticker, price_exc,
        )
        current_price = entry  # stale-entry guard will still run using entry ≈ entry

    result = trader.place_bracket_order(
        ticker=ticker,
        side=side,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        current_price=current_price,
    )

    http_code = 200 if result.get("ok") else 422
    return jsonify(result), http_code


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="127.0.0.1")
