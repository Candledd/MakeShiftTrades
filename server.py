"""MakeShiftTrades — Flask + HTML chart server

Run with:
    python server.py

Then open  http://localhost:5000  in your browser.
"""
import logging
import os
import threading
import time
import traceback
import collections
import math

import config

import pandas as pd
import plotly.io as pio
from flask import Flask, Response, jsonify, render_template, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_overnight_telemetry.log", encoding="utf-8")
    ]
)

class DequeLogHandler(logging.Handler):
    def __init__(self, maxlen=200):
        super().__init__()
        self.logs = collections.deque(maxlen=maxlen)
        self.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            "%Y-%m-%d %H:%M:%S"
        ))

    def emit(self, record):
        try:
            msg = self.format(record)
            self.logs.append(msg)
        except Exception:
            self.handleError(record)

deque_handler = DequeLogHandler()
logging.getLogger().addHandler(deque_handler)

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

# Suppress routine Flask HTTP GET logs to keep the console clean
import logging as _logging
_werkzeug_log = _logging.getLogger('werkzeug')
_werkzeug_log.setLevel(_logging.ERROR)

# ── API Key authentication ────────────────────────────────────────────────────
# All /api/* routes require a valid API key passed via the X-API-Key header.
# Set API_KEY in your .env file or environment. If unset, a mock key is used
# for local development; change this before deploying to production.
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable is required")

@app.before_request
def _require_api_key():
    if not request.path.startswith("/api/"):
        return None
    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        return jsonify({"ok": False, "error": "Unauthorized — missing or invalid API key."}), 401

TICKERS = ["NQ=F", "ES=F", "YM=F", "RTY=F", "SPY", "QQQ", "AAPL", "TSLA", "GC=F", "CL=F", "BTC-USD", "ETH-USD"]

# ── Background ML training (fires once at server start) ────────────────────────
from src.ml_model import get_model as _get_ml_model


def _bg_train() -> None:
    logging.getLogger(__name__).info("Background ML training started…")
    try:
        _get_ml_model().fit()
    except Exception as _exc:
        logging.getLogger(__name__).error("ML training failed: %s", _exc)


threading.Thread(target=_bg_train, daemon=True).start()


def _bg_trade_feedback() -> None:
    """Continuously sync Alpaca trade outcomes into the ML feedback dataset."""
    log = logging.getLogger(__name__)
    while True:
        try:
            trader = _get_alpaca()
            if trader is not None:
                sync_res = trader.sync_closed_trades()
                feedback_rows = trader.drain_ml_feedback_queue()
                ingested = 0
                if feedback_rows:
                    model = _get_ml_model()
                    for row in feedback_rows:
                        if model.add_trade_feedback(row):
                            ingested += 1
                if sync_res.get("processed") or ingested:
                    log.info(
                        "Trade feedback sync: processed=%s queued=%s ingested=%s",
                        sync_res.get("processed", 0),
                        sync_res.get("queued", 0),
                        ingested,
                    )
        except Exception as exc:
            log.warning("Trade feedback loop error: %s", exc)

        time.sleep(30)


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
    interval = request.args.get("interval", "1m")
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
    interval = request.args.get("interval",   "1m")
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

    Pipeline
    --------
      1. SMCStrategy.analyze()  — strict SMC setup (sweep + CHoCH + FVG + OB)
      2. SMCStrategy.find_setup() — pending levels for UI display
      3. ML.evaluate_signal()   — Veto Filter on the SMC signal
      4. Combine → final signal + confidence + alignment

    Response JSON keys:
      ok, ticker, signal, confidence, alignment,
      ml  : { signal, confidence, probabilities, trained },
      smc : { signal, entry, stop_loss, take_profit, risk_reward,
              confidence, reason, smc_score }
    """
    ticker   = request.args.get("ticker",   "SPY").strip().upper()
    interval = request.args.get("interval", "1m")
    period   = request.args.get("period",   "5d")

    try:
        df = fetch_ohlcv(ticker, period=period, interval=interval)

        # ── SMC signal ──────────────────────────────────────────────
        from src.strategy import SMCStrategy
        strategy     = SMCStrategy(ticker, interval=interval, period=period)
        strict_sig   = strategy.analyze(df)
        pending_sig  = strategy.find_setup(df)
        smc_signal   = strict_sig or pending_sig

        smc_dir = smc_signal.direction if smc_signal else None

        # ── SMC score (0–6) mapped from strategy confidence ─────────
        strategy_conf = smc_signal.confidence if smc_signal else 0.0
        if strategy_conf >= 80:
            smc_score = 6
        elif strategy_conf >= 60:
            smc_score = 5
        elif strategy_conf >= 50:
            smc_score = 4
        elif strategy_conf >= 40:
            smc_score = 3
        elif strategy_conf >= 25:
            smc_score = 2
        elif strategy_conf >= 1:
            smc_score = 1
        else:
            smc_score = 0

        smc_block = {
            "signal":        smc_dir,
            "entry":         smc_signal.entry       if smc_signal else None,
            "stop_loss":     smc_signal.stop_loss   if smc_signal else None,
            "take_profit":   smc_signal.take_profit if smc_signal else None,
            "risk_reward":   smc_signal.risk_reward if smc_signal else None,
            "confidence":    strategy_conf,
            "reason":        smc_signal.reason      if smc_signal else None,
            "smc_score":     smc_score,
            "price_at_zone": strict_sig is not None,
        }

        # ── Combine SMC + ML Veto Filter ────────────────────────────
        if strict_sig is not None and smc_dir:
            # We have a real SMC setup — run the ML Veto Filter
            ml_eval = _get_ml_model().evaluate_signal(df, smc_dir)

            if ml_eval["veto"]:
                # ML says this setup is unlikely to succeed — block the trade
                final_signal = "HOLD"
                final_conf   = strategy_conf * 0.5  # mark confidence down
                alignment    = "disagreement"
            else:
                # ML confirms the setup — blend confidences
                final_signal = smc_dir
                final_conf   = min(95.0, (strategy_conf + ml_eval["confidence"]) / 2)
                alignment    = "aligned"

            # Prepare ml block for frontend display
            ml_result = {
                "signal":        final_signal,
                "confidence":    round(ml_eval["confidence"], 1),
                "probabilities": ml_eval["details"],
                "trained":       ml_eval["trained"],
                "training":      ml_eval["training"],
            }

        elif pending_sig is not None:
            # No strict SMC setup, but there's a pending setup for UI
            final_signal = "HOLD"
            final_conf   = strategy_conf * 0.6  # low — pending only
            alignment    = "pending"
            ml_result    = _get_ml_model().predict(df)

        else:
            # No SMC signal at all — fall back to ML standalone
            ml_result = _get_ml_model().predict(df)
            final_signal = ml_result["signal"]
            final_conf   = ml_result["confidence"]
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
_alpaca_lock = threading.Lock()


def _get_alpaca():
    """Return the AlpacaTrader singleton, initialising on first call."""
    global _alpaca_trader, _alpaca_init_error
    if _alpaca_trader is not None:
        return _alpaca_trader

    with _alpaca_lock:
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


threading.Thread(target=_bg_trade_feedback, daemon=True).start()


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


@app.route("/api/paper/outcomes")
def api_paper_outcomes():
    """Return recent settled paper-trade outcomes with reason labels."""
    trader = _get_alpaca()
    if trader is None:
        return jsonify({"ok": False, "error": _alpaca_init_error or "AlpacaTrader unavailable"}), 503
    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        limit = 50
    return jsonify(trader.get_recent_outcomes(limit=limit))


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



# ── Order validation helper ─────────────────────────────────────────────────────

def validate_order(
    entry: float,
    stop_loss: float,
    take_profit: float,
) -> None:
    """Validate numeric order fields: must be finite, non-NaN, and > 0.

    Raises ValueError with a descriptive message if any field is invalid.
    """
    for name, val in [("entry", entry), ("stop_loss", stop_loss), ("take_profit", take_profit)]:
        if math.isnan(val):
            raise ValueError(f"{name} is NaN")
        if math.isinf(val):
            raise ValueError(f"{name} is infinite")
        if val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")

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
    interval   = str(data.get("interval", "1m")).strip() or "1m"
    period     = str(data.get("period", "5d")).strip() or "5d"

    # Parse numeric order levels
    try:
        entry       = float(data["entry"])
        stop_loss   = float(data["stop_loss"])
        take_profit = float(data["take_profit"])
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": f"Missing/invalid numeric field: {exc}"}), 400

    # Validate numeric fields: finite, non-NaN, > 0
    try:
        validate_order(entry, stop_loss, take_profit)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    if not ticker or not side:
        return jsonify({"ok": False, "error": "Fields 'ticker' and 'side' are required."}), 400

    # Guard: paper trading must be explicitly enabled for this ticker
    if not trader.is_enabled(ticker):
        return jsonify({
            "ok":    False,
            "error": f"Paper trading is not enabled for {ticker}. "
                     "Toggle it ON in the dashboard first.",
        }), 409

    # Guard: minimum confidence threshold (from slider override or config default)
    import config as _cfg
    _raw_min = data.get("min_confidence")
    min_conf = float(_raw_min) if _raw_min is not None else getattr(_cfg, "PAPER_MIN_CONFIDENCE", 60.0)
    min_conf = max(50.0, min(95.0, min_conf))  # clamp to valid range
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

    # Build an ML feature snapshot at decision time so closed trades can
    # become supervised feedback samples.
    feature_row = None
    signal_reason = str(data.get("reason", "")).strip()
    try:
        from src.ml_model import FEATURE_NAMES, extract_features

        feat_df = fetch_ohlcv(ticker, period=period, interval=interval)
        feat_mat = extract_features(feat_df)
        last = feat_mat[FEATURE_NAMES].iloc[[-1]].fillna(0.0).clip(-10, 10).iloc[0]
        feature_row = {k: float(last[k]) for k in FEATURE_NAMES}
    except Exception as feat_exc:
        logging.getLogger(__name__).warning("Could not snapshot ML features for %s: %s", ticker, feat_exc)

    result = trader.place_bracket_order(
        ticker=ticker,
        side=side,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        current_price=current_price,
        metadata={
            "feature_row": feature_row,
            "signal_reason": signal_reason,
        },
    )

    if result.get("ok"):
        http_code = 200
    elif result.get("error_code") == "preflight_conflict":
        http_code = 409
    else:
        http_code = 422
    return jsonify(result), http_code


@app.route("/api/paper/test-order", methods=["POST"])
def api_paper_test_order():
    """Place a small test market order at the current price — no is_enabled guard.

    Submits a SIMPLE GTC market order directly at the requested notional so the
    amount is exactly what you asked for (typically $10 for a smoke-test).
    Does NOT run through the full bracket/sizing pipeline.

    Body JSON (all optional):
      ticker  : str   — yfinance symbol (default "BTC-USD")
      side    : str   — "BUY" or "SELL"  (default "BUY")
      notional: float — dollar amount    (default 10.0)
    """
    from alpaca.trading.enums import OrderClass as _OC, OrderSide as _OS, TimeInForce as _TIF
    from alpaca.trading.requests import MarketOrderRequest as _MOR

    trader = _get_alpaca()
    if trader is None:
        return jsonify({"ok": False, "error": _alpaca_init_error or "AlpacaTrader unavailable"}), 503

    data     = request.get_json(silent=True) or {}
    ticker   = str(data.get("ticker",   "BTC-USD")).strip().upper()
    side     = str(data.get("side",     "BUY")).strip().upper()
    notional = float(data.get("notional", 10.0))

    if side not in ("BUY", "SELL"):
        return jsonify({"ok": False, "error": "side must be BUY or SELL"}), 400
    if notional < 1.0:
        return jsonify({"ok": False, "error": "notional must be >= 1.0"}), 400

    alpaca_symbol = trader.map_ticker(ticker)
    order_side    = _OS.BUY if side == "BUY" else _OS.SELL

    try:
        order = trader._client.submit_order(_MOR(
            symbol=alpaca_symbol,
            side=order_side,
            time_in_force=_TIF.GTC,
            order_class=_OC.SIMPLE,
            notional=notional,
        ))
        return jsonify({
            "ok":       True,
            "test":     True,
            "order_id": str(order.id),
            "symbol":   alpaca_symbol,
            "side":     side,
            "notional": notional,
            "status":   str(order.status.value) if hasattr(order.status, "value") else str(order.status),
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 422


# ── Autonomous Trading Bot Control APIs ──────────────────────────────────────────

_bot_engine = None
_bot_thread = None
_bot_lock = threading.Lock()

@app.route("/api/bot/status")
def api_bot_status():
    global _bot_engine
    is_running = _bot_engine is not None and _bot_engine.running
    
    status = {
        "ok": True,
        "running": is_running,
        "uptime": int(time.time() - _bot_engine.start_time) if (_bot_engine is not None and is_running) else 0,
        "cycle_count": _bot_engine.cycle_count if _bot_engine else 0,
        "signals_today": _bot_engine.signals_today if _bot_engine else 0,
        "orders_today": _bot_engine.orders_today if _bot_engine else 0,
        "ai_regime": _bot_engine.ai_tuner.current_regimes if (_bot_engine and hasattr(_bot_engine, "ai_tuner")) else {"Equity": "Unknown", "Crypto": "Unknown", "Gold": "Unknown", "Broad Commodity": "Unknown"},
        "config": {
            "dry_run": config.DRY_RUN,
            "max_risk_pct": config.MAX_RISK_PCT,
            "max_positions": config.MAX_POSITIONS,
            "scan_interval": config.SCAN_INTERVAL,
        },
        "instruments": []
    }
    
    if _bot_engine:
        for inst in _bot_engine.instruments:
            status["instruments"].append({
                "ticker": inst["ticker"],
                "strategy": inst["strategy"].name if hasattr(inst["strategy"], "name") else str(inst["strategy"]),
                "timeframe": inst["strategy"].timeframe if hasattr(inst["strategy"], "timeframe") else "?",
                "last_scan": int(inst["last_scan"]) if inst["last_scan"] > 0 else 0,
                "interval_seconds": inst["interval_seconds"],
            })
            
    return jsonify(status)

@app.route("/api/bot/toggle", methods=["POST"])
def api_bot_toggle():
    global _bot_engine, _bot_thread
    data = request.get_json(silent=True) or {}
    enable = bool(data.get("enabled", False))
    
    with _bot_lock:
        is_running = _bot_engine is not None and _bot_engine.running
        
        if enable:
            if is_running:
                return jsonify({"ok": True, "running": True, "message": "Bot is already running."})
            
            from src.engine import TradingEngine
            try:
                _bot_engine = TradingEngine()
                _bot_thread = threading.Thread(target=_bot_engine.run, daemon=True)
                _bot_thread.start()
                return jsonify({"ok": True, "running": True, "message": "Bot engine started."})
            except Exception as exc:
                return jsonify({"ok": False, "error": f"Failed to start bot engine: {exc}"}), 500
        else:
            if not is_running or _bot_engine is None:
                return jsonify({"ok": True, "running": False, "message": "Bot is already stopped."})
            
            _bot_engine.stop()
            # Wait briefly for thread shutdown
            for _ in range(10):
                if _bot_engine is None or not _bot_engine.running:
                    break
                time.sleep(0.1)
            return jsonify({"ok": True, "running": False, "message": "Bot engine stopped."})

@app.route("/api/bot/configure", methods=["POST"])
def api_bot_configure():
    global _bot_engine
    data = request.get_json(silent=True) or {}
    
    try:
        if "dry_run" in data:
            config.DRY_RUN = bool(data["dry_run"])
        if "max_risk_pct" in data:
            val = float(data["max_risk_pct"])
            if not (0.001 <= val <= 0.1):
                return jsonify({"ok": False, "error": "max_risk_pct must be between 0.001 and 0.1"}), 400
            config.MAX_RISK_PCT = val
        if "max_positions" in data:
            val = int(data["max_positions"])
            if not (1 <= val <= 20):
                return jsonify({"ok": False, "error": "max_positions must be between 1 and 20"}), 400
            config.MAX_POSITIONS = val
        if "scan_interval" in data:
            val = float(data["scan_interval"])
            if not (60 <= val <= 3600):
                return jsonify({"ok": False, "error": "scan_interval must be between 60 and 3600 seconds"}), 400
            config.SCAN_INTERVAL = val
            
        # Dynamically sync config changes to the active engine instances if running
        if _bot_engine:
            if _bot_engine.risk_manager:
                _bot_engine.risk_manager.max_risk_pct = config.MAX_RISK_PCT
                _bot_engine.risk_manager.max_positions = config.MAX_POSITIONS
            
        return jsonify({
            "ok": True,
            "message": "Configuration updated successfully.",
            "config": {
                "dry_run": config.DRY_RUN,
                "max_risk_pct": config.MAX_RISK_PCT,
                "max_positions": config.MAX_POSITIONS,
                "scan_interval": config.SCAN_INTERVAL,
            }
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Failed to update config: {exc}"}), 400

@app.route("/api/bot/logs")
def api_bot_logs():
    # Return last 100 log statements from the deque handler
    return jsonify({
        "ok": True,
        "logs": list(deque_handler.logs)
    })

@app.route("/api/bot/broker")
def api_bot_broker():
    trader = _get_alpaca()
    if not trader:
        return jsonify({"ok": False, "error": _alpaca_init_error or "AlpacaTrader unavailable"}), 503
    
    positions = trader.get_positions()
    orders = trader.get_active_orders().get("orders", [])

    broker_exits = {}
    for o in orders:
        sym = o.get("symbol")
        if not sym: continue
        otype = o.get("type", "")
        if otype == "stop":
            broker_exits.setdefault(sym, {})["stop"] = o.get("stop_price")
        elif otype == "limit":
            broker_exits.setdefault(sym, {})["limit"] = o.get("limit_price")

    for pos in positions:
        sym = pos.get("symbol")
        st = _bot_engine._position_state.get(sym, {}) if _bot_engine else {}
        exits = broker_exits.get(sym, {})
        sl = exits.get("stop") if "stop" in exits else st.get("stop_loss")
        tp = exits.get("limit") if "limit" in exits else st.get("take_profit")
        pos["stop_loss"] = sl
        pos["take_profit"] = tp

    return jsonify({
        "ok": True,
        "positions": positions,
        "orders": orders
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true", port=5000, host="127.0.0.1")
