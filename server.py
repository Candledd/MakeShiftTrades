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

        # ── SMC signal ──────────────────────────────────────────────────
        from src.strategy import SMCStrategy
        strategy   = SMCStrategy(ticker, interval=interval, period=period)
        smc_signal = strategy.analyze(df)

        smc_dir   = smc_signal.direction if smc_signal else None
        smc_score = 0
        if smc_signal:
            smc_score = 2  # trend + FVG always present when signal fires
            if smc_signal.raw_data.get("has_ob"):
                smc_score += 1
            if smc_signal.raw_data.get("has_engulf"):
                smc_score += 1

        smc_block = {
            "signal":      smc_dir,
            "entry":       smc_signal.entry       if smc_signal else None,
            "stop_loss":   smc_signal.stop_loss   if smc_signal else None,
            "take_profit": smc_signal.take_profit if smc_signal else None,
            "risk_reward": smc_signal.risk_reward if smc_signal else None,
            "confidence":  smc_signal.confidence  if smc_signal else None,
            "reason":      smc_signal.reason       if smc_signal else None,
            "smc_score":   smc_score,
        }

        # ── ML signal ───────────────────────────────────────────────────
        ml_result = _get_ml_model().predict(df)
        ml_dir    = ml_result["signal"]
        ml_conf   = ml_result["confidence"]

        # ── Combine SMC + ML ────────────────────────────────────────────
        if smc_dir and smc_dir == ml_dir:
            # Both agree → boost confidence proportional to SMC score
            final_signal = smc_dir
            final_conf   = min(100.0, ml_conf + 15.0 * (smc_score / 4))
            alignment    = "aligned"
        elif smc_dir and ml_dir != "HOLD" and smc_dir != ml_dir:
            # Explicit disagreement → show ML, reduce confidence
            final_signal = ml_dir
            final_conf   = ml_conf * 0.75
            alignment    = "disagreement"
        elif smc_dir:
            # SMC fired, ML says HOLD → use SMC direction at slightly reduced confidence
            final_signal = smc_dir
            final_conf   = ml_conf * 0.85
            alignment    = "smc_only"
        else:
            # No SMC setup; rely entirely on ML
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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="127.0.0.1")
