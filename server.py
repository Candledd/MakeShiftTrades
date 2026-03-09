"""MakeShiftTrades — Flask + HTML chart server

Run with:
    python server.py

Then open  http://localhost:5000  in your browser.
"""
import traceback

import pandas as pd
import plotly.io as pio
from flask import Flask, Response, jsonify, render_template, request

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

app = Flask(__name__)

TICKERS = ["NQ=F", "ES=F", "YM=F", "RTY=F", "SPY", "QQQ", "AAPL", "TSLA", "GC=F", "CL=F"]


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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="127.0.0.1")
