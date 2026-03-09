"""MakeShiftTrades — Real-Time Chart App

Run with:
    python chart_app.py
Then open http://localhost:8050 in your browser.
"""

import copy
import pandas as pd
import dash
from dash import dcc, html, ctx, ALL
from dash.dependencies import Input, Output, State

from charts.data import fetch_ohlcv
from charts.renderer import build_chart

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="MakeShiftTrades")
server = app.server  # expose for production WSGI

# ── Preset tickers ────────────────────────────────────────────────────────────
PRESET_TICKERS = [
    ("NQ",   "NQ=F"),    # NASDAQ-100 Futures
    ("ES",   "ES=F"),    # S&P 500 Futures
    ("YM",   "YM=F"),    # Dow Jones Futures
    ("RTY",  "RTY=F"),   # Russell 2000 Futures
    ("SPY",  "SPY"),
    ("QQQ",  "QQQ"),
    ("AAPL", "AAPL"),
    ("TSLA", "TSLA"),
    ("GC",   "GC=F"),    # Gold Futures
    ("CL",   "CL=F"),    # Crude Oil Futures
]

# ── Interval options ──────────────────────────────────────────────────────────
INTERVAL_OPTIONS = [
    {"label": "1m",  "value": "1m"},
    {"label": "3m",  "value": "3m"},
    {"label": "5m",  "value": "5m"},
    {"label": "15m", "value": "15m"},
    {"label": "30m", "value": "30m"},
    {"label": "1h",  "value": "1h"},
    {"label": "1D",  "value": "1d"},
    {"label": "1W",  "value": "1wk"},
]

# Period dropdown options per interval family (only valid yfinance strings)
_P_1M  = [{"label": "1 Day", "value": "1d"}, {"label": "5 Days", "value": "5d"}]
_P_5M  = [{"label": "1 Day", "value": "1d"}, {"label": "5 Days", "value": "5d"},
          {"label": "1 Month", "value": "1mo"}]
_P_30M = [{"label": "5 Days", "value": "5d"}, {"label": "1 Month", "value": "1mo"},
          {"label": "3 Months", "value": "3mo"}]
_P_DAY = [{"label": "1 Month", "value": "1mo"}, {"label": "3 Months", "value": "3mo"},
          {"label": "6 Months", "value": "6mo"}, {"label": "1 Year", "value": "1y"},
          {"label": "2 Years", "value": "2y"}]

_INTERVAL_PERIOD_MAP: dict[str, tuple[list, str]] = {
    "1m":  (_P_1M,  "5d"),
    "3m":  (_P_1M,  "5d"),
    "5m":  (_P_5M,  "1mo"),
    "15m": (_P_5M,  "1mo"),
    "30m": (_P_30M, "1mo"),
    "1h":  (_P_DAY, "3mo"),
    "1d":  (_P_DAY, "6mo"),
    "1wk": (_P_DAY, "1y"),
}

# ── Indicator options ─────────────────────────────────────────────────────────
INDICATOR_OPTIONS = [
    {"label": "FVG / IFVG",       "value": "fvg"},
    {"label": "Engulfing",         "value": "engulfing"},
    {"label": "Liquidity Levels", "value": "liquidity"},
    {"label": "Order Blocks",     "value": "ob"},
    {"label": "Market Structure",  "value": "ms"},
    {"label": "Swing H/L",        "value": "swings"},
]
ALL_INDICATORS = [o["value"] for o in INDICATOR_OPTIONS]

# ── Auto-refresh rates ────────────────────────────────────────────────────────
REFRESH_OPTIONS = [
    {"label": "15s", "value": 15_000},
    {"label": "30s", "value": 30_000},
    {"label": "1m",  "value": 60_000},
    {"label": "5m",  "value": 300_000},
]

# ── Colour tokens ─────────────────────────────────────────────────────────────
BG      = "#131722"
SURFACE = "#1e2230"
BORDER  = "#363a45"
TEXT    = "#B2B5BE"
ACCENT  = "#089981"
DANGER  = "#f23645"
MUTED   = "#4e5263"

_base = {
    "background": SURFACE, "color": TEXT,
    "border": f"1px solid {BORDER}", "borderRadius": "4px",
    "padding": "6px 10px", "fontSize": "13px", "outline": "none",
}
_dd_style  = {**_base, "width": "105px", "minWidth": "80px"}
_btn_style = {**_base, "cursor": "pointer", "fontWeight": "600"}


def _interval_btn_style(active: bool) -> dict:
    return {
        **{k: v for k, v in _base.items() if k not in ("border", "borderRadius")},
        "border": "none",
        "borderRight": f"1px solid {BORDER}",
        "borderRadius": "0",
        "padding": "5px 10px",
        "fontSize": "12px",
        "fontWeight": "700",
        "cursor": "pointer",
        "color": BG if active else TEXT,
        "background": ACCENT if active else SURFACE,
        "whiteSpace": "nowrap",
    }


def _fmt_ts(ts) -> str:
    """Format a pandas Timestamp for Plotly xaxis_range."""
    if isinstance(ts, pd.Timestamp):
        ts = ts.tz_convert(None) if ts.tzinfo else ts
        if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
            return ts.strftime("%Y-%m-%d")
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)


# ── Viewport-filter helpers ──────────────────────────────────────────────────────
def _to_ts(s) -> pd.Timestamp | None:
    """Safely convert a string to pd.Timestamp; return None on failure."""
    if s is None:
        return None
    try:
        return pd.Timestamp(s)
    except Exception:
        return None


def _shape_visible(shape: dict, x0: pd.Timestamp, x1: pd.Timestamp) -> bool:
    sx0 = _to_ts(shape.get("x0"))
    sx1 = _to_ts(shape.get("x1"))
    if sx0 is None or sx1 is None:
        return True   # non-temporal shape → keep
    return sx0 <= x1 and sx1 >= x0


def _annotation_visible(ann: dict, x0: pd.Timestamp, x1: pd.Timestamp) -> bool:
    ax = _to_ts(ann.get("x"))
    if ax is None:
        return True   # no x position → keep
    return x0 <= ax <= x1


# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={
        "background": BG, "height": "100vh",
        "display": "flex", "flexDirection": "column",
        "fontFamily": "Inter, sans-serif", "overflow": "hidden",
    },
    children=[
        # Hidden stores & timer
        dcc.Interval(id="auto-refresh", interval=30_000, n_intervals=0, disabled=True),
        dcc.Store(id="interval-store", data="5m"),
        dcc.Store(id="ticker-store",   data="SPY"),
        dcc.Store(id="figure-store",   storage_type="memory"),

        # ── Toolbar ──────────────────────────────────────────────────────────
        html.Div(
            style={
                "display": "flex", "alignItems": "center", "gap": "8px",
                "padding": "7px 14px", "borderBottom": f"1px solid {BORDER}",
                "background": SURFACE, "flexWrap": "wrap", "flexShrink": "0",
            },
            children=[
                # Logo
                html.Span(
                    "MakeShiftTrades",
                    style={
                        "color": ACCENT, "fontWeight": "800", "fontSize": "15px",
                        "letterSpacing": "0.5px", "whiteSpace": "nowrap",
                        "marginRight": "4px",
                    },
                ),
                html.Div(style={"width": "1px", "height": "22px", "background": BORDER, "flexShrink": "0"}),

                # Preset ticker buttons
                html.Div(
                    style={"display": "flex", "gap": "3px", "flexWrap": "nowrap"},
                    children=[
                        html.Button(
                            label,
                            id={"type": "preset-btn", "ticker": ticker},
                            n_clicks=0,
                            title=ticker,
                            style={
                                **_base,
                                "padding": "4px 8px", "fontSize": "11px",
                                "fontWeight": "700", "letterSpacing": "0.4px",
                                "cursor": "pointer", "color": TEXT, "borderColor": MUTED,
                            },
                        )
                        for label, ticker in PRESET_TICKERS
                    ],
                ),
                html.Div(style={"width": "1px", "height": "22px", "background": BORDER, "flexShrink": "0"}),

                # Custom ticker input
                dcc.Input(
                    id="ticker-input", type="text", value="SPY",
                    debounce=False, placeholder="TICKER",
                    style={
                        **_base, "width": "72px", "textTransform": "uppercase",
                        "textAlign": "center", "fontWeight": "700",
                    },
                ),

                # Interval tab-strip
                html.Div(
                    style={
                        "display": "flex",
                        "border": f"1px solid {BORDER}",
                        "borderRadius": "4px",
                        "overflow": "hidden",
                        "flexShrink": "0",
                    },
                    children=[
                        html.Button(
                            opt["label"],
                            id={"type": "interval-btn", "value": opt["value"]},
                            n_clicks=0,
                            style=_interval_btn_style(opt["value"] == "5m"),
                        )
                        for opt in INTERVAL_OPTIONS
                    ],
                ),

                # Period dropdown (options updated dynamically by callback)
                dcc.Dropdown(
                    id="period-dd", options=_P_5M, value="1mo",
                    clearable=False, style=_dd_style,
                ),

                # Load button
                html.Button(
                    "Load",
                    id="load-btn", n_clicks=0,
                    style={
                        **_btn_style, "color": ACCENT, "borderColor": ACCENT,
                        "padding": "6px 18px",
                    },
                ),

                html.Div(style={"width": "1px", "height": "22px", "background": BORDER, "flexShrink": "0"}),

                # Live-refresh toggle
                html.Span("Live:", style={"color": TEXT, "fontSize": "12px", "whiteSpace": "nowrap"}),
                html.Button(
                    "OFF",
                    id="refresh-toggle", n_clicks=0,
                    style={
                        **_btn_style, "color": DANGER, "borderColor": DANGER,
                        "padding": "5px 10px", "fontSize": "12px",
                    },
                ),
                dcc.Dropdown(
                    id="refresh-rate-dd", options=REFRESH_OPTIONS, value=30_000,
                    clearable=False, style={**_dd_style, "width": "75px"},
                ),

                # Status / error message
                html.Span(
                    id="status-msg",
                    style={"color": DANGER, "fontSize": "12px", "marginLeft": "6px"},
                ),
            ],
        ),

        # ── Indicator row + candle-count slider ───────────────────────────────
        html.Div(
            style={
                "display": "flex", "alignItems": "center", "gap": "12px",
                "padding": "5px 14px", "borderBottom": f"1px solid {BORDER}",
                "flexWrap": "wrap", "flexShrink": "0",
            },
            children=[
                html.Span(
                    "Indicators:",
                    style={"color": MUTED, "fontSize": "11px", "whiteSpace": "nowrap"},
                ),
                dcc.Checklist(
                    id="indicator-checklist",
                    options=INDICATOR_OPTIONS,
                    value=ALL_INDICATORS,
                    inline=True,
                    inputStyle={"marginRight": "4px", "accentColor": ACCENT},
                    labelStyle={
                        "color": TEXT, "fontSize": "12px",
                        "marginRight": "12px", "cursor": "pointer",
                    },
                ),
                html.Div(style={"width": "1px", "height": "18px", "background": BORDER, "flexShrink": "0"}),
                html.Span(
                    "↕ Drag price axis to scale",
                    style={"color": MUTED, "fontSize": "10px", "whiteSpace": "nowrap", "fontStyle": "italic"},
                ),
                html.Div(style={"width": "1px", "height": "18px", "background": BORDER, "flexShrink": "0"}),
                html.Span(
                    "Candles visible:",
                    style={"color": MUTED, "fontSize": "11px", "whiteSpace": "nowrap"},
                ),
                html.Div(
                    style={"width": "240px"},
                    children=dcc.Slider(
                        id="n-candles-slider",
                        min=20, max=600, step=10, value=150,
                        marks={
                            20:  {"label": "20",  "style": {"color": MUTED, "fontSize": "10px"}},
                            100: {"label": "100", "style": {"color": MUTED, "fontSize": "10px"}},
                            200: {"label": "200", "style": {"color": MUTED, "fontSize": "10px"}},
                            400: {"label": "400", "style": {"color": MUTED, "fontSize": "10px"}},
                            600: {"label": "600", "style": {"color": MUTED, "fontSize": "10px"}},
                        },
                        tooltip={"placement": "top", "always_visible": False},
                        updatemode="mouseup",
                    ),
                ),
            ],
        ),

        # ── Chart ─────────────────────────────────────────────────────────────
        dcc.Graph(
            id="main-chart",
            style={"flex": "1", "minHeight": "0"},
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["autoScale2d", "lasso2d", "select2d"],
                "toImageButtonOptions": {"format": "png", "scale": 2},
                "doubleClick": "reset",
            },
        ),
    ],
)

# ── Callbacks ─────────────────────────────────────────────────────────────────

# 1) Preset ticker → fill input + triggers chart reload via ticker-store
@app.callback(
    Output("ticker-input", "value"),
    Output("ticker-store", "data"),
    Input({"type": "preset-btn", "ticker": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def on_preset_click(_n_clicks):
    triggered = ctx.triggered_id
    if not triggered:
        return dash.no_update, dash.no_update
    ticker = triggered["ticker"]
    return ticker, ticker


# 2) Interval tab-strip → update interval-store + highlight active button
@app.callback(
    Output("interval-store", "data"),
    Output({"type": "interval-btn", "value": ALL}, "style"),
    Input({"type": "interval-btn", "value": ALL}, "n_clicks"),
    State("interval-store", "data"),
    prevent_initial_call=False,
)
def on_interval_click(_n_clicks, current_interval):
    triggered = ctx.triggered_id
    new_interval = current_interval
    if triggered and isinstance(triggered, dict) and "value" in triggered:
        new_interval = triggered["value"]
    styles = [_interval_btn_style(opt["value"] == new_interval) for opt in INTERVAL_OPTIONS]
    return new_interval, styles


# 3) Interval change → sync period dropdown options & default value
#    (update_chart depends on period-dd.value so this chains correctly)
@app.callback(
    Output("period-dd", "options"),
    Output("period-dd", "value"),
    Input("interval-store", "data"),
)
def sync_period_options(interval: str):
    opts, default = _INTERVAL_PERIOD_MAP.get(interval, (_P_DAY, "6mo"))
    return opts, default


# 4) Refresh toggle / rate → control the Interval component
@app.callback(
    Output("auto-refresh", "disabled"),
    Output("auto-refresh", "interval"),
    Output("refresh-toggle", "children"),
    Output("refresh-toggle", "style"),
    Input("refresh-toggle", "n_clicks"),
    Input("refresh-rate-dd", "value"),
    State("auto-refresh", "disabled"),
    prevent_initial_call=True,
)
def toggle_refresh(_n_clicks, rate_ms, currently_disabled):
    triggered = ctx.triggered_id
    now_disabled = (not currently_disabled) if triggered == "refresh-toggle" else currently_disabled
    label = "OFF" if now_disabled else "ON"
    color = DANGER if now_disabled else ACCENT
    style = {**_btn_style, "padding": "5px 10px", "fontSize": "12px",
             "color": color, "borderColor": color}
    return now_disabled, rate_ms, label, style


# 5) Main chart — triggered by load, auto-refresh, interval, period, indicators, slider, preset
@app.callback(
    Output("main-chart", "figure"),
    Output("status-msg", "children"),
    Output("figure-store", "data"),
    Input("load-btn", "n_clicks"),
    Input("ticker-store", "data"),       # preset ticker click
    Input("period-dd", "value"),         # period change (also chains from interval change)
    Input("auto-refresh", "n_intervals"),
    Input("indicator-checklist", "value"),
    Input("n-candles-slider", "value"),
    State("ticker-input", "value"),
    State("interval-store", "data"),
    prevent_initial_call=False,
)
def update_chart(
    _n_clicks, _ticker_store, period, _n_intervals,
    active_indicators, n_candles,
    ticker, interval,
):
    ticker = (ticker or "SPY").strip().upper()
    try:
        df = fetch_ohlcv(ticker, period=period, interval=interval)
        fig = build_chart(df, ticker, active_indicators or [])

        # Set initial x-axis zoom to the last n_candles bars
        if n_candles and len(df) > n_candles:
            fig.update_layout(
                xaxis_range=[
                    _fmt_ts(df.index[-n_candles]),
                    _fmt_ts(df.index[-1]),
                ]
            )

        fig_dict = fig.to_dict()
        return fig, "", fig_dict
    except ValueError as exc:
        return dash.no_update, str(exc), dash.no_update
    except Exception as exc:
        return dash.no_update, f"Error: {exc}", dash.no_update


# 6) Viewport filter — fires on every zoom / pan, hides out-of-view indicator shapes
@app.callback(
    Output("main-chart", "figure", allow_duplicate=True),
    Input("main-chart", "relayoutData"),
    State("figure-store", "data"),
    prevent_initial_call=True,
)
def filter_on_zoom(relayout_data, fig_json):
    """Re-filter shapes / annotations to only those overlapping the visible x range."""
    if not fig_json or not relayout_data:
        return dash.no_update

    # Double-click reset → return full unfiltered figure
    if relayout_data.get("xaxis.autorange") is True:
        full = copy.deepcopy(fig_json)
        full.setdefault("layout", {}).pop("xaxis", None)
        return full

    x0_str = relayout_data.get("xaxis.range[0]")
    x1_str = relayout_data.get("xaxis.range[1]")
    if not x0_str or not x1_str:
        return dash.no_update

    x0 = _to_ts(x0_str)
    x1 = _to_ts(x1_str)
    if x0 is None or x1 is None:
        return dash.no_update

    fig = copy.deepcopy(fig_json)
    layout = fig.setdefault("layout", {})

    # Filter shapes (FVG boxes, OB boxes, liquidity lines)
    layout["shapes"] = [
        s for s in layout.get("shapes", [])
        if _shape_visible(s, x0, x1)
    ]

    # Filter annotations (Market Structure BOS / CHoCH labels)
    layout["annotations"] = [
        a for a in layout.get("annotations", [])
        if _annotation_visible(a, x0, x1)
    ]

    # Preserve the zoom range so Plotly doesn't re-autorange
    layout.setdefault("xaxis", {})["range"] = [x0_str, x1_str]

    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
