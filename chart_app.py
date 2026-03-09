"""MakeShiftTrades — Chart App

Run with:
    python chart_app.py
Then open http://localhost:8050 in your browser.
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

from charts.data import fetch_ohlcv
from charts.renderer import build_chart

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="MakeShiftTrades")
server = app.server   # expose for production WSGI servers

# ── Options ───────────────────────────────────────────────────────────────────
INTERVAL_OPTIONS = [
    {"label": "5 Min",   "value": "5m"},
    {"label": "15 Min",  "value": "15m"},
    {"label": "30 Min",  "value": "30m"},
    {"label": "1 Hour",  "value": "1h"},
    {"label": "Daily",   "value": "1d"},
    {"label": "Weekly",  "value": "1wk"},
]

PERIOD_OPTIONS = [
    {"label": "5 Days",    "value": "5d"},
    {"label": "1 Month",   "value": "1mo"},
    {"label": "3 Months",  "value": "3mo"},
    {"label": "6 Months",  "value": "6mo"},
    {"label": "1 Year",    "value": "1y"},
    {"label": "2 Years",   "value": "2y"},
]

INDICATOR_OPTIONS = [
    {"label": "FVG / IFVG",         "value": "fvg"},
    {"label": "Engulfing",           "value": "engulfing"},
    {"label": "Liquidity Levels",   "value": "liquidity"},
    {"label": "Order Blocks",       "value": "ob"},
    {"label": "Market Structure",    "value": "ms"},
    {"label": "Swing H/L",          "value": "swings"},
]

ALL_INDICATORS = [o["value"] for o in INDICATOR_OPTIONS]

# ── Shared style tokens ───────────────────────────────────────────────────────
BG        = "#131722"
SURFACE   = "#1e2230"
BORDER    = "#363a45"
TEXT      = "#B2B5BE"
ACCENT    = "#089981"
DANGER    = "#f23645"

_input_style = {
    "background": SURFACE,
    "color": TEXT,
    "border": f"1px solid {BORDER}",
    "borderRadius": "4px",
    "padding": "6px 10px",
    "fontSize": "13px",
    "outline": "none",
}
_dd_style = {**_input_style, "width": "130px"}
_btn_style = {
    **_input_style,
    "cursor": "pointer",
    "fontWeight": "600",
    "color": ACCENT,
    "borderColor": ACCENT,
    "padding": "6px 18px",
}

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"background": BG, "minHeight": "100vh", "fontFamily": "Inter, sans-serif"},
    children=[
        # ── Top bar ──────────────────────────────────────────────────────────
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "14px",
                "padding": "10px 20px",
                "borderBottom": f"1px solid {BORDER}",
                "flexWrap": "wrap",
            },
            children=[
                html.Span(
                    "MakeShiftTrades",
                    style={
                        "color": ACCENT,
                        "fontWeight": "700",
                        "fontSize": "17px",
                        "marginRight": "6px",
                        "letterSpacing": "0.5px",
                    },
                ),
                # Ticker input
                dcc.Input(
                    id="ticker-input",
                    type="text",
                    value="SPY",
                    debounce=False,
                    placeholder="Ticker (e.g. SPY)",
                    style={**_input_style, "width": "90px", "textTransform": "uppercase"},
                ),
                # Interval dropdown
                dcc.Dropdown(
                    id="interval-dd",
                    options=INTERVAL_OPTIONS,
                    value="1d",
                    clearable=False,
                    style=_dd_style,
                ),
                # Period dropdown
                dcc.Dropdown(
                    id="period-dd",
                    options=PERIOD_OPTIONS,
                    value="6mo",
                    clearable=False,
                    style=_dd_style,
                ),
                # Load button
                html.Button(
                    "Load Chart",
                    id="load-btn",
                    n_clicks=0,
                    style=_btn_style,
                ),
                # Status message (errors shown here)
                html.Span(id="status-msg", style={"color": DANGER, "fontSize": "12px"}),
            ],
        ),

        # ── Indicator toggles ─────────────────────────────────────────────────
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "6px",
                "padding": "8px 20px",
                "borderBottom": f"1px solid {BORDER}",
                "flexWrap": "wrap",
            },
            children=[
                html.Span("Indicators:", style={"color": TEXT, "fontSize": "12px", "marginRight": "4px"}),
                dcc.Checklist(
                    id="indicator-checklist",
                    options=INDICATOR_OPTIONS,
                    value=ALL_INDICATORS,
                    inline=True,
                    inputStyle={"marginRight": "4px", "accentColor": ACCENT},
                    labelStyle={
                        "color": TEXT,
                        "fontSize": "12px",
                        "marginRight": "14px",
                        "cursor": "pointer",
                    },
                ),
            ],
        ),

        # ── Main chart ────────────────────────────────────────────────────────
        dcc.Graph(
            id="main-chart",
            style={"height": "calc(100vh - 100px)"},
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["autoScale2d", "lasso2d", "select2d"],
                "toImageButtonOptions": {"format": "png", "scale": 2},
            },
        ),
    ],
)

# ── Callback ──────────────────────────────────────────────────────────────────
@app.callback(
    Output("main-chart", "figure"),
    Output("status-msg", "children"),
    Input("load-btn", "n_clicks"),
    Input("indicator-checklist", "value"),
    State("ticker-input", "value"),
    State("interval-dd", "value"),
    State("period-dd", "value"),
    prevent_initial_call=False,
)
def update_chart(
    _n_clicks: int,
    active_indicators: list[str],
    ticker: str,
    interval: str,
    period: str,
) -> tuple:
    ticker = (ticker or "SPY").strip().upper()
    try:
        df = fetch_ohlcv(ticker, period=period, interval=interval)
        fig = build_chart(df, ticker, active_indicators or [])
        return fig, ""
    except ValueError as exc:
        return {}, str(exc)
    except Exception as exc:
        return {}, f"Error loading {ticker}: {exc}"


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
