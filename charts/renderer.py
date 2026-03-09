"""Plotly figure builder — combines all four indicator layers onto one chart."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .indicators.fvg import detect_fvg
from .indicators.engulfing import detect_engulfing
from .indicators.liquidity import detect_liquidity_levels
from .indicators.price_action import (
    detect_swing_points,
    detect_market_structure,
    detect_order_blocks,
)

# ── Colour palette ───────────────────────────────────────────────────────────
# Each indicator family uses a distinct hue so layers never look the same:
#   Candlestick  → teal / red        (standard)
#   FVG / IFVG   → cornflower-blue / amber
#   Order Blocks → teal / red        (bolder fill than candles)
#   Engulfing    → cyan / magenta
#   Liquidity    → amber / violet
#   Market Str.  → orange / sky-blue
#   Swing H/L    → red / teal        (own keys, same hue as candles)
C = {
    "bg":            "#131722",
    "grid":          "#363a45",
    "text":          "#B2B5BE",
    # Candlestick — standard TradingView green / red
    "bull":          "#089981",
    "bear":          "#f23645",
    # FVG / IFVG — cornflower-blue / amber
    "fvg_bull":      "rgba(100,149,237,0.22)",
    "fvg_bear":      "rgba(255,165,0,0.22)",
    "fvg_bull_line": "#6495ed",
    "fvg_bear_line": "#ffa500",
    "ifvg_bull":     "rgba(100,149,237,0.08)",
    "ifvg_bear":     "rgba(255,165,0,0.08)",
    # Order Blocks — teal / crimson (same hue as candles, bolder fill)
    "ob_bull":       "rgba(8,153,129,0.32)",
    "ob_bear":       "rgba(242,54,69,0.32)",
    # Engulfing markers — cyan / magenta
    "eng_bull":      "#00bcd4",
    "eng_bear":      "#e040fb",
    # Liquidity levels — amber / violet
    "liq_high":      "#ffb300",
    "liq_low":       "#ab47bc",
    # Market structure
    "choch":         "#ffa726",
    "bos":           "#29b6f6",
    # Swing High / Low (independent keys)
    "swing_high":    "#f23645",
    "swing_low":     "#089981",
}

_OPACITY_BY_STRENGTH = {1: 0.20, 2: 0.35, 3: 0.50, 4: 0.65, 5: 0.80}

# Cap how many FVG boxes we draw to keep the chart readable
_MAX_FVG = 60


def _ts(ts) -> str:
    """Convert a pandas Timestamp (possibly tz-aware) to a plain date/datetime string."""
    if isinstance(ts, pd.Timestamp):
        if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
            return ts.strftime("%Y-%m-%d")
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)


def build_chart(
    df: pd.DataFrame,
    ticker: str,
    indicators: list[str] | None = None,
) -> go.Figure:
    """Build and return a Plotly Figure for `df`.

    `indicators` is a list of active layer keys:
        'fvg', 'engulfing', 'liquidity', 'ob', 'ms', 'swings'
    All layers are active when `indicators` is None.
    """
    if indicators is None:
        indicators = ["fvg", "engulfing", "liquidity", "ob", "ms", "swings"]
    active = set(indicators)

    fig = go.Figure()

    last_date = df.index[-1]

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Candlestick
    # ─────────────────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=ticker,
            increasing=dict(line=dict(color=C["bull"], width=1), fillcolor=C["bull"]),
            decreasing=dict(line=dict(color=C["bear"], width=1), fillcolor=C["bear"]),
            showlegend=True,
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. FVG / IFVG zones
    # ─────────────────────────────────────────────────────────────────────────
    if "fvg" in active:
        fvg_df = detect_fvg(df)
        if not fvg_df.empty:
            # Show only the most recent FVGs
            recent = fvg_df.tail(_MAX_FVG)
            for _, row in recent.iterrows():
                is_bull = row["type"] == "bullish"
                if row["ifvg"]:
                    fill  = C["ifvg_bull"] if is_bull else C["ifvg_bear"]
                    dash  = "dot"
                else:
                    fill  = C["fvg_bull"] if is_bull else C["fvg_bear"]
                    dash  = "solid"
                border = C["fvg_bull_line"] if is_bull else C["fvg_bear_line"]
                end_d  = row["end_date"] if row["active"] is False else last_date

                fig.add_shape(
                    type="rect",
                    x0=_ts(row["date"]),
                    x1=_ts(end_d),
                    y0=row["bottom"],
                    y1=row["top"],
                    fillcolor=fill,
                    line=dict(color=border, width=0.5, dash=dash),
                    layer="below",
                )

            # Legend proxies
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color=C["fvg_bull_line"], symbol="square", opacity=0.75),
                    name="Bullish FVG (blue)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color=C["fvg_bear_line"], symbol="square", opacity=0.75),
                    name="Bearish FVG (amber)",
                )
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Liquidity Engulfing markers
    # ─────────────────────────────────────────────────────────────────────────
    if "engulfing" in active:
        eng_df = detect_engulfing(df)
        if not eng_df.empty:
            bull_e = eng_df[eng_df["type"] == "bullish"]
            bear_e = eng_df[eng_df["type"] == "bearish"]
            if not bull_e.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bull_e["date"],
                        y=bull_e["price"] * 0.9995,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=10, color=C["eng_bull"]),
                        name="Bullish Engulf (cyan)",
                    )
                )
            if not bear_e.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bear_e["date"],
                        y=bear_e["price"] * 1.0005,
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=10, color=C["eng_bear"]),
                        name="Bearish Engulf (magenta)",
                    )
                )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Liquidity Heatmap — horizontal level lines
    # ─────────────────────────────────────────────────────────────────────────
    if "liquidity" in active:
        levels = detect_liquidity_levels(df)
        # Only draw strength ≥ 3 to avoid clutter; deduplicate by rounded price
        strong = [lv for lv in levels if lv["strength"] >= 3]
        seen_prices: set[float] = set()
        for lv in strong:
            p_key = round(lv["price"], 1)
            if p_key in seen_prices:
                continue
            seen_prices.add(p_key)
            color   = C["liq_high"] if lv["dir"] == "high" else C["liq_low"]
            opacity = _OPACITY_BY_STRENGTH[lv["strength"]]
            fig.add_shape(
                type="line",
                x0=_ts(lv["date"]),
                x1=_ts(last_date),
                y0=lv["price"],
                y1=lv["price"],
                line=dict(color=color, width=1, dash="dot"),
                opacity=opacity,
            )

        # Legend proxies
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=C["liq_high"], dash="dot", width=1),
                name="Sell-side Liq (amber)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=C["liq_low"], dash="dot", width=1),
                name="Buy-side Liq (violet)",
            )
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Order Blocks
    # ─────────────────────────────────────────────────────────────────────────
    if "ob" in active:
        obs = detect_order_blocks(df)
        for ob in obs:
            is_bull = ob["type"] == "bullish"
            fill   = C["ob_bull"] if is_bull else C["ob_bear"]
            border = C["bull"]    if is_bull else C["bear"]
            fig.add_shape(
                type="rect",
                x0=_ts(ob["date"]),
                x1=_ts(last_date),
                y0=ob["bottom"],
                y1=ob["top"],
                fillcolor=fill,
                line=dict(color=border, width=1),
            )
        if obs:
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color=C["ob_bull"], symbol="square"),
                    name="Bullish OB",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color=C["ob_bear"], symbol="square"),
                    name="Bearish OB",
                )
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Market Structure — BOS / CHoCH labels
    # ─────────────────────────────────────────────────────────────────────────
    if "ms" in active:
        ms_events = detect_market_structure(df)
        for ev in ms_events:
            color = C["choch"] if ev["label"] == "CHoCH" else C["bos"]
            fig.add_annotation(
                x=_ts(ev["date"]),
                y=ev["price"],
                text=ev["label"],
                showarrow=False,
                font=dict(color=color, size=8, family="monospace"),
                bgcolor="rgba(19,23,34,0.7)",
                borderpad=2,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Swing Highs / Lows
    # ─────────────────────────────────────────────────────────────────────────
    if "swings" in active:
        sw_df = detect_swing_points(df)
        if not sw_df.empty:
            sh = sw_df[sw_df["type"] == "high"]
            sl = sw_df[sw_df["type"] == "low"]
            if not sh.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sh["date"],
                        y=sh["price"],
                        mode="markers+text",
                        text=["SH"] * len(sh),
                        textposition="top center",
                        marker=dict(symbol="triangle-down", size=7, color=C["swing_high"]),
                        textfont=dict(size=7, color=C["swing_high"]),
                        name="Swing High",
                    )
                )
            if not sl.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sl["date"],
                        y=sl["price"],
                        mode="markers+text",
                        text=["SL"] * len(sl),
                        textposition="bottom center",
                        marker=dict(symbol="triangle-up", size=7, color=C["swing_low"]),
                        textfont=dict(size=7, color=C["swing_low"]),
                        name="Swing Low",
                    )
                )

    # ─────────────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>{ticker}</b> — ICT / SMC Indicator Suite",
            font=dict(size=16, color=C["text"]),
        ),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        font=dict(color=C["text"], family="sans-serif"),
        xaxis=dict(
            gridcolor=C["grid"],
            linecolor=C["grid"],
            rangeslider=dict(visible=False),
            type="date",
        ),
        yaxis=dict(
            gridcolor=C["grid"],
            linecolor=C["grid"],
            side="right",
            tickformat=".2f",
            fixedrange=False,  # allow Y-axis drag to scale price
        ),
        legend=dict(
            bgcolor="rgba(19,23,34,0.8)",
            bordercolor=C["grid"],
            borderwidth=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(bgcolor=C["bg"], font_color=C["text"]),
        height=720,
        margin=dict(l=10, r=70, t=50, b=30),
        dragmode="pan",
        hovermode="x unified",
    )

    return fig
