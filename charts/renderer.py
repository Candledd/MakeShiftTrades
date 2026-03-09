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

# ── Colour palette (TradingView dark-mode inspired) ──────────────────────────
C = {
    "bg":         "#131722",
    "grid":       "#363a45",
    "text":       "#B2B5BE",
    "bull":       "#089981",
    "bear":       "#f23645",
    "choch":      "#e0c36a",
    "bos":        "#8888aa",
    # FVG fills (RGBA strings)
    "fvg_bull":   "rgba(8,153,129,0.18)",
    "fvg_bear":   "rgba(242,54,69,0.18)",
    "ifvg_bull":  "rgba(8,153,129,0.07)",
    "ifvg_bear":  "rgba(242,54,69,0.07)",
    # Order block fills
    "ob_bull":    "rgba(8,153,129,0.28)",
    "ob_bear":    "rgba(242,54,69,0.28)",
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
                border = C["bull"] if is_bull else C["bear"]
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
                    marker=dict(size=10, color=C["fvg_bull"], symbol="square"),
                    name="Bullish FVG",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color=C["fvg_bear"], symbol="square"),
                    name="Bearish FVG",
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
                        marker=dict(symbol="triangle-up", size=9, color=C["bull"]),
                        name="Bullish Engulf",
                    )
                )
            if not bear_e.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bear_e["date"],
                        y=bear_e["price"] * 1.0005,
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=9, color=C["bear"]),
                        name="Bearish Engulf",
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
            color   = C["bear"] if lv["dir"] == "high" else C["bull"]
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
                line=dict(color=C["bear"], dash="dot", width=1),
                name="Sell-side Liquidity",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=C["bull"], dash="dot", width=1),
                name="Buy-side Liquidity",
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
                        marker=dict(symbol="triangle-down", size=7, color=C["bear"]),
                        textfont=dict(size=7, color=C["bear"]),
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
                        marker=dict(symbol="triangle-up", size=7, color=C["bull"]),
                        textfont=dict(size=7, color=C["bull"]),
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
