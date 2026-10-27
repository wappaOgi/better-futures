"""Pokretanje strategija iz backtesting/strategies/ (Backtesting.py)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from backtesting import Strategy as BtStrategy
from backtesting.lib import FractionalBacktest
from dash import html
from plotly.subplots import make_subplots

from data import load_5m, resample
from indicators import BB_PERIOD, BB_STD, RSI_PERIOD, bollinger, rsi
from ny_orb import compute_ny_orb, compute_ny_vwap, ny_session_dim_shapes

STRATEGIES_DIR = Path(__file__).parent / "backtesting" / "strategies"
CHART_BG, CHART_GRID = "#0b0d0e", "#2a2d31"
GREEN, RED = "#c5ff4a", "#ff5c5c"
BLUE = "#42a5f5"

STAT_KEYS = [
    "Return [%]",
    "Buy & Hold Return [%]",
    "Sharpe Ratio",
    "# Trades",
    "Win Rate [%]",
    "Max. Drawdown [%]",
    "Equity Final [$]",
    "Exposure Time [%]",
]


def discover_strategies() -> list[dict]:
    opts = []
    for path in sorted(STRATEGIES_DIR.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        cls = _load_strategy(path.stem)
        label = getattr(cls, "label", None) or (cls.__doc__ or path.stem).strip().split("\n")[0][:60]
        opts.append({"label": label, "value": path.stem})
    return opts or [{"label": "strategy", "value": "strategy"}]


def _load_strategy(name: str) -> type:
    path = STRATEGIES_DIR / f"{name}.py"
    if not path.is_file():
        raise ValueError(f"Strategija ne postoji: {name}")
    spec = importlib.util.spec_from_file_location(f"bt_strategy_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, "Strategy", None)
    if cls is None or not (isinstance(cls, type) and issubclass(cls, BtStrategy)):
        raise ValueError(f"{name}.py mora definirati Strategy klasu (Backtesting.py)")
    return cls


def _prepare_df(ticker: str, days: int, timeframe: str):
    df5 = load_5m(ticker, days)
    if timeframe == "1h":
        return resample(df5, "1h")
    if timeframe == "4h":
        return resample(df5, "4h")
    return df5


def run_backtest(
    strategy_name: str,
    ticker: str,
    days: int = 30,
    timeframe: str = "1h",
    cash: float = 10_000,
    risk_pct: float = 0.01,
    commission: float = 0.0002,
):
    StrategyCls = _load_strategy(strategy_name)
    req_tf = getattr(StrategyCls, "required_timeframe", None)
    if req_tf:
        timeframe = req_tf
    min_days = getattr(StrategyCls, "min_days", None)
    if min_days:
        days = max(days, int(min_days))
    df = _prepare_df(ticker, days, timeframe)
    min_bars = getattr(StrategyCls, "min_bars", 50)
    if len(df) < min_bars:
        raise ValueError(f"Premalo podataka za backtest (min ~{min_bars} barova)")
    trade_on_close = getattr(StrategyCls, "trade_on_close", True)
    spread = getattr(StrategyCls, "spread", 0.0)
    bt = FractionalBacktest(
        df, StrategyCls,
        cash=float(cash),
        commission=float(commission),
        spread=float(spread),
        trade_on_close=trade_on_close,
        exclusive_orders=True,
        finalize_trades=True,
    )
    risk_pct = max(0.001, min(float(risk_pct), 0.25))
    run_kw = {}
    if hasattr(StrategyCls, "risk_pct"):
        run_kw["risk_pct"] = risk_pct
    if hasattr(StrategyCls, "rr_ratio"):
        run_kw["rr_ratio"] = getattr(StrategyCls, "rr_ratio")
    stats = bt.run(**run_kw)
    return stats, df, StrategyCls


def stats_cards(stats) -> html.Div:
    items = []
    for key in STAT_KEYS:
        if key not in stats.index:
            continue
        val = stats[key]
        if isinstance(val, float):
            txt = f"{val:.2f}" if abs(val) < 1e6 else f"{val:,.0f}"
        else:
            txt = str(val)
        cls = "text-muted"
        if key == "Return [%]":
            cls = "text-positive" if float(stats[key]) >= 0 else "text-negative"
        items.append(html.Span([html.Strong(f"{key}: "), html.Span(txt, className=cls), " · "]))
    return html.Div(items, className="badge-row")


def _uses_ny_chart(strategy_cls) -> bool:
    if strategy_cls is None:
        return False
    if getattr(strategy_cls, "chart_profile", None) == "ny":
        return True
    return hasattr(strategy_cls, "rvol_threshold")


def _orb_overlays(strategy_cls, df):
    if not hasattr(strategy_cls, "rvol_threshold"):
        return []
    sig = compute_ny_orb(
        df,
        rvol_lookback=getattr(strategy_cls, "rvol_lookback", 14),
        atr_period=getattr(strategy_cls, "atr_period", 5),
    )
    vwap = compute_ny_vwap(df)
    return [
        ("OR High", sig["or_high"], "#ffeb3b"),
        ("OR Low", sig["or_low"], "#ff9800"),
        ("VWAP", vwap, "#4fc3f7"),
    ]


def _sma_overlays(strategy_cls, df):
    """SMA linije za strategije s fast_period / slow_period."""
    if strategy_cls is None:
        return []
    fast_n = getattr(strategy_cls, "fast_period", None)
    slow_n = getattr(strategy_cls, "slow_period", None)
    if not fast_n or not slow_n:
        return []
    close = df["Close"]
    return [
        ("SMA fast", close.rolling(int(fast_n)).mean(), GREEN),
        ("SMA slow", close.rolling(int(slow_n)).mean(), "#ba68c8"),
    ]


def _volume_colors(df) -> list[str]:
    up = (df["Close"] >= df["Open"]).to_numpy()
    return np.where(up, GREEN, RED).tolist()


def _add_candles(fig, df, row: int, secondary_y: bool = False, name: str = "OHLC"):
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name=name,
            increasing_line_color=GREEN, decreasing_line_color=RED,
            increasing_fillcolor=GREEN, decreasing_fillcolor=RED,
        ),
        row=row, col=1, secondary_y=secondary_y,
    )


def _add_bollinger(fig, df, row: int = 1):
    bb = bollinger(df["Close"], BB_PERIOD, BB_STD)
    x = df.index
    fig.add_trace(
        go.Scatter(
            x=x, y=bb["bb_upper"], name=f"BB upper ({BB_PERIOD})",
            line=dict(color="rgba(186,104,200,0.9)", width=1),
        ),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=bb["bb_lower"], name=f"BB lower ({BB_PERIOD})",
            line=dict(color="rgba(186,104,200,0.9)", width=1),
            fill="tonexty", fillcolor="rgba(186,104,200,0.08)",
        ),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=bb["bb_mid"], name=f"BB mid ({BB_PERIOD})",
            line=dict(color="rgba(186,104,200,0.45)", width=1, dash="dot"),
        ),
        row=row, col=1,
    )


def _add_rsi(fig, df, row: int = 3, ob: float = 70, os_level: float = 30):
    rsi_s = rsi(df["Close"], RSI_PERIOD)
    x = df.index
    zones = [
        (rsi_s > ob, RED, f"RSI > {ob:g}"),
        (rsi_s < os_level, GREEN, f"RSI < {os_level:g}"),
        ((rsi_s >= os_level) & (rsi_s <= ob), "#808191", f"RSI({RSI_PERIOD})"),
    ]
    for mask, color, name in zones:
        fig.add_trace(
            go.Scatter(
                x=x, y=rsi_s.where(mask), mode="lines", name=name,
                line=dict(color=color, width=1.5), connectgaps=False,
            ),
            row=row, col=1,
        )
    fig.add_hline(y=ob, line=dict(color=RED, width=1, dash="dash"), row=row, col=1)
    fig.add_hline(y=os_level, line=dict(color=GREEN, width=1, dash="dash"), row=row, col=1)


def _rgba(color: str, a: float) -> str:
    c = color.lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{a:.2f})"


def _ny_dim_shapes(df, rows: tuple[int, ...] = (1, 2, 3)) -> list:
    yrefs = {1: "y", 2: "y3", 3: "y4"}
    ylims = {
        1: (df["Low"].min(), df["High"].max()),
        2: (0, df["Volume"].max() * 1.05),
        3: (0, 100),
    }
    shapes = []
    for row in rows:
        yref = yrefs.get(row, "y")
        y0, y1 = ylims.get(row, (0, 1))
        shapes.extend(ny_session_dim_shapes(df, yref=yref, y0=y0, y1=y1))
    return shapes


def _trade_label(x, y, text, color, anchor="left", valign="middle"):
    return dict(
        x=x, y=y, xref="x", yref="y", text=text, showarrow=False,
        xanchor=anchor, yanchor=valign, font=dict(size=8, color=color),
        bgcolor="rgba(0,0,0,0.45)", borderpad=2, name="trade",
    )


def _backtest_trade_overlays(stats, df) -> tuple[list, list]:
    """Pravokutnik entry→exit; SL/TP = donja/gornja granica; ✓/✗ po ishodu."""
    trades = stats.get("_trades")
    shapes, annotations = [], []
    if trades is None or not len(trades):
        return shapes, annotations

    n = len(df)
    bar_delta = pd.Timedelta(minutes=5)
    if n >= 2:
        bar_delta = max(df.index[1] - df.index[0], bar_delta)
    fixed = dict(layer="above", name="trade", editable=False)

    for _, t in trades.iterrows():
        eb, xb = int(t["EntryBar"]), int(t["ExitBar"])
        if eb < 0 or xb < 0 or eb >= n or xb >= n:
            continue
        x0, x1 = df.index[min(eb, xb)], df.index[max(eb, xb)]
        if x1 <= x0:
            x1 = x0 + bar_delta

        sl, tp = t.get("SL"), t.get("TP")
        if pd.isna(sl) or pd.isna(tp):
            continue
        sl, tp = float(sl), float(tp)
        entry = float(t["EntryPrice"])
        box_lo, box_hi = min(sl, tp), max(sl, tp)
        if box_lo >= box_hi:
            continue

        win = float(t["ReturnPct"]) >= 0
        col = GREEN if win else RED
        ret_pct = float(t["ReturnPct"]) * 100.0
        pct_txt = f"+{ret_pct:.2f}%" if win else f"{ret_pct:.2f}%"
        direction = "long" if float(t["Size"]) > 0 else "short"
        sym = "▲ LONG" if direction == "long" else "▼ SHORT"
        t_lbl = x0 + (x1 - x0) * 0.04

        shapes.extend([
            dict(
                type="rect", xref="x", yref="y", x0=x0, x1=x1,
                y0=box_lo, y1=box_hi,
                fillcolor=_rgba(col, 0.10),
                line=dict(color=col, width=2),
                **fixed,
            ),
            dict(
                type="line", xref="x", yref="y", x0=x0, x1=x1,
                y0=entry, y1=entry,
                line=dict(color=BLUE, width=1, dash="dash"),
                **fixed,
            ),
        ])

        annotations.extend([
            _trade_label(t_lbl, box_hi, sym, BLUE, valign="top"),
            _trade_label(t_lbl, box_hi, f"TP {tp:.2f}", GREEN, valign="bottom"),
            _trade_label(t_lbl, entry, f"E {entry:.2f}", BLUE),
            _trade_label(t_lbl, box_lo, f"SL {sl:.2f}", RED, valign="top"),
            dict(
                x=x1, y=box_hi, xref="x", yref="y",
                text="✓" if win else "✗", showarrow=False,
                xanchor="right", yanchor="top",
                font=dict(size=16, color=col),
                bgcolor="rgba(0,0,0,0.5)", borderpad=3, name="trade",
            ),
            _trade_label(t_lbl, (box_lo + box_hi) / 2, pct_txt, col),
        ])

    return shapes, annotations


def _apply_chart_extras(fig, df, stats, *, ny_dim: bool = False):
    shapes = _ny_dim_shapes(df) if ny_dim else []
    trade_shapes, trade_ann = _backtest_trade_overlays(stats, df)
    fig.update_layout(
        shapes=shapes + trade_shapes,
        annotations=trade_ann,
        margin=dict(l=8, r=72, t=24, b=48),
    )


def _ny_session_figure(stats, df, strategy_cls) -> go.Figure:
    eq = stats.get("_equity_curve")
    rsi_ob = getattr(strategy_cls, "rsi_short", 70)
    rsi_os = getattr(strategy_cls, "rsi_long", 30)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.54, 0.16, 0.22],
        specs=[[{"secondary_y": True}], [{}], [{}]],
    )

    _add_candles(fig, df, row=1)
    _add_bollinger(fig, df, row=1)
    for name, series, color in _orb_overlays(strategy_cls, df):
        fig.add_trace(
            go.Scatter(
                x=df.index, y=series, name=name,
                line=dict(color=color, width=1.5),
            ),
            row=1, col=1,
        )

    if eq is not None and len(eq):
        fig.add_trace(
            go.Scatter(
                x=eq.index, y=eq["Equity"], name="Equity",
                line=dict(color=GREEN, width=2),
                fill="tozeroy", fillcolor="rgba(197,255,74,0.08)",
            ),
            row=1, col=1, secondary_y=True,
        )

    fig.add_trace(
        go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=_volume_colors(df), opacity=0.75,
        ),
        row=2, col=1,
    )

    _add_rsi(fig, df, row=3, ob=rsi_ob, os_level=rsi_os)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        legend=dict(orientation="h", y=1.06),
        xaxis_rangeslider_visible=False,
        xaxis2=dict(gridcolor=CHART_GRID),
        xaxis3=dict(gridcolor=CHART_GRID),
        yaxis=dict(title="Cijena", gridcolor=CHART_GRID, side="left"),
        yaxis2=dict(title="Equity $", overlaying="y", side="right", showgrid=False),
        yaxis3=dict(title="Vol", gridcolor=CHART_GRID),
        yaxis4=dict(title=f"RSI({RSI_PERIOD})", gridcolor=CHART_GRID, range=[0, 100]),
    )
    fig.update_xaxes(showticklabels=True, row=3, col=1)
    _apply_chart_extras(fig, df, stats, ny_dim=True)
    return fig


def _default_equity_figure(stats, df, strategy_cls) -> go.Figure:
    eq = stats.get("_equity_curve")
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.78, 0.22],
        specs=[[{"secondary_y": True}], [{}]],
    )
    _add_candles(fig, df, row=1)
    for name, series, color in _sma_overlays(strategy_cls, df):
        fig.add_trace(
            go.Scatter(
                x=df.index, y=series, name=name,
                line=dict(color=color, width=1.5),
            ),
            row=1, col=1,
        )
    if eq is not None and len(eq):
        fig.add_trace(
            go.Scatter(
                x=eq.index, y=eq["Equity"], name="Equity",
                line=dict(color=GREEN, width=2),
                fill="tozeroy", fillcolor="rgba(197,255,74,0.08)",
            ),
            row=1, col=1, secondary_y=True,
        )
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=_volume_colors(df), opacity=0.75,
        ),
        row=2, col=1,
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        legend=dict(orientation="h", y=1.04),
        xaxis_rangeslider_visible=False,
        yaxis=dict(title="Cijena", gridcolor=CHART_GRID, side="left"),
        yaxis2=dict(title="Equity $", overlaying="y", side="right", showgrid=False),
        yaxis3=dict(title="Vol", gridcolor=CHART_GRID),
    )
    fig.update_xaxes(showticklabels=True, row=2, col=1)
    _apply_chart_extras(fig, df, stats, ny_dim=False)
    return fig


def equity_figure(stats, df, strategy_cls=None) -> go.Figure:
    if _uses_ny_chart(strategy_cls):
        return _ny_session_figure(stats, df, strategy_cls)
    return _default_equity_figure(stats, df, strategy_cls)
