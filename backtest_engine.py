"""Pokretanje strategija iz backtesting/strategies/ (Backtesting.py)."""
import importlib.util
from pathlib import Path

import plotly.graph_objects as go
from backtesting import Strategy as BtStrategy
from backtesting.lib import FractionalBacktest
from dash import html

from data import load_5m, resample
from ny_orb import compute_ny_orb

STRATEGIES_DIR = Path(__file__).parent / "backtesting" / "strategies"
CHART_BG, CHART_GRID = "#0b0d0e", "#2a2d31"
GREEN, RED = "#c5ff4a", "#ff5c5c"

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
    bt = FractionalBacktest(
        df, StrategyCls,
        cash=float(cash),
        commission=float(commission),
        trade_on_close=True,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run()
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


def _orb_overlays(strategy_cls, df):
    if strategy_cls is None or not hasattr(strategy_cls, "rvol_threshold"):
        return []
    sig = compute_ny_orb(
        df,
        rvol_lookback=getattr(strategy_cls, "rvol_lookback", 14),
        atr_period=getattr(strategy_cls, "atr_period", 5),
    )
    return [
        ("OR High", sig["or_high"], "#ffeb3b"),
        ("OR Low", sig["or_low"], "#ff9800"),
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


def equity_figure(stats, df, strategy_cls=None) -> go.Figure:
    eq = stats.get("_equity_curve")
    fig = go.Figure()
    if eq is not None and len(eq):
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq["Equity"], name="Equity",
            line=dict(color=GREEN, width=2),
            fill="tozeroy", fillcolor="rgba(197,255,74,0.08)",
        ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name="Close",
        line=dict(color="#808191", width=1), yaxis="y2",
    ))
    for name, series, color in _sma_overlays(strategy_cls, df) + _orb_overlays(strategy_cls, df):
        fig.add_trace(go.Scatter(
            x=df.index, y=series, name=name,
            line=dict(color=color, width=1.5), yaxis="y2",
        ))
    trades = stats.get("_trades")
    if trades is not None and len(trades):
        bars = [int(i) for i in trades["EntryBar"] if int(i) < len(df)]
        if bars:
            fig.add_trace(go.Scatter(
                x=df.index[bars], y=df["Close"].iloc[bars].tolist(),
                mode="markers", name="Entry",
                marker=dict(symbol="triangle-up", size=9, color=GREEN),
            ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        margin=dict(l=8, r=72, t=24, b=48),
        legend=dict(orientation="h", y=1.02),
        yaxis=dict(title="Equity $", side="right", gridcolor=CHART_GRID),
        yaxis2=dict(title="Cijena", overlaying="y", side="left", showgrid=False),
        xaxis=dict(gridcolor=CHART_GRID),
    )
    return fig
