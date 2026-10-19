"""UI: Dash + Plotly. Pokretanje: python ui.py  ->  http://127.0.0.1:8050"""
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, clientside_callback, ctx, dcc, html, no_update

from bank import load as load_bank, recalc, risk_eur, save as save_bank, update_settings
from atr import PERIOD as ATR_PERIOD, last as last_atr, series as atr_series, suggest_sl
from data import DEFAULT_TICKER, cache_path, clear_cache, load_5m, resample
from sessions import overlays as session_overlays
from timelines import overlays as timeline_overlays
from trades import (
    FILE as TRADES_FILE,
    add_trade, chart_overlays, delete_trade, get as get_trade, load as load_trades,
    make_draft_manual, resolve_all, save as save_trades, seed_test_trades, sort_by_opened,
    update_trade,
)
from trend import analyze, trend_overlays
from zones import get_zones

GREEN, RED, AMBER = "#26a69a", "#ef5350", "#ffc107"
SWING_N = {"5min": 6, "1h": 3}
X_TICKS = [
    dict(dtickrange=[None, 600_000], value="%H:%M"),
    dict(dtickrange=[600_000, 3_600_000], value="%H:%M"),
    dict(dtickrange=[3_600_000, 86_400_000], value="%d.%m. %H:%M"),
    dict(dtickrange=[86_400_000, 604_800_000], value="%d.%m."),
    dict(dtickrange=[604_800_000, None], value="%d.%m.%Y"),
]
PANEL = {
    "background": "#1c1f26", "border": "1px solid #333", "borderRadius": "6px",
    "padding": "10px 12px", "marginTop": "6px",
}
INP = {"width": "110px", "marginRight": "8px"}
MODAL_HIDE = {"display": "none"}
MODAL_SHOW = {
    "display": "flex", "position": "fixed", "top": 0, "left": 0, "right": 0, "bottom": 0,
    "backgroundColor": "rgba(0,0,0,0.65)", "zIndex": 1000,
    "alignItems": "center", "justifyContent": "center",
}
MODAL_BOX = {
    "background": "#1c1f26", "border": "1px solid #444", "borderRadius": "8px",
    "padding": "20px 24px", "minWidth": "340px", "maxWidth": "420px", "boxShadow": "0 8px 32px rgba(0,0,0,0.5)",
}
MODAL_ROW = {"marginBottom": "12px"}
MODAL_LBL = {"display": "block", "color": "#aaa", "fontSize": "12px", "marginBottom": "4px"}
FAVORITE_TICKERS = [
    {"label": "MNQ · Nasdaq Micro", "value": "MNQ=F"},
    {"label": "MES · S&P Micro", "value": "MES=F"},
    {"label": "GC · Gold", "value": "GC=F"},
    {"label": "CL · Oil", "value": "CL=F"},
    {"label": "NG · Gas", "value": "NG=F"},
]
OTHER_TICKERS = [
    {"label": "NQ · Nasdaq", "value": "NQ=F"},
    {"label": "ES · S&P", "value": "ES=F"},
    {"label": "MYM · Dow Micro", "value": "MYM=F"},
]
TICKER_OPTS = [
    {"label": "★ Favoriti", "options": FAVORITE_TICKERS},
    {"label": "Ostali", "options": OTHER_TICKERS},
]


def build_market(ticker: str, days: int) -> dict:
    df5 = load_5m(ticker, days)
    data = {"5min": df5, "1h": resample(df5, "1h")}
    h1 = data["1h"].index
    gaps = pd.date_range(h1[0], h1[-1], freq="1h").difference(h1).tolist() if len(h1) > 1 else []
    return {
        "ticker": ticker,
        "days": days,
        "df5": df5,
        "data": data,
        "zones": get_zones(df5),
        "trend": {tf: analyze(df5, tf, n=SWING_N[tf]) for tf in data},
        "atr": {tf: last_atr(data[tf], ATR_PERIOD) for tf in data},
        "gaps": gaps,
    }


MARKET = None
TRADES = load_trades()
BANK = recalc(TRADES) if TRADES else load_bank()


def empty_fig(msg: str = "Odaberi ticker i klikni Učitaj") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#14161a", plot_bgcolor="#14161a",
        margin=dict(l=8, r=72, t=40, b=62),
        title=dict(text=msg, font=dict(size=14, color="#888"), x=0.5, xanchor="center"),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def rgba(color: str, a: float) -> str:
    c = color.lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{a:.2f})"


def build_fig(market, tf, layers, store, trades):
    df = market["data"][tf]
    df5 = market["df5"]
    zones = market["zones"]
    trend = market["trend"]
    gaps = market["gaps"]
    view_days = market["days"]
    x = df.index
    x_end = x[-1]
    fig = go.Figure(go.Candlestick(
        x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=GREEN, decreasing_line_color=RED, showlegend=False))

    if "zones" in layers:
        for z in zones.itertuples():
            col = {"support": GREEN, "resistance": RED}.get(z.kind, AMBER)
            x0 = z.start
            fig.add_trace(go.Scatter(
                x=[x0, x_end, x_end, x0, x0], y=[z.low, z.low, z.high, z.high, z.low],
                mode="lines", line=dict(color=rgba(col, 0.8), width=1),
                fill="toself", fillcolor=rgba(col, min(0.12 + 0.04 * z.touches, 0.45)),
                hoveron="fills", hoverinfo="text", showlegend=False,
                text=f"{z.kind} {z.low:.2f} - {z.high:.2f} ({z.touches} dodira)"))

    trend_shapes, trend_ann = [], []
    if "swings" in layers:
        sw_full = trend[tf]["swings"]
        sw = sw_full.dropna(subset=["label"])
        for kind, pos, sym in (("H", "top center", "triangle-down"), ("L", "bottom center", "triangle-up")):
            s = sw[sw["kind"] == kind]
            cols = [GREEN if lb in ("HH", "HL") else RED if lb in ("LH", "LL") else "#9e9e9e" for lb in s["label"]]
            fig.add_trace(go.Scatter(
                x=s.index, y=s["price"], mode="markers+text", text=s["label"],
                textposition=pos, textfont=dict(color=cols, size=11),
                marker=dict(symbol=sym, size=6, color=cols), hoverinfo="skip", showlegend=False))
        trend_shapes, trend_ann = trend_overlays(sw_full, x_end)

    last = df.iloc[-1]
    last_close = float(last["Close"])
    last_col = GREEN if last_close >= last["Open"] else RED
    price_line = dict(
        type="line", xref="paper", yref="y", x0=0, x1=1,
        y0=last_close, y1=last_close, layer="above", name="lastprice",
        line=dict(color=last_col, width=1, dash="dot"),
    )
    price_ann = dict(
        x=1.0, y=last_close, xref="paper", yref="y", xanchor="left",
        text=f"  {last_close:.2f}  ", showarrow=False, name="lastprice",
        bgcolor=last_col, bordercolor=last_col, font=dict(color="#fff", size=11),
    )

    tl_shapes, tl_ann = timeline_overlays(df5.index[0], df5.index[-1])
    sess_shapes, sess_ann = [], []
    if "sessions" in layers:
        sess_shapes, sess_ann = session_overlays(df5.index[0], df5.index[-1])

    trade_shapes, trade_ann = [], []
    if "trades" in layers:
        for t in trades:
            sh, an = chart_overlays(t, x_end)
            trade_shapes.extend(sh)
            trade_ann.extend(an)

    atr_s = atr_series(df, ATR_PERIOD).dropna().tail(ATR_PERIOD)
    if len(atr_s) >= 2:
        fig.add_trace(go.Scatter(
            x=atr_s.index, y=atr_s.values, mode="lines",
            line=dict(color="#ba68c8", width=1.5), name="atr",
            yaxis="y2", hovertemplate="ATR: %{y:.2f} pts<extra></extra>", showlegend=False,
        ))

    x_start = df5.index[-1] - pd.Timedelta(days=view_days)
    yaxis = dict(side="right", fixedrange=False, gridcolor="#22252b",
                 tickformat=".2f", showexponent="none")
    layout_extra = {}
    if len(atr_s) >= 2:
        yaxis["domain"] = [0.22, 1.0]
        layout_extra["yaxis2"] = dict(
            title=dict(text=f"ATR({ATR_PERIOD})", font=dict(size=10, color="#ba68c8")),
            domain=[0.08, 0.18], anchor="x", overlaying=None, side="right",
            showgrid=True, gridcolor="#22252b", tickformat=".1f", fixedrange=True,
            zeroline=False,
        )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#14161a", plot_bgcolor="#14161a",
        margin=dict(l=8, r=72, t=8, b=62), dragmode="pan", uirevision="keep",
        xaxis=dict(type="date", rangeslider_visible=False,
                   range=[x_start, df5.index[-1] + pd.Timedelta(hours=2)],
                   tickformatstops=X_TICKS, rangebreaks=[dict(values=gaps, dvalue=3_600_000)],
                   gridcolor="#22252b"),
        yaxis=yaxis,
        **layout_extra,
        newshape=dict(line=dict(color="#ffeb3b", width=2)),
        shapes=tl_shapes + sess_shapes + trend_shapes + trade_shapes + [price_line] + store.get("shapes", []),
        annotations=tl_ann + sess_ann + trend_ann + trade_ann + [price_ann] + store.get("annotations", []),
    )
    return fig


def trend_badges(trend, atr_map=None, tf="5min"):
    out = []
    for tframe, r in trend.items():
        arrow = {"UP": "▲", "DOWN": "▼", "RANGE": "◆"}[r["trend"]]
        col = {"UP": GREEN, "DOWN": RED, "RANGE": AMBER}[r["trend"]]
        txt = f"{tframe}: {arrow} {r['trend']}"
        if r["invalidation"] is not None:
            txt += f" · invalidacija {r['invalidation']:.2f}" + (" (probijena)" if r["broken"] else "")
        out.append(html.Span(txt, style={"color": col, "marginRight": "18px"}))
    if atr_map:
        v = atr_map.get(tf)
        if v is not None:
            out.append(html.Span(f"ATR({ATR_PERIOD}): {v:.1f} pts", style={"color": "#ba68c8", "marginRight": "18px"}))
    return out


def bank_panel(bank):
    r = risk_eur(bank)
    pnl = bank["balance_eur"] - bank["initial_eur"]
    sign = "+" if pnl >= 0 else ""
    return html.Div([
        html.Span(f"Banka: {bank['balance_eur']:.2f} € ", style={"color": GREEN if pnl >= 0 else RED}),
        html.Span(f"({sign}{pnl:.2f} €) · ", style={"color": "#aaa"}),
        html.Span(f"risk {bank['risk_pct']}% = {r:.2f} € · margina {bank['margin']}x",
                  style={"color": "#aaa"}),
    ])


BTN_SM = {"padding": "2px 8px", "fontSize": "12px", "cursor": "pointer", "marginRight": "4px",
          "border": "none", "borderRadius": "3px"}


def trades_table(trades):
    if not trades:
        return html.P("Nema trejdova.", style={"color": "#888"})
    header = html.Tr([html.Th(c) for c in
                      ["Datum", "ID", "Test", "Dir", "Entry", "SL pts", "TP pts", "Status", "PnL pts", "PnL €", ""]])
    rows = []
    for t in sort_by_opened(trades):
        st = t.get("status", "open")
        col = GREEN if st == "win" else RED if st == "loss" else "#aaa"
        pnl_pts = t.get("pnl_points")
        pnl_e = t.get("pnl_eur")
        tid = t["id"]
        rows.append(html.Tr([
            html.Td(t["opened_at"][:16], style={"whiteSpace": "nowrap"}),
            html.Td(tid), html.Td("✓" if t.get("test") else ""),
            html.Td(t["direction"]), html.Td(f"{t['entry_price']:.2f}"),
            html.Td(f"{abs(t['sl_points']):.0f}"), html.Td(f"{abs(t['tp_points']):.0f}"),
            html.Td(st, style={"color": col}),
            html.Td(f"{pnl_pts:.0f}" if pnl_pts is not None else "—"),
            html.Td(f"{pnl_e:.2f}" if pnl_e is not None else "—", style={"color": col}),
            html.Td([
                html.Button("Uredi", id={"type": "edit-trade", "index": tid}, n_clicks=0,
                            style={**BTN_SM, "background": "#444", "color": "#ccc"}),
                html.Button("Obriši", id={"type": "delete-trade", "index": tid}, n_clicks=0,
                            style={**BTN_SM, "background": "#5d2a2a", "color": "#ef5350"}),
            ], style={"whiteSpace": "nowrap"}),
        ]))
    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"},
    )


def edit_trade_modal():
    btn = {"padding": "6px 14px", "borderRadius": "4px", "border": "none", "cursor": "pointer", "fontSize": "14px"}
    return html.Div(id="edit-modal", style=MODAL_HIDE, children=[
        html.Div(style=MODAL_BOX, children=[
            html.H3("Uredi trejd", style={"margin": "0 0 16px", "fontSize": "18px"}),
            html.Div(id="edit-trade-label", style={"color": "#888", "fontSize": "12px", "marginBottom": "12px"}),
            html.Div(style=MODAL_ROW, children=[
                html.Label("Entry", style=MODAL_LBL),
                dcc.Input(id="edit-entry", type="number", style={**INP, "width": "100%"}),
            ]),
            html.Div(style=MODAL_ROW, children=[
                html.Label("Smjer", style=MODAL_LBL),
                dcc.RadioItems(
                    id="edit-direction", value="long",
                    options=[{"label": " Long", "value": "long"}, {"label": " Short", "value": "short"}],
                    labelStyle={"marginRight": "16px", "cursor": "pointer"},
                    inputStyle={"marginRight": "4px"},
                ),
            ]),
            html.Div(style=MODAL_ROW, children=[
                html.Label("Stop loss (bodovi)", style=MODAL_LBL),
                dcc.Input(id="edit-sl-pts", type="number", min=1, style={**INP, "width": "100%"}),
            ]),
            html.Div(style=MODAL_ROW, children=[
                html.Label("Take profit (bodovi)", style=MODAL_LBL),
                dcc.Input(id="edit-tp-pts", type="number", min=1, style={**INP, "width": "100%"}),
            ]),
            html.Div(id="edit-preview", style={"color": "#aaa", "fontSize": "13px", "marginBottom": "12px"}),
            html.Div(style=MODAL_ROW, children=[
                html.Label("Iznos €", style=MODAL_LBL),
                dcc.Input(id="edit-size", type="number", style={**INP, "width": "100%"}),
            ]),
            html.Div(style=MODAL_ROW, children=[
                html.Label("Margina", style=MODAL_LBL),
                dcc.Input(id="edit-margin", type="number", min=1, style={**INP, "width": "100%"}),
            ]),
            html.Div(style={"display": "flex", "gap": "10px", "marginTop": "16px", "justifyContent": "flex-end"}, children=[
                html.Button("Odustani", id="edit-cancel", n_clicks=0,
                            style={**btn, "background": "#444", "color": "#ccc"}),
                html.Button("Spremi", id="edit-save", n_clicks=0,
                            style={**btn, "background": GREEN, "color": "#fff"}),
            ]),
        ]),
    ])


def trade_modal():
    btn = {"padding": "6px 14px", "borderRadius": "4px", "border": "none", "cursor": "pointer", "fontSize": "14px"}
    return html.Div(id="trade-modal", style=MODAL_HIDE, children=[
        html.Div(style=MODAL_BOX, children=[
            html.H3("Novi trejd", style={"margin": "0 0 16px", "fontSize": "18px"}),
            html.Div(style=MODAL_ROW, children=[
                html.Label("Entry (trenutna cijena)", style=MODAL_LBL),
                dcc.Input(id="modal-entry", type="number", disabled=True, style={**INP, "width": "100%", "opacity": 0.85}),
            ]),
            html.Div(style=MODAL_ROW, children=[
                html.Label("Smjer", style=MODAL_LBL),
                dcc.RadioItems(
                    id="modal-direction", value="long",
                    options=[{"label": " Long", "value": "long"}, {"label": " Short", "value": "short"}],
                    labelStyle={"marginRight": "16px", "cursor": "pointer"},
                    inputStyle={"marginRight": "4px"},
                ),
            ]),
            html.Div(style=MODAL_ROW, children=[
                html.Label("Stop loss (bodovi)", style=MODAL_LBL),
                dcc.Input(id="modal-sl-pts", type="number", value=30, min=1, style={**INP, "width": "100%"}),
            ]),
            html.Div(id="modal-sl-hint", style={"color": "#ba68c8", "fontSize": "12px", "margin": "-4px 0 8px"}),
            html.Div(style=MODAL_ROW, children=[
                html.Label("Take profit (bodovi)", style=MODAL_LBL),
                dcc.Input(id="modal-tp-pts", type="number", value=60, min=1, style={**INP, "width": "100%"}),
            ]),
            html.Div(id="modal-preview", style={"color": "#aaa", "fontSize": "13px", "marginBottom": "12px"}),
            html.Div(style=MODAL_ROW, children=[
                html.Label("Iznos €", style=MODAL_LBL),
                dcc.Input(id="modal-size", type="number", style={**INP, "width": "100%"}),
            ]),
            html.Div(style=MODAL_ROW, children=[
                html.Label("Margina", style=MODAL_LBL),
                dcc.Input(id="modal-margin", type="number", value=1, min=1, style={**INP, "width": "100%"}),
            ]),
            html.Div(style={"display": "flex", "gap": "10px", "marginTop": "16px", "justifyContent": "flex-end"}, children=[
                html.Button("Odustani", id="modal-cancel", n_clicks=0,
                            style={**btn, "background": "#444", "color": "#ccc"}),
                html.Button("Spremi trejd", id="modal-save", n_clicks=0,
                            style={**btn, "background": GREEN, "color": "#fff"}),
            ]),
        ]),
    ])


CONFIG = {
    "scrollZoom": True, "displaylogo": False, "doubleClick": "reset",
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "drawcircle", "eraseshape"],
    "edits": {"shapePosition": True, "annotationPosition": True, "annotationText": True},
}
BAR = {"display": "flex", "gap": "18px", "alignItems": "center", "flexWrap": "wrap", "padding": "6px 4px"}
ITEM = {"marginRight": "12px", "display": "inline-block"}


def cache_badge(ticker: str) -> html.Span:
    path = cache_path(ticker or DEFAULT_TICKER)
    if path.exists():
        kb = path.stat().st_size // 1024
        return html.Span(f"cache: {path.name} ({kb} KB)", style={"color": "#888"})
    return html.Span("nema cachea", style={"color": "#666"})


app = Dash(__name__)
app.layout = html.Div(
    style={"background": "#14161a", "color": "#d7dae0", "minHeight": "100vh", "padding": "6px",
           "fontFamily": "system-ui, sans-serif", "fontSize": "14px"},
    children=[
        dcc.Store(id="trades-store", data=TRADES),
        dcc.Store(id="bank-store", data=BANK),
        dcc.Store(id="market-store", data={"status": "idle"}),
        dcc.Store(id="store", data={}),
        dcc.Store(id="edit-trade-id", data=None),
        dcc.Store(id="d_fit"), dcc.Store(id="d_txt"), dcc.Store(id="d_clr"),
        dcc.Tabs(id="main-tabs", value="Graf", children=[
            dcc.Tab(label="Graf", value="Graf", children=[
                html.Div(style=BAR, children=[
                    dcc.Dropdown(id="ticker", value=DEFAULT_TICKER, options=TICKER_OPTS,
                                 clearable=False, style={"width": "168px"}),
                    dcc.Input(id="days", type="number", value=5, min=1, max=60, style={"width": "52px"}),
                    html.Label(" dana ", style={"marginRight": "8px"}),
                    html.Button("Učitaj", id="load-data", n_clicks=0,
                                style={"marginRight": "4px", "padding": "2px 10px", "cursor": "pointer"}),
                    html.Button("×", id="clear-cache", n_clicks=0, title="Obriši CSV cache",
                                style={"marginRight": "12px", "padding": "0 8px", "cursor": "pointer",
                                       "fontSize": "18px", "lineHeight": "1", "color": "#ef5350"}),
                    html.Span(id="cache-badge", children=cache_badge(DEFAULT_TICKER),
                              style={"marginRight": "12px", "fontSize": "12px"}),
                    dcc.RadioItems(id="tf", value="5min", labelStyle=ITEM,
                                   options=[{"label": "5 min", "value": "5min"}, {"label": "1 h", "value": "1h"}]),
                    dcc.Checklist(id="layers", value=["zones", "swings", "sessions", "trades"], labelStyle=ITEM,
                                  options=[
                                      {"label": "S/R zone (1h)", "value": "zones"},
                                      {"label": "HH/HL/LH/LL", "value": "swings"},
                                      {"label": "Sesije (x-os)", "value": "sessions"},
                                      {"label": "Trejdovi", "value": "trades"},
                                  ]),
                    html.Button("Novi trejd", id="new-trade-btn", n_clicks=0,
                                style={"padding": "2px 12px", "cursor": "pointer", "marginRight": "8px"}),
                    dcc.Input(id="txt", type="text", placeholder="tekst bilješke", style={"width": "150px"}),
                    html.Button("Dodaj tekst", id="add_txt"),
                    html.Button("Očisti crteže", id="clear"),
                    html.Div(id="bank-badge", children=bank_panel(BANK)),
                    html.Div(id="trend-badges", children=html.Span("—", style={"color": "#666"})),
                ]),
                trade_modal(),
                edit_trade_modal(),
                dcc.Graph(id="chart", config=CONFIG, style={"height": "78vh"}),
            ]),
            dcc.Tab(label="Trejdovi", value="Trejdovi", children=[
                html.Div(style={**PANEL, "margin": "12px 0"}, children=[
                    html.H3("Banka", style={"margin": "0 0 8px"}),
                    html.Div(id="bank-tab-info", children=bank_panel(BANK)),
                    html.Div([
                        html.Label("Početni € "), dcc.Input(id="bank-initial", type="number", value=BANK["initial_eur"], style=INP),
                        html.Label("Risk % "), dcc.Input(id="bank-risk", type="number", value=BANK["risk_pct"], style=INP),
                        html.Label("Margina "), dcc.Input(id="bank-margin", type="number", value=BANK["margin"], style=INP),
                        html.Button("Spremi banku", id="save-bank", n_clicks=0),
                    ], style={"marginTop": "10px"}),
                ]),
                html.Div(style={**PANEL, "margin": "12px 0", "overflowX": "auto"}, children=[
                    html.H3("Povijest trejdova", style={"margin": "0 0 8px"}),
                    html.Div(id="trades-table", children=trades_table(TRADES)),
                ]),
            ]),
        ]),
    ],
)


@app.callback(
    Output("cache-badge", "children"),
    Input("ticker", "value"),
    Input("clear-cache", "n_clicks"),
    Input("load-data", "n_clicks"),
)
def update_cache_badge(ticker, *_):
    return cache_badge(ticker or DEFAULT_TICKER)


@app.callback(
    Output("market-store", "data", allow_duplicate=True),
    Input("clear-cache", "n_clicks"),
    State("ticker", "value"),
    prevent_initial_call=True,
)
def delete_cache(_, ticker):
    global MARKET
    if not _:
        return no_update
    clear_cache(ticker or DEFAULT_TICKER)
    MARKET = None
    return {"status": "idle"}


@app.callback(
    Output("market-store", "data"),
    Output("trades-store", "data"),
    Output("bank-store", "data"),
    Output("bank-badge", "children"),
    Output("bank-tab-info", "children"),
    Output("trades-table", "children"),
    Input("load-data", "n_clicks"),
    State("ticker", "value"),
    State("days", "value"),
    prevent_initial_call=True,
)
def reload_market(_, ticker, days):
    global MARKET, TRADES, BANK
    if not _:
        return (no_update,) * 6
    ticker = ticker or DEFAULT_TICKER
    days = max(1, min(int(days or 5), 60))
    try:
        MARKET = build_market(ticker, days)
        if not TRADES_FILE.exists():
            TRADES = seed_test_trades(MARKET["df5"])
        else:
            TRADES = resolve_all(load_trades(), MARKET["df5"])
            save_trades(TRADES)
        BANK = recalc(TRADES)
        panel = bank_panel(BANK)
        meta = {"ticker": ticker, "days": days, "status": "ok", "rows": len(MARKET["df5"])}
        return meta, TRADES, BANK, panel, panel, trades_table(TRADES)
    except Exception as e:
        meta = {"ticker": ticker, "days": days, "status": "error", "msg": str(e)}
        panel = bank_panel(BANK)
        return meta, TRADES, BANK, panel, panel, trades_table(TRADES)


@app.callback(
    Output("chart", "figure"),
    Output("trend-badges", "children"),
    Input("market-store", "data"),
    Input("tf", "value"), Input("layers", "value"),
    Input("trades-store", "data"),
    State("store", "data"),
)
def update_chart(meta, tf, layers, trades, store):
    if not meta or meta.get("status") == "idle" or MARKET is None:
        return empty_fig(), html.Span("—", style={"color": "#666"})
    if meta.get("status") == "error":
        return empty_fig(meta.get("msg", "Greška")), html.Span(meta.get("msg", "Greška"), style={"color": RED})
    tf = tf or "5min"
    return (
        build_fig(MARKET, tf, layers or [], store or {}, trades or []),
        trend_badges(MARKET["trend"], MARKET.get("atr"), tf),
    )


def _atr_suggest(tf: str) -> tuple[int, int, str]:
    if MARKET is None:
        return 30, 60, ""
    df = MARKET["data"].get(tf or "5min", MARKET["df5"])
    sl = suggest_sl(df, ATR_PERIOD)
    v = last_atr(df, ATR_PERIOD)
    hint = f"Sugestija SL: ATR({ATR_PERIOD}) = {v:.1f} pts" if v is not None else ""
    return sl, sl * 2, hint


def _modal_preview(entry, direction, sl_pts, tp_pts):
    if entry is None or sl_pts is None or tp_pts is None:
        return "—"
    entry, sl_pts, tp_pts = float(entry), abs(float(sl_pts)), abs(float(tp_pts))
    if direction == "long":
        sl, tp = entry - sl_pts, entry + tp_pts
    else:
        sl, tp = entry + sl_pts, entry - tp_pts
    return f"SL {sl:.2f} · TP {tp:.2f}"


@app.callback(
    Output("trade-modal", "style"),
    Output("modal-entry", "value"),
    Output("modal-direction", "value"),
    Output("modal-sl-pts", "value"),
    Output("modal-tp-pts", "value"),
    Output("modal-size", "value"),
    Output("modal-margin", "value"),
    Output("modal-preview", "children"),
    Output("modal-sl-hint", "children"),
    Input("new-trade-btn", "n_clicks"),
    Input("modal-cancel", "n_clicks"),
    Input("modal-save", "n_clicks"),
    Input("modal-entry", "value"),
    Input("modal-direction", "value"),
    Input("modal-sl-pts", "value"),
    Input("modal-tp-pts", "value"),
    State("bank-store", "data"),
    State("tf", "value"),
    prevent_initial_call=True,
)
def toggle_trade_modal(open_clicks, cancel_clicks, save_clicks, entry, direction, sl_pts, tp_pts, bank, tf):
    tid = ctx.triggered_id
    if tid == "modal-cancel" or tid == "modal-save":
        return MODAL_HIDE, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    if tid == "new-trade-btn":
        if MARKET is None:
            return MODAL_HIDE, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        bank = bank or load_bank()
        entry = float(MARKET["df5"].iloc[-1]["Close"])
        sl, tp, hint = _atr_suggest(tf)
        preview = _modal_preview(entry, "long", sl, tp)
        return MODAL_SHOW, entry, "long", sl, tp, bank["balance_eur"], bank["margin"], preview, hint
    preview = _modal_preview(entry, direction, sl_pts, tp_pts)
    return no_update, no_update, no_update, no_update, no_update, no_update, no_update, preview, no_update


@app.callback(
    Output("trades-store", "data", allow_duplicate=True),
    Output("bank-store", "data", allow_duplicate=True),
    Output("trades-table", "children", allow_duplicate=True),
    Output("bank-badge", "children", allow_duplicate=True),
    Output("bank-tab-info", "children", allow_duplicate=True),
    Input("modal-save", "n_clicks"),
    State("modal-direction", "value"),
    State("modal-sl-pts", "value"),
    State("modal-tp-pts", "value"),
    State("modal-size", "value"),
    State("modal-margin", "value"),
    prevent_initial_call=True,
)
def save_trade_modal(n, direction, sl_pts, tp_pts, size, margin):
    if MARKET is None or not n or not direction or sl_pts is None or tp_pts is None:
        return (no_update,) * 5
    sl_pts, tp_pts = abs(float(sl_pts)), abs(float(tp_pts))
    if sl_pts < 1 or tp_pts < 1:
        return (no_update,) * 5
    draft = make_draft_manual(
        MARKET["df5"], direction, sl_pts, tp_pts,
        float(size or load_bank()["balance_eur"]),
        int(margin or load_bank()["margin"]),
    )
    add_trade(draft, MARKET["df5"])
    return _trade_refresh(MARKET["df5"])


@app.callback(
    Output("bank-store", "data", allow_duplicate=True),
    Output("bank-badge", "children", allow_duplicate=True),
    Output("bank-tab-info", "children", allow_duplicate=True),
    Input("save-bank", "n_clicks"),
    State("bank-initial", "value"), State("bank-risk", "value"), State("bank-margin", "value"),
    State("trades-store", "data"),
    prevent_initial_call=True,
)
def save_bank_cb(n, initial, risk, margin, trades):
    if not n:
        return no_update, no_update, no_update
    bank = update_settings(initial_eur=initial, risk_pct=risk, margin=margin)
    bank = recalc(trades or [])
    panel = bank_panel(bank)
    return bank, panel, panel


def _trade_refresh(df5=None):
    """Učitaj trejdove iz trades.json i osvježi UI."""
    global TRADES, BANK
    TRADES = sort_by_opened(load_trades())
    if df5 is not None and TRADES:
        TRADES = sort_by_opened(resolve_all(TRADES, df5))
        save_trades(TRADES)
    BANK = recalc(TRADES)
    panel = bank_panel(BANK)
    return TRADES, BANK, trades_table(TRADES), panel, panel


@app.callback(
    Output("edit-modal", "style"),
    Output("edit-trade-id", "data"),
    Output("edit-trade-label", "children"),
    Output("edit-entry", "value"),
    Output("edit-direction", "value"),
    Output("edit-sl-pts", "value"),
    Output("edit-tp-pts", "value"),
    Output("edit-size", "value"),
    Output("edit-margin", "value"),
    Output("edit-preview", "children"),
    Input({"type": "edit-trade", "index": ALL}, "n_clicks"),
    Input("edit-cancel", "n_clicks"),
    Input("edit-save", "n_clicks"),
    Input("edit-entry", "value"),
    Input("edit-direction", "value"),
    Input("edit-sl-pts", "value"),
    Input("edit-tp-pts", "value"),
    State("trades-store", "data"),
    State("edit-trade-id", "data"),
    prevent_initial_call=True,
)
def toggle_edit_modal(_clicks, _cancel, _save, entry, direction, sl_pts, tp_pts, trades, trade_id):
    tid = ctx.triggered_id
    if tid in ("edit-cancel", "edit-save"):
        return MODAL_HIDE, None, "", no_update, no_update, no_update, no_update, no_update, no_update, no_update
    if isinstance(tid, dict) and tid.get("type") == "edit-trade":
        if not any(_clicks):
            return (no_update,) * 10
        trade_id = tid["index"]
        t = get_trade(trade_id)
        if not t:
            return (no_update,) * 10
        sl_abs, tp_abs = abs(t["sl_points"]), abs(t["tp_points"])
        preview = _modal_preview(t["entry_price"], t["direction"], sl_abs, tp_abs)
        label = f"ID {trade_id} · {t['opened_at'][:16]}"
        return (
            MODAL_SHOW, trade_id, label, t["entry_price"], t["direction"],
            sl_abs, tp_abs, t["size_eur"], t["margin"], preview,
        )
    preview = _modal_preview(entry, direction, sl_pts, tp_pts)
    return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, preview


@app.callback(
    Output("trades-store", "data", allow_duplicate=True),
    Output("bank-store", "data", allow_duplicate=True),
    Output("trades-table", "children", allow_duplicate=True),
    Output("bank-badge", "children", allow_duplicate=True),
    Output("bank-tab-info", "children", allow_duplicate=True),
    Input("edit-save", "n_clicks"),
    State("edit-trade-id", "data"),
    State("edit-entry", "value"),
    State("edit-direction", "value"),
    State("edit-sl-pts", "value"),
    State("edit-tp-pts", "value"),
    State("edit-size", "value"),
    State("edit-margin", "value"),
    prevent_initial_call=True,
)
def save_edit_trade(n, trade_id, entry, direction, sl_pts, tp_pts, size, margin):
    if not n or not trade_id or entry is None or not direction or sl_pts is None or tp_pts is None:
        return (no_update,) * 5
    sl_pts, tp_pts = abs(float(sl_pts)), abs(float(tp_pts))
    if sl_pts < 1 or tp_pts < 1:
        return (no_update,) * 5
    df5 = MARKET["df5"] if MARKET is not None else None
    update_trade(
        trade_id,
        entry_price=float(entry),
        direction=direction,
        sl_pts_abs=sl_pts,
        tp_pts_abs=tp_pts,
        size_eur=float(size or load_bank()["balance_eur"]),
        margin=int(margin or load_bank()["margin"]),
        df5=df5,
    )
    return _trade_refresh(df5)


@app.callback(
    Output("trades-store", "data", allow_duplicate=True),
    Output("bank-store", "data", allow_duplicate=True),
    Output("trades-table", "children", allow_duplicate=True),
    Output("bank-badge", "children", allow_duplicate=True),
    Output("bank-tab-info", "children", allow_duplicate=True),
    Input({"type": "delete-trade", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def delete_trade_cb(clicks):
    if not any(clicks):
        return (no_update,) * 5
    tid = ctx.triggered_id
    if not isinstance(tid, dict) or tid.get("type") != "delete-trade":
        return (no_update,) * 5
    df5 = MARKET["df5"] if MARKET is not None else None
    delete_trade(tid["index"], df5)
    return _trade_refresh(df5)


@app.callback(
    Output("trades-store", "data", allow_duplicate=True),
    Output("bank-store", "data", allow_duplicate=True),
    Output("trades-table", "children", allow_duplicate=True),
    Output("bank-badge", "children", allow_duplicate=True),
    Output("bank-tab-info", "children", allow_duplicate=True),
    Input("main-tabs", "value"),
    prevent_initial_call=True,
)
def sync_trades_from_json(tab):
    if tab != "Trejdovi":
        return (no_update,) * 5
    df5 = MARKET["df5"] if MARKET is not None else None
    return _trade_refresh(df5)


clientside_callback(
    """function(rd) {
        if (!rd) return window.dash_clientside.no_update;
        const keys = Object.keys(rd);
        if (!keys.some(k => k.startsWith('shapes[') || k.startsWith('annotations[')))
            return window.dash_clientside.no_update;
        const gd = document.querySelector('#chart .js-plotly-plot');
        if (!gd) return window.dash_clientside.no_update;
        const locked = ['timeline','session','trendstrip','lastprice','trade','draft','draft-sl','draft-tp'];
        const isLocked = n => locked.includes(n);
        if (gd._lockedSnap) {
            const fix = {};
            for (const k of keys) {
                const sm = k.match(/^shapes\\[(\\d+)\\]\\.(\\w+)$/);
                if (sm) {
                    const i = +sm[1], s = (gd._fullLayout.shapes || [])[i];
                    if (s && isLocked(s.name) && gd._lockedSnap.shapes[i])
                        fix[k] = gd._lockedSnap.shapes[i][sm[2]];
                }
                const am = k.match(/^annotations\\[(\\d+)\\]\\.(\\w+)$/);
                if (am) {
                    const i = +am[1], a = (gd._fullLayout.annotations || [])[i];
                    if (a && isLocked(a.name) && gd._lockedSnap.annotations[i])
                        fix[k] = gd._lockedSnap.annotations[i][am[2]];
                }
            }
            if (Object.keys(fix).length) Plotly.relayout(gd, fix);
        }
        const skip = n => !isLocked(n);
        const sh = (gd.layout.shapes || []).filter(s => skip(s.name));
        const an = (gd.layout.annotations || []).filter(a => skip(a.name));
        return JSON.parse(JSON.stringify({shapes: sh, annotations: an}));
    }""",
    Output("store", "data"), Input("chart", "relayoutData"), prevent_initial_call=True)

clientside_callback(
    """function(n, txt) {
        if (!n) return window.dash_clientside.no_update;
        const gd = document.querySelector('#chart .js-plotly-plot');
        const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
        const ann = (gd.layout.annotations || []).slice();
        ann.push({x: xa.p2d(xa._length / 2), y: ya.p2d(ya._length / 2), text: txt || 'tekst', showarrow: false,
                  font: {color: '#ffeb3b', size: 16}, bgcolor: 'rgba(0,0,0,0.45)'});
        Plotly.relayout(gd, {annotations: ann});
        return window.dash_clientside.no_update;
    }""",
    Output("d_txt", "data"), Input("add_txt", "n_clicks"), State("txt", "value"))

clientside_callback(
    """function(n) {
        if (!n) return window.dash_clientside.no_update;
        Plotly.relayout(document.querySelector('#chart .js-plotly-plot'), {shapes: [], annotations: []});
        return window.dash_clientside.no_update;
    }""",
    Output("d_clr", "data"), Input("clear", "n_clicks"))

clientside_callback(
    """function(fig) {
        setTimeout(function() {
            const gd = document.querySelector('#chart .js-plotly-plot');
            if (!gd || !gd._fullLayout || !gd.data.length) return;
            gd._lockedSnap = {
                shapes: (gd.layout.shapes || []).map(s => JSON.parse(JSON.stringify(s))),
                annotations: (gd.layout.annotations || []).map(a => JSON.parse(JSON.stringify(a))),
            };
            const tr = gd.data[0], r = gd._fullLayout.xaxis.range;
            const t0 = new Date(r[0]).getTime(), t1 = new Date(r[1]).getTime(), span = t1 - t0;
            let lo = Infinity, hi = -Infinity;
            for (let i = 0; i < tr.x.length; i++) {
                const t = new Date(tr.x[i]).getTime();
                if (t >= t0 && t <= t1) { if (tr.low[i] < lo) lo = tr.low[i]; if (tr.high[i] > hi) hi = tr.high[i]; }
            }
            const upd = {};
            if (span <= 6 * 3600000) upd['xaxis.dtick'] = 300000;
            else if (span <= 24 * 3600000) upd['xaxis.dtick'] = 900000;
            else if (span <= 7 * 86400000) upd['xaxis.dtick'] = 3600000;
            else upd['xaxis.dtick'] = null;
            if (isFinite(lo)) {
                const pad = (hi - lo) * 0.06 || 1;
                upd['yaxis.range'] = [lo - pad, hi + pad];
            }
            if (Object.keys(upd).length) Plotly.relayout(gd, upd);
            if (!gd._wheelBound) {
                gd._wheelBound = true;
                gd.addEventListener('wheel', function(e) {
                    if (!gd._fullLayout || !gd.querySelector('.nsewdrag')) return;
                    const fl = gd._fullLayout;
                    const plot = gd.querySelector('.nsewdrag');
                    const bb = plot.getBoundingClientRect();
                    if (e.clientX < bb.left || e.clientX > bb.right || e.clientY < bb.top || e.clientY > bb.bottom) return;
                    e.preventDefault();
                    const f = Math.pow(1.08, e.deltaY > 0 ? 1 : -1);
                    const xa = fl.xaxis, ya = fl.yaxis;
                    const xp = e.clientX - bb.left, yp = bb.bottom - e.clientY;
                    const xv = xa.p2d(xa._offset + xp);
                    const yv = ya.p2d(ya._offset + yp);
                    function zr(rng, c, fac, isDate) {
                        let a = isDate ? new Date(rng[0]).getTime() : rng[0];
                        let b = isDate ? new Date(rng[1]).getTime() : rng[1];
                        let cv = isDate ? new Date(c).getTime() : c;
                        const n0 = cv - (cv - a) * fac, n1 = cv + (b - cv) * fac;
                        return isDate ? [new Date(n0), new Date(n1)] : [n0, n1];
                    }
                    const isDate = xa.type === 'date';
                    Plotly.relayout(gd, {
                        'xaxis.range': zr(xa.range, xv, f, isDate),
                        'yaxis.range': zr(ya.range, yv, f, false),
                    });
                }, {passive: false});
            }
        }, 150);
        return window.dash_clientside.no_update;
    }""",
    Output("d_fit", "data"), Input("chart", "figure"))

clientside_callback(
    """function(rd) {
        if (!rd) return window.dash_clientside.no_update;
        setTimeout(function() {
            const gd = document.querySelector('#chart .js-plotly-plot');
            if (!gd || !gd._fullLayout) return;
            const r = gd._fullLayout.xaxis.range;
            const span = new Date(r[1]).getTime() - new Date(r[0]).getTime();
            const upd = {};
            if (span <= 6 * 3600000) upd['xaxis.dtick'] = 300000;
            else if (span <= 24 * 3600000) upd['xaxis.dtick'] = 900000;
            else if (span <= 7 * 86400000) upd['xaxis.dtick'] = 3600000;
            else upd['xaxis.dtick'] = null;
            if (Object.keys(upd).length) Plotly.relayout(gd, upd);
        }, 50);
        return window.dash_clientside.no_update;
    }""",
    Output("d_fit", "data", allow_duplicate=True), Input("chart", "relayoutData"), prevent_initial_call=True)


if __name__ == "__main__":
    app.run(debug=False)
