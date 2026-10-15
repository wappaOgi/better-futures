"""UI: Dash + Plotly. Pokretanje: python ui.py  ->  http://127.0.0.1:8050"""
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, clientside_callback, dcc, html, no_update

from bank import load as load_bank, recalc, risk_eur, save as save_bank, update_settings
from data import DEFAULT_TICKER, load_5m, resample
from sessions import overlays as session_overlays
from trades import (
    add_trade, chart_overlays, draft_overlays, load as load_trades,
    make_draft, seed_test_trades, update_draft_levels,
)
from trend import analyze
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
TICKER_OPTS = [
    {"label": "MNQ", "value": "MNQ=F"},
    {"label": "NQ", "value": "NQ=F"},
    {"label": "MES", "value": "MES=F"},
    {"label": "ES", "value": "ES=F"},
    {"label": "MYM", "value": "MYM=F"},
    {"label": "GC", "value": "GC=F"},
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
        "gaps": gaps,
    }


MARKET = build_market(DEFAULT_TICKER, 5)
TRADES = seed_test_trades(MARKET["df5"])
BANK = recalc(TRADES)


def rgba(color: str, a: float) -> str:
    c = color.lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{a:.2f})"


def build_fig(market, tf, layers, store, trades, draft):
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

    if "swings" in layers:
        sw = trend[tf]["swings"].dropna(subset=["label"])
        for kind, pos, sym in (("H", "top center", "triangle-down"), ("L", "bottom center", "triangle-up")):
            s = sw[sw["kind"] == kind]
            cols = [GREEN if lb in ("HH", "HL") else RED if lb in ("LH", "LL") else "#9e9e9e" for lb in s["label"]]
            fig.add_trace(go.Scatter(
                x=s.index, y=s["price"], mode="markers+text", text=s["label"],
                textposition=pos, textfont=dict(color=cols, size=11),
                marker=dict(symbol=sym, size=6, color=cols), hoverinfo="skip", showlegend=False))

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

    sess_shapes, sess_ann = [], []
    if "sessions" in layers:
        sess_shapes, sess_ann = session_overlays(df5.index[0], df5.index[-1])

    trade_shapes, trade_ann = [], []
    if "trades" in layers:
        for t in trades:
            sh, an = chart_overlays(t, x_end)
            trade_shapes.extend(sh)
            trade_ann.extend(an)

    draft_sh, draft_an = draft_overlays(draft, x_end) if draft else ([], [])

    x_start = df5.index[-1] - pd.Timedelta(days=view_days)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#14161a", plot_bgcolor="#14161a",
        margin=dict(l=8, r=72, t=8, b=8), dragmode="pan", uirevision="keep",
        xaxis=dict(type="date", rangeslider_visible=False,
                   range=[x_start, df5.index[-1] + pd.Timedelta(hours=2)],
                   tickformatstops=X_TICKS, rangebreaks=[dict(values=gaps, dvalue=3_600_000)],
                   gridcolor="#22252b"),
        yaxis=dict(side="right", fixedrange=True, gridcolor="#22252b",
                   tickformat=".2f", showexponent="none"),
        newshape=dict(line=dict(color="#ffeb3b", width=2)),
        shapes=sess_shapes + trade_shapes + draft_sh + [price_line] + store.get("shapes", []),
        annotations=sess_ann + trade_ann + draft_an + [price_ann] + store.get("annotations", []),
    )
    return fig


def trend_badges(trend):
    out = []
    for tf, r in trend.items():
        arrow = {"UP": "▲", "DOWN": "▼", "RANGE": "◆"}[r["trend"]]
        col = {"UP": GREEN, "DOWN": RED, "RANGE": AMBER}[r["trend"]]
        txt = f"{tf}: {arrow} {r['trend']}"
        if r["invalidation"] is not None:
            txt += f" · invalidacija {r['invalidation']:.2f}" + (" (probijena)" if r["broken"] else "")
        out.append(html.Span(txt, style={"color": col, "marginRight": "18px"}))
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


def trades_table(trades):
    if not trades:
        return html.P("Nema trejdova.", style={"color": "#888"})
    header = html.Tr([html.Th(c) for c in
                      ["ID", "Test", "Dir", "Entry", "SL pts", "TP pts", "Status", "PnL pts", "PnL €", "Datum"]])
    rows = []
    for t in sorted(trades, key=lambda x: x["opened_at"], reverse=True):
        st = t.get("status", "open")
        col = GREEN if st == "win" else RED if st == "loss" else "#aaa"
        pnl_pts = t.get("pnl_points")
        pnl_e = t.get("pnl_eur")
        rows.append(html.Tr([
            html.Td(t["id"]), html.Td("✓" if t.get("test") else ""),
            html.Td(t["direction"]), html.Td(f"{t['entry_price']:.2f}"),
            html.Td(f"{t['sl_points']:.0f}"), html.Td(f"{t['tp_points']:.0f}"),
            html.Td(st, style={"color": col}),
            html.Td(f"{pnl_pts:.0f}" if pnl_pts is not None else "—"),
            html.Td(f"{pnl_e:.2f}" if pnl_e is not None else "—", style={"color": col}),
            html.Td(t["opened_at"][:16]),
        ]))
    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"},
    )


def trade_form(draft):
    if not draft or not draft.get("active"):
        return html.P("Uključi Trejd mod i klikni na graf.", style={"color": "#888", "margin": "4px 0"})
    d = draft["direction"]
    col = GREEN if d == "long" else RED
    return html.Div([
        html.Div([
            html.Strong("Entry: ", style={"color": col}),
            html.Span(f"{draft['entry_price']:.2f} · {d.upper()} · {draft['opened_at'][:16]}"),
        ]),
        html.Div([
            html.Label("SL "), dcc.Input(id="inp-sl", type="number", value=draft["stop_loss"], style=INP),
            html.Label("TP "), dcc.Input(id="inp-tp", type="number", value=draft["take_profit"], style=INP),
            html.Label("Iznos € "), dcc.Input(id="inp-size", type="number", value=draft["size_eur"], style=INP),
            html.Label("Margina "), dcc.Input(id="inp-margin", type="number", value=draft["margin"], style=INP),
        ], style={"marginTop": "8px"}),
        html.Div([
            html.Span(f"SL {draft['sl_points']:.0f} pts · TP {draft['tp_points']:.0f} pts · risk {draft['risk_eur']:.2f} €",
                      style={"color": "#aaa", "marginRight": "12px"}),
            html.Button("Spremi trejd ✓", id="save-trade", n_clicks=0,
                        style={"background": GREEN, "color": "#fff", "border": "none", "padding": "4px 12px",
                               "borderRadius": "4px", "cursor": "pointer"}),
            html.Button("Odustani", id="cancel-trade", n_clicks=0,
                        style={"marginLeft": "8px", "background": "#444", "color": "#ccc", "border": "none",
                               "padding": "4px 12px", "borderRadius": "4px", "cursor": "pointer"}),
        ], style={"marginTop": "10px"}),
        html.P("Povuci crveni (SL) i zeleni (TP) pravokutnik na grafu.", style={"color": "#666", "fontSize": "12px"}),
    ])


CONFIG = {
    "scrollZoom": True, "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "drawcircle", "eraseshape"],
    "edits": {"shapePosition": True, "annotationPosition": True, "annotationText": True},
}
BAR = {"display": "flex", "gap": "18px", "alignItems": "center", "flexWrap": "wrap", "padding": "6px 4px"}
ITEM = {"marginRight": "12px", "display": "inline-block"}

app = Dash(__name__)
app.layout = html.Div(
    style={"background": "#14161a", "color": "#d7dae0", "minHeight": "100vh", "padding": "6px",
           "fontFamily": "system-ui, sans-serif", "fontSize": "14px"},
    children=[
        dcc.Store(id="trades-store", data=TRADES),
        dcc.Store(id="bank-store", data=BANK),
        dcc.Store(id="market-store", data={"ticker": DEFAULT_TICKER, "days": 5, "status": "ok"}),
        dcc.Store(id="draft-trade", data={}),
        dcc.Store(id="store", data={}),
        dcc.Store(id="d_fit"), dcc.Store(id="d_txt"), dcc.Store(id="d_clr"),
        dcc.Tabs([
            dcc.Tab(label="Graf", children=[
                html.Div(style=BAR, children=[
                    dcc.Dropdown(id="ticker", value=DEFAULT_TICKER, options=TICKER_OPTS,
                                 clearable=False, style={"width": "100px"}),
                    dcc.Input(id="days", type="number", value=5, min=1, max=60, style={"width": "52px"}),
                    html.Label(" dana ", style={"marginRight": "8px"}),
                    html.Button("Učitaj", id="load-data", n_clicks=0,
                                style={"marginRight": "12px", "padding": "2px 10px", "cursor": "pointer"}),
                    dcc.RadioItems(id="tf", value="5min", labelStyle=ITEM,
                                   options=[{"label": "5 min", "value": "5min"}, {"label": "1 h", "value": "1h"}]),
                    dcc.Checklist(id="layers", value=["zones", "swings", "sessions", "trades"], labelStyle=ITEM,
                                  options=[
                                      {"label": "S/R zone (1h)", "value": "zones"},
                                      {"label": "HH/HL/LH/LL", "value": "swings"},
                                      {"label": "Sesije", "value": "sessions"},
                                      {"label": "Trejdovi", "value": "trades"},
                                  ]),
                    dcc.Checklist(id="trade-mode", value=[], labelStyle=ITEM,
                                  options=[{"label": "Trejd mod (klik na graf)", "value": "on"}]),
                    dcc.Input(id="txt", type="text", placeholder="tekst bilješke", style={"width": "150px"}),
                    html.Button("Dodaj tekst", id="add_txt"),
                    html.Button("Očisti crteže", id="clear"),
                    html.Div(id="bank-badge", children=bank_panel(BANK)),
                    html.Div(id="trend-badges", children=trend_badges(MARKET["trend"])),
                ]),
                html.Details(id="trade-expander", open=False, children=[
                    html.Summary("Novi trejd", style={"cursor": "pointer", "padding": "4px 0"}),
                    html.Div(id="trade-form", style=PANEL, children=trade_form({})),
                ]),
                dcc.Graph(id="chart", config=CONFIG, style={"height": "78vh"}),
            ]),
            dcc.Tab(label="Trejdovi", children=[
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
    Output("market-store", "data"),
    Input("load-data", "n_clicks"),
    State("ticker", "value"),
    State("days", "value"),
    prevent_initial_call=False,
)
def reload_market(_, ticker, days):
    global MARKET
    ticker = ticker or DEFAULT_TICKER
    days = max(1, min(int(days or 5), 60))
    try:
        MARKET = build_market(ticker, days)
        return {"ticker": ticker, "days": days, "status": "ok", "rows": len(MARKET["df5"])}
    except Exception as e:
        return {"ticker": ticker, "days": days, "status": "error", "msg": str(e)}


@app.callback(
    Output("chart", "figure"),
    Output("trend-badges", "children"),
    Input("market-store", "data"),
    Input("tf", "value"), Input("layers", "value"),
    Input("trades-store", "data"), Input("draft-trade", "data"),
    State("store", "data"),
)
def update_chart(meta, tf, layers, trades, draft, store):
    if meta and meta.get("status") == "error":
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title=f"Greška: {meta.get('msg', '?')}")
        return fig, html.Span(meta.get("msg", "Greška"), style={"color": RED})
    return (
        build_fig(MARKET, tf, layers or [], store or {}, trades or [], draft or {}),
        trend_badges(MARKET["trend"]),
    )


@app.callback(
    Output("draft-trade", "data"),
    Output("trade-expander", "open"),
    Input("chart", "clickData"),
    State("trade-mode", "value"),
    State("bank-store", "data"),
    prevent_initial_call=True,
)
def on_chart_click(click, mode, bank):
    if not mode or "on" not in mode or not click:
        return no_update, no_update
    pt = click["points"][0]
    bank = bank or load_bank()
    draft = make_draft(MARKET["df5"], pt["x"], pt["y"], bank["balance_eur"], bank["margin"])
    return draft, True


@app.callback(
    Output("draft-trade", "data", allow_duplicate=True),
    Input("inp-sl", "value"), Input("inp-tp", "value"),
    Input("inp-size", "value"), Input("inp-margin", "value"),
    State("draft-trade", "data"),
    prevent_initial_call=True,
)
def sync_draft_inputs(sl, tp, size, margin, draft):
    if not draft or not draft.get("active"):
        return no_update
    d = update_draft_levels(draft, sl=float(sl) if sl is not None else None,
                            tp=float(tp) if tp is not None else None)
    if size is not None:
        d["size_eur"] = float(size)
    if margin is not None:
        d["margin"] = int(margin)
    return d


@app.callback(
    Output("trades-store", "data"),
    Output("bank-store", "data"),
    Output("draft-trade", "data", allow_duplicate=True),
    Output("trade-expander", "open", allow_duplicate=True),
    Output("trades-table", "children"),
    Output("bank-badge", "children"),
    Output("bank-tab-info", "children"),
    Input("save-trade", "n_clicks"),
    State("draft-trade", "data"),
    State("inp-sl", "value"), State("inp-tp", "value"),
    State("inp-size", "value"), State("inp-margin", "value"),
    prevent_initial_call=True,
)
def save_trade_cb(n, draft, sl, tp, size, margin):
    if not n or not draft or not draft.get("active"):
        return (no_update,) * 7
    d = update_draft_levels(draft, sl=float(sl), tp=float(tp))
    d["size_eur"] = float(size or d["size_eur"])
    d["margin"] = int(margin or d["margin"])
    trades = add_trade(d, MARKET["df5"])
    bank = recalc(trades)
    return trades, bank, {}, False, trades_table(trades), bank_panel(bank), bank_panel(bank)


@app.callback(
    Output("draft-trade", "data", allow_duplicate=True),
    Output("trade-expander", "open", allow_duplicate=True),
    Input("cancel-trade", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_trade(n):
    return {}, False


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


@app.callback(Output("trade-form", "children"), Input("draft-trade", "data"))
def refresh_trade_form(draft):
    return trade_form(draft or {})


clientside_callback(
    """function(_) {
        const gd = document.querySelector('#chart .js-plotly-plot');
        if (!gd) return window.dash_clientside.no_update;
        const skip = n => !['session','lastprice','trade','draft','draft-sl','draft-tp'].includes(n);
        const sh = (gd.layout.shapes || []).filter(s => skip(s.name));
        const an = (gd.layout.annotations || []).filter(a => skip(a.name));
        return JSON.parse(JSON.stringify({shapes: sh, annotations: an}));
    }""",
    Output("store", "data"), Input("chart", "relayoutData"), prevent_initial_call=True)

clientside_callback(
    """function(rd, draft) {
        if (!rd || !draft || !draft.active) return window.dash_clientside.no_update;
        const gd = document.querySelector('#chart .js-plotly-plot');
        if (!gd) return window.dash_clientside.no_update;
        let sl = null, tp = null;
        (gd.layout.shapes || []).forEach(function(s, i) {
            const y0 = rd['shapes[' + i + '].y0'] ?? s.y0;
            const y1 = rd['shapes[' + i + '].y1'] ?? s.y1;
            if (s.name === 'draft-sl') sl = (y0 + y1) / 2;
            if (s.name === 'draft-tp') tp = (y0 + y1) / 2;
        });
        if (sl === null && tp === null) return window.dash_clientside.no_update;
        const d = Object.assign({}, draft);
        if (sl !== null) {
            d.stop_loss = sl;
            const diff = sl - d.entry_price;
            d.sl_points = Math.round((d.direction === 'long' ? diff : -diff) * 100) / 100;
        }
        if (tp !== null) {
            d.take_profit = tp;
            const diff = tp - d.entry_price;
            d.tp_points = Math.round((d.direction === 'long' ? diff : -diff) * 100) / 100;
        }
        return d;
    }""",
    Output("draft-trade", "data", allow_duplicate=True),
    Input("chart", "relayoutData"), State("draft-trade", "data"),
    prevent_initial_call=True)

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
    """function(rd, fig) {
        setTimeout(function() {
            const gd = document.querySelector('#chart .js-plotly-plot');
            if (!gd || !gd._fullLayout || !gd.data.length) return;
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
                const pad = (hi - lo) * 0.06 || 1, cur = gd._fullLayout.yaxis.range;
                const ylo = lo - pad, yhi = hi + pad;
                if (Math.abs(cur[0] - ylo) >= 1e-6 || Math.abs(cur[1] - yhi) >= 1e-6)
                    upd['yaxis.range'] = [ylo, yhi];
            }
            if (Object.keys(upd).length) Plotly.relayout(gd, upd);
        }, 150);
        return window.dash_clientside.no_update;
    }""",
    Output("d_fit", "data"), Input("chart", "relayoutData"), Input("chart", "figure"))


if __name__ == "__main__":
    app.run(debug=False)
