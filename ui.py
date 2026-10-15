"""UI: Dash + Plotly. Pokretanje: python ui.py  ->  http://127.0.0.1:8050

- candlesticks 5 min / 1 h (prebacivanje radio buttonom)
- S/R zone računate na 1h (zones.py), vidljive i na 5m
- HH/HL/LH/LL oznake + trend (trend.py) za odabrani timeframe
- crtanje preko grafa: modebar alati (linija, slobodna olovka, pravokutnik, krug, gumica)
  i tekst bilješke (klik na tekst = uredi, povuci = pomjeri)
"""
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, clientside_callback, dcc, html, no_update

from data import load_5m, resample
from trend import analyze
from zones import get_zones

GREEN, RED, AMBER = "#26a69a", "#ef5350", "#ffc107"
SWING_N = {"5min": 6, "1h": 3}  # veći prozor na 5m da oznake ne zatrpaju graf
FMT = "%Y-%m-%d %H:%M:%S"

# ---------- podaci i proračun (jednom na startu) ----------
df5 = load_5m()
DATA = {"5min": df5, "1h": resample(df5, "1h")}
ZONES = get_zones(df5)  # uvijek iz 1h
TREND = {tf: analyze(df5, tf, n=SWING_N[tf]) for tf in DATA}

# sati bez trgovanja (vikend, dnevna pauza) sakrivamo da graf nema rupa
_h1 = DATA["1h"].index
GAPS = pd.date_range(_h1[0], _h1[-1], freq="1h").difference(_h1).strftime(FMT).tolist()


def rgba(color: str, a: float) -> str:
    c = color.lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{a:.2f})"


def build_fig(tf: str, layers: list, store: dict) -> go.Figure:
    df = DATA[tf]
    x = df.index.strftime(FMT)
    x_end = x[-1]
    fig = go.Figure(go.Candlestick(
        x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=GREEN, decreasing_line_color=RED, showlegend=False))

    if "zones" in layers:  # deblje, poluprozirne zone; jače (više dodira) = vidljivije
        for z in ZONES.itertuples():
            col = {"support": GREEN, "resistance": RED}.get(z.kind, AMBER)
            x0 = z.start.strftime(FMT)
            fig.add_trace(go.Scatter(
                x=[x0, x_end, x_end, x0, x0], y=[z.low, z.low, z.high, z.high, z.low],
                mode="lines", line=dict(color=rgba(col, 0.8), width=1),
                fill="toself", fillcolor=rgba(col, min(0.12 + 0.04 * z.touches, 0.45)),
                hoveron="fills", hoverinfo="text", showlegend=False,
                text=f"{z.kind} {z.low:.2f} - {z.high:.2f} ({z.touches} dodira)"))

    if "swings" in layers:
        sw = TREND[tf]["swings"].dropna(subset=["label"])
        for kind, pos, sym in (("H", "top center", "triangle-down"), ("L", "bottom center", "triangle-up")):
            s = sw[sw["kind"] == kind]
            cols = [GREEN if lb in ("HH", "HL") else RED if lb in ("LH", "LL") else "#9e9e9e" for lb in s["label"]]
            fig.add_trace(go.Scatter(
                x=s.index.strftime(FMT), y=s["price"], mode="markers+text", text=s["label"],
                textposition=pos, textfont=dict(color=cols, size=11),
                marker=dict(symbol=sym, size=6, color=cols), hoverinfo="skip", showlegend=False))

    x_start = (df5.index[-1] - pd.Timedelta(days=3)).strftime(FMT)  # početni pogled: zadnja 3 dana
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#14161a", plot_bgcolor="#14161a",
        margin=dict(l=8, r=8, t=8, b=8), dragmode="pan", uirevision="keep",
        xaxis=dict(rangeslider_visible=False, range=[x_start, (df5.index[-1] + pd.Timedelta(hours=2)).strftime(FMT)],
                   rangebreaks=[dict(values=GAPS, dvalue=3_600_000)], gridcolor="#22252b"),
        yaxis=dict(side="right", fixedrange=True, gridcolor="#22252b"),  # y se sam prilagođava (callback dolje)
        newshape=dict(line=dict(color="#ffeb3b", width=2)),
        shapes=store.get("shapes", []), annotations=store.get("annotations", []))
    return fig


def trend_badges():
    out = []
    for tf, r in TREND.items():
        arrow = {"UP": "▲", "DOWN": "▼", "RANGE": "◆"}[r["trend"]]
        col = {"UP": GREEN, "DOWN": RED, "RANGE": AMBER}[r["trend"]]
        txt = f"{tf}: {arrow} {r['trend']}"
        if r["invalidation"] is not None:
            txt += f" · invalidacija {r['invalidation']:.2f}" + (" (probijena)" if r["broken"] else "")
        out.append(html.Span(txt, style={"color": col, "marginRight": "18px"}))
    return out


# ---------- layout ----------
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
        html.Div(style=BAR, children=[
            dcc.RadioItems(id="tf", value="5min", labelStyle=ITEM,
                           options=[{"label": "5 min", "value": "5min"}, {"label": "1 h", "value": "1h"}]),
            dcc.Checklist(id="layers", value=["zones", "swings"], labelStyle=ITEM,
                          options=[{"label": "S/R zone (1h)", "value": "zones"},
                                   {"label": "HH/HL/LH/LL", "value": "swings"}]),
            dcc.Input(id="txt", type="text", placeholder="tekst bilješke", style={"width": "150px"}),
            html.Button("Dodaj tekst", id="add_txt"),
            html.Button("Očisti crteže", id="clear"),
            html.Div(trend_badges()),
        ]),
        dcc.Graph(id="chart", config=CONFIG, style={"height": "88vh"}),
        dcc.Store(id="store", data={}),
        dcc.Store(id="d_fit"), dcc.Store(id="d_txt"), dcc.Store(id="d_clr"),
    ],
)


@app.callback(Output("chart", "figure"), Input("tf", "value"), Input("layers", "value"), State("store", "data"))
def update(tf, layers, store):
    return build_fig(tf, layers or [], store or {})


# ---------- client-side: crteži, tekst, y-fit ----------
# 1) zapamti sve što je korisnik nacrtao/napisao (da preživi promjenu timeframea)
clientside_callback(
    """function(_) {
        const gd = document.querySelector('#chart .js-plotly-plot');
        if (!gd) return window.dash_clientside.no_update;
        return JSON.parse(JSON.stringify({shapes: gd.layout.shapes || [], annotations: gd.layout.annotations || []}));
    }""",
    Output("store", "data"), Input("chart", "relayoutData"), prevent_initial_call=True)

# 2) tekst bilješka u sredinu trenutnog pogleda (zatim je povučeš gdje treba, klik = uredi)
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

# 3) očisti sve crteže i tekstove
clientside_callback(
    """function(n) {
        if (!n) return window.dash_clientside.no_update;
        Plotly.relayout(document.querySelector('#chart .js-plotly-plot'), {shapes: [], annotations: []});
        return window.dash_clientside.no_update;
    }""",
    Output("d_clr", "data"), Input("clear", "n_clicks"))

# 4) y os se automatski prilagodi vidljivom x rasponu (Plotly to sam ne radi)
clientside_callback(
    """function(rd, fig) {
        setTimeout(function() {
            const gd = document.querySelector('#chart .js-plotly-plot');
            if (!gd || !gd._fullLayout || !gd.data.length) return;
            const tr = gd.data[0], r = gd._fullLayout.xaxis.range;
            const a = String(r[0]).replace('T', ' '), b = String(r[1]).replace('T', ' ');
            let lo = Infinity, hi = -Infinity;
            for (let i = 0; i < tr.x.length; i++) {
                if (tr.x[i] >= a && tr.x[i] <= b) { if (tr.low[i] < lo) lo = tr.low[i]; if (tr.high[i] > hi) hi = tr.high[i]; }
            }
            if (!isFinite(lo)) return;
            const pad = (hi - lo) * 0.06 || 1, cur = gd._fullLayout.yaxis.range;
            if (Math.abs(cur[0] - (lo - pad)) < 1e-6 && Math.abs(cur[1] - (hi + pad)) < 1e-6) return;
            Plotly.relayout(gd, {'yaxis.range': [lo - pad, hi + pad]});
        }, 150);
        return window.dash_clientside.no_update;
    }""",
    Output("d_fit", "data"), Input("chart", "relayoutData"), Input("chart", "figure"))


if __name__ == "__main__":
    app.run(debug=False)
