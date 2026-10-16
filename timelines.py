"""Dvije timeline trake uz x-os: ET (izvor) i Zagreb."""
from zoneinfo import ZoneInfo

import pandas as pd

from data import CACHE_TZ

ET = ZoneInfo(CACHE_TZ)
ZG = ZoneInfo("Europe/Zagreb")


def _step(t_start, t_end) -> pd.Timedelta:
    span = pd.Timestamp(t_end) - pd.Timestamp(t_start)
    if span <= pd.Timedelta(hours=6):
        return pd.Timedelta(minutes=30)
    if span <= pd.Timedelta(days=1):
        return pd.Timedelta(hours=1)
    if span <= pd.Timedelta(days=3):
        return pd.Timedelta(hours=2)
    return pd.Timedelta(hours=4)


def overlays(t_start, t_end) -> tuple[list, list]:
    """Donji red = Zagreb, srednji = ET (originalni podaci)."""
    shapes, annotations = [], []
    t0, t1 = pd.Timestamp(t_start), pd.Timestamp(t_end)
    step = _step(t0, t1)

    shapes.extend([
        dict(type="rect", xref="paper", yref="paper", x0=0, x1=1, y0=0, y1=0.022,
             fillcolor="rgba(38,166,154,0.14)", line_width=0, layer="below", name="timeline"),
        dict(type="rect", xref="paper", yref="paper", x0=0, x1=1, y0=0.022, y1=0.044,
             fillcolor="rgba(66,165,245,0.14)", line_width=0, layer="below", name="timeline"),
        dict(type="line", xref="paper", yref="paper", x0=0, x1=1, y0=0.022, y1=0.022,
             line=dict(color="#444", width=1), layer="below", name="timeline"),
    ])
    annotations.extend([
        dict(x=0.005, y=0.011, xref="paper", yref="paper", text="Zagreb",
             showarrow=False, xanchor="left", font=dict(size=9, color="#26a69a"), name="timeline"),
        dict(x=0.005, y=0.033, xref="paper", yref="paper", text="ET",
             showarrow=False, xanchor="left", font=dict(size=9, color="#42a5f5"), name="timeline"),
    ])

    t = t0.floor("h")
    prev_et_day = prev_zg_day = None
    while t <= t1:
        et = t.tz_localize(ZG).tz_convert(ET)
        et_day, zg_day = et.date(), t.date()

        et_txt = et.strftime("%H:%M")
        if et_day != prev_et_day:
            et_txt = f"{et.strftime('%d.%m.')} {et_txt}"
            prev_et_day = et_day
        annotations.append(dict(
            x=t, y=0.033, xref="x", yref="paper", text=et_txt, showarrow=False,
            xanchor="center", font=dict(size=9, color="#90caf9"), name="timeline",
        ))

        zg_txt = t.strftime("%H:%M")
        if zg_day != prev_zg_day:
            zg_txt = f"{t.strftime('%d.%m.')} {zg_txt}"
            prev_zg_day = zg_day
        annotations.append(dict(
            x=t, y=0.011, xref="x", yref="paper", text=zg_txt, showarrow=False,
            xanchor="center", font=dict(size=9, color="#80cbc4"), name="timeline",
        ))

        t += step
    return shapes, annotations
