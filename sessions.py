"""Trading session trake (definirano u ET, prikaz u Europe/Zagreb).

- Asia:   18:00–03:00 ET (Tokyo/Sydney)
- London: 03:00–09:30 ET
- NY:     09:30–17:00 ET (RTH + after)
- Open 1h: 09:30–10:30 ET (prvi sat cash opena)
"""
from datetime import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

MARKET = ZoneInfo("America/New_York")
DISPLAY = ZoneInfo("Europe/Zagreb")

# (id, start, end) u ET; end <= start znači prelazak ponoći
WINDOWS = [
    ("asia", "18:00", "03:00"),
    ("london", "03:00", "09:30"),
    ("ny", "09:30", "17:00"),
    ("open1h", "09:30", "10:30"),
]

# boja, alpha trake na vremenskoj liniji (samo dolje, ispod grafa)
STYLE = {
    "asia": ("#9575cd", 0.55),
    "london": ("#ffb74d", 0.60),
    "ny": ("#4fc3f7", 0.55),
    "open1h": ("#ffeb3b", 0.75),
}

LABELS = {"asia": "Asia", "london": "London", "ny": "NY", "open1h": "Open 1h"}

VWAP_SESSIONS = ("asia", "london", "ny")
VWAP_WINDOWS = {sid: (start, end) for sid, start, end in WINDOWS if sid in VWAP_SESSIONS}


def _rgba(color: str, a: float) -> str:
    c = color.lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{a:.2f})"


def _parse_hm(hm: str) -> tuple[int, int]:
    h, m = hm.split(":")
    return int(h), int(m)


def _to_et(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize(DISPLAY).tz_convert(MARKET)
    return index.tz_convert(MARKET)


def _time_hm(hm: str) -> time:
    h, m = _parse_hm(hm)
    return time(h, m)


def _in_session_et(ts: pd.Timestamp, start: str, end: str) -> bool:
    t = ts.time()
    t0, t1 = _time_hm(start), _time_hm(end)
    if t1 > t0:
        return t0 <= t < t1
    return t >= t0 or t < t1


def _session_bar_allowed(ts: pd.Timestamp, session_id: str) -> bool:
    wd = ts.weekday()
    if wd >= 5:
        return False
    if wd == 6 and session_id != "asia":
        return False
    return True


def compute_session_vwap(df: pd.DataFrame, session_id: str) -> pd.Series:
    """VWAP sidrište na početku sesije (Asia / London / NY), reset svaki session open."""
    if session_id not in VWAP_WINDOWS or df is None or df.empty:
        return pd.Series(dtype=float, name=f"vwap_{session_id}")
    start, end = VWAP_WINDOWS[session_id]
    et = _to_et(df.index)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    out = np.full(len(df), np.nan)
    cum_pv, cum_v = 0.0, 0.0
    in_sess = False

    for i, t in enumerate(et):
        if not _session_bar_allowed(t, session_id):
            in_sess = False
            continue
        inside = _in_session_et(t, start, end)
        if inside:
            if not in_sess:
                cum_pv, cum_v = 0.0, 0.0
                in_sess = True
            v = float(vol.iloc[i])
            if v > 0:
                cum_pv += float(tp.iloc[i]) * v
                cum_v += v
            if cum_v > 0:
                out[i] = cum_pv / cum_v
        else:
            in_sess = False

    return pd.Series(out, index=df.index, name=f"vwap_{session_id}")


def _et_window(day: pd.Timestamp, start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    sh, sm = _parse_hm(start)
    eh, em = _parse_hm(end)
    t0 = pd.Timestamp(day.year, day.month, day.day, sh, sm, tzinfo=MARKET)
    t1 = pd.Timestamp(day.year, day.month, day.day, eh, em, tzinfo=MARKET)
    if t1 <= t0:
        t1 += pd.Timedelta(days=1)
    z0 = t0.astimezone(DISPLAY).replace(tzinfo=None)
    z1 = t1.astimezone(DISPLAY).replace(tzinfo=None)
    return z0, z1


def overlays(t_start: pd.Timestamp, t_end: pd.Timestamp) -> tuple[list, list]:
    """Samo traka uz dno (x-os), bez overlaya iznad candlestickova."""
    shapes, annotations = [], []
    z0 = pd.Timestamp(t_start).tz_localize(DISPLAY).tz_convert(MARKET)
    z1 = pd.Timestamp(t_end).tz_localize(DISPLAY).tz_convert(MARKET)
    day = pd.Timestamp(z0.year, z0.month, z0.day, tzinfo=MARKET)
    end_day = pd.Timestamp(z1.year, z1.month, z1.day, tzinfo=MARKET)

    while day <= end_day + pd.Timedelta(days=1):
        if day.weekday() == 5:  # subota — CME ne trguje
            day += pd.Timedelta(days=1)
            continue
        day_x0, day_x1 = None, None
        for sid, start, end in WINDOWS:
            if day.weekday() == 6 and sid != "asia":  # nedjelja: samo večernji Asia open
                continue
            x0, x1 = _et_window(day, start, end)
            if x1 < t_start or x0 > t_end:
                continue
            day_x0 = x0 if day_x0 is None else min(day_x0, x0)
            day_x1 = x1 if day_x1 is None else max(day_x1, x1)
            col, strip_a = STYLE[sid]
            shapes.append(dict(
                type="rect", xref="x", yref="paper", x0=x0, x1=x1,
                y0=0.048, y1=0.066, fillcolor=_rgba(col, strip_a), line_width=0,
                layer="below", name="session",
            ))
            if (x1 - x0) > pd.Timedelta(minutes=50):
                annotations.append(dict(
                    x=x0 + (x1 - x0) / 2, y=0.057, xref="x", yref="paper",
                    text=LABELS[sid], showarrow=False, xanchor="center", yanchor="middle",
                    font=dict(size=8, color=col), name="session",
                ))
        day += pd.Timedelta(days=1)
    return shapes, annotations
