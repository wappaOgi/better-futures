"""Trading session trake (definirano u ET, prikaz u Europe/Zagreb).

- Asia:   18:00–03:00 ET (Tokyo/Sydney)
- London: 03:00–09:30 ET
- NY:     09:30–17:00 ET (RTH + after)
- Open 1h: 09:30–10:30 ET (prvi sat cash opena)
"""
from zoneinfo import ZoneInfo

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

# boja, alpha trake na vremenskoj liniji, alpha pozadine grafa
STYLE = {
    "asia": ("#9575cd", 0.45, 0.035),
    "london": ("#ffb74d", 0.50, 0.04),
    "ny": ("#4fc3f7", 0.45, 0.035),
    "open1h": ("#ffeb3b", 0.70, 0.08),
}

LABELS = {"asia": "Asia", "london": "London", "ny": "NY", "open1h": "Open 1h"}


def _rgba(color: str, a: float) -> str:
    c = color.lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{a:.2f})"


def _parse_hm(hm: str) -> tuple[int, int]:
    h, m = hm.split(":")
    return int(h), int(m)


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
    """Vrect trake + kratke oznake na vremenskoj liniji."""
    shapes, annotations = [], []
    z0 = pd.Timestamp(t_start).tz_localize(DISPLAY).tz_convert(MARKET)
    z1 = pd.Timestamp(t_end).tz_localize(DISPLAY).tz_convert(MARKET)
    day = pd.Timestamp(z0.year, z0.month, z0.day, tzinfo=MARKET)
    end_day = pd.Timestamp(z1.year, z1.month, z1.day, tzinfo=MARKET)

    while day <= end_day + pd.Timedelta(days=1):
        if day.weekday() == 5:  # subota — CME ne trguje
            day += pd.Timedelta(days=1)
            continue
        for sid, start, end in WINDOWS:
            if day.weekday() == 6 and sid != "asia":  # nedjelja: samo večernji Asia open
                continue
            x0, x1 = _et_window(day, start, end)
            if x1 < t_start or x0 > t_end:
                continue
            col, strip_a, bg_a = STYLE[sid]
            shapes.append(dict(
                type="rect", xref="x", yref="paper", x0=x0, x1=x1,
                y0=0, y1=0.04, fillcolor=_rgba(col, strip_a), line_width=0,
                layer="below", name="session",
            ))
            if bg_a:
                shapes.append(dict(
                    type="rect", xref="x", yref="paper", x0=x0, x1=x1,
                    y0=0.04, y1=1, fillcolor=_rgba(col, bg_a), line_width=0,
                    layer="below", name="session",
                ))
            if (x1 - x0) > pd.Timedelta(minutes=50):
                annotations.append(dict(
                    x=x0 + (x1 - x0) / 2, y=0.02, xref="x", yref="paper",
                    text=LABELS[sid], showarrow=False, xanchor="center",
                    font=dict(size=8, color=col), name="session",
                ))
        day += pd.Timedelta(days=1)
    return shapes, annotations
