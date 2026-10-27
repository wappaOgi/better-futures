"""NY open-range breakout pomoćne funkcije (09:30–10:30 ET)."""
from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

DISPLAY = ZoneInfo("Europe/Zagreb")
MARKET = ZoneInfo("America/New_York")

OR_START = time(9, 30)
OR_END = time(10, 30)
NY_END = time(17, 0)


def _to_et(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize(DISPLAY).tz_convert(MARKET)
    return index.tz_convert(MARKET)


def _in_or_window(ts: pd.Timestamp) -> bool:
    t = ts.time()
    return OR_START <= t < OR_END


def _post_or_window(ts: pd.Timestamp) -> bool:
    t = ts.time()
    return OR_END <= t < NY_END and ts.weekday() < 5


def _daily_atr(daily: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = daily["High"], daily["Low"], daily["Close"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=max(2, period // 2)).mean()


def compute_ny_orb(
    df: pd.DataFrame,
    *,
    rvol_lookback: int = 14,
    atr_period: int = 5,
    min_range_atr: float = 0.15,
    max_range_atr: float = 2.0,
) -> pd.DataFrame:
    """
    Po baru: OR high/low, RVOL, ATR(5) dnevni, post_or flag.

    RVOL = volumen prvog sata / prosjek istog sata zadnjih N trgovačkih dana.
    ATR filter: OR širina mora biti između min/max × dnevni ATR(5).
    """
    et = _to_et(df.index)
    n = len(df)
    or_high = np.full(n, np.nan)
    or_low = np.full(n, np.nan)
    rvol = np.full(n, np.nan)
    atr5 = np.full(n, np.nan)
    post_or = np.zeros(n, dtype=bool)
    volatile = np.zeros(n, dtype=bool)
    day_id = np.full(n, -1, dtype=int)

    # --- dnevni ATR (bez lookahead: bar koristi ATR do jučer) ---
    daily = df.copy()
    daily.index = et
    daily["_date"] = et.date
    daily_ohlc = daily.groupby("_date").agg(
        High=("High", "max"), Low=("Low", "min"), Close=("Close", "last")
    )
    daily_ohlc = daily_ohlc[daily_ohlc.index.map(lambda d: pd.Timestamp(d).weekday() < 5)]
    atr_by_date = _daily_atr(daily_ohlc, atr_period).shift(1)  # jučerašnji ATR

    # --- OR po trgovačkom danu ---
    or_rows: list[dict] = []
    for i in range(n):
        if et[i].weekday() >= 5:
            continue
        if _in_or_window(et[i]):
            d = et[i].date()
            # proširi postojeći red ili novi
            found = next((r for r in or_rows if r["date"] == d), None)
            if found is None:
                found = {"date": d, "hi": df["High"].iloc[i], "lo": df["Low"].iloc[i],
                         "vol": df["Volume"].iloc[i], "day_idx": len(or_rows)}
                or_rows.append(found)
            else:
                found["hi"] = max(found["hi"], df["High"].iloc[i])
                found["lo"] = min(found["lo"], df["Low"].iloc[i])
                found["vol"] += df["Volume"].iloc[i]

    vol_hist: list[float] = []
    for row in or_rows:
        d = row["date"]
        if len(vol_hist) >= rvol_lookback:
            avg_vol = float(np.mean(vol_hist[-rvol_lookback:]))
            row_rvol = row["vol"] / avg_vol if avg_vol > 0 else np.nan
        else:
            row_rvol = np.nan
        vol_hist.append(row["vol"])

        atr_val = atr_by_date.get(d, np.nan)
        atr_val = np.nan if pd.isna(atr_val) else float(atr_val)

        or_range = row["hi"] - row["lo"]
        vol_ok = (
            not np.isnan(atr_val)
            and atr_val > 0
            and min_range_atr * atr_val <= or_range <= max_range_atr * atr_val
        )

        day_mask = np.array([t.date() == d for t in et])
        post_mask = np.array([_post_or_window(t) for t in et])
        mask = day_mask & post_mask
        or_high[mask] = row["hi"]
        or_low[mask] = row["lo"]
        rvol[mask] = row_rvol
        atr5[mask] = atr_val
        post_or[mask] = True
        volatile[mask] = vol_ok
        day_id[mask] = row["day_idx"]

    return pd.DataFrame(
        {
            "or_high": or_high,
            "or_low": or_low,
            "rvol": rvol,
            "atr5": atr5,
            "post_or": post_or,
            "volatile": volatile,
            "day_id": day_id,
        },
        index=df.index,
    )


def _ny_session_bar(ts: pd.Timestamp) -> bool:
    return ts.weekday() < 5 and OR_START <= ts.time() < NY_END


def ny_session_mask(df: pd.DataFrame) -> np.ndarray:
    et = _to_et(df.index)
    return np.array([_ny_session_bar(t) for t in et], dtype=bool)


def ny_session_dim_shapes(
    df: pd.DataFrame,
    *,
    yref: str = "y",
    y0: float | None = None,
    y1: float | None = None,
    opacity: float = 0.72,
) -> list[dict]:
    """Zamrači periode izvan NY sesije (09:30–17:00 ET)."""
    et = _to_et(df.index)
    y0 = float(y0 if y0 is not None else df["Low"].min())
    y1 = float(y1 if y1 is not None else df["High"].max())
    fill = f"rgba(0,0,0,{opacity:.2f})"
    shapes, in_block, block_start = [], False, None

    for i, t in enumerate(et):
        outside = not _ny_session_bar(t)
        if outside and not in_block:
            in_block, block_start = True, df.index[i]
        elif not outside and in_block:
            shapes.append(dict(
                type="rect", xref="x", yref=yref,
                x0=block_start, x1=df.index[i - 1],
                y0=y0, y1=y1, fillcolor=fill, line_width=0, layer="below",
            ))
            in_block = False
    if in_block:
        shapes.append(dict(
            type="rect", xref="x", yref=yref,
            x0=block_start, x1=df.index[-1],
            y0=y0, y1=y1, fillcolor=fill, line_width=0, layer="below",
        ))
    return shapes


def compute_ny_vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP sidrište na početku prvog sata NY sesije (09:30 ET).
    Kumulativno se računa kroz cijelu NY sesiju (09:30–17:00 ET).
    """
    et = _to_et(df.index)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].fillna(0.0)
    out = np.full(len(df), np.nan)
    cum_pv, cum_v = 0.0, 0.0
    cur_day = None

    for i, t in enumerate(et):
        if not _ny_session_bar(t):
            continue
        d = t.date()
        if d != cur_day:
            cum_pv, cum_v = 0.0, 0.0
            cur_day = d
        v = float(vol.iloc[i])
        if v > 0:
            cum_pv += float(tp.iloc[i]) * v
            cum_v += v
        if cum_v > 0:
            out[i] = cum_pv / cum_v

    return pd.Series(out, index=df.index, name="vwap")
