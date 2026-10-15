"""Dohvat 5m podataka (yfinance) + resampling u veće timeframe-ove.

Ovaj file zovu i zones.py i trend.py, tako da je logika pretvaranja 5m -> 1h na jednom mjestu.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKER = "MNQ=F"
TZ = "America/New_York"
CACHE = Path(__file__).with_name("mnq_5m.csv")
OHLCV = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}


def fetch_5m(ticker: str = TICKER, period: str = "60d") -> pd.DataFrame:
    df = yf.download(ticker, interval="5m", period=period, progress=False, multi_level_index=False)
    if df.empty:
        raise RuntimeError("yfinance nije vratio podatke")
    if df.index.tz is not None:  # naive New York vrijeme, lakše za crtanje
        df.index = df.index.tz_convert(TZ).tz_localize(None)
    df.index.name = "Time"
    return df[list(OHLCV)].dropna()


def load_5m() -> pd.DataFrame:
    """Povuče svježe podatke i spoji ih s cacheom, pa se istorija vremenom produžuje iznad 60 dana.
    Ako Yahoo ne radi, koristi se samo cache."""
    old = pd.read_csv(CACHE, index_col=0, parse_dates=True) if CACHE.exists() else None
    try:
        new = fetch_5m()
    except Exception as e:
        if old is None:
            raise
        print(f"yfinance greška ({e}), koristim cache")
        return old
    df = pd.concat([old, new]) if old is not None else new
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(CACHE)
    return df


def resample(df: pd.DataFrame, rule: str = "1h") -> pd.DataFrame:
    """5m OHLCV -> bilo koji veći timeframe ('1h', '4h', ...). '5min' vraća original."""
    if rule in ("5min", "5m"):
        return df
    out = df.resample(rule, closed="left", label="left").agg(OHLCV)
    return out.dropna(subset=["Open"])
