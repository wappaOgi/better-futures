"""Dohvat 5m podataka (yfinance) + resampling u veće timeframe-ove."""
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_TICKER = "MNQ=F"
TZ = "Europe/Zagreb"
CACHE_TZ = "America/New_York"
OHLCV = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}


def cache_path(ticker: str) -> Path:
    safe = "".join(c if c.isalnum() else "_" for c in ticker).strip("_") or "ticker"
    return Path(__file__).with_name(f"{safe}_5m.csv")


def clear_cache(ticker: str) -> bool:
    """Obriši lokalni CSV cache za ticker. Vraća True ako je datoteka obrisana."""
    path = cache_path(ticker)
    if path.exists():
        path.unlink()
        return True
    return False


def _to_zagreb(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize(CACHE_TZ, ambiguous="infer").tz_convert(TZ).tz_localize(None)
    else:
        df.index = df.index.tz_convert(TZ).tz_localize(None)
    return df


def fetch_5m(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, interval="5m", period=period, progress=False, multi_level_index=False)
    if df.empty:
        raise RuntimeError(f"yfinance nije vratio podatke za {ticker}")
    df = _to_zagreb(df)
    df.index.name = "Time"
    return df[list(OHLCV)].dropna()


def load_5m(ticker: str = DEFAULT_TICKER, days: int = 5) -> pd.DataFrame:
    """Povuče podatke za ticker i vrati zadnjih `days` dana (5m max ~60d na Yahoo)."""
    days = max(1, min(int(days), 60))
    cache = cache_path(ticker)
    fetch_days = min(max(days + 5, 7), 60)
    period = f"{fetch_days}d"

    old = pd.read_csv(cache, index_col=0, parse_dates=True) if cache.exists() else None
    if old is not None:
        old = _to_zagreb(old)
    try:
        new = fetch_5m(ticker, period)
    except Exception as e:
        if old is None:
            raise
        print(f"yfinance greška ({e}), koristim cache")
        df = old
    else:
        df = pd.concat([old, new]) if old is not None else new
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_csv(cache)

    cutoff = df.index[-1] - pd.Timedelta(days=days)
    return df[df.index >= cutoff]


def resample(df: pd.DataFrame, rule: str = "1h") -> pd.DataFrame:
    if rule in ("5min", "5m"):
        return df
    out = df.resample(rule, closed="left", label="left").agg(OHLCV)
    return out.dropna(subset=["Open"])
