"""Dohvat 5m podataka (yfinance) + resampling u veće timeframe-ove."""
import pandas as pd
import yfinance as yf

DEFAULT_TICKER = "MNQ=F"
TZ = "Europe/Zagreb"
CACHE_TZ = "America/New_York"
OHLCV = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}

# RAM cache: ticker -> puni 5m DataFrame za sesiju
_MEMORY: dict[str, pd.DataFrame] = {}


def memory_tickers() -> list[str]:
    return list(_MEMORY.keys())


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
    """Povuče podatke za ticker, spremi u RAM i vrati zadnjih `days` dana."""
    days = max(1, min(int(days), 60))
    fetch_days = min(max(days + 5, 7), 60)
    df = fetch_5m(ticker, f"{fetch_days}d")
    _MEMORY[ticker] = df
    cutoff = df.index[-1] - pd.Timedelta(days=days)
    return df[df.index >= cutoff]


def resample(df: pd.DataFrame, rule: str = "1h") -> pd.DataFrame:
    if rule in ("5min", "5m"):
        return df
    pd_rule = {"4h": "4h", "1h": "1h"}.get(rule, rule)
    out = df.resample(pd_rule, closed="left", label="left").agg(OHLCV)
    return out.dropna(subset=["Open"])
