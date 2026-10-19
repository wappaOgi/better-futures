"""Average True Range (ATR) — zadnjih N svijeća."""
import pandas as pd

PERIOD = 24
COLOR = "#ba68c8"


def series(df: pd.DataFrame, period: int = PERIOD) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=max(2, period // 2)).mean()


def last(df: pd.DataFrame, period: int = PERIOD) -> float | None:
    s = series(df, period).dropna()
    if s.empty:
        return None
    return round(float(s.iloc[-1]), 2)


def suggest_sl(df: pd.DataFrame, period: int = PERIOD) -> int:
    """Zaokruženi ATR u bodovima — sugestija za stop loss."""
    v = last(df, period)
    return max(1, round(v)) if v is not None else 30
