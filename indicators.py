"""BB, RSI i pomoćne funkcije za graf / strategije."""
import numpy as np
import pandas as pd

BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
SMA_TREND = 200


def bollinger(close: pd.Series, period: int = BB_PERIOD, std: float = BB_STD) -> pd.DataFrame:
    mid = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std()
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": mid + std * sd, "bb_lower": mid - std * sd},
        index=close.index,
    )


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.rename("rsi")


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=max(2, period // 2)).mean().rename("atr")


def bollinger_width(close: pd.Series, period: int = BB_PERIOD, std: float = BB_STD) -> pd.Series:
    bb = bollinger(close, period, std)
    mid = bb["bb_mid"].replace(0, np.nan)
    return ((bb["bb_upper"] - bb["bb_lower"]) / mid).rename("bbw")


def rolling_percentile(s: pd.Series, window: int = 100) -> pd.Series:
    """Percentil trenutne vrijednosti unutar prošlih `window` barova (0–1)."""

    def _pct(arr: np.ndarray) -> float:
        cur = arr[-1]
        past = arr[:-1]
        if len(past) == 0 or np.isnan(cur):
            return np.nan
        past = past[~np.isnan(past)]
        if len(past) == 0:
            return np.nan
        return float(np.sum(past < cur) / len(past))

    return s.rolling(window, min_periods=max(20, window // 5)).apply(_pct, raw=True).rename("pct")


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean().rename(f"ema_{span}")


def sma(close: pd.Series, period: int = SMA_TREND) -> pd.Series:
    return close.rolling(period, min_periods=period).mean().rename(f"sma_{period}")


def hammers_at_bollinger(
    df: pd.DataFrame,
    bb: pd.DataFrame,
    *,
    touch_frac: float = 0.10,
) -> pd.Index:
    """Hammer kod donjeg BB: donji wick duži od bodyja, low blizu bb_lower."""
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    lo = df["Low"].astype(float)
    c = df["Close"].astype(float)
    span = (h - lo).replace(0, np.nan)
    body = (c - o).abs()
    body_eff = np.maximum(body, span * 0.08)

    lower_wick = np.minimum(o, c) - lo
    upper_wick = h - np.maximum(o, c)

    shape = (lower_wick > body_eff) & (upper_wick <= body_eff * 1.25)
    band_w = (bb["bb_upper"] - bb["bb_lower"]).replace(0, np.nan)
    at_lower = lo <= bb["bb_lower"] + touch_frac * band_w

    mask = shape & at_lower & span.notna()
    return df.index[mask]


def hammer_bb_circle_shapes(
    df: pd.DataFrame,
    hammer_times: pd.Index,
    bar_half_ms: float,
    *,
    color: str = "#42a5f5",
) -> list[dict]:
    """Plavi krug oko svijeće (Plotly shape type=circle)."""
    shapes = []
    half_ms = max(float(bar_half_ms), 1.0)
    for t in hammer_times:
        row = df.loc[t]
        hi, low = float(row["High"]), float(row["Low"])
        cy = (hi + low) / 2
        half_y = max((hi - low) * 0.58, 1e-9)
        t0 = t - pd.Timedelta(milliseconds=half_ms)
        t1 = t + pd.Timedelta(milliseconds=half_ms)
        shapes.append(
            {
                "type": "circle",
                "xref": "x",
                "yref": "y",
                "x0": t0,
                "x1": t1,
                "y0": cy - half_y,
                "y1": cy + half_y,
                "line": {"color": color, "width": 2},
                "fillcolor": "rgba(66,165,245,0)",
                "layer": "above",
            }
        )
    return shapes
