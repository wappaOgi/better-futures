"""BB, RSI i pomoćne funkcije za graf / strategije."""
import numpy as np
import pandas as pd

BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14


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
