"""Breakout setup — samo podaci dostupni u trenutku odluke (shift/rolling)."""
import pandas as pd

from indicators import SMA_TREND, atr, bollinger_width, ema, rolling_percentile, sma

RANGE_LEN = 20
ATR_LEN = 14
RVOL_LEN = 20
BBW_PCT_WINDOW = 100
EMA_LEN = 50


def compute_breakout_long(df: pd.DataFrame) -> pd.DataFrame:
    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]

    range_high = high.shift(1).rolling(RANGE_LEN, min_periods=RANGE_LEN).max()
    range_low = low.shift(1).rolling(RANGE_LEN, min_periods=RANGE_LEN).min()
    atr_s = atr(df, ATR_LEN)

    vol_base = volume.shift(1).rolling(RVOL_LEN, min_periods=RVOL_LEN).mean()
    rvol = volume / vol_base.replace(0, pd.NA)

    bbw = bollinger_width(close, RANGE_LEN, 2.0)
    bbw_pct = rolling_percentile(bbw, BBW_PCT_WINDOW)
    ema50 = ema(close, EMA_LEN)
    sma200 = sma(close, SMA_TREND)

    breakout = close > range_high + 0.1 * atr_s
    setup = (
        breakout
        & (rvol > 1.5)
        & (bbw_pct < 0.80)
        & (close > ema50)
        & (close > sma200)
    )

    return pd.DataFrame(
        {
            "range_high": range_high,
            "range_low": range_low,
            "atr": atr_s,
            "rvol": rvol,
            "bbw_pct": bbw_pct,
            "ema50": ema50,
            "setup_long": setup.fillna(False),
        },
        index=df.index,
    )
