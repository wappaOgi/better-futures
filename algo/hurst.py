"""Copyright (C) 2025 James Sawyer
All rights reserved.

This script and the associated files are private
and confidential property. Unauthorized copying of
this file, via any medium, and the divulgence of any
contained information without express written consent
is strictly prohibited.

This script is intended for personal use only and should
not be distributed or used in any commercial or public
setting unless otherwise authorized by the copyright holder.
By using this script, you agree to abide by these terms.

DISCLAIMER: This script is provided 'as is' without warranty
of any kind, either express or implied, including, but not
limited to, the implied warranties of merchantability,
fitness for a particular purpose, or non-infringement. In no
event shall the authors or copyright holders be liable for any
claim, damages, or other liability, whether in an action of
contract, tort or otherwise, arising from, out of, or in
connection with the script or the use or other dealings in
the script.
"""

# -*- coding: utf-8 -*-
# pylint: disable=C0116, W0621, W1203, C0103, C0301, W1201, W0511, E0401, E1101, E0606

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hurst import compute_Hc
from sklearn.linear_model import TheilSenRegressor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

CLOSE_COLS = ("close", "Close", "CLOSE")
TIME_COLS = ("snapshotTime", "Time", "time", "Datetime", "datetime")
MIN_SERIES = 100


class HurstAnalyzer:
    def __init__(self, window: int = 100, kind: str = "random_walk"):
        self.window = window
        self.kind = kind
        self.hurst_series = None
        self.latest_value = None
        self.trend_line = None

    def parse_ohlc(self, df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty:
            logger.error("DataFrame is empty")
            raise ValueError("Empty DataFrame")

        df = df.copy()
        close_col = next((c for c in CLOSE_COLS if c in df.columns), None)
        if close_col is None:
            logger.error("DataFrame must contain a close column")
            raise ValueError("Missing required columns in DataFrame")

        if "snapshotTime" in df.columns:
            parsed = pd.to_datetime(df["snapshotTime"], format="%Y:%m:%d-%H:%M:%S", errors="coerce")
            if parsed.isna().all():
                parsed = pd.to_datetime(df["snapshotTime"], errors="coerce")
            df["snapshotTime"] = parsed
            df.dropna(subset=["snapshotTime"], inplace=True)
            df.set_index("snapshotTime", inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            time_col = next((c for c in TIME_COLS if c in df.columns), None)
            if time_col is None:
                logger.error("DataFrame must contain 'snapshotTime' and 'close' columns")
                raise ValueError("Missing required columns in DataFrame")
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df.dropna(subset=[time_col], inplace=True)
            df.set_index(time_col, inplace=True)

        series = pd.to_numeric(df[close_col], errors="coerce").dropna()
        logger.info("Parsed OHLC data: %s rows retained after cleaning", len(series))
        return series

    def compute_rolling(self, series: pd.Series) -> pd.Series:
        window = max(int(self.window), MIN_SERIES)
        self.window = window
        logger.info("Computing rolling Hurst exponent with window = %s", window)
        series_values = series.to_numpy()
        hurst_vals = np.full(len(series_values), np.nan)
        first_error = None

        for i in range(window, len(series_values)):
            window_data = series_values[i - window : i]
            if np.isnan(window_data).any() or len(window_data) < MIN_SERIES:
                continue
            try:
                h, _, _ = compute_Hc(window_data, kind=self.kind, min_window=5, simplified=True)
                hurst_vals[i] = np.clip(h, 0.0, 1.0)
            except Exception as e:
                if first_error is None:
                    first_error = e
                    logger.warning("Error computing Hurst at index %s: %s", i, e)

        self.hurst_series = pd.Series(hurst_vals, index=series.index)
        logger.info("Computed %s valid Hurst values", np.count_nonzero(~np.isnan(hurst_vals)))
        return self.hurst_series

    def plot(self, output_file: str = None, headless: bool = False) -> float:
        if self.hurst_series is None:
            logger.error("Hurst series has not been computed.")
            raise ValueError("Run compute_rolling() first")

        valid_hurst = self.hurst_series.dropna()
        if valid_hurst.empty:
            logger.error("Hurst series is empty after dropping NaNs")
            raise ValueError("Hurst series contains no valid values")

        x = np.arange(len(valid_hurst)).reshape(-1, 1)
        y = valid_hurst.values

        try:
            model = TheilSenRegressor()
            model.fit(x, y)
            trend_line = model.predict(x)
            self.trend_line = pd.Series(trend_line, index=valid_hurst.index)
        except Exception as e:
            logger.warning("Theil-Sen trend failed: %s", e)
            trend_line = np.full(len(valid_hurst), np.nan)
            self.trend_line = pd.Series(trend_line, index=valid_hurst.index)

        if not headless:
            plt.figure(figsize=(14, 6))
            plt.plot(valid_hurst.index, valid_hurst, label="Rolling Hurst", color="blue")
            plt.plot(valid_hurst.index, trend_line, label="Theil-Sen Trend", color="red", linestyle="--")
            plt.axhline(0.5, color="gray", linestyle=":", label="H = 0.5 (Random Walk)")

            plt.fill_between(valid_hurst.index, 0, 1, where=(valid_hurst < 0.45), color="green", alpha=0.1, label="Mean-Reverting")
            plt.fill_between(valid_hurst.index, 0, 1, where=((valid_hurst >= 0.45) & (valid_hurst <= 0.55)), color="gray", alpha=0.1, label="Random Walk")
            plt.fill_between(valid_hurst.index, 0, 1, where=(valid_hurst > 0.55), color="orange", alpha=0.1, label="Trending")

            plt.title("Rolling Hurst Exponent with Trend Detection")
            plt.xlabel("Time")
            plt.ylabel("Hurst Exponent")
            plt.legend()
            plt.grid(True)

            if output_file:
                plt.savefig(output_file, bbox_inches="tight")
                logger.info("Saved plot to %s", output_file)
            else:
                plt.show()

        self.latest_value = float(valid_hurst.iloc[-1])
        logger.info("Latest Hurst Value: %.4f", self.latest_value)
        return self.latest_value

    def summarize(self) -> dict:
        if self.hurst_series is None:
            raise ValueError("Run compute_rolling() first")
        latest = self.plot(headless=True)
        info = self.interpret(latest)
        return {
            "hurst": round(float(latest), 4),
            "code": info["code"],
            "label": info["label"],
            "strategy": info["strategy"],
            "window": int(self.window),
            "n_valid": int(self.hurst_series.notna().sum()),
        }

    @staticmethod
    def interpret(h: float) -> dict:
        if h > 0.65:
            return {
                "code": "MOMENTUM",
                "label": "Strong persistence",
                "strategy": "Trend is strong and likely to continue --->  favor momentum strategies",
            }
        if 0.55 < h <= 0.65:
            return {
                "code": "MOMENTUM",
                "label": "Mild persistence",
                "strategy": "Possible trend, but not strongly established --->  trade trend cautiously",
            }
        if 0.45 < h <= 0.55:
            return {
                "code": "NEUTRAL",
                "label": "Random walk",
                "strategy": "No clear memory or structure --->  avoid directional bias",
            }
        return {
            "code": "MEAN_REV",
            "label": "Mean-reversion (anti-persistence)",
            "strategy": "Price likely to reverse --->  favor fading/extreme-reversal strategies",
        }


def analyze_ohlc(df: pd.DataFrame, window: int = 100, kind: str = "random_walk", lookback: int = 400) -> tuple[dict, pd.Series, pd.Series]:
    """Vrati (summary, hurst_series, trend_line) za OHLC DataFrame iz UI-ja."""
    n = 0 if df is None else len(df)
    w = max(int(window), MIN_SERIES)
    if n <= w:
        raise ValueError(f"Premalo barova za Hurst (treba > {w})")

    analyzer = HurstAnalyzer(window=w, kind=kind)
    close = analyzer.parse_ohlc(df)
    keep = lookback + analyzer.window
    if len(close) > keep:
        close = close.iloc[-keep:]
    analyzer.compute_rolling(close)
    summary = analyzer.summarize()
    return summary, analyzer.hurst_series, analyzer.trend_line


if __name__ == "__main__":
    import sys

    csv_path = "backtest_prices.csv" if len(sys.argv) < 2 else sys.argv[1]
    df = pd.read_csv(csv_path)

    analyzer = HurstAnalyzer(window=np.min([20, len(df) // 2]), kind="random_walk")
    close_series = analyzer.parse_ohlc(df)
    hurst_series = analyzer.compute_rolling(close_series)
    latest = analyzer.plot(headless=False)

    interpretation = HurstAnalyzer.interpret(latest)
    print(
        "\n"
        + json.dumps(
            {
                "hurst": round(latest, 4),
                "code": interpretation["code"],
                "interpretation": interpretation["label"],
                "strategy": interpretation["strategy"],
            },
            indent=2,
        ),
    )
