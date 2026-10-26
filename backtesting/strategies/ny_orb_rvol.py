"""NY open-range breakout — RVOL (1h) + ATR(5) filter, R:R 1:3."""
import numpy as np
from backtesting import Strategy

from ny_orb import compute_ny_orb


def _as_indicator(arr):
    def _fn(*_):
        return arr
    return _fn


class NyOrbRvol(Strategy):
    label = "NY ORB + RVOL + ATR (1:3)"
    required_timeframe = "5min"
    min_days = 20
    min_bars = 200

    rvol_threshold = 2.0
    rvol_lookback = 14
    atr_period = 5
    rr_ratio = 3.0

    def init(self):
        sig = compute_ny_orb(
            self.data.df,
            rvol_lookback=self.rvol_lookback,
            atr_period=self.atr_period,
        )
        self._or_high = sig["or_high"].to_numpy()
        self._or_low = sig["or_low"].to_numpy()
        self._rvol = sig["rvol"].to_numpy()
        self._post_or = sig["post_or"].to_numpy()
        self._volatile = sig["volatile"].to_numpy()
        self._day_id = sig["day_id"].to_numpy()
        self._last_traded_day = -1

        self.or_high_line = self.I(
            _as_indicator(self._or_high), self.data.Close,
            name="OR High", overlay=True,
        )
        self.or_low_line = self.I(
            _as_indicator(self._or_low), self.data.Close,
            name="OR Low", overlay=True,
        )

    def next(self):
        if self.position:
            return

        i = len(self.data) - 1
        if i < 1 or not self._post_or[i]:
            return
        if self._day_id[i] == self._last_traded_day:
            return
        if np.isnan(self._rvol[i]) or self._rvol[i] < self.rvol_threshold:
            return
        if not self._volatile[i]:
            return

        or_h, or_l = self._or_high[i], self._or_low[i]
        if np.isnan(or_h) or np.isnan(or_l):
            return

        prev_close = self.data.Close[-2]
        high, low, close = self.data.High[-1], self.data.Low[-1], self.data.Close[-1]

        # proboj OR high → long
        if high > or_h and prev_close <= or_h:
            risk = close - or_l
            if risk <= 0:
                return
            self.buy(sl=or_l, tp=close + self.rr_ratio * risk)
            self._last_traded_day = self._day_id[i]
            return

        # proboj OR low → short
        if low < or_l and prev_close >= or_l:
            risk = or_h - close
            if risk <= 0:
                return
            self.sell(sl=or_h, tp=close - self.rr_ratio * risk)
            self._last_traded_day = self._day_id[i]


Strategy = NyOrbRvol
