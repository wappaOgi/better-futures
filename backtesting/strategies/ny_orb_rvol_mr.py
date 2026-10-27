"""NY ORB mean reversion — failed breakout + VWAP, RVOL + ATR filter, R:R 1:3."""
import numpy as np

from bt_risk import RiskSizedStrategy
from indicators import BB_PERIOD, BB_STD, RSI_PERIOD, bollinger, rsi
from ny_orb import compute_ny_orb, compute_ny_vwap


def _as_indicator(arr):
    def _fn(*_):
        return arr
    return _fn


class NyOrbRvolMr(RiskSizedStrategy):
    label = "NY ORB + RVOL + ATR + MR (1:3)"
    required_timeframe = "5min"
    min_days = 20
    min_bars = 200

    rvol_threshold = 2.0
    rvol_lookback = 14
    atr_period = 5
    rr_ratio = 3.0
    bb_period = BB_PERIOD
    bb_std = BB_STD
    rsi_period = RSI_PERIOD

    def init(self):
        df = self.data.df
        sig = compute_ny_orb(
            df,
            rvol_lookback=self.rvol_lookback,
            atr_period=self.atr_period,
        )
        bb = bollinger(df["Close"], self.bb_period, self.bb_std)
        self._or_high = sig["or_high"].to_numpy()
        self._or_low = sig["or_low"].to_numpy()
        self._rvol = sig["rvol"].to_numpy()
        self._post_or = sig["post_or"].to_numpy()
        self._volatile = sig["volatile"].to_numpy()
        self._day_id = sig["day_id"].to_numpy()
        self._vwap = compute_ny_vwap(df).to_numpy()
        self._bb_upper = bb["bb_upper"].to_numpy()
        self._bb_lower = bb["bb_lower"].to_numpy()
        self._rsi = rsi(df["Close"], self.rsi_period).to_numpy()
        self._last_traded_day = -1
        self._track_day = -1
        self._broke_high = False
        self._broke_low = False
        self._session_high = -np.inf
        self._session_low = np.inf

        self.or_high_line = self.I(
            _as_indicator(self._or_high), self.data.Close,
            name="OR High", overlay=True,
        )
        self.or_low_line = self.I(
            _as_indicator(self._or_low), self.data.Close,
            name="OR Low", overlay=True,
        )

    def _reset_day(self, day: int):
        self._track_day = day
        self._broke_high = False
        self._broke_low = False
        self._session_high = -np.inf
        self._session_low = np.inf

    def _passes_filters(self, i: int) -> bool:
        if not self._post_or[i]:
            return False
        if self._day_id[i] == self._last_traded_day:
            return False
        if np.isnan(self._rvol[i]) or self._rvol[i] < self.rvol_threshold:
            return False
        if not self._volatile[i]:
            return False
        or_h, or_l = self._or_high[i], self._or_low[i]
        return not (np.isnan(or_h) or np.isnan(or_l))

    def next(self):
        if self.position:
            return

        i = len(self.data) - 1
        if i < 1:
            return

        day = self._day_id[i]
        if day != self._track_day:
            self._reset_day(day)

        high, low, close = self.data.High[-1], self.data.Low[-1], self.data.Close[-1]
        prev_close = self.data.Close[-2]
        or_h, or_l = self._or_high[i], self._or_low[i]

        if self._post_or[i] and not (np.isnan(or_h) or np.isnan(or_l)):
            if high > or_h:
                self._broke_high = True
            if low < or_l:
                self._broke_low = True
            self._session_high = max(self._session_high, high)
            self._session_low = min(self._session_low, low)

        if not self._passes_filters(i):
            return

        vwap = self._vwap[i]

        rsi_v = self._rsi[i]
        bb_up, bb_lo = self._bb_upper[i], self._bb_lower[i]

        # failed upside break → short (MR + RSI>70 + BB upper)
        if self._broke_high and prev_close >= or_h and close < or_h:
            if (
                not np.isnan(vwap) and close > vwap
                and not np.isnan(rsi_v) and rsi_v > 70
                and not np.isnan(bb_up) and close >= bb_up
            ):
                sl = max(self._session_high, high)
                risk = sl - close
                if risk <= 0:
                    return
                tp = min(close - self.rr_ratio * risk, vwap)
                self.sell_risk(sl=sl, tp=tp)
                self._last_traded_day = day
            return

        # failed downside break → long (MR + RSI<30 + BB lower)
        if self._broke_low and prev_close <= or_l and close > or_l:
            if (
                not np.isnan(vwap) and close < vwap
                and not np.isnan(rsi_v) and rsi_v < 30
                and not np.isnan(bb_lo) and close <= bb_lo
            ):
                sl = min(self._session_low, low)
                risk = close - sl
                if risk <= 0:
                    return
                tp = max(close + self.rr_ratio * risk, vwap)
                self.buy_risk(sl=sl, tp=tp)
                self._last_traded_day = day


Strategy = NyOrbRvolMr
