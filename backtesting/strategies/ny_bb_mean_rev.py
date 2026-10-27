"""NY sesija — BB mean reversion, RSI ekstremi."""
import numpy as np

from bt_risk import RiskSizedStrategy
from indicators import BB_PERIOD, BB_STD, RSI_PERIOD, SMA_TREND, bollinger, rsi, sma
from ny_orb import ny_session_mask


class NyBbMeanRev(RiskSizedStrategy):
    label = "NY BB Mean Rev (R:R 1:3)"
    chart_profile = "ny"
    required_timeframe = "5min"
    min_days = 10
    min_bars = 200

    bb_period = BB_PERIOD
    bb_std = BB_STD
    rsi_period = RSI_PERIOD
    rsi_short = 75
    rsi_long = 30
    sl_points = 5
    rr_ratio = 3.0

    def init(self):
        df = self.data.df
        bb = bollinger(df["Close"], self.bb_period, self.bb_std)
        self._bb_upper = bb["bb_upper"].to_numpy()
        self._bb_lower = bb["bb_lower"].to_numpy()
        self._rsi = rsi(df["Close"], self.rsi_period).to_numpy()
        self._sma200 = sma(df["Close"], SMA_TREND).to_numpy()
        self._in_ny = ny_session_mask(df)
        med = float(np.nanmedian(self.data.Close))
        self._sl_off = self.sl_points * 1e-8 if med < 1.0 else float(self.sl_points)

    def _rsi_vals(self, i: int, n: int = 3) -> list[float] | None:
        if i < n - 1:
            return None
        vals = [self._rsi[i - k] for k in range(n - 1, -1, -1)]
        if any(np.isnan(v) for v in vals):
            return None
        return vals

    def _rsi_long_turn(self, i: int) -> bool:
        """RSI < 30, viši dno od prethodnog, kreće prema gore (prema sredini)."""
        v = self._rsi_vals(i, 3)
        if not v:
            return False
        r0, r1, r2 = v
        if r0 >= self.rsi_long:
            return False
        if r0 <= r1:
            return False
        return r1 < r2 and r0 > r1

    def _rsi_short_turn(self, i: int) -> bool:
        """RSI > 75, niži vrh od prethodnog, kreće prema dolje (prema sredini)."""
        v = self._rsi_vals(i, 3)
        if not v:
            return False
        r0, r1, r2 = v
        if r0 <= self.rsi_short:
            return False
        if r0 >= r1:
            return False
        return r1 > r2 and r0 < r1

    def next(self):
        if self.position:
            return

        i = len(self.data) - 1
        if i < max(self.bb_period, self.rsi_period, SMA_TREND, 2):
            return
        if not self._in_ny[i]:
            return

        sma200 = self._sma200[i]
        if np.isnan(sma200):
            return

        rsi_v = self._rsi[i]
        bb_up = self._bb_upper[i]
        bb_lo = self._bb_lower[i]
        if any(np.isnan(x) for x in (rsi_v, bb_up, bb_lo)):
            return

        open_ = self.data.Open[-1]
        high, low, close = self.data.High[-1], self.data.Low[-1], self.data.Close[-1]
        bearish = close < open_
        bullish = close >= open_

        # short: crvena svijeća na gornjem BB + RSI okret, ispod SMA200
        if high >= bb_up and bearish and close < sma200 and self._rsi_short_turn(i):
            sl = high + self._sl_off
            risk = sl - close
            if risk <= 0:
                return
            tp = close - self.rr_ratio * risk
            if sl > tp:
                self.sell_risk(sl=sl, tp=tp)
            return

        # long: zelena svijeća na donjem BB + RSI okret, iznad SMA200
        if low <= bb_lo and bullish and close > sma200 and self._rsi_long_turn(i):
            sl = low - self._sl_off
            risk = close - sl
            if risk <= 0:
                return
            tp = close + self.rr_ratio * risk
            if sl < tp and sl > 0:
                self.buy_risk(sl=sl, tp=tp)


Strategy = NyBbMeanRev
