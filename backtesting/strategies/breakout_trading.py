"""Breakout Trading — signal na zatvaranju bara, fill na sljedećem open (Backtesting.py redoslijed)."""
import numpy as np

from breakout_signals import compute_breakout_long
from bt_risk import RiskSizedStrategy


class BreakoutTrading(RiskSizedStrategy):
    label = "Breakout Trading"
    min_bars = 210
    min_days = 15  # run_backtest() povećava days ako treba

    trade_on_close = False
    rr_ratio = 3.0  # testiraj bt.run(rr_ratio=2|3|4)
    atr_sl_mult = 1.5
    breakout_buffer_atr = 0.1
    slippage_frac = 0.00015
    reentry_cooldown_bars = 5

    def init(self):
        sig = compute_breakout_long(self.data.df)
        self._range_high = sig["range_high"].to_numpy()
        self._atr = sig["atr"].to_numpy()
        self._setup = sig["setup_long"].to_numpy()
        self._last_entry_bar = -10_000

    def _long_stops(self, entry: float, breakout_level: float, atr_v: float) -> tuple[float, float] | None:
        """Strukturni SL ispod razine proboja; ATR SL; uži stop samo ako je ispod entry."""
        if atr_v <= 0 or entry <= 0:
            return None
        sl_struct = breakout_level - self.breakout_buffer_atr * atr_v
        sl_atr = entry - self.atr_sl_mult * atr_v
        valid = [s for s in (sl_struct, sl_atr) if s < entry]
        if not valid:
            return None
        sl = max(valid)
        risk = entry - sl
        if risk <= 0:
            return None
        tp = entry + self.rr_ratio * risk
        return sl, tp

    def next(self):
        i = len(self.data) - 1
        if i < self.min_bars or self.position:
            return
        if i - self._last_entry_bar < self.reentry_cooldown_bars:
            return
        if not self._setup[i]:
            return

        atr_v = float(self._atr[i])
        breakout_level = float(self._range_high[i])
        if np.isnan(atr_v) or np.isnan(breakout_level):
            return

        # Procjena filla (sljedeći open); broker validira SL/TP prema Close[-1] pri slanju naloga
        close = float(self.data.Close[-1])
        entry_est = close * (1 + self.slippage_frac)
        stops = self._long_stops(entry_est, breakout_level, atr_v)
        if stops is None:
            return
        sl, tp = stops
        self.buy_risk(sl=sl, tp=tp, entry=entry_est, fill=close)
        self._last_entry_bar = i


Strategy = BreakoutTrading
