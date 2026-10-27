"""Position sizing — fiksni % rizika kapitala po trejdu."""
from backtesting import Strategy


def size_for_risk(equity: float, risk_pct: float, entry: float, sl: float) -> float:
    """
    Udio equity (0–1) za Backtesting.py.
    Gubitak do SL ≈ equity × risk_pct: size = risk_pct × entry / |entry − sl|
    """
    entry, sl = float(entry), float(sl)
    dist = abs(entry - sl)
    if dist <= 0 or risk_pct <= 0 or equity <= 0 or entry <= 0:
        return 0.0
    frac = (risk_pct * entry) / dist
    if frac <= 0:
        return 0.0
    return min(0.999, frac)


class RiskSizedStrategy(Strategy):
    risk_pct = 0.01  # 1% — override u bt.run(risk_pct=...)

    def _risk_size(self, entry: float, sl: float) -> float:
        return size_for_risk(self.equity, self.risk_pct, entry, sl)

    def buy_risk(self, sl: float, tp: float, entry: float | None = None):
        entry = float(self.data.Close[-1] if entry is None else entry)
        size = self._risk_size(entry, sl)
        if size > 0:
            self.buy(size=size, sl=sl, tp=tp)

    def sell_risk(self, sl: float, tp: float, entry: float | None = None):
        entry = float(self.data.Close[-1] if entry is None else entry)
        size = self._risk_size(entry, sl)
        if size > 0:
            self.sell(size=size, sl=sl, tp=tp)
