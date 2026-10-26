"""SMA crossover — brza MA presječe sporu MA → long, obrnuto → short."""
from backtesting import Strategy
from backtesting.lib import crossover


def _sma(values, period: int):
    import pandas as pd
    return pd.Series(values).rolling(period).mean()


class SmaCrossover(Strategy):
    label = "SMA Crossover (20/50)"
    fast_period = 20
    slow_period = 50

    def init(self):
        close = self.data.Close
        self.fast = self.I(_sma, close, self.fast_period)
        self.slow = self.I(_sma, close, self.slow_period)

    def next(self):
        if crossover(self.fast, self.slow):
            self.buy()
        elif crossover(self.slow, self.fast):
            self.sell()


Strategy = SmaCrossover
