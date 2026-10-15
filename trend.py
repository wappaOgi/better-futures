"""Potvrda trenda preko higher highs / higher lows (uptrend) i lower highs / lower lows (downtrend)."""
import pandas as pd

from data import resample


def swings(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Naizmjenični swing high/low pivoti, svaki označen kao HH/LH/EH (high) ili HL/LL/EL (low)
    u odnosu na prethodni pivot istog tipa. Prvi pivot svakog tipa nema labelu."""
    w = 2 * n + 1
    ph = df["High"][df["High"] == df["High"].rolling(w, center=True).max()]
    pl = df["Low"][df["Low"] == df["Low"].rolling(w, center=True).min()]
    pts = sorted([(t, p, "H") for t, p in ph.items()] + [(t, p, "L") for t, p in pl.items()])

    seq = []  # dva uzastopna ista tipa -> zadrži ekstremniji
    for t, p, k in pts:
        if seq and seq[-1][2] == k:
            if (k == "H" and p > seq[-1][1]) or (k == "L" and p < seq[-1][1]):
                seq[-1] = (t, p, k)
        else:
            seq.append((t, p, k))

    prev, rows = {"H": None, "L": None}, []
    for t, p, k in seq:
        q, label = prev[k], None
        if q is not None:
            if k == "H":
                label = "HH" if p > q else "LH" if p < q else "EH"
            else:
                label = "HL" if p > q else "LL" if p < q else "EL"
        prev[k] = p
        rows.append((t, p, k, label))
    return pd.DataFrame(rows, columns=["time", "price", "kind", "label"]).set_index("time")


def analyze(df5: pd.DataFrame, rule: str = "1h", n: int = 3, confirm: int = 2) -> dict:
    """Trend za zadani timeframe.
    UP   = zadnjih `confirm` high-ova su HH i zadnjih `confirm` low-ova su HL
    DOWN = zadnjih `confirm` high-ova su LH i zadnjih `confirm` low-ova su LL
    inače RANGE. `broken` = zadnji close je probio invalidaciju (zadnji HL u uptrendu / LH u downtrendu)."""
    d = resample(df5, rule)
    sw = swings(d, n)
    lab = sw.dropna(subset=["label"])
    hs = lab[lab["kind"] == "H"]["label"].tail(confirm).tolist()
    ls = lab[lab["kind"] == "L"]["label"].tail(confirm).tolist()
    full = len(hs) == confirm and len(ls) == confirm

    trend = "RANGE"
    if full and set(hs) == {"HH"} and set(ls) == {"HL"}:
        trend = "UP"
    elif full and set(hs) == {"LH"} and set(ls) == {"LL"}:
        trend = "DOWN"

    last_close, inval, broken = d["Close"].iloc[-1], None, False
    if trend == "UP":
        inval = sw[sw["kind"] == "L"]["price"].iloc[-1]
        broken = last_close < inval
    elif trend == "DOWN":
        inval = sw[sw["kind"] == "H"]["price"].iloc[-1]
        broken = last_close > inval
    return dict(trend=trend, invalidation=inval, broken=bool(broken), last_highs=hs, last_lows=ls, swings=sw)


if __name__ == "__main__":
    from data import load_5m

    for tf in ("1h", "5min"):
        r = analyze(load_5m(), tf)
        print(tf, r["trend"], r["last_highs"], r["last_lows"], "inval:", r["invalidation"], "broken:", r["broken"])
