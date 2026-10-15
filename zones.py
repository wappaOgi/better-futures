"""Support/resistance zone iz 1h candleova.

Ideja: nađi swing pivote (lokalne high/low), grupiraj one koji su blizu jedan drugome
(tolerancija = dio ATR-a) i svaku grupu s dovoljno dodira proglasi zonom.
"""
import pandas as pd

from data import resample


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def pivots(df: pd.DataFrame, n: int = 3):
    """Pivot high/low: ekstrem unutar n candleova lijevo i desno."""
    w = 2 * n + 1
    ph = df["High"][df["High"] == df["High"].rolling(w, center=True).max()]
    pl = df["Low"][df["Low"] == df["Low"].rolling(w, center=True).min()]
    return ph, pl


def get_zones(df5: pd.DataFrame, rule: str = "1h", n: int = 3, tol_atr: float = 0.5,
              min_touches: int = 3, min_width_atr: float = 0.2, top: int = 12) -> pd.DataFrame:
    """Vraća `top` najjačih zona kao DataFrame: low, high, touches, start, kind (support/resistance/inside)."""
    h = resample(df5, rule)
    a = atr(h).iloc[-1]
    ph, pl = pivots(h, n)
    pts = sorted([(p, t) for t, p in ph.items()] + [(p, t) for t, p in pl.items()])
    if not pts:
        return pd.DataFrame(columns=["low", "high", "touches", "start", "kind"])

    # grupiranje: nova grupa kad pivot odmakne više od tol_atr * ATR od dna grupe
    groups, cur = [], [pts[0]]
    for p, t in pts[1:]:
        if p - cur[0][0] <= tol_atr * a:
            cur.append((p, t))
        else:
            groups.append(cur)
            cur = [(p, t)]
    groups.append(cur)

    last, rows = h["Close"].iloc[-1], []
    for g in groups:
        if len(g) < min_touches:
            continue
        lo, hi = min(p for p, _ in g), max(p for p, _ in g)
        if hi - lo < min_width_atr * a:  # zona mora imati debljinu, ne samo liniju
            mid, half = (lo + hi) / 2, min_width_atr * a / 2
            lo, hi = mid - half, mid + half
        kind = "support" if hi < last else "resistance" if lo > last else "inside"
        rows.append(dict(low=lo, high=hi, touches=len(g), start=min(t for _, t in g), kind=kind))
    df = pd.DataFrame(rows, columns=["low", "high", "touches", "start", "kind"])
    return df.sort_values("touches", ascending=False).head(top).sort_values("low").reset_index(drop=True)


if __name__ == "__main__":
    from data import load_5m

    print(get_zones(load_5m()).to_string())
