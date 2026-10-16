"""Potvrda trenda preko higher highs / higher lows (uptrend) i lower highs / lower lows (downtrend)."""
import pandas as pd

from data import resample

GREEN, RED = "#26a69a", "#ef5350"
STRIP = {"UP": ("#1b5e20", 0.32), "DOWN": ("#8b1a1a", 0.32)}
STRIP_Y = (0.066, 0.080)


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


def _rgba(color: str, a: float) -> str:
    c = color.lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{a:.2f})"


UP_SEQ = [("L", "HL"), ("H", "HH"), ("L", "HL")]
DOWN_SEQ = [("H", "LH"), ("L", "LL"), ("H", "LH")]


def _seq_match(recent: list[tuple[str, str]], pattern: list[tuple[str, str]]) -> bool:
    return len(recent) >= len(pattern) and recent[-len(pattern):] == pattern


def _breaks_up(kind: str, label: str) -> bool:
    return (kind == "H" and label == "LH") or (kind == "L" and label == "LL")


def _breaks_down(kind: str, label: str) -> bool:
    return (kind == "H" and label == "HH") or (kind == "L" and label == "HL")


def _walk_trends(labeled: pd.DataFrame) -> tuple[list[dict], str | None]:
    """Vraća periode i trenutni trend na kraju niza."""
    recent, periods, current = [], [], None
    for t, row in labeled.iterrows():
        k, label = row["kind"], row["label"]
        recent.append((k, label))

        if current:
            brk = _breaks_up(k, label) if current["trend"] == "UP" else _breaks_down(k, label)
            if brk:
                current["end"] = t
                periods.append(current)
                current = None

        if _seq_match(recent, UP_SEQ):
            if not current or current["trend"] != "UP":
                if current:
                    current["end"] = t
                    periods.append(current)
                current = {"trend": "UP", "start": t}
        elif _seq_match(recent, DOWN_SEQ):
            if not current or current["trend"] != "DOWN":
                if current:
                    current["end"] = t
                    periods.append(current)
                current = {"trend": "DOWN", "start": t}

    if current:
        periods.append(current)
    active = current["trend"] if current else None
    return periods, active


def trend_periods(sw: pd.DataFrame) -> list[dict]:
    """UP tek nakon HL → HH → HL; DOWN nakon LH → LL → LH."""
    labeled = sw.dropna(subset=["label"])
    if labeled.empty:
        return []
    periods, _ = _walk_trends(labeled)
    return periods


def trend_overlays(sw: pd.DataFrame, x_end) -> tuple[list, list]:
    """Trake potencijalnog trenda ispod grafa (paper y)."""
    shapes, annotations = [], []
    y0, y1 = STRIP_Y
    for p in trend_periods(sw):
        t0 = pd.Timestamp(p["start"])
        t1 = pd.Timestamp(p.get("end") or x_end)
        if t1 <= t0:
            continue
        col, alpha = STRIP[p["trend"]]
        shapes.append(dict(
            type="rect", xref="x", yref="paper", x0=t0, x1=t1, y0=y0, y1=y1,
            fillcolor=_rgba(col, alpha), line=dict(color=_rgba(col, alpha + 0.25), width=1),
            layer="below", name="trendstrip",
        ))
        if t1 - t0 >= pd.Timedelta(hours=1):
            sym = "▲ UP" if p["trend"] == "UP" else "▼ DOWN"
            txt_col = GREEN if p["trend"] == "UP" else RED
            annotations.append(dict(
                x=t0 + (t1 - t0) / 2, y=(y0 + y1) / 2, xref="x", yref="paper",
                text=sym, showarrow=False, xanchor="center", yanchor="middle",
                font=dict(size=8, color=txt_col), name="trendstrip",
            ))
    return shapes, annotations


def analyze(df5: pd.DataFrame, rule: str = "1h", n: int = 3, confirm: int = 2) -> dict:
    """Trend za zadani timeframe.
    UP   = potvrđen nizom HL → HH → HL
    DOWN = potvrđen nizom LH → LL → LH
    inače RANGE. `broken` = zadnji close je probio invalidaciju (zadnji HL u uptrendu / LH u downtrendu)."""
    d = resample(df5, rule)
    sw = swings(d, n)
    lab = sw.dropna(subset=["label"])
    _, state = _walk_trends(lab)
    trend = state if state else "RANGE"
    hs = lab[lab["kind"] == "H"]["label"].tolist()
    ls = lab[lab["kind"] == "L"]["label"].tolist()

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
