"""Trejdovi: JSON storage, bodovi, rezolucija, overlay na grafu."""
import json
import uuid
from pathlib import Path

import pandas as pd

from bank import load as load_bank, risk_eur, recalc

FILE = Path(__file__).with_name("trades.json")
GREEN, RED, BLUE, AMBER = "#26a69a", "#ef5350", "#42a5f5", "#ffc107"


def load() -> list:
    if FILE.exists():
        return json.loads(FILE.read_text())
    return []


def save(trades: list) -> None:
    FILE.write_text(json.dumps(trades, indent=2, default=str))


def pts(entry: float, price: float, direction: str) -> float:
    diff = price - entry
    return round(diff if direction == "long" else -diff, 2)


def pnl_eur_for(trade: dict, bank: dict | None = None) -> float:
    b = bank or load_bank()
    risk = trade.get("risk_eur") or risk_eur(b)
    sl_pts = abs(trade.get("sl_points") or 0)
    pnl_pts = trade.get("pnl_points") or 0
    if not sl_pts:
        return 0.0
    return round(pnl_pts * (risk / sl_pts), 2)


def candle_at(df5: pd.DataFrame, ts) -> dict:
    idx = df5.index.searchsorted(pd.Timestamp(ts), side="right") - 1
    idx = max(0, min(idx, len(df5) - 1))
    row = df5.iloc[idx]
    t = df5.index[idx]
    return {
        "time": str(t),
        "open": float(row["Open"]),
        "high": float(row["High"]),
        "low": float(row["Low"]),
        "close": float(row["Close"]),
    }


def entry_from_click(df5: pd.DataFrame, click_x) -> tuple[pd.Timestamp, float, dict]:
    ts = pd.Timestamp(click_x)
    idx = df5.index.searchsorted(ts, side="right") - 1
    idx = max(0, min(idx, len(df5) - 1))
    row = df5.iloc[idx]
    t = df5.index[idx]
    entry = float(row["Close"])
    return t, entry, candle_at(df5, t)


def make_draft(df5: pd.DataFrame, click_x, click_y, size_eur: float, margin: int) -> dict:
    opened_at, entry, candle = entry_from_click(df5, click_x)
    direction = "long" if float(click_y) >= entry else "short"
    sl_dist, tp_dist = 30.0, 60.0
    if direction == "long":
        sl, tp = entry - sl_dist, entry + tp_dist
    else:
        sl, tp = entry + sl_dist, entry - tp_dist
    bank = load_bank()
    return {
        "active": True,
        "opened_at": str(opened_at),
        "direction": direction,
        "entry_price": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "sl_points": pts(entry, sl, direction),
        "tp_points": pts(entry, tp, direction),
        "size_eur": size_eur,
        "margin": margin,
        "risk_eur": risk_eur(bank),
        "candle": candle,
    }


def from_draft(draft: dict, test: bool = False) -> dict:
    bank = load_bank()
    return {
        "id": str(uuid.uuid4())[:8],
        "test": test,
        "status": "open",
        "opened_at": draft["opened_at"],
        "direction": draft["direction"],
        "entry_price": draft["entry_price"],
        "stop_loss": draft["stop_loss"],
        "take_profit": draft["take_profit"],
        "sl_points": draft["sl_points"],
        "tp_points": draft["tp_points"],
        "size_eur": draft.get("size_eur", bank["balance_eur"]),
        "margin": draft.get("margin", bank["margin"]),
        "risk_eur": draft.get("risk_eur", risk_eur(bank)),
        "candle": draft["candle"],
        "closed_at": None,
        "exit_price": None,
        "pnl_points": None,
        "pnl_eur": None,
    }


def resolve_one(trade: dict, df5: pd.DataFrame) -> dict:
    if trade.get("status") in ("win", "loss"):
        return trade
    t0 = pd.Timestamp(trade["opened_at"])
    future = df5[df5.index > t0]
    if future.empty:
        return trade
    entry = trade["entry_price"]
    sl, tp = trade["stop_loss"], trade["take_profit"]
    direction = trade["direction"]
    for ts, row in future.iterrows():
        h, l = float(row["High"]), float(row["Low"])
        if direction == "long":
            hit_sl, hit_tp = l <= sl, h >= tp
        else:
            hit_sl, hit_tp = h >= sl, l <= tp
        if hit_sl and hit_tp:
            # isti candle — bliži entry pogođen prvi (konzervativno SL)
            trade = _close(trade, "loss", sl, ts, df5)
            break
        if hit_sl:
            trade = _close(trade, "loss", sl, ts, df5)
            break
        if hit_tp:
            trade = _close(trade, "win", tp, ts, df5)
            break
    return trade


def _close(trade: dict, status: str, exit_price: float, closed_at, df5: pd.DataFrame) -> dict:
    t = trade.copy()
    pnl_pts = pts(t["entry_price"], exit_price, t["direction"])
    t.update({
        "status": status,
        "exit_price": round(exit_price, 2),
        "closed_at": str(closed_at),
        "pnl_points": pnl_pts,
    })
    t["pnl_eur"] = pnl_eur_for(t)
    return t


def resolve_all(trades: list, df5: pd.DataFrame) -> list:
    return [resolve_one(t, df5) for t in trades]


def add_trade(draft: dict, df5: pd.DataFrame) -> list:
    trades = resolve_all(load(), df5)
    trades.append(from_draft(draft))
    trades = resolve_all(trades, df5)
    save(trades)
    recalc(trades)
    return trades


def seed_test_trades(df5: pd.DataFrame) -> list:
    trades = load()
    if trades:
        trades = resolve_all(trades, df5)
        save(trades)
        recalc(trades)
        return trades

    bank = load_bank()
    days = sorted({t.date() for t in df5.index})[-5:]
    for i, day in enumerate(days):
        day_df = df5[df5.index.date == day]
        if len(day_df) < 4:
            continue
        mid = day_df.iloc[len(day_df) // 2]
        ts = day_df.index[len(day_df) // 2]
        entry = float(mid["Close"])
        direction = "long" if i % 2 == 0 else "short"
        sl = entry - 35 if direction == "long" else entry + 35
        tp = entry + 70 if direction == "long" else entry - 70
        candle = {
            "time": str(ts),
            "open": float(mid["Open"]),
            "high": float(mid["High"]),
            "low": float(mid["Low"]),
            "close": entry,
        }
        trades.append({
            "id": f"test{i+1}",
            "test": True,
            "status": "open",
            "opened_at": str(ts),
            "direction": direction,
            "entry_price": entry,
            "stop_loss": sl,
            "take_profit": tp,
            "sl_points": pts(entry, sl, direction),
            "tp_points": pts(entry, tp, direction),
            "size_eur": bank["balance_eur"],
            "margin": bank["margin"],
            "risk_eur": risk_eur(bank),
            "candle": candle,
            "closed_at": None,
            "exit_price": None,
            "pnl_points": None,
            "pnl_eur": None,
        })

    trades = resolve_all(trades, df5)
    save(trades)
    recalc(trades)
    return trades


def _rgba(color: str, a: float) -> str:
    c = color.lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{a:.2f})"


def _trade_end(trade: dict, x_end) -> pd.Timestamp:
    if trade.get("closed_at"):
        return pd.Timestamp(trade["closed_at"])
    return pd.Timestamp(x_end)


def _box_label(x, y, text, color, anchor="left", valign="middle"):
    return dict(
        x=x, y=y, xref="x", yref="y", text=text, showarrow=False,
        xanchor=anchor, yanchor=valign, font=dict(size=9, color=color),
        bgcolor="rgba(0,0,0,0.45)", borderpad=2, name="trade",
    )


def _box_pad(lo: float, hi: float) -> tuple[float, float]:
    pad = max((hi - lo) * 0.22, 18.0)
    return lo - pad, hi + pad


def chart_overlays(trade: dict, x_end) -> tuple[list, list]:
    """Plavi box s oznakama: smjer, entry, SL, TP; ✓/✗ gore desno."""
    shapes, ann = [], []
    t0 = pd.Timestamp(trade["opened_at"])
    t1 = _trade_end(trade, x_end)
    entry = trade["entry_price"]
    sl, tp = trade["stop_loss"], trade["take_profit"]
    sl_pts, tp_pts = trade.get("sl_points", 0), trade.get("tp_points", 0)
    direction = trade["direction"]
    status = trade.get("status", "open")
    lo, hi = min(sl, tp, entry), max(sl, tp, entry)
    box_lo, box_hi = _box_pad(lo, hi)
    t_lbl = t0 + (t1 - t0) * 0.04

    shapes.extend([
        dict(type="rect", xref="x", yref="y", x0=t0, x1=t1, y0=box_lo, y1=box_hi,
             fillcolor=_rgba(BLUE, 0.10), line=dict(color=BLUE, width=2), layer="above", name="trade"),
        dict(type="line", xref="x", yref="y", x0=t0, x1=t1, y0=entry, y1=entry,
             line=dict(color=BLUE, width=1, dash="dash"), layer="above", name="trade"),
        dict(type="line", xref="x", yref="y", x0=t0, x1=t1, y0=sl, y1=sl,
             line=dict(color=RED, width=1, dash="dot"), layer="above", name="trade"),
        dict(type="line", xref="x", yref="y", x0=t0, x1=t1, y0=tp, y1=tp,
             line=dict(color=GREEN, width=1, dash="dot"), layer="above", name="trade"),
    ])

    sym = "▲ LONG" if direction == "long" else "▼ SHORT"
    if trade.get("test"):
        sym = f"[test] {sym}"
    ann.append(_box_label(t_lbl, box_hi, sym, BLUE, valign="top"))

    ann.append(_box_label(t_lbl, tp, f"TP {tp:.2f} · {tp_pts:+.0f} pts", GREEN))
    ann.append(_box_label(t_lbl, entry, f"E {entry:.2f} · 0 pts", BLUE))
    ann.append(_box_label(t_lbl, sl, f"SL {sl:.2f} · {sl_pts:+.0f} pts", RED))

    if status in ("win", "loss"):
        icon = "✓" if status == "win" else "✗"
        icon_col = GREEN if status == "win" else RED
        pnl_pts = trade.get("pnl_points") or 0
        pnl_e = trade.get("pnl_eur") or 0
        sign = "+" if pnl_e >= 0 else ""
        ann.append(dict(
            x=t1, y=box_hi, xref="x", yref="y", text=icon, showarrow=False,
            xanchor="right", yanchor="top", font=dict(size=16, color=icon_col),
            bgcolor="rgba(0,0,0,0.5)", borderpad=3, name="trade",
        ))
        mid_y = entry + (tp - entry) / 2 if status == "win" else entry + (sl - entry) / 2
        ann.append(_box_label(
            t_lbl, mid_y, f"{sign}{pnl_pts:.0f} pts · {sign}{pnl_e:.2f} €", icon_col,
        ))

    return shapes, ann


def draft_overlays(draft: dict, x_end) -> tuple[list, list]:
    """Draft box + povlačive SL/TP trake (vertikalno)."""
    if not draft or not draft.get("active"):
        return [], []
    t0 = pd.Timestamp(draft["opened_at"])
    t1 = pd.Timestamp(x_end)
    entry = draft["entry_price"]
    sl, tp = draft["stop_loss"], draft["take_profit"]
    sl_pts, tp_pts = draft.get("sl_points", 0), draft.get("tp_points", 0)
    lo, hi = min(sl, tp, entry), max(sl, tp, entry)
    box_lo, box_hi = _box_pad(lo, hi)

    shapes = [
        dict(type="rect", xref="x", yref="y", x0=t0, x1=t1, y0=box_lo, y1=box_hi,
             fillcolor=_rgba(BLUE, 0.08), line=dict(color=BLUE, width=2, dash="dash"),
             layer="above", name="draft"),
        dict(type="line", xref="x", yref="y", x0=t0, x1=t1, y0=entry, y1=entry,
             line=dict(color=BLUE, width=1.5, dash="dash"), name="draft"),
        dict(type="rect", xref="x", yref="y", x0=t0, x1=t1, y0=sl - 6, y1=sl + 6,
             fillcolor=_rgba(RED, 0.35), line=dict(color=RED, width=2), layer="above", name="draft-sl"),
        dict(type="rect", xref="x", yref="y", x0=t0, x1=t1, y0=tp - 6, y1=tp + 6,
             fillcolor=_rgba(GREEN, 0.35), line=dict(color=GREEN, width=2), layer="above", name="draft-tp"),
    ]
    t_lbl = t0 + (t1 - t0) * 0.04
    sym = "▲ LONG" if draft["direction"] == "long" else "▼ SHORT"
    ann = [
        _box_label(t_lbl, box_hi, sym, BLUE, valign="top"),
        _box_label(t_lbl, tp, f"TP {tp:.2f} · {tp_pts:+.0f} pts", GREEN),
        _box_label(t_lbl, entry, f"E {entry:.2f} · 0 pts", BLUE),
        _box_label(t_lbl, sl, f"SL {sl:.2f} · {sl_pts:+.0f} pts", RED),
    ]
    for a in ann:
        a["name"] = "draft"
    return shapes, ann


def update_draft_levels(draft: dict, sl: float | None = None, tp: float | None = None) -> dict:
    d = draft.copy()
    entry = d["entry_price"]
    direction = d["direction"]
    if sl is not None:
        d["stop_loss"] = sl
        d["sl_points"] = pts(entry, sl, direction)
    if tp is not None:
        d["take_profit"] = tp
        d["tp_points"] = pts(entry, tp, direction)
    return d
