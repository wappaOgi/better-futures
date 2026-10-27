"""Yahoo day gainers/losers + dnevna povijest u JSON (cijena, % promjena, komentar)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

FILE = Path(__file__).with_name("market_movers.json")
ET = "America/New_York"
MOVER_COUNT = 25  # 25 gainers + 25 losers = 50


def trading_day_key(when: pd.Timestamp | None = None) -> str:
    ts = when or pd.Timestamp.now(tz=ET)
    return ts.strftime("%Y-%m-%d")


def _quote_row(q: dict, side: str, comment: str = "") -> dict[str, Any]:
    return {
        "symbol": q.get("symbol"),
        "shortName": q.get("shortName") or q.get("longName") or q.get("symbol"),
        "regularMarketPrice": q.get("regularMarketPrice"),
        "regularMarketChangePercent": q.get("regularMarketChangePercent"),
        "regularMarketVolume": q.get("regularMarketVolume"),
        "side": side,
        "comment": comment or "",
    }


def _storage_row(row: dict, side: str) -> dict[str, Any]:
    """Samo polja koja idu u JSON."""
    return {
        "symbol": row.get("symbol"),
        "regularMarketPrice": row.get("regularMarketPrice"),
        "regularMarketChangePercent": row.get("regularMarketChangePercent"),
        "side": side,
        "comment": (row.get("comment") or "").strip(),
    }


def _comments_map(day: str) -> dict[tuple[str, str], str]:
    hist = load_history()
    block = hist["by_date"].get(day) or {}
    out: dict[tuple[str, str], str] = {}
    for side_key, side in (("gainers", "gainer"), ("losers", "loser")):
        for row in block.get(side_key) or []:
            sym = row.get("symbol")
            if sym:
                out[(side, sym)] = row.get("comment") or ""
    return out


def _screen(name: str, count: int) -> list[dict]:
    raw = yf.screen(name, count=count)
    if not isinstance(raw, dict):
        raise RuntimeError(f"yfinance screen {name!r} nije vratio dict")
    quotes = raw.get("quotes") or []
    return [q for q in quotes if q.get("symbol")]


def get_market_movers(count: int = MOVER_COUNT) -> tuple[list[dict], list[dict]]:
    gainers_q = _screen("day_gainers", count)
    losers_q = _screen("day_losers", count)
    gainers = [_quote_row(q, "gainer") for q in gainers_q]
    losers = [_quote_row(q, "loser") for q in losers_q]
    return gainers, losers


def load_history() -> dict[str, Any]:
    if not FILE.exists():
        return {"by_date": {}}
    data = json.loads(FILE.read_text())
    data.setdefault("by_date", {})
    return data


def save_history(data: dict[str, Any]) -> None:
    FILE.write_text(json.dumps(data, indent=2, default=str))


def save_daily_snapshot(gainers: list[dict], losers: list[dict], day: str | None = None) -> str:
    day = day or trading_day_key()
    comments = _comments_map(day)
    hist = load_history()
    by_date = hist["by_date"]

    def _apply_comments(rows: list[dict], side: str) -> list[dict]:
        out = []
        for r in rows:
            sym = r.get("symbol")
            c = comments.get((side, sym), r.get("comment") or "")
            merged = {**r, "comment": c}
            out.append(_storage_row(merged, side))
        return out

    by_date[day] = {
        "gainers": _apply_comments(gainers, "gainer"),
        "losers": _apply_comments(losers, "loser"),
        "saved_at": pd.Timestamp.now(tz=ET).isoformat(),
    }
    save_history(hist)
    return day


def save_comment(day: str, symbol: str, side: str, comment: str) -> None:
    day = day or trading_day_key()
    sym = (symbol or "").strip().upper()
    side_key = "gainers" if side == "gainer" else "losers"
    hist = load_history()
    block = hist["by_date"].setdefault(day, {"gainers": [], "losers": []})
    rows = block.get(side_key) or []
    found = False
    for row in rows:
        if (row.get("symbol") or "").upper() == sym:
            row["comment"] = comment
            found = True
            break
    if not found:
        rows.append({
            "symbol": sym,
            "regularMarketPrice": None,
            "regularMarketChangePercent": None,
            "side": side,
            "comment": comment,
        })
        block[side_key] = rows
    save_history(hist)


def attach_comments(day: str, gainers: list[dict], losers: list[dict]) -> tuple[list[dict], list[dict]]:
    comments = _comments_map(day)

    def _merge(rows: list[dict], side: str) -> list[dict]:
        out = []
        for r in rows:
            sym = r.get("symbol")
            out.append({**r, "comment": comments.get((side, sym), r.get("comment") or "")})
        return out

    return _merge(gainers, "gainer"), _merge(losers, "loser")


def fetch_and_persist(count: int = MOVER_COUNT) -> dict[str, Any]:
    gainers, losers = get_market_movers(count)
    day = trading_day_key()
    gainers, losers = attach_comments(day, gainers, losers)
    day = save_daily_snapshot(gainers, losers, day=day)
    return {
        "day": day,
        "gainers": gainers,
        "losers": losers,
        "total": len(gainers) + len(losers),
    }


def all_symbols_in_history() -> list[str]:
    seen: set[str] = set()
    for day_data in load_history()["by_date"].values():
        for side in ("gainers", "losers"):
            for row in day_data.get(side) or []:
                sym = row.get("symbol")
                if sym:
                    seen.add(sym)
    return sorted(seen)


def symbol_history(symbol: str) -> list[dict[str, Any]]:
    sym = (symbol or "").strip().upper()
    if not sym:
        return []
    out: list[dict[str, Any]] = []
    by_date = load_history()["by_date"]
    for day in sorted(by_date.keys()):
        block = by_date[day]
        for side_key, side_label in (("gainers", "gainer"), ("losers", "loser")):
            for row in block.get(side_key) or []:
                if (row.get("symbol") or "").upper() != sym:
                    continue
                out.append({
                    "date": day,
                    "side": side_label,
                    "change_pct": row.get("regularMarketChangePercent"),
                    "price": row.get("regularMarketPrice"),
                    "comment": row.get("comment") or "",
                })
    return out
