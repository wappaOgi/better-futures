"""EP / premarket screener — intraday metrike isključivo na 1h barovima (prepost).

Dnevni barovi služe samo za ADR, ADV, prethodni close i konsolidaciju.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import time
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

ET = "America/New_York"
CATALYST_OK = {
    "earnings_beat",
    "raised_guidance",
    "major_contract",
    "material_partnership",
    "regulatory",
    "clinical_trial",
    "other_material_news",
}

SCREENER_DEFAULT_SYMBOLS = [
    "TSLA", "AAPL", "NVDA", "AMD", "PLTR", "SOFI", "RIVN", "UPST", "DKNG", "SNAP",
    "HOOD", "IONQ", "SMCI", "ARM", "CRWD", "NET", "DDOG", "MARA", "COIN", "RBLX",
    "ABNB", "UBER", "LYFT", "PINS", "SNOW", "ZM", "DOCU", "ROKU", "ETSY", "SHOP",
    "PL", "AFRM", "OPEN", "LCID", "NIO", "XPEV", "MRNA", "BNTX", "SAVA", "VKTX",
]

_WEIGHTS = {
    "catalyst": 0.30,
    "rvol_pm": 0.25,
    "gap": 0.15,
    "consolidation": 0.15,
    "liquidity": 0.15,
}

_CATALYST_SCORE = {
    "earnings_beat": 100,
    "raised_guidance": 95,
    "major_contract": 85,
    "material_partnership": 80,
    "regulatory": 75,
    "clinical_trial": 70,
    "other_material_news": 60,
}

_NEWS_HINTS = {
    "raised_guidance": ("guidance", "raises", "raised", "outlook", "forecast"),
    "major_contract": ("contract", "award", "deal", "agreement", "wins"),
    "material_partnership": ("partnership", "collaborat", "strategic"),
    "regulatory": ("fda", "sec", "approval", "regulat", "clearance"),
    "clinical_trial": ("trial", "phase", "clinical", "study results"),
    "earnings_beat": ("earnings", "eps", "beats", "surprise", "results"),
    "other_material_news": ("acquisition", "merger", "launch", "breakthrough"),
}


@dataclass
class ScreenerConfig:
    min_gap_pct: float = 10.0
    min_mcap: float = 100_000_000
    max_mcap: float = 10_000_000_000
    min_adr_pct: float = 5.0
    min_adv: float = 500_000
    min_price: float = 2.0
    min_pm_dollar_volume: float = 250_000
    max_spread_pct: float = 1.0
    scan_time_et: time = time(9, 20)
    consolidation_min_sessions: int = 40
    consolidation_max_sessions: int = 65


def parse_symbols(text: str | None) -> list[str]:
    if not text or not str(text).strip():
        return list(SCREENER_DEFAULT_SYMBOLS)
    parts = re.split(r"[\s,;]+", text.strip().upper())
    out, seen = [], set()
    for p in parts:
        p = p.strip()
        if not p or "=" in p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out or list(SCREENER_DEFAULT_SYMBOLS)


def _fetch_daily(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, interval="1d", period="120d", progress=False, multi_level_index=False)
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize(ET)
    else:
        df.index = df.index.tz_convert(ET)
    return df.dropna(subset=["Close"])


def _fetch_hourly(symbol: str) -> pd.DataFrame:
    df = yf.download(
        symbol, interval="1h", period="60d", prepost=True, progress=False, multi_level_index=False,
    )
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize(ET)
    else:
        df.index = df.index.tz_convert(ET)
    return df.dropna(subset=["Close"])


def adr_pct(daily: pd.DataFrame, n: int = 20) -> float | None:
    d = daily.iloc[:-1] if len(daily) > 1 else daily
    d = d.tail(n)
    if len(d) < n:
        return None
    ranges = (d["High"] - d["Low"]) / d["Close"].replace(0, np.nan)
    v = float(ranges.mean() * 100)
    return v if np.isfinite(v) else None


def adv_20(daily: pd.DataFrame) -> float | None:
    d = daily.iloc[:-1] if len(daily) > 1 else daily
    d = d.tail(20)
    if len(d) < 20:
        return None
    v = float(d["Volume"].mean())
    return v if np.isfinite(v) and v > 0 else None


def prev_regular_close(daily: pd.DataFrame) -> float | None:
    if daily.empty:
        return None
    d = daily.iloc[:-1] if len(daily) > 1 else daily
    if d.empty:
        return None
    return float(d["Close"].iloc[-1])


def _premarket_bars(h1: pd.DataFrame, day: pd.Timestamp, cutoff: time) -> pd.DataFrame:
    """1h barovi u premarketu (04:00–09:30 ET) do cutoff sata."""
    day = day.tz_convert(ET) if day.tz else day.tz_localize(ET)
    start = day + pd.Timedelta(hours=4)
    rth_open = day + pd.Timedelta(hours=9, minutes=30)
    cutoff_dt = day + pd.Timedelta(hours=cutoff.hour, minutes=cutoff.minute)
    if cutoff_dt > rth_open:
        cutoff_dt = rth_open
    mask = (h1.index >= start) & (h1.index < cutoff_dt)
    return h1.loc[mask]


def premarket_snapshot(
    h1: pd.DataFrame, daily: pd.DataFrame, cfg: ScreenerConfig,
) -> dict[str, Any] | None:
    if h1.empty or daily.empty:
        return None
    now_et = pd.Timestamp.now(tz=ET)
    scan_day = now_et.normalize()
    pm = _premarket_bars(h1, scan_day, cfg.scan_time_et)
    if pm.empty:
        last_days = h1.index.normalize().unique()
        if len(last_days) < 2:
            return None
        scan_day = last_days[-1]
        pm = _premarket_bars(h1, scan_day, cfg.scan_time_et)
        if pm.empty:
            return None

    last = pm.iloc[-1]
    p_pre = float(last["Close"])
    c_prev = prev_regular_close(daily)
    if c_prev is None or c_prev <= 0:
        return None
    gap_pct = 100.0 * (p_pre / c_prev - 1.0)
    vol = float(pm["Volume"].sum())
    dollar_vol = vol * p_pre
    spread_pct = 100.0 * (float(last["High"]) - float(last["Low"])) / p_pre if p_pre else 999.0

    rvol = _rvol_pm(h1, scan_day, cfg.scan_time_et, vol)
    return {
        "gap_pct": gap_pct,
        "last_price": p_pre,
        "premarket_volume": vol,
        "premarket_dollar_volume": dollar_vol,
        "spread_pct": spread_pct,
        "rvol_pm": rvol,
        "scan_day": scan_day,
    }


def _rvol_pm(h1: pd.DataFrame, today: pd.Timestamp, cutoff: time, today_vol: float) -> float | None:
    prior_vols: list[float] = []
    days = sorted(h1.index.normalize().unique())
    today = today.normalize()
    for d in days:
        if d >= today:
            continue
        pm = _premarket_bars(h1, d, cutoff)
        if not pm.empty:
            prior_vols.append(float(pm["Volume"].sum()))
    prior_vols = prior_vols[-20:]
    if len(prior_vols) < 5 or today_vol <= 0:
        return None
    med = float(np.median(prior_vols))
    if med <= 0:
        return None
    return today_vol / med


def market_cap(symbol: str) -> float | None:
    try:
        cap = yf.Ticker(symbol).info.get("marketCap")
        if cap is None:
            return None
        v = float(cap)
        return v if v > 0 else None
    except Exception:
        return None


def _catalyst_from_earnings(symbol: str) -> tuple[str | None, float | None]:
    try:
        t = yf.Ticker(symbol)
        hist = t.get_earnings_history()
        if hist is None or (hasattr(hist, "empty") and hist.empty):
            return None, None
        if isinstance(hist, pd.DataFrame):
            row = hist.iloc[0]
            surprise = row.get("epsDifference") or row.get("surprisePercent")
            if surprise is not None:
                try:
                    sp = float(surprise)
                    if abs(sp) >= 3:
                        return "earnings_beat", min(100.0, 50.0 + abs(sp) * 2)
                except (TypeError, ValueError):
                    pass
    except Exception:
        pass
    return None, None


def _catalyst_from_news(symbol: str) -> tuple[str | None, str]:
    try:
        items = yf.Ticker(symbol).news or []
    except Exception:
        return None, ""
    now = pd.Timestamp.now(tz="UTC")
    for item in items[:12]:
        pub = item.get("providerPublishTime") or item.get("publishedAt")
        if pub is not None:
            try:
                ts = pd.Timestamp(pub, unit="s", tz="UTC")
                if (now - ts).total_seconds() > 36 * 3600:
                    continue
            except Exception:
                pass
        title = (item.get("title") or "").lower()
        if not title:
            continue
        for cat, words in _NEWS_HINTS.items():
            if any(w in title for w in words):
                return cat, item.get("title", "")
    return None, ""


def find_catalyst(symbol: str) -> dict[str, Any]:
    cat, score = _catalyst_from_earnings(symbol)
    headline = ""
    if cat:
        return {"category": cat, "verified": True, "headline": "Earnings surprise (yfinance)", "score_hint": score}
    cat, headline = _catalyst_from_news(symbol)
    if cat:
        return {
            "category": cat,
            "verified": True,
            "headline": headline,
            "score_hint": _CATALYST_SCORE.get(cat, 60),
        }
    return {"category": None, "verified": False, "headline": "", "score_hint": 0}


def consolidation_quality(daily: pd.DataFrame, cfg: ScreenerConfig) -> tuple[bool, float]:
    """Prefer 40–65 session base: range compression vs prior swing."""
    d = daily.iloc[:-1] if len(daily) > 1 else daily
    n = min(cfg.consolidation_max_sessions, len(d))
    if n < cfg.consolidation_min_sessions:
        return False, 0.0
    window = d.tail(n)
    prior = d.iloc[-(n + 20):-n] if len(d) >= n + 20 else d.iloc[:-n]
    if prior.empty:
        return False, 30.0
    w_range = (window["High"].max() - window["Low"].min()) / window["Close"].iloc[-1]
    p_range = (prior["High"].max() - prior["Low"].min()) / prior["Close"].iloc[-1]
    if p_range <= 0:
        return False, 30.0
    ratio = w_range / p_range
    in_band = cfg.consolidation_min_sessions <= n <= cfg.consolidation_max_sessions
    extended = float(window["Close"].iloc[-1]) >= float(window["High"].max()) * 0.95
    consolidating = in_band and ratio < 0.85 and not extended
    score = max(0.0, min(100.0, 100.0 * (1.0 - ratio)))
    if consolidating:
        score = min(100.0, score + 15.0)
    return consolidating, score


def _norm_gap(gap_pct: float, min_gap: float) -> float:
    g = abs(gap_pct)
    if g < min_gap:
        return 0.0
    # Sweet spot ~10–25%; penalize extreme gaps
    if g <= 25:
        return min(100.0, 40.0 + (g - min_gap) * 3.0)
    return max(30.0, 100.0 - (g - 25) * 2.0)


def _norm_rvol(rvol: float | None) -> float:
    if rvol is None:
        return 0.0
    if rvol >= 5:
        return 100.0
    if rvol >= 2:
        return 60.0 + (rvol - 2) * (40.0 / 3.0)
    return min(100.0, rvol * 30.0)


def _liquidity_score(dollar_vol: float, spread_pct: float, adv: float) -> float:
    dv = min(100.0, dollar_vol / 50_000)
    sp = max(0.0, 100.0 - spread_pct * 80.0)
    adv_s = min(100.0, adv / 10_000)
    return 0.5 * dv + 0.3 * sp + 0.2 * adv_s


def rank_score(row: dict[str, Any], cfg: ScreenerConfig) -> float:
    cat = row.get("catalyst") or ""
    cat_s = _CATALYST_SCORE.get(cat, row.get("catalyst_score_hint") or 0)
    parts = {
        "catalyst": float(cat_s),
        "rvol_pm": _norm_rvol(row.get("rvol_pm")),
        "gap": _norm_gap(float(row.get("gap_pct", 0)), cfg.min_gap_pct),
        "consolidation": float(row.get("consolidation_score", 0)),
        "liquidity": _liquidity_score(
            float(row.get("premarket_dollar_volume", 0)),
            float(row.get("spread_pct", 99)),
            float(row.get("adv", 0)),
        ),
    }
    return sum(parts[k] * _WEIGHTS[k] for k in _WEIGHTS)


def scan_symbol(symbol: str, cfg: ScreenerConfig) -> dict[str, Any] | None:
    daily = _fetch_daily(symbol)
    h1 = _fetch_hourly(symbol)
    if daily.empty or h1.empty:
        return {"symbol": symbol, "error": "Nema podataka", "passed": False}

    adr = adr_pct(daily)
    adv = adv_20(daily)
    mcap = market_cap(symbol)
    pm = premarket_snapshot(h1, daily, cfg)
    catalyst = find_catalyst(symbol)
    consolidating, cons_score = consolidation_quality(daily, cfg)

    row: dict[str, Any] = {
        "symbol": symbol,
        "gap_pct": pm["gap_pct"] if pm else None,
        "market_cap": mcap,
        "adr_pct": adr,
        "adv": adv,
        "premarket_volume": pm["premarket_volume"] if pm else None,
        "premarket_dollar_volume": pm["premarket_dollar_volume"] if pm else None,
        "spread_pct": pm["spread_pct"] if pm else None,
        "rvol_pm": pm["rvol_pm"] if pm else None,
        "last_price": pm["last_price"] if pm else None,
        "catalyst": catalyst["category"],
        "catalyst_headline": catalyst["headline"],
        "catalyst_score_hint": catalyst.get("score_hint"),
        "consolidating": consolidating,
        "consolidation_score": cons_score,
        "passed": False,
        "fail_reason": "",
    }

    reasons: list[str] = []
    if pm is None:
        reasons.append("premarket 1h")
    else:
        if row["gap_pct"] is not None and abs(row["gap_pct"]) < cfg.min_gap_pct:
            reasons.append(f"gap<{cfg.min_gap_pct}%")
        if row["last_price"] is not None and row["last_price"] < cfg.min_price:
            reasons.append("cijena")
        if row["premarket_dollar_volume"] is not None and row["premarket_dollar_volume"] < cfg.min_pm_dollar_volume:
            reasons.append("PM $vol")
        if row["spread_pct"] is not None and row["spread_pct"] > cfg.max_spread_pct:
            reasons.append("spread")
    if mcap is None or not (cfg.min_mcap <= mcap <= cfg.max_mcap):
        reasons.append("market cap")
    if adr is None or adr < cfg.min_adr_pct:
        reasons.append("ADR")
    if adv is None or adv < cfg.min_adv:
        reasons.append("ADV")
    if not catalyst["verified"] or catalyst["category"] not in CATALYST_OK:
        reasons.append("katalizator")

    if reasons:
        row["fail_reason"] = ", ".join(reasons)
        row["score"] = rank_score(row, cfg) if pm else 0.0
        return row

    row["passed"] = True
    row["score"] = rank_score(row, cfg)
    return row


def run_screener(symbols: list[str], cfg: ScreenerConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or ScreenerConfig()
    rows: list[dict[str, Any]] = []
    for sym in symbols:
        try:
            rows.append(scan_symbol(sym, cfg))
        except Exception as e:
            rows.append({"symbol": sym, "passed": False, "error": str(e), "fail_reason": "greška"})
    passed = [r for r in rows if r.get("passed")]
    failed = [r for r in rows if not r.get("passed")]
    passed.sort(key=lambda r: r.get("score", 0), reverse=True)
    failed.sort(key=lambda r: r.get("score", 0), reverse=True)
    return passed + failed
