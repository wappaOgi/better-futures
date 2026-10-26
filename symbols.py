"""Pretraga tickera (dionice, ETF, futures, crypto) preko Yahoo Finance."""
from yfinance import Search

# Brzi odabir — futures + popularno
FAVORITE_TICKERS = [
    {"label": "MNQ · Nasdaq Micro", "value": "MNQ=F"},
    {"label": "MES · S&P Micro", "value": "MES=F"},
    {"label": "GC · Gold", "value": "GC=F"},
    {"label": "CL · Oil", "value": "CL=F"},
    {"label": "NG · Gas", "value": "NG=F"},
    {"label": "NQ · Nasdaq", "value": "NQ=F"},
    {"label": "ES · S&P", "value": "ES=F"},
    {"label": "MYM · Dow Micro", "value": "MYM=F"},
    {"label": "TSLA · Tesla", "value": "TSLA"},
    {"label": "AAPL · Apple", "value": "AAPL"},
    {"label": "NVDA · NVIDIA", "value": "NVDA"},
    {"label": "SPY · S&P 500 ETF", "value": "SPY"},
    {"label": "BTC-USD · Bitcoin", "value": "BTC-USD"},
]

_TYPE_SHORT = {
    "EQUITY": "dionica",
    "ETF": "ETF",
    "FUTURE": "future",
    "MUTUALFUND": "fond",
    "CRYPTOCURRENCY": "crypto",
    "INDEX": "index",
    "CURRENCY": "valuta",
}

_cache: dict[str, list[dict]] = {}


def _format_label(q: dict) -> str:
    sym = q.get("symbol") or "?"
    name = (q.get("shortname") or q.get("longname") or sym).strip()
    typ = _TYPE_SHORT.get(q.get("quoteType", ""), q.get("typeDisp") or "")
    if len(name) > 36:
        name = name[:33] + "…"
    return f"{sym} · {name}" + (f" ({typ})" if typ else "")


def search_symbols(query: str, max_results: int = 15) -> list[dict]:
    """Yahoo pretraga — npr. 'tesla' → TSLA, STSLA=F, …"""
    key = query.strip().lower()
    if len(key) < 2:
        return []
    if key in _cache:
        return _cache[key]

    try:
        res = Search(query, max_results=max_results)
        quotes = res.all.get("quotes", []) if isinstance(res.all, dict) else []
    except Exception:
        return []

    seen, out = set(), []
    for q in quotes:
        sym = q.get("symbol")
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append({"label": _format_label(q), "value": sym})
    _cache[key] = out
    return out


def ticker_options(search: str | None = None, selected: str | None = None) -> list[dict]:
    """Rezultati pretrage (ako ima upita) + favoriti + odabrani ticker."""
    seen, opts = set(), []

    if search and len(search.strip()) >= 2:
        for hit in search_symbols(search):
            if hit["value"] not in seen:
                opts.append(hit)
                seen.add(hit["value"])

    for fav in FAVORITE_TICKERS:
        if fav["value"] not in seen:
            opts.append(fav)
            seen.add(fav["value"])

    if selected and selected not in seen:
        opts.insert(0, {"label": selected, "value": selected})
    return opts
