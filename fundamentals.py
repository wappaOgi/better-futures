"""Fundamentalni podaci i vijesti (yfinance) — samo za prikaz u UI, ne sprema se u JSON."""
from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf


def _fmt_pct(v) -> str | None:
    if v is None:
        return None
    try:
        return f"{float(v) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(v)


def _fmt_num(v) -> str | None:
    if v is None:
        return None
    try:
        x = float(v)
        if abs(x) >= 1e9:
            return f"{x / 1e9:.2f}B"
        if abs(x) >= 1e6:
            return f"{x / 1e6:.2f}M"
        if abs(x) >= 1e3:
            return f"{x / 1e3:.2f}K"
        return f"{x:.2f}"
    except (TypeError, ValueError):
        return str(v)


def fetch_ticker_news(symbol: str, count: int = 10) -> list[dict[str, Any]]:
    stock = yf.Ticker(symbol)
    items: list = []
    try:
        items = stock.get_news(count=count, tab="news") or []
    except Exception:
        items = []
    if not items:
        try:
            items = stock.news or []
        except Exception:
            items = []

    out: list[dict[str, Any]] = []
    for raw in items[:count]:
        if not isinstance(raw, dict):
            continue
        content = raw.get("content") if isinstance(raw.get("content"), dict) else raw
        title = raw.get("title") or content.get("title") or content.get("headline")
        link = (
            content.get("canonicalUrl")
            or content.get("clickThroughUrl")
            or raw.get("link")
            or raw.get("url")
        )
        pub = raw.get("providerPublishTime") or content.get("pubDate") or raw.get("publishedAt")
        publisher = content.get("provider") or raw.get("publisher") or raw.get("source")
        if isinstance(publisher, dict):
            publisher = publisher.get("displayName") or publisher.get("name")
        if not title:
            continue
        out.append({"title": title, "link": link, "published": pub, "publisher": publisher})
    return out


def _df_preview(df: pd.DataFrame | None, max_rows: int = 8, max_cols: int = 5) -> list[list[str]]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    sub = df.iloc[:max_rows, :max_cols]
    cols = [str(c)[:10] for c in sub.columns]
    rows = [[str(sub.index[i])[:12]] + [ _fmt_num(sub.iloc[i, j]) or "—" for j in range(len(cols))] for i in range(len(sub))]
    return [[""] + cols] + rows


def _table_from_df(df: pd.DataFrame | None, title: str) -> dict[str, Any]:
    preview = _df_preview(df)
    return {"title": title, "rows": preview}


def analyze_fundamentals(symbol: str) -> dict[str, Any]:
    stock = yf.Ticker(symbol)
    info = stock.info or {}

    metrics = {
        "Company": info.get("shortName") or info.get("longName"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Price": _fmt_num(info.get("currentPrice") or info.get("regularMarketPrice")),
        "Market Cap": _fmt_num(info.get("marketCap")),
        "Revenue TTM": _fmt_num(info.get("totalRevenue")),
        "Revenue Growth": _fmt_pct(info.get("revenueGrowth")),
        "Earnings Growth": _fmt_pct(info.get("earningsGrowth")),
        "Gross Margin": _fmt_pct(info.get("grossMargins")),
        "Operating Margin": _fmt_pct(info.get("operatingMargins")),
        "Profit Margin": _fmt_pct(info.get("profitMargins")),
        "Free Cash Flow": _fmt_num(info.get("freeCashflow")),
        "Debt / Equity": info.get("debtToEquity"),
        "Forward P/E": info.get("forwardPE"),
        "Price / Sales": info.get("priceToSalesTrailing12Months"),
        "EV / EBITDA": info.get("enterpriseToEbitda"),
        "Shares Outstanding": _fmt_num(info.get("sharesOutstanding")),
        "Short % of Float": _fmt_pct(info.get("shortPercentOfFloat")),
    }

    def _safe(attr: str):
        try:
            return getattr(stock, attr, None)
        except Exception:
            return None

    return {
        "symbol": symbol,
        "metrics": metrics,
        "income": _table_from_df(_safe("quarterly_income_stmt"), "Quarterly income"),
        "balance_sheet": _table_from_df(_safe("quarterly_balance_sheet"), "Quarterly balance sheet"),
        "cashflow": _table_from_df(_safe("quarterly_cashflow"), "Quarterly cash flow"),
        "earnings_history": _table_from_df(_safe("earnings_history"), "Earnings history"),
        "eps_estimates": _table_from_df(_safe("earnings_estimate"), "EPS estimates"),
        "eps_revisions": _table_from_df(_safe("eps_revisions"), "EPS revisions"),
        "revenue_estimates": _table_from_df(_safe("revenue_estimate"), "Revenue estimates"),
        "insider": _table_from_df(_safe("insider_transactions"), "Insider transactions"),
    }
