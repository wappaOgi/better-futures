"""Banka: balans, risk po trejdu, margina."""
import json
from pathlib import Path

FILE = Path(__file__).with_name("bank.json")
DEFAULT = {"balance_eur": 2000.0, "initial_eur": 2000.0, "risk_pct": 1.0, "margin": 10}


def load() -> dict:
    if FILE.exists():
        return {**DEFAULT, **json.loads(FILE.read_text())}
    save(DEFAULT)
    return DEFAULT.copy()


def save(bank: dict) -> None:
    FILE.write_text(json.dumps(bank, indent=2))


def risk_eur(bank: dict | None = None) -> float:
    b = bank or load()
    return round(b["balance_eur"] * b["risk_pct"] / 100, 2)


def recalc(trades: list) -> dict:
    """Balans = početni + zbroj PnL zatvorenih trejdova."""
    bank = load()
    pnl = sum(t.get("pnl_eur") or 0 for t in trades if t.get("status") in ("win", "loss"))
    bank["balance_eur"] = round(bank["initial_eur"] + pnl, 2)
    save(bank)
    return bank


def update_settings(initial_eur=None, risk_pct=None, margin=None) -> dict:
    bank = load()
    if initial_eur is not None:
        bank["initial_eur"] = float(initial_eur)
    if risk_pct is not None:
        bank["risk_pct"] = float(risk_pct)
    if margin is not None:
        bank["margin"] = int(margin)
    save(bank)
    return bank
