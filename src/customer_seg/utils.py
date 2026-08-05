from __future__ import annotations


def format_currency(value: float, currency_symbol: str = "$") -> str:
    return f"{currency_symbol}{value:,.2f}"
