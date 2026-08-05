from customer_seg.utils import format_currency


def test_format_currency_defaults_to_dollars():
    assert format_currency(1234.5) == "$1,234.50"


def test_format_currency_with_euro_symbol():
    assert format_currency(432.1, currency_symbol="€") == "€432.10"
