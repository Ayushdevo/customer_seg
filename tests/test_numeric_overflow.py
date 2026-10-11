import pytest

from customer_seg.predict import validate_numeric_input
from test_vercel_api import endpoint


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_unrepresentable_integers_produce_validation_errors(value):
    with pytest.raises(ValueError, match="finite number"):
        validate_numeric_input(value, "Recency")


def test_large_json_integer_is_rejected_before_loading_artifacts(monkeypatch):
    def unavailable():
        raise AssertionError("Invalid input must not attempt model loading")

    monkeypatch.setattr(endpoint, "models", unavailable)
    with pytest.raises(ValueError, match="finite number"):
        endpoint.predict({"recency": 10**400, "frequency": 2, "monetary": 3})
