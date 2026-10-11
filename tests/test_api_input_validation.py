import pytest

from test_vercel_api import endpoint


@pytest.mark.parametrize("name", ["recency", "frequency", "monetary"])
@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf"), -float("inf")])
def test_invalid_features_are_rejected_before_artifacts_load(monkeypatch, name, invalid):
    def unavailable():
        raise AssertionError("Invalid input must not attempt model loading")

    monkeypatch.setattr(endpoint, "models", unavailable)
    payload = {"recency": 1, "frequency": 2, "monetary": 3}
    payload[name] = invalid
    with pytest.raises(ValueError, match=name):
        endpoint.predict(payload)
