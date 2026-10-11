from urllib.error import HTTPError, URLError

import pytest

from customer_seg.ai import AIServiceError
from test_vercel_api import endpoint


@pytest.fixture
def configured_prediction(monkeypatch):
    class Scaler:
        def transform(self, values):
            return values

    class Model:
        def predict(self, values):
            return [0]

    monkeypatch.setattr(endpoint, "models", lambda: (Model(), Scaler()))
    return {"recency": 1, "frequency": 2, "monetary": 3, "generate_strategy": True}


@pytest.mark.parametrize("error", [
    AIServiceError("no text"), TimeoutError(), URLError("offline"),
    HTTPError("https://provider.invalid", 429, "rate limited", {}, None),
])
def test_provider_failure_preserves_segmentation(monkeypatch, configured_prediction, error):
    def unavailable(*args):
        raise error

    monkeypatch.setattr(endpoint, "generate_marketing_strategy", unavailable)
    result = endpoint.predict(configured_prediction)
    assert result["cluster"] == 0
    assert result["label"] == "The VIPs"
    assert result["strategy"] is None
    assert result["strategy_status"] == "unavailable"


def test_successful_strategy_has_explicit_status(monkeypatch, configured_prediction):
    monkeypatch.setattr(endpoint, "generate_marketing_strategy", lambda *args: "Offer rewards.")
    result = endpoint.predict(configured_prediction)
    assert result["strategy"] == "Offer rewards."
    assert result["strategy_status"] == "generated"
