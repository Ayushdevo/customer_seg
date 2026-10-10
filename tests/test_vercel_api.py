"""Contract tests for the Vercel prediction endpoint."""
import importlib.util
from pathlib import Path

import pytest

API = Path(__file__).resolve().parents[1] / "api" / "predict.py"
spec = importlib.util.spec_from_file_location("vercel_predict", API)
endpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(endpoint)


def test_prediction_response_contract(monkeypatch):
    class Scaler:
        def transform(self, values):
            return values

    class Model:
        def predict(self, values):
            return [0]

    monkeypatch.setattr(endpoint, "models", lambda: (Model(), Scaler()))
    result = endpoint.predict({"recency": 30, "frequency": 5, "monetary": 250})
    assert result["cluster"] == 0
    assert result["label"] == "The VIPs"
    assert result["profile"]


@pytest.mark.parametrize("invalid", [None, True, "30", [], {}])
def test_reject_non_numeric_features(monkeypatch, invalid):
    with pytest.raises(ValueError):
        endpoint.predict({"recency": invalid, "frequency": 5, "monetary": 250})


def test_missing_secret_does_not_block_prediction(monkeypatch):
    class Scaler:
        def transform(self, values):
            return values

    class Model:
        def predict(self, values):
            return [1]

    monkeypatch.setattr(endpoint, "models", lambda: (Model(), Scaler()))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = endpoint.predict({"recency": 10, "frequency": 2, "monetary": 45, "generate_strategy": True})
    assert result["strategy"] is None
    assert result["strategy_status"] == "not_configured"
