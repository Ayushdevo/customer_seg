import io
import json

import pytest

from customer_seg.ai import AIServiceError
from test_vercel_api import endpoint


@pytest.mark.parametrize("result", [None, [], {}, {"candidates": None},
    {"candidates": [None, {"content": None}, {"content": {"parts": None}}]},
    {"candidates": [{"content": {"parts": [{"text": 123}, {"text": " "}]}}]},
])
def test_malformed_provider_responses_raise_service_errors(result):
    with pytest.raises(AIServiceError, match="no strategy text"):
        endpoint.extract_strategy_text(result)


def test_strategy_ignores_nontext_and_thinking_blocks():
    result = {"candidates": [{"content": {"parts": [
        None, {"text": 42}, {"text": "internal", "thought": True},
        {"text": " Offer loyalty rewards. "}, {"inlineData": {}},
        {"text": "Encourage a repeat purchase."},
    ]}}]}
    assert endpoint.extract_strategy_text(result) == (
        "Offer loyalty rewards.\nEncourage a repeat purchase."
    )


def test_invalid_provider_json_is_a_service_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(endpoint, "urlopen", lambda *args, **kwargs: io.BytesIO(b"{oops"))
    with pytest.raises(AIServiceError, match="invalid JSON"):
        endpoint.generate_marketing_strategy(0, 1, 2, 3)
