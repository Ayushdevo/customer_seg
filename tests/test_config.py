"""Tests for deployment credential precedence and empty secret handling."""

from customer_seg.config import google_api_key_from_environment


def test_google_api_key_has_priority(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", " google-secret ")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    assert google_api_key_from_environment() == "google-secret"


def test_gemini_key_fallback(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    assert google_api_key_from_environment() == "gemini-secret"


def test_blank_primary_uses_fallback(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "  ")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    assert google_api_key_from_environment() == "gemini-secret"


def test_no_credentials_is_optional(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert google_api_key_from_environment() is None
