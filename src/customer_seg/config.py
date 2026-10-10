"""Runtime configuration for local development and managed hosting."""

import os


def google_api_key_from_environment() -> str | None:
    """Return the first configured API key without exposing it in logs."""
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None
