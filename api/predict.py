"""Vercel serverless endpoint for customer segmentation and optional AI strategy."""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from customer_seg.loaders import LoaderError, load_models
from customer_seg.ai import AIServiceError
from customer_seg.predict import (
    CLUSTER_LABELS, CLUSTER_PROFILES, predict_customer_cluster, validate_numeric_input,
)


@lru_cache(maxsize=1)
def models():
    return load_models(ROOT / "models/kmeans_model.pkl", ROOT / "models/rfm_scaler.pkl")


def generate_marketing_strategy(cluster: int, recency: float, frequency: float, monetary: float) -> str | None:
    key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        return None
    from customer_seg.ai import build_marketing_prompt
    prompt = build_marketing_prompt(cluster, recency, frequency, monetary)
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if not model.replace("-", "").replace(".", "").isalnum():
        raise AIServiceError("Invalid Gemini model identifier")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    request = Request(url, data=payload, headers={
        "Content-Type": "application/json", "x-goog-api-key": key
    }, method="POST")
    try:
        with urlopen(request, timeout=20) as response:
            result = json.load(response)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AIServiceError("Gemini returned invalid JSON") from exc
    return extract_strategy_text(result)


def extract_strategy_text(result: object) -> str:
    candidates = result.get("candidates") if isinstance(result, dict) else None
    if not isinstance(candidates, list):
        raise AIServiceError("Gemini returned no strategy text")
    text_parts = []
    for candidate in candidates:
        content = candidate.get("content") if isinstance(candidate, dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict) or part.get("thought") is True:
                continue
            value = part.get("text")
            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())
    strategy = "\n".join(text_parts)
    if not strategy:
        raise AIServiceError("Gemini returned no strategy text")
    return strategy


def predict(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    values = []
    for name in ("recency", "frequency", "monetary"):
        value = payload.get(name)
        # Do not convert strings or bools to numeric values.
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"{name} must be a number")
        values.append(validate_numeric_input(value, name))
    cluster = predict_customer_cluster(*models(), *values)
    result = {
        "cluster": cluster,
        "label": CLUSTER_LABELS[cluster],
        "profile": CLUSTER_PROFILES[cluster],
    }
    if payload.get("generate_strategy") is True:
        result["strategy"] = generate_marketing_strategy(cluster, *values)
        if result["strategy"] is None:
            result["strategy_status"] = "not_configured"
    return result


class handler(BaseHTTPRequestHandler):
    def respond(self, code: int, value: dict):
        output = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(output)))
        self.end_headers()
        self.wfile.write(output)

    def do_GET(self):
        self.respond(200, {"status": "ok", "service": "customer-segmentation"})

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 8192:
                return self.respond(413, {"error": "Request must contain at most 8 KB of JSON"})
            payload = json.loads(self.rfile.read(size))
            result = predict(payload)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self.respond(400, {"error": str(exc)})
        except LoaderError:
            return self.respond(503, {"error": "Model artifacts are unavailable"})
        except (HTTPError, URLError, TimeoutError):
            return self.respond(502, {"error": "AI strategy service is temporarily unavailable"})
        except Exception:
            return self.respond(500, {"error": "Prediction service encountered an unexpected error"})
        self.respond(200, result)
