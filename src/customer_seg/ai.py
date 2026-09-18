from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_google_genai import ChatGoogleGenerativeAI


class AIServiceError(ValueError):
    """A strategy could not be generated or interpreted."""


def build_marketing_prompt(
    predicted_cluster: int,
    recency: float,
    frequency: float,
    monetary: float,
) -> str:
    from .predict import CLUSTER_LABELS, validate_numeric_input
    if isinstance(predicted_cluster, bool) or predicted_cluster not in CLUSTER_LABELS:
        raise ValueError("Unknown customer cluster")
    recency = validate_numeric_input(recency, "Recency")
    frequency = validate_numeric_input(frequency, "Frequency")
    monetary = validate_numeric_input(monetary, "Monetary")
    return (
        f"You are an expert marketing strategist. A machine learning K-Means model has assigned a customer to "
        f"Cluster {predicted_cluster}.\n\n"
        f"Here are the customer metrics:\n"
        f"- Recency: {recency} days since last purchase.\n"
        f"- Frequency: {frequency} total lifetime purchases.\n"
        f"- Monetary Value: ${monetary:.2f} total lifetime spend.\n\n"
        f"Write a concise, one-paragraph actionable marketing plan for this customer. "
        f"Focus on retention or upselling based on these exact numbers and their cluster assignment. "
        f"Do not use emojis in the response."
    )


def create_llm(
    google_api_key: str,
    model_name: str = "gemini-3.6-flash",
    temperature: float = 0.7,
) -> ChatGoogleGenerativeAI:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise AIServiceError("Install langchain-google-genai to generate AI strategies") from exc
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_api_key,
        temperature=temperature,
    )


def generate_strategy(llm: Any, prompt: str) -> str:
    response = llm.invoke(prompt)
    content = getattr(response, "content", None)
    if isinstance(content, list):
        content = "\n".join(
            block if isinstance(block, str) else block["text"]
            for block in content
            if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str))
        )
    if not isinstance(content, str) or not content.strip():
        raise AIServiceError("LLM response did not contain nonempty text")
    return content.strip()
