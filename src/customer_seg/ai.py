from __future__ import annotations

from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI


class AIServiceError(ValueError):
    """A strategy could not be generated or interpreted."""


def build_marketing_prompt(
    predicted_cluster: int,
    recency: float,
    frequency: float,
    monetary: float,
) -> str:
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
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_api_key,
        temperature=temperature,
    )


def generate_strategy(llm: Any, prompt: str) -> str:
    response = llm.invoke(prompt)
    if not hasattr(response, "content"):
        raise ValueError("LLM response object did not contain expected content")
    return response.content
