from __future__ import annotations

import numpy as np

CLUSTER_LABELS = {
    0: "The VIPs",
    1: "Churn Risk",
    2: "The Bargain Hunters",
}

CLUSTER_PROFILES = {
    0: "High-value customers with strong purchase frequency and loyalty.",
    1: "Customers showing signs of churn due to reduced recent activity.",
    2: "Price-sensitive buyers who respond to value-driven offers.",
}


def validate_numeric_input(value: float, name: str, minimum: float = 0.0) -> float:
    if value is None:
        raise ValueError(f"{name} must be provided")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def predict_customer_cluster(model, scaler, recency: float, frequency: float, monetary: float) -> int:
    recency = validate_numeric_input(recency, "Recency", minimum=0.0)
    frequency = validate_numeric_input(frequency, "Frequency", minimum=0.0)
    monetary = validate_numeric_input(monetary, "Monetary", minimum=0.0)

    input_data = np.array([[recency, frequency, monetary]])
    scaled_data = scaler.transform(input_data)
    cluster = int(model.predict(scaled_data)[0])
    if cluster not in CLUSTER_LABELS:
        raise ValueError(f"Unknown cluster id returned by model: {cluster}")
    return cluster
