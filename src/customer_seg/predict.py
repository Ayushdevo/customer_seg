from __future__ import annotations

import math
from numbers import Real

import numpy as np
import pandas as pd

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
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def predict_customer_cluster(model, scaler, recency: float, frequency: float, monetary: float) -> int:
    recency = validate_numeric_input(recency, "Recency", minimum=0.0)
    frequency = validate_numeric_input(frequency, "Frequency", minimum=0.0)
    monetary = validate_numeric_input(monetary, "Monetary", minimum=0.0)

    features = {"Recency": recency, "Frequency": frequency, "Monetary": monetary}
    if hasattr(scaler, "feature_names_in_"):
        names = list(scaler.feature_names_in_)
        if len(names) != 3 or set(names) != set(features):
            raise ValueError("Scaler feature names must be Recency, Frequency, and Monetary")
        input_data = pd.DataFrame([[features[name] for name in names]], columns=names)
    else:
        input_data = np.array([[recency, frequency, monetary]])
    scaled_data = np.asarray(scaler.transform(input_data))
    if scaled_data.shape != (1, 3) or not np.issubdtype(scaled_data.dtype, np.number) or not np.isfinite(scaled_data).all():
        raise ValueError("Scaler must return one row of three finite numeric features")
    predictions = np.asarray(model.predict(scaled_data))
    if predictions.shape != (1,):
        raise ValueError("Model must return exactly one cluster label")
    label = predictions[0]
    if isinstance(label, (bool, np.bool_)) or not isinstance(label, Real) or not math.isfinite(label) or label != int(label):
        raise ValueError("Model cluster label must be a finite integer")
    cluster = int(label)
    if cluster not in CLUSTER_LABELS:
        raise ValueError(f"Unknown cluster id returned by model: {cluster}")
    return cluster
