from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


class LoaderError(Exception):
    pass


def load_models(
    model_path: str | Path = "models/kmeans_model.pkl",
    scaler_path: str | Path = "models/rfm_scaler.pkl",
):
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)

    if not model_path.exists():
        raise LoaderError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise LoaderError(f"Scaler file not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def load_data(data_path: str | Path = "data/clustered_rfm_data.csv") -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise LoaderError(f"Data file not found: {data_path}")

    return pd.read_csv(data_path)
