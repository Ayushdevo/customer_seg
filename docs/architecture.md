# Project Architecture

This project is structured as a Streamlit application with a small ML inference layer.

## Core components

- `app.py` - Streamlit user interface and application flow.
- `models/` - Serialized model artifacts for clustering and scaling.
- `data/clustered_rfm_data.csv` - Input dataset used for cluster analysis and model training.
- `pyproject.toml` - Project metadata and dependency definitions.
- `.github/workflows/python-app.yml` - CI workflow for tests and dependency installation.
