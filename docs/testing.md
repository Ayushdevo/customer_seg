# Testing Guide

## Run unit tests

Install the development dependencies and run:

```bash
pip install -r requirements-dev.txt
pytest
```

## Test strategy

- Validate Streamlit app helper functions.
- Ensure model loading and data loading operate without exceptions.
- Cover expected cluster labels and API prompt generation logic.
