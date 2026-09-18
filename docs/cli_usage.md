# Customer segmentation CLI

Use Python 3.11 or newer and run `python -m pip install -e '.[dev]'` from the repository root.
The install exposes `customer-seg` and makes the package available to `streamlit run app.py`.

```bash
customer-seg --recency 30 --frequency 5 --monetary 250
customer-seg --recency 30 --frequency 5 --monetary 250 --json
customer-seg --recency 30 --frequency 5 --monetary 250 --model /path/model.pkl --scaler /path/scaler.pkl
python -m pytest -q
```

Default artifact paths are relative to the current working directory. Load only trusted
joblib artifacts: deserialization can execute Python code. Success exits with status 0;
invalid inputs or missing artifacts exit with status 2. JSON output contains `cluster`
and `label`. Predictions are cluster assignments, not verified customer outcomes.
The CLI does not call the AI provider or require an API key.
