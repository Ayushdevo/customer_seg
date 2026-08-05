# Usage Guide

## Run locally

Activate the virtual environment and install dependencies:

```powershell
.
\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the dashboard:

```powershell
streamlit run app.py
```

## Input values

- `Recency`: days since last purchase
- `Frequency`: number of purchases
- `Monetary`: total spend in dollars

## AI integration

Enter a Google AI Studio API key to generate targeted marketing recommendations from Gemini.
