# Customer Intelligence Dashboard

An interactive Customer Segmentation dashboard built using Streamlit, Scikit-Learn, and Google Generative AI (Gemini). This application segments customers based on Recency, Frequency, and Monetary (RFM) values using a pre-trained K-Means clustering model, and leverages Gemini to generate real-time, actionable marketing strategies for each segment.

---

## 🚀 Key Features

* **Real-Time Customer Segmentation:** Inputs such as Recency, Frequency, and Monetary value are mapped to a pre-trained K-Means clustering model to predict customer segments instantaneously.
* **Segment Profiling:** Predicts and classifies customer clusters into distinct profiles:
  * **Cluster 0:** *The VIPs* (High-frequency, high-spending customers)
  * **Cluster 1:** *Churn Risk* (Customers with long periods of inactivity)
  * **Cluster 2:** *The Bargain Hunters* (Value-driven, cost-conscious buyers)
* **AI-Generated Marketing Strategies:** Integration with **Google Gemini (gemini-3.6-flash)** using the `langchain-google-genai` library to create highly targeted, context-aware marketing copies and retention strategies based on the customer's unique RFM profile.
* **Raw Data Exploration:** Expandable raw data view displaying the source dataset (`data/clustered_rfm_data.csv`) directly in the dashboard.
* **API Configuration:** Interactive sidebar setting to provide the Google AI Studio API key securely.

---

## 🛠️ Tech Stack

* **Front-End & Dashboard:** [Streamlit](https://streamlit.io/)
* **Machine Learning & Pipeline:** [Scikit-Learn](https://scikit-learn.org/), [Joblib](https://joblib.github.io/joblib/)
* **Generative AI Orchestration:** [LangChain](https://www.langchain.com/) & [Google Gemini](https://ai.google.dev/) (`langchain-google-genai`)
* **Data Wrangling:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)

---

## 📂 Project Structure

```text
├── data/
│   └── clustered_rfm_data.csv   # Dataset used to train & preview raw profiles
├── models/
│   ├── kmeans_model.pkl         # Pre-trained K-Means Clustering model
│   └── rfm_scaler.pkl           # StandardScaler used for RFM data scaling
├── env/                         # Python Virtual Environment
├── app.py                       # Main Streamlit dashboard code
├── .gitignore                   # Exclusions for virtual envs and caches
└── README.md                    # Project documentation
```

---

## ⚙️ How to Setup & Run

### Prerequisites
Make sure you have **Python 3.8+** installed.

### 1. Set Up the Virtual Environment
Activate the pre-existing virtual environment in the `env` folder:
* **Windows (PowerShell):**
  ```powershell
  .\env\Scripts\Activate.ps1
  ```
* **Windows (Command Prompt):**
  ```cmd
  .\env\Scripts\activate.bat
  ```

### 2. Run the Dashboard
Run the Streamlit application using:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💡 How it Works
1. **Interactive Inputs:** Drag sliders in the sidebar to define a customer's purchasing behaviour.
2. **Clustering & Prediction:** Values are scaled via `rfm_scaler.pkl` and predictions are run using `kmeans_model.pkl`.
3. **AI Insight Activation:** Provide your Google AI Studio API key to fetch personalized recommendations for that specific customer's cluster behavior directly from Gemini.


---

## Deployment (Docker / Render / Railway / Hugging Face Spaces)

The deployment image starts Streamlit on **0.0.0.0:7860** and exposes a health endpoint at `/_stcore/health`. The trained model and scaler in `models/`, and `data/clustered_rfm_data.csv`, must remain in the repository.

### Local Docker smoke test

```bash
docker build -t customer-seg .
docker run --rm -p 7860:7860 customer-seg
```

Open http://localhost:7860 and check http://localhost:7860/_stcore/health. Verify model prediction first. The AI strategy feature is optional and needs a Google AI Studio key.

To enable AI-generated strategies without requiring users to enter a key, pass the key at runtime (never put it in Git or the Docker image):

```bash
docker run --rm -p 7860:7860 -e GOOGLE_API_KEY=YOUR_KEY customer-seg
```

For production, set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in the hosting platform's **secret environment variables**, rather than using a shell command that exposes it in history. If neither key is configured, visitors may supply their own key in the password-type sidebar field; clustering itself still works without AI credentials.

### Hosted Docker deployment

Create a web service from this repository using its `Dockerfile`. Set the external/container port to `7860`, make the service publicly reachable, and configure the health path `/_stcore/health`. Configure secrets in the hosting dashboard, not in repository files. On platforms which expect a configurable `PORT` variable instead of the image's port, set the service/container port to `7860` or adapt the launch command accordingly.

**Verification checklist:** image builds, process starts without root privileges, health endpoint responds, model prediction works, and (only if configured) a test Gemini request produces a marketing strategy. A healthy HTTP response alone does not confirm that external model credentials are working.

### Streamlit Community Cloud (alternative)

Set the entrypoint to `app.py` and Python 3.11 if the platform supports choosing it. This repository's `pyproject.toml` defines the installable package in `src/`; a deployment environment must install the package (e.g. `pip install .`) so that `import customer_seg` works. Docker deployment is the primary supported configuration.

**Security note:** Load `.pkl` files only from trusted sources; Python pickle/joblib files can execute code when loaded.
