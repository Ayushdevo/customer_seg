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
