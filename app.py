import streamlit as st
import pandas as pd
import numpy as np
import joblib
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("Customer Intelligence Dashboard")
st.markdown("Predict customer segments in real-time and generate AI-driven marketing strategies.")
st.markdown("---")

# --- 1. DATA & MODEL LOADING ---
@st.cache_resource
def load_models():
    model = joblib.load('models/kmeans_model.pkl')
    scaler = joblib.load('models/rfm_scaler.pkl')
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv('data/clustered_rfm_data.csv')
    return df

kmeans_model, scaler = load_models()
df = load_data()

with st.expander("View Raw Customer Data"):
    st.dataframe(df.head(100), use_container_width=True)

# --- 2. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("1. API Configuration")
    google_api_key = st.text_input("Google AI Studio API Key(as it is free)", type="password")
    st.caption("Required to generate AI marketing strategies.")
    
    st.markdown("---")
    
    st.header("2. Input Customer Data")
    recency = st.slider("Recency (Days since last purchase)", min_value=1, max_value=365, value=30)
    frequency = st.slider("Frequency (Number of purchases)", min_value=1, max_value=100, value=5)
    monetary = st.slider("Monetary (Total spend in $)", min_value=10.0, max_value=5000.0, value=250.0)
    
    st.markdown("---")
    predict_btn = st.button("Predict Segment", use_container_width=True)

# --- 3. PREDICTION & AI GENERATION ---
if predict_btn:
    
    # 1. Run the Machine Learning Model
    input_data = np.array([[recency, frequency, monetary]])
    scaled_data = scaler.transform(input_data)
    predicted_cluster = kmeans_model.predict(scaled_data)[0]
    
    # Display basic model results
    st.subheader("Model Prediction")
    
    if predicted_cluster == 0:
        st.success(f"Cluster {predicted_cluster}: The VIPs")
    elif predicted_cluster == 1:
        st.warning(f"Cluster {predicted_cluster}: Churn Risk")
    else:
        st.info(f"Cluster {predicted_cluster}: The Bargain Hunters")

    st.markdown("---")
    
    # 2. Run the AI Strategist
    st.subheader("AI Strategy Analysis")
    
    if not google_api_key:
        st.warning("Please enter your Google API Key in the sidebar to view the AI-generated strategy for this prediction.")
    else:
        with st.spinner("Generating targeted marketing strategy..."):
            try:
                # Initialize the standard LLM using the updated model alias
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.6-flash", 
                    google_api_key=google_api_key, 
                    temperature=0.7
                )
                
                # Create a prompt combining the user inputs and the model's prediction
                prompt = f"""
                You are an expert marketing strategist. A machine learning K-Means model just segmented a customer into Cluster {predicted_cluster}.
                
                Here are the specific metrics for this customer:
                - Recency: {recency} days since their last purchase.
                - Frequency: {frequency} total lifetime purchases.
                - Monetary Value: ${monetary} total lifetime spend.
                
                Based on these specific numbers and their cluster, write a concise, one-paragraph actionable marketing plan for this user. 
                Focus on retention or upselling based on their exact behavior. Do not use emojis in your response.
                """
                
                # Call the API
                response = llm.invoke(prompt)
                
                # Display the AI's explanation
                st.write(response.content)
                
            except Exception as e:
                st.error(f"Error generating AI strategy: {str(e)}")