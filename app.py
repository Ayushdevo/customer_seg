import logging

import streamlit as st

from customer_seg.ai import build_marketing_prompt, create_llm, generate_strategy
from customer_seg.loaders import LoaderError, load_data, load_models
from customer_seg.predict import CLUSTER_LABELS, CLUSTER_PROFILES, predict_customer_cluster


logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("Customer Intelligence Dashboard")
st.markdown("Predict customer segments in real-time and generate AI-driven marketing strategies.")
st.markdown("---")


@st.cache_resource
def cached_load_models():
    return load_models()


@st.cache_data
def cached_load_data():
    return load_data()


try:
    kmeans_model, scaler = cached_load_models()
    df = cached_load_data()
except LoaderError as error:
    st.error(str(error))
    st.stop()

with st.expander("View Raw Customer Data"):
    st.dataframe(df.head(100), use_container_width=True)

with st.sidebar:
    st.header("1. API Configuration")
    google_api_key = st.text_input("Google AI Studio API Key (as it is free)", type="password")
    st.caption("Required to generate AI marketing strategies.")

    st.markdown("---")

    st.header("2. Input Customer Data")
    recency = st.slider("Recency (Days since last purchase)", min_value=1, max_value=365, value=30)
    frequency = st.slider("Frequency (Number of purchases)", min_value=1, max_value=100, value=5)
    monetary = st.slider("Monetary (Total spend in $)", min_value=10.0, max_value=5000.0, value=250.0)

    st.markdown("---")
    predict_btn = st.button("Predict Segment", use_container_width=True)

if predict_btn:
    try:
        predicted_cluster = predict_customer_cluster(kmeans_model, scaler, recency, frequency, monetary)
    except ValueError as error:
        st.error(str(error))
    else:
        label = CLUSTER_LABELS.get(predicted_cluster, f"Cluster {predicted_cluster}")
        profile = CLUSTER_PROFILES.get(predicted_cluster, "Customer profile unavailable.")

        st.subheader("Model Prediction")
        st.success(f"{label}")
        st.info(profile)
        st.markdown("---")

        st.subheader("AI Strategy Analysis")

        if not google_api_key:
            st.warning("Please enter your Google API Key in the sidebar to view the AI-generated strategy for this prediction.")
        else:
            with st.spinner("Generating targeted marketing strategy..."):
                try:
                    llm = create_llm(google_api_key)
                    prompt = build_marketing_prompt(predicted_cluster, recency, frequency, monetary)
                    response_text = generate_strategy(llm, prompt)
                    st.write(response_text)
                except Exception:
                    logging.error("Failed to generate AI strategy", exc_info=True)
                    st.error("Error generating AI strategy. Please verify your API key, check your network connection, or try again later.")