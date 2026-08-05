# Deployment Guide

This guide describes simple deployment options for the Streamlit customer segmentation app.

## Local deployment

1. Activate the virtual environment:
   ```powershell
   .\env\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run the app:
   ```powershell
   streamlit run app.py
   ```

## Docker deployment

A `Dockerfile` is included in the repository for container-based deployment. Build and run the container using Docker or a compatible runtime.
