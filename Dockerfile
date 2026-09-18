FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
EXPOSE 7860
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
