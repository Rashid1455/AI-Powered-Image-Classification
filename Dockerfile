# Serves the Streamlit demo with the committed model checkpoint (CPU inference).
#   docker build -t flower-classifier .
#   docker run -p 8501:8501 flower-classifier
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Inference-only dependencies (no notebooks / training extras).
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        numpy pandas pillow opencv-python-headless matplotlib streamlit pyyaml

COPY config.yaml ./
COPY src/ ./src/
COPY app/ ./app/
COPY models/*.pt ./models/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
