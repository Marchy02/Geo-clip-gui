# GeoOSINT — immagine completa (CPU). Build: docker build -t geoosint .
# Avvio:  docker run -p 7860:7860 --env-file .env -v geoosint-cache:/root/.cache geoosint
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 REPORTS_DIR=/data/reports GEOOSINT_CONFIG=/data/settings.json

# Librerie di sistema richieste da OpenCV (EasyOCR / ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# PyTorch CPU-only: molto più leggero della build CUDA
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY . .
RUN pip install ".[all]"

VOLUME ["/data", "/root/.cache"]
EXPOSE 7860
CMD ["geoosint", "ui", "--host", "0.0.0.0", "--port", "7860"]
