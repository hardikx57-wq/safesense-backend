FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Writable config/cache dirs — harmless and safe to keep even on Render.
ENV YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp \
    PYTHONUNBUFFERED=1

WORKDIR /app

# No "Backend/" prefix: Render's build context already starts inside
# Backend/ because Root Directory is set to "Backend" in the dashboard.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets $PORT itself (no fixed value like Hugging Face's 7860) —
# the container must bind to whatever Render assigns.
# 1 worker: four models are loaded per process, more workers = more RAM.
CMD gunicorn -w 1 --threads 4 -t 300 -b 0.0.0.0:${PORT:-10000} app:app