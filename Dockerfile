# Single Container Multi-Agent ASL Conversation Assistant
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies for OpenCV, MediaPipe, audio/TTS, and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libasound2-dev \
    libespeak1 \
    ffmpeg \
    espeak \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and assets
COPY src/ ./src/
COPY static/ ./static/
COPY tests/ ./tests/
COPY *.csv ./
COPY project.md AGENTS.md README.md ./

# Expose HTTP/WebSocket port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/agents/status || exit 1

# Start FastAPI application
CMD ["python", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
