FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Create non-root user
RUN useradd -m -u 1000 raguser && chown -R raguser:raguser /app
USER raguser

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]