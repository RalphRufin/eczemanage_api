**`Dockerfile`** (same as before but optimized for HF):
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Download model during build (HF has plenty of RAM for this)
RUN python download_model.py

# Expose port (HF Spaces uses 8000 by default)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
