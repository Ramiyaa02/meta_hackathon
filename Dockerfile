FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY models.py ./
COPY inference.py ./
COPY openenv.yaml ./
COPY server /app/server
COPY __init__.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir uvicorn[standard]>=0.20 httpx

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENM_ENABLE_WEB_INTERFACE=true
ENV OPENM_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=2)" || exit 1

# Start server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
