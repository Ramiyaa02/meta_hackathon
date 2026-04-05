FROM python:3.11-slim

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all files (flat structure)
COPY pyproject.toml ./
COPY models.py ./
COPY sql_query_environment.py ./
COPY app.py ./
COPY inference.py ./
COPY __init__.py ./
COPY client.py ./

# Install dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir uvicorn[standard]>=0.20 httpx

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]