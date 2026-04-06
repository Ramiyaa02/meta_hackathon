
FROM python:3.11-slim

RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

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
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic httpx aiosqlite sqlparse openenv-core

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]