FROM python:3.11-slim

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all Python files and config
COPY pyproject.toml README.md ./
COPY models.py ./
COPY openenv.yaml ./
COPY inference.py ./
COPY __init__.py ./
COPY app.py ./
COPY sql_query_environment.py ./
COPY client.py ./

# Install dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir uvicorn[standard]>=0.20 httpx

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]