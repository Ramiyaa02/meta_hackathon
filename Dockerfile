FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files (all at root)
COPY pyproject.toml README.md ./
COPY models.py ./
COPY openenv.yaml ./
COPY inference.py ./
COPY __init__.py ./
COPY app.py ./
COPY sql_query_environment.py ./
COPY client.py ./
# Copy any other needed files (optional)
# COPY requirements.txt ./

# Install the package and dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir uvicorn[standard]>=0.20 httpx

# Expose the port (HF Spaces expects 7860 for Docker)
EXPOSE 7860

# Run the FastAPI app (app.py)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]