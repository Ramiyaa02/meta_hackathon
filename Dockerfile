# AI Customer Support Ticket System — Docker
#
# Build:  docker build -t ai-customer-support .
# Run:    docker run -p 8000:8000 ai-customer-support
# With API key for inference:
#         docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... ai-customer-support

FROM python:3.11-slim

LABEL maintainer="OpenEnv Contributors"
LABEL description="AI Customer Support Ticket System — OpenEnv Environment"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY env/ ./env/
COPY data/ ./data/
COPY openenv.yaml .
COPY app.py .
COPY inference.py .

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
