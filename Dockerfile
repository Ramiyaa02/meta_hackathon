FROM python:3.11-slim

RUN pip install --no-cache-dir fastapi uvicorn

WORKDIR /app
COPY app.py .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]