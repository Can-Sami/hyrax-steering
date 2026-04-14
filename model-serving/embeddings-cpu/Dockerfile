FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV EMBEDDING_CPU_MODEL_NAME=Qwen/Qwen3-Embedding-4B
ENV EMBEDDING_API_KEY=local-dev-key

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
