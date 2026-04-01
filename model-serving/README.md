# Model Serving Project (vLLM + Whisper)

This project runs a separate model service for backend intent recognition.

- `embedding-vllm`: serves embeddings via vLLM (GPU-accelerated)
- `embeddings-cpu`: serves embeddings via Transformers on CPU
- `model-gateway`: serves `/v1/audio/transcriptions` using faster-whisper and proxies embeddings based on backend mode

## Backend modes

The service supports two embedding backend modes via `EMBEDDING_BACKEND_MODE`:

- `EMBEDDING_BACKEND_MODE=vllm` => vLLM-only embeddings, fail fast on upstream issues
- `EMBEDDING_BACKEND_MODE=transformers_cpu` => CPU embeddings service only

## Run locally

### Production mode (vLLM with GPU)

```bash
cp .env.prod.vllm.example .env && ../scripts/model-serving.sh start
```

### Development mode (CPU only)

```bash
cp .env.dev.cpu.example .env && ../scripts/model-serving.sh start
```

Gateway endpoint: `http://127.0.0.1:8002/v1`

For `POST /v1/embeddings`, the gateway contract is strict:
- Allowed request field: `input`
- Rejected request fields: `model`, `encoding_format`, or any unknown fields (returns HTTP 400 with structured error payload)
- Response `model` metadata always reflects the backend-configured effective model

## Backend Configuration

Configure the main backend to use the model service:

```bash
export STT_ENGINE=openai_compatible
export EMBEDDING_ENGINE=openai_compatible
export OPENAI_BASE_URL=http://127.0.0.1:8002/v1
export OPENAI_API_KEY=local-dev-key
export WHISPER_MODEL_NAME=openai/whisper-large-v3
```

The embedding model variable depends on your chosen mode:
- vLLM mode: `EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5` (or other vLLM-compatible models)
- CPU mode: `EMBEDDING_CPU_MODEL_NAME=jinaai/jina-embeddings-v3`

For production deployments, set `APP_ENV=production` and replace the default API keys with secure values for `GATEWAY_API_KEY` and `EMBEDDING_API_KEY`. If the embedding model is gated or private on Hugging Face, also set `HUGGING_FACE_HUB_TOKEN`.

## Advanced Configuration

For low-VRAM GPUs in vLLM mode, adjust in `.env`:

```bash
VLLM_MAX_MODEL_LEN=512
VLLM_DTYPE=half
```
