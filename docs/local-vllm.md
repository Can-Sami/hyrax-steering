# Local model-service integration

Use the dedicated model-serving project in `model-serving/`.

## 1) Start model service

Choose the appropriate mode for your environment and copy the matching `.env` file before starting the stack:

**Production mode (`EMBEDDING_BACKEND_MODE=vllm`):**
```bash
cd model-serving
cp .env.prod.vllm.example .env
../scripts/model-serving.sh start
```

**Development mode (`EMBEDDING_BACKEND_MODE=transformers_cpu`):**
```bash
cd model-serving
cp .env.dev.cpu.example .env
../scripts/model-serving.sh start
```

This exposes a single OpenAI-compatible gateway at `http://127.0.0.1:8002/v1`.

`EMBEDDING_BACKEND_MODE=vllm` keeps embeddings on vLLM only and should fail fast if the upstream GPU service is unhealthy. `EMBEDDING_BACKEND_MODE=transformers_cpu` routes embeddings to the CPU service only.

## 2) Configure backend

```bash
export STT_ENGINE=openai_compatible
export EMBEDDING_ENGINE=openai_compatible
export OPENAI_BASE_URL=http://127.0.0.1:8002/v1
export OPENAI_API_KEY=local-dev-key
export WHISPER_MODEL_NAME=openai/whisper-large-v3
```

Set the embedding model variable to match your chosen mode:
- vLLM mode: `export EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5`
- CPU mode: `export EMBEDDING_CPU_MODEL_NAME=jinaai/jina-embeddings-v3`

For production deployments, set `APP_ENV=production` and use secure values for `GATEWAY_API_KEY` and `EMBEDDING_API_KEY`. If the embedding model is gated or private on Hugging Face, also export `HUGGING_FACE_HUB_TOKEN` before starting the stack.

## 3) Run backend

```bash
uvicorn app.main:app --reload
```

Then call `POST /api/v1/inference/intent` with multipart `audio_file`.
