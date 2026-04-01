from __future__ import annotations

import os
import secrets
import shutil
from typing import Literal

from huggingface_hub import hf_hub_download
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="CPU Embeddings Service")
security = HTTPBearer()

model = None
model_name = os.getenv("EMBEDDING_CPU_MODEL_NAME", "jinaai/jina-embeddings-v3")
api_key = os.getenv("EMBEDDING_API_KEY", "local-dev-key")
max_batch_size = int(os.getenv("EMBEDDING_MAX_BATCH_SIZE", "128"))
max_input_chars = int(os.getenv("EMBEDDING_MAX_INPUT_CHARS", "8192"))


def _repair_jina_flash_module_cache(error_message: str) -> bool:
    marker = "xlm_hyphen_roberta_hyphen_flash_hyphen_implementation"
    if marker not in error_message:
        return False

    missing_path = error_message.split("'", 2)[1] if "'" in error_message else ""
    if not missing_path:
        return False

    missing_file = os.path.basename(missing_path)
    downloaded = hf_hub_download(
        repo_id="jinaai/xlm-roberta-flash-implementation",
        filename=missing_file,
    )
    os.makedirs(os.path.dirname(missing_path), exist_ok=True)
    shutil.copy(downloaded, missing_path)
    return True


def load_model():
    global model
    if model is None:
        try:
            model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device="cpu"
            )
        except FileNotFoundError as e:
            if _repair_jina_flash_module_cache(str(e)):
                model = SentenceTransformer(
                    model_name,
                    trust_remote_code=True,
                    device="cpu"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Failed to load model {model_name}: {str(e)}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to load model {model_name}: {str(e)}"
            )
    return model


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    encoding_format: Literal["float"] = "float"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "sentence-transformers"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not secrets.compare_digest(credentials.credentials, api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )


@app.get("/v1/models", response_model=ModelsResponse)
def list_models():
    return ModelsResponse(
        data=[
            ModelInfo(id=model_name)
        ]
    )


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embeddings(
    request: EmbeddingRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    verify_api_key(credentials)
    
    texts = [request.input] if isinstance(request.input, str) else request.input
    
    if not texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input list cannot be empty"
        )
    
    if any(not text or not text.strip() for text in texts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input strings cannot be empty"
        )
    
    if len(texts) > max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch size {len(texts)} exceeds maximum {max_batch_size}"
        )
    
    if any(len(text) > max_input_chars for text in texts):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Input text exceeds maximum length of {max_input_chars} characters"
        )
    
    model_instance = load_model()
    
    embeddings = model_instance.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    
    embedding_list = embeddings.tolist()
    
    data = [
        EmbeddingData(embedding=emb, index=idx)
        for idx, emb in enumerate(embedding_list)
    ]
    
    total_tokens = sum(len(text.split()) for text in texts)
    
    return EmbeddingResponse(
        data=data,
        model=model_name,
        usage=EmbeddingUsage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens
        )
    )


@app.get("/health")
def health():
    return {"status": "healthy"}
