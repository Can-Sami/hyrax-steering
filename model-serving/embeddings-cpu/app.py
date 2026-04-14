from __future__ import annotations

import os
import secrets
import shutil
from typing import Literal

from huggingface_hub import hf_hub_download
import numpy as np
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="CPU Embeddings Service")
security = HTTPBearer()

model = None
model_name = os.getenv("EMBEDDING_CPU_MODEL_NAME", "Qwen/Qwen3-Embedding-4B")
api_key = os.getenv("EMBEDDING_API_KEY", "local-dev-key")
max_batch_size = int(os.getenv("EMBEDDING_MAX_BATCH_SIZE", "128"))
max_input_chars = int(os.getenv("EMBEDDING_MAX_INPUT_CHARS", "8192"))
max_score_docs = int(os.getenv("EMBEDDING_MAX_SCORE_DOCS", "64"))
max_score_query_chars = int(os.getenv("EMBEDDING_MAX_SCORE_QUERY_CHARS", "4096"))
max_score_doc_chars = int(os.getenv("EMBEDDING_MAX_SCORE_DOC_CHARS", "4096"))
max_score_total_chars = int(os.getenv("EMBEDDING_MAX_SCORE_TOTAL_CHARS", "32768"))


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


class ScoreRequest(BaseModel):
    model: str | None = None
    text_1: str
    text_2: list[str]


class ScoreData(BaseModel):
    index: int
    score: float


class ScoreResponse(BaseModel):
    object: str = "list"
    data: list[ScoreData]
    model: str


ScoreResponse.model_rebuild()


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


@app.post("/v1/score", response_model=ScoreResponse)
def create_scores(
    request: ScoreRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    verify_api_key(credentials)

    if not request.text_1.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="text_1 must be a non-empty string"
        )
    if not request.text_2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="text_2 must be a non-empty list of strings"
        )
    if len(request.text_2) > max_score_docs:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"text_2 exceeds maximum document count of {max_score_docs}"
        )
    if any((not item.strip()) for item in request.text_2):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="text_2 must contain non-empty strings only"
        )
    if len(request.text_1) > max_score_query_chars:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"text_1 exceeds maximum length of {max_score_query_chars} characters"
        )
    if any(len(item) > max_score_doc_chars for item in request.text_2):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"text_2 contains documents exceeding max length of {max_score_doc_chars} characters"
        )
    total_chars = len(request.text_1) + sum(len(item) for item in request.text_2)
    if total_chars > max_score_total_chars:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"score request exceeds total character limit of {max_score_total_chars}"
        )

    model_instance = load_model()
    query_embedding = model_instance.encode(
        [request.text_1],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    docs_embedding = model_instance.encode(
        request.text_2,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    scores = np.dot(docs_embedding, query_embedding).tolist()

    return ScoreResponse(
        data=[ScoreData(index=index, score=float(score)) for index, score in enumerate(scores)],
        model=request.model or model_name,
    )
