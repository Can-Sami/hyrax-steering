from __future__ import annotations

import os


def get_embedding_backend_url(mode: str | None = None) -> tuple[str, str]:
    """
    Resolve embedding backend URL based on mode.
    
    Args:
        mode: Backend mode ('vllm' or 'transformers_cpu'). If None, reads from EMBEDDING_BACKEND_MODE env var.
    
    Returns:
        Tuple of (backend_name, url)
    
    Raises:
        ValueError: If mode is invalid (not 'vllm' or 'transformers_cpu')
    """
    if mode is None:
        mode = os.getenv('EMBEDDING_BACKEND_MODE', 'vllm')
    
    mode = mode.strip().lower()
    
    if mode == 'vllm':
        url = os.getenv('EMBEDDING_VLLM_BASE_URL', 'http://embedding-vllm:8000/v1').rstrip('/')
        return ('vllm', url)
    elif mode == 'transformers_cpu':
        url = os.getenv('EMBEDDING_CPU_BASE_URL', 'http://embeddings-cpu:8000/v1').rstrip('/')
        return ('transformers_cpu', url)
    else:
        raise ValueError(
            f'Invalid backend mode: {mode!r}. '
            f'EMBEDDING_BACKEND_MODE must be "vllm" or "transformers_cpu".'
        )
