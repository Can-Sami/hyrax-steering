from fastapi import Header

from app.api.errors import AppError
from app.config.settings import get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if settings.api_key and x_api_key != settings.api_key:
        raise AppError(code='unauthorized', message='Invalid API key.', status_code=401)
