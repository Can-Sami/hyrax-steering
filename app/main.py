import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import AppError, app_error_handler
from app.api.middleware import RequestIdMiddleware
from app.api.routes import router
from app.config.logging import configure_logging
from app.config.settings import get_settings

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info('application_started', extra={'request_id': 'system'})
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)
app.include_router(router, prefix=settings.api_prefix)
app.add_exception_handler(AppError, app_error_handler)
