from collections.abc import Callable
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        request_id = request.headers.get('x-request-id', str(uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers['x-request-id'] = request_id
        return response
