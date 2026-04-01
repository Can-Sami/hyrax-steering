from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse


@dataclass(slots=True)
class AppError(Exception):
    code: str
    message: str
    status_code: int


async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'error': {
                'code': exc.code,
                'message': exc.message,
            }
        },
    )
