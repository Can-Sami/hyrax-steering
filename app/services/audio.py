from dataclasses import dataclass
from pathlib import Path

from app.api.errors import AppError


@dataclass(frozen=True)
class AudioConstraints:
    max_bytes: int = 10 * 1024 * 1024
    allowed_extensions: tuple[str, ...] = ('.wav', '.mp3', '.m4a', '.ogg')


class AudioValidator:
    def __init__(self, constraints: AudioConstraints | None = None) -> None:
        self.constraints = constraints or AudioConstraints()

    def validate(self, filename: str, content: bytes) -> None:
        extension = Path(filename).suffix.lower()
        if extension not in self.constraints.allowed_extensions:
            raise AppError(code='invalid_audio_format', message='Unsupported audio format.', status_code=400)
        if len(content) == 0:
            raise AppError(code='empty_audio', message='Audio file is empty.', status_code=400)
        if len(content) > self.constraints.max_bytes:
            raise AppError(code='audio_too_large', message='Audio file exceeds size limit.', status_code=413)
