from pathlib import Path
from uuid import uuid4

from app.api.errors import AppError


class LocalStorageProvider:
    def __init__(self, root_dir: str = 'data/audio') -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def store(self, filename: str, content: bytes) -> str:
        safe_name = Path(filename).name
        if safe_name != filename:
            raise AppError(code='invalid_audio_filename', message='Invalid audio filename.', status_code=400)
        target = self.root / f'{uuid4()}-{safe_name}'
        target.write_bytes(content)
        return str(target)
