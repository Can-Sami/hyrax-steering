import pytest

from app.api.errors import AppError
from app.services.storage import LocalStorageProvider


def test_store_rejects_path_traversal_filename(tmp_path) -> None:
    provider = LocalStorageProvider(root_dir=str(tmp_path))

    with pytest.raises(AppError) as exc:
        provider.store('../evil.wav', b'abc')

    assert exc.value.code == 'invalid_audio_filename'


def test_store_writes_file_for_safe_name(tmp_path) -> None:
    provider = LocalStorageProvider(root_dir=str(tmp_path))

    stored = provider.store('safe.wav', b'abc')

    assert stored.endswith('safe.wav')
    assert (tmp_path / stored.split('/')[-1]).read_bytes() == b'abc'
