from pathlib import Path

from app.services.storage import LocalStorageProvider


def test_local_storage_provider_writes_file(tmp_path: Path) -> None:
    provider = LocalStorageProvider(root_dir=str(tmp_path))
    stored_path = provider.store('sample.wav', b'data')

    written = Path(stored_path)
    assert written.exists()
    assert written.read_bytes() == b'data'
