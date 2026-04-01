import pytest

from app.api.errors import AppError
from app.services.audio import AudioValidator


def test_audio_validator_accepts_supported_file() -> None:
    AudioValidator().validate('test.wav', b'abc')


def test_audio_validator_accepts_mp3_file() -> None:
    AudioValidator().validate('test.mp3', b'abc')


@pytest.mark.parametrize('filename', ['test.txt', 'test.pdf'])
def test_audio_validator_rejects_invalid_format(filename: str) -> None:
    with pytest.raises(AppError) as exc:
        AudioValidator().validate(filename, b'abc')
    assert exc.value.code == 'invalid_audio_format'


def test_audio_validator_rejects_empty_file() -> None:
    with pytest.raises(AppError) as exc:
        AudioValidator().validate('test.wav', b'')
    assert exc.value.code == 'empty_audio'
