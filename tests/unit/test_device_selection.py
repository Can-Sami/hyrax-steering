import os

from app.config.settings import get_settings


def test_inference_device_defaults_to_auto() -> None:
    os.environ.pop('INFERENCE_DEVICE', None)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.inference_device == 'auto'
