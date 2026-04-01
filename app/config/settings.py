from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_name: str = Field(default='callsteering-backend')
    app_env: str = Field(default='dev')
    app_version: str = Field(default='0.1.0')
    api_prefix: str = Field(default='/api')
    log_level: str = Field(default='INFO')
    database_url: str = Field(default='postgresql+psycopg://postgres:postgres@localhost:5432/callsteering')
    inference_device: str = Field(default='auto')
    model_precision: str = Field(default='fp32')
    stt_engine: str = Field(default='stub')
    embedding_engine: str = Field(default='stub')
    openai_base_url: str = Field(default='http://127.0.0.1:8001/v1')
    openai_api_key: str = Field(default='local-dev-key')
    whisper_model_name: str = Field(default='whisper-large-v3')
    embedding_model_name: str = Field(default='Qwen/Qwen3-Embedding-8B')
    api_key: str = Field(default='')


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
