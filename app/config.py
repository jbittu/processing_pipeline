from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    nvidia_api_key: str
    nvidia_model: str = "meta/llama-3.2-90b-vision-instruct"
    log_level: str = "INFO"
    max_file_size_mb: int = 20
    max_concurrent_agents: int = 3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()