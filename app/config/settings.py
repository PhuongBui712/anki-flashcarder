from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # API keys
    azure_openai_api_key: str
    azure_openai_endpoint: str
    openai_api_version: str

    # Directories
    audio_download_dir: str = "data/audio"
    input_dir: str = "data/input"
    output_dir: str = "data/output"

    # LLM settings
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7

    # Word generation settings
    min_word_types: int = 3
    max_word_types: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def ensure_directories():
    """Create necessary directories if they don't exist"""
    global settings
    Path(settings.audio_download_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.input_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)


settings = Settings()