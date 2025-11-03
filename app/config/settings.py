from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Azure OpenAI
    azure_openai_api_key: str
    azure_openai_endpoint: str
    openai_api_version: str

    # LLM settings
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7

    # Directories
    audio_download_dir: str = "data/audio"
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    failed_dir: str = "data/failed"

    # Concurrency settings
    max_concurrent_scrapes: int = 5  # Max concurrent Cambridge scrapes
    max_concurrent_llm: int = 3      # Max concurrent LLM requests
    llm_batch_size: int = 8          # Number of words per LLM batch request
    scraper_rate_limit: float = 0.5  # Delay between scrapes (seconds)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    return Settings()


def ensure_directories():
    """Create necessary directories if they don't exist"""
    settings = get_settings()
    Path(settings.audio_download_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.input_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.failed_dir).mkdir(parents=True, exist_ok=True)