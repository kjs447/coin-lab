from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration container."""

    database_url: str = Field(
        default=f"sqlite:///{(Path(__file__).resolve().parent.parent / 'coin_lab.db')}"
    )
    upbit_base_url: str = "https://api.upbit.com/v1"
    default_market: str = "KRW-BTC"
    max_candle_request: int = 200
    random_search_trials: int = 20
    random_seed: int = 42
    upbit_max_retries: int = 5
    upbit_retry_backoff: float = 0.8

    class Config:
        env_prefix = "COIN_LAB_"
        case_sensitive = False


settings = Settings()
