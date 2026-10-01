"""
Configuration settings using Pydantic
"""
from secrets import token_urlsafe

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Database
    database_url: str = "sqlite+aiosqlite:///./churn_predictions.db"

    # Model
    model_path: str = "saved_models/churn_model.pkl"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # Rate Limiting
    rate_limit_predict: str = "60/minute"
    rate_limit_batch: str = "10/minute"
    rate_limit_general: str = "100/minute"

    # JWT Authentication
    # Local development gets a per-process random key if SECRET_KEY is not set.
    # Set SECRET_KEY explicitly for stable deployments or multiple workers.
    secret_key: str = token_urlsafe(32)

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields from .env


settings = Settings()
