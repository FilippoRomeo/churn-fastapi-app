"""
Configuration management using pydantic-settings
"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
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
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
