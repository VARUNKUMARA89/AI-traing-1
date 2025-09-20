from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""
    PROJECT_NAME: str = "AI Trading System"
    VERSION: str = "1.0.0"
    
    # DhanHQ API settings
    DHAN_CLIENT_ID: str
    DHAN_ACCESS_TOKEN: str
    
    # Telegram settings
    TELEGRAM_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./trading.db"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

settings = get_settings()