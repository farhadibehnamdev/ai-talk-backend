"""
Configuration module for AI-Talk backend.
Uses pydantic-settings for environment variable management.
"""

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # ===========================================
    # Server Configuration
    # ===========================================
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment name")

    # ===========================================
    # LLM Model Configuration
    # ===========================================
    llm_model_name: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct-AWQ",
        description="Hugging Face model name for LLM",
    )
    llm_max_model_len: int = Field(
        default=4096, description="Maximum context length for LLM"
    )
    llm_gpu_memory_utilization: float = Field(
        default=0.6, description="Fraction of GPU memory to use for LLM"
    )

    # ===========================================
    # ASR Model Configuration
    # ===========================================
    asr_model_name: str = Field(
        default="kyutai/moshi-asr", description="Model name for ASR (Speech-to-Text)"
    )
    asr_sample_rate: int = Field(
        default=16000, description="Audio sample rate for ASR input"
    )

    # ===========================================
    # TTS Model Configuration
    # ===========================================
    tts_model_name: str = Field(
        default="kyutai/moshi-tts", description="Model name for TTS (Text-to-Speech)"
    )
    tts_sample_rate: int = Field(
        default=24000, description="Audio sample rate for TTS output"
    )

    # ===========================================
    # Model Loading Options
    # ===========================================
    preload_models: bool = Field(
        default=True, description="Whether to load models on startup"
    )
    models_cache_dir: str = Field(
        default="/app/models", description="Directory to cache downloaded models"
    )

    # ===========================================
    # WebSocket Configuration
    # ===========================================
    audio_chunk_ms: int = Field(
        default=100, description="Audio chunk size in milliseconds"
    )
    max_conversation_duration: int = Field(
        default=1800,
        description="Maximum conversation duration in seconds (0 = unlimited)",
    )
    ws_ping_interval: int = Field(
        default=30, description="WebSocket ping interval in seconds"
    )

    # ===========================================
    # Redis Configuration (Optional)
    # ===========================================
    redis_enabled: bool = Field(
        default=False, description="Enable Redis for session management"
    )
    redis_host: str = Field(default="redis", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")

    # ===========================================
    # CORS Configuration
    # ===========================================
    cors_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated list of allowed CORS origins",
    )

    # ===========================================
    # Logging
    # ===========================================
    log_level: str = Field(default="INFO", description="Logging level")

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string to list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def redis_url(self) -> str:
        """Construct Redis URL from components."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Global settings instance for convenience
settings = get_settings()
