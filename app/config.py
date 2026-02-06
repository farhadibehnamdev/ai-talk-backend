"""
Configuration module for AI-Talk backend.
Uses pydantic-settings for environment variable management.
"""

from functools import lru_cache
from typing import List, Optional

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
        default=0.35,
        description="Fraction of GPU memory to use for LLM (0.35 = ~8.4GB on 24GB GPU)",
    )
    llm_tensor_parallel_size: int = Field(
        default=1, description="Number of GPUs for tensor parallelism"
    )
    llm_dtype: str = Field(
        default="auto", description="Data type for LLM (auto, float16, bfloat16)"
    )

    # ===========================================
    # ASR (Speech-to-Text) Model Configuration
    # ===========================================
    asr_model_name: str = Field(
        default="kyutai/stt-2.6b-en",
        description="Kyutai STT model for Speech-to-Text (2.6B English model)",
    )
    asr_sample_rate: int = Field(
        default=16000, description="Audio sample rate for ASR input (Hz)"
    )
    asr_device: Optional[str] = Field(
        default=None,
        description="Device for ASR model (None = auto-detect, 'cuda:0', 'cpu')",
    )
    asr_dtype: str = Field(
        default="float16",
        description="Data type for ASR model (float16, bfloat16, float32)",
    )
    # STT streaming delay: 2.6b-en has 2.5s delay, 1b-en_fr has 0.5s delay
    asr_stream_delay_ms: int = Field(
        default=2500,
        description="STT streaming delay in milliseconds (2500 for 2.6b-en, 500 for 1b-en_fr)",
    )
    asr_prefer_moshi: bool = Field(
        default=True,
        description="Prefer Moshi backend for Kyutai STT models (more stable on CUDA)",
    )
    asr_enable_kyutai_transformers: bool = Field(
        default=False,
        description="Enable Kyutai STT via transformers backend (disabled by default for stability)",
    )
    asr_kyutai_attn_implementation: str = Field(
        default="eager",
        description="Attention implementation for Kyutai transformers backend (eager, sdpa, flash_attention_2)",
    )
    asr_whisper_fallback_model: str = Field(
        default="openai/whisper-large-v3",
        description="Whisper model used when Kyutai STT backends are unavailable",
    )
    asr_silence_rms_threshold: float = Field(
        default=0.006,
        description="Minimum RMS for considering decoded audio as speech",
    )
    asr_silence_peak_threshold: float = Field(
        default=0.08,
        description="Minimum peak amplitude for considering decoded audio as speech",
    )
    asr_filter_noise_transcripts: bool = Field(
        default=True,
        description="Drop low-information transcripts from low-energy chunks (e.g., repeated 'you')",
    )
    asr_noise_max_duration_s: float = Field(
        default=2.0,
        description="Apply low-information noise transcript filter only for short chunks",
    )

    # ===========================================
    # TTS (Text-to-Speech) Model Configuration
    # ===========================================
    tts_model_name: str = Field(
        default="kyutai/tts-1.6b-en_fr",
        description="Kyutai TTS model for Text-to-Speech (1.8B English/French model)",
    )
    tts_sample_rate: int = Field(
        default=24000, description="Audio sample rate for TTS output (Hz)"
    )
    tts_device: Optional[str] = Field(
        default=None,
        description="Device for TTS model (None = auto-detect, 'cuda:0', 'cpu')",
    )
    tts_dtype: str = Field(
        default="float16",
        description="Data type for TTS model (float16, bfloat16, float32)",
    )
    tts_voice: str = Field(
        default="default",
        description="TTS voice to use (from kyutai/tts-voices repository)",
    )
    # TTS has 1.28s audio shift w.r.t. text
    tts_audio_shift_ms: int = Field(
        default=1280, description="TTS audio shift in milliseconds"
    )

    # ===========================================
    # GPU Memory Management
    # ===========================================
    gpu_memory_reserved_gb: float = Field(
        default=2.0, description="GPU memory to reserve for system (GB)"
    )
    enable_memory_efficient_loading: bool = Field(
        default=True, description="Enable sequential model loading to manage memory"
    )
    cuda_memory_fraction: float = Field(
        default=0.95, description="Maximum fraction of CUDA memory to use"
    )

    # ===========================================
    # Model Loading Options
    # ===========================================
    preload_models: bool = Field(
        default=True, description="Whether to load models on startup"
    )
    models_cache_dir: str = Field(
        default="./models", description="Directory to cache downloaded models"
    )
    use_flash_attention: bool = Field(
        default=True, description="Use Flash Attention 2 if available"
    )
    trust_remote_code: bool = Field(
        default=True, description="Trust remote code when loading models"
    )

    # ===========================================
    # Inference Configuration
    # ===========================================
    llm_max_tokens: int = Field(
        default=256, description="Default max tokens for LLM generation"
    )
    llm_temperature: float = Field(
        default=0.7, description="Default temperature for LLM generation"
    )
    llm_top_p: float = Field(
        default=0.9, description="Default top-p for LLM generation"
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
    ws_ping_timeout: int = Field(
        default=10, description="WebSocket ping timeout in seconds"
    )

    # ===========================================
    # Audio Processing Configuration
    # ===========================================
    vad_enabled: bool = Field(
        default=True, description="Enable Voice Activity Detection"
    )
    vad_threshold: float = Field(
        default=0.5, description="VAD threshold for speech detection"
    )
    audio_normalize: bool = Field(
        default=True, description="Normalize audio input/output"
    )
    silence_threshold_db: float = Field(
        default=-40.0, description="Silence threshold in dB"
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
    redis_password: Optional[str] = Field(default=None, description="Redis password")

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
    log_format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Loguru log format",
    )

    # ===========================================
    # Computed Properties
    # ===========================================
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string to list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def redis_url(self) -> str:
        """Construct Redis URL from components."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    @property
    def total_model_vram_estimate_gb(self) -> float:
        """Estimate total VRAM needed for all models (approximate)."""
        # Estimates based on model sizes:
        # STT 2.6B fp16: ~6-8 GB
        # TTS 1.8B fp16: ~4-5 GB
        # LLM 7B AWQ 4-bit: ~4-5 GB
        stt_vram = 7.0  # GB
        tts_vram = 4.5  # GB
        llm_vram = 5.0  # GB (AWQ quantized)
        return stt_vram + tts_vram + llm_vram

    def get_device(self, preference: Optional[str] = None) -> str:
        """Get the appropriate device string."""
        import torch

        if preference:
            return preference
        return "cuda" if torch.cuda.is_available() else "cpu"

    def validate_gpu_memory(self) -> dict:
        """Validate if GPU has enough memory for all models."""
        try:
            import torch

            if not torch.cuda.is_available():
                return {
                    "sufficient": False,
                    "message": "No CUDA GPU available",
                    "available_gb": 0,
                    "required_gb": self.total_model_vram_estimate_gb,
                }

            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            available_memory = total_memory - self.gpu_memory_reserved_gb
            required_memory = self.total_model_vram_estimate_gb

            return {
                "sufficient": available_memory >= required_memory,
                "message": f"GPU: {torch.cuda.get_device_name(0)}",
                "total_gb": round(total_memory, 2),
                "available_gb": round(available_memory, 2),
                "required_gb": round(required_memory, 2),
            }
        except Exception as e:
            return {
                "sufficient": False,
                "message": f"Error checking GPU: {e}",
                "available_gb": 0,
                "required_gb": self.total_model_vram_estimate_gb,
            }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Global settings instance for convenience
settings = get_settings()
