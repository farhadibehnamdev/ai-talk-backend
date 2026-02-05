"""
ASR (Automatic Speech Recognition) Service using Kyutai Moshi STT.

This service handles real-time speech-to-text conversion for the AI-Talk application.
"""

import asyncio
import io
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import numpy as np
import torch
from loguru import logger

from app.config import settings


@dataclass
class TranscriptionResult:
    """Result from ASR transcription."""

    text: str
    is_final: bool
    confidence: float = 1.0
    language: str = "en"


class ASRService:
    """
    Automatic Speech Recognition service using Kyutai Moshi STT.

    This service provides:
    - Audio preprocessing and format conversion
    - Real-time streaming transcription
    - Batch transcription for complete audio files
    """

    def __init__(self):
        """Initialize the ASR service."""
        self._model = None
        self._processor = None
        self._device = None
        self._is_loaded = False
        self._lock = asyncio.Lock()

        # Audio configuration
        self.sample_rate = settings.asr_sample_rate  # 16000 Hz expected
        self.chunk_duration_ms = settings.audio_chunk_ms

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded

    async def load_model(self) -> None:
        """
        Load the ASR model into memory.

        This should be called during application startup if PRELOAD_MODELS is True,
        or lazily on first transcription request.
        """
        async with self._lock:
            if self._is_loaded:
                logger.debug("ASR model already loaded, skipping")
                return

            logger.info(f"Loading ASR model: {settings.asr_model_name}")

            try:
                # Determine device
                self._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                logger.info(f"ASR using device: {self._device}")

                # Load model in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync)

                self._is_loaded = True
                logger.info("ASR model loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load ASR model: {e}")
                raise RuntimeError(f"Failed to load ASR model: {e}") from e

    def _load_model_sync(self) -> None:
        """Synchronous model loading (runs in thread pool)."""
        try:
            # Try to load Kyutai Moshi ASR
            # Note: The actual import may vary based on Moshi's API
            # This is a placeholder that should be updated based on actual Moshi documentation
            from moshi.models import loaders

            self._model, self._processor = loaders.load_asr_model(
                settings.asr_model_name,
                device=self._device,
                cache_dir=settings.models_cache_dir,
            )
        except ImportError:
            # Fallback: Use Whisper or another ASR model for development
            logger.warning(
                "Moshi ASR not available, falling back to Whisper for development"
            )
            self._load_whisper_fallback()

    def _load_whisper_fallback(self) -> None:
        """Load Whisper as a fallback ASR model for development."""
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            model_name = "openai/whisper-base"
            self._processor = WhisperProcessor.from_pretrained(model_name)
            self._model = WhisperForConditionalGeneration.from_pretrained(
                model_name
            ).to(self._device)
            logger.info("Loaded Whisper fallback model for ASR")
        except Exception as e:
            logger.error(f"Failed to load fallback ASR model: {e}")
            raise

    async def unload_model(self) -> None:
        """Unload the model to free GPU memory."""
        async with self._lock:
            if not self._is_loaded:
                return

            logger.info("Unloading ASR model")
            del self._model
            del self._processor
            self._model = None
            self._processor = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._is_loaded = False
            logger.info("ASR model unloaded")

    def preprocess_audio(self, audio_bytes: bytes) -> np.ndarray:
        """
        Preprocess audio bytes into the format expected by the model.

        Args:
            audio_bytes: Raw audio bytes (expected format: 16-bit PCM, mono)

        Returns:
            Numpy array of audio samples normalized to [-1, 1]
        """
        try:
            # Try to decode as raw PCM first
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            # Normalize to [-1, 1]
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception:
            # Try using soundfile for other formats
            try:
                import soundfile as sf

                audio_array, sr = sf.read(io.BytesIO(audio_bytes))

                # Convert to mono if stereo
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1)

                # Resample if needed
                if sr != self.sample_rate:
                    audio_array = self._resample(audio_array, sr, self.sample_rate)

                return audio_array.astype(np.float32)
            except Exception as e:
                logger.error(f"Failed to preprocess audio: {e}")
                raise ValueError(f"Could not process audio data: {e}") from e

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import librosa

            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Simple resampling without librosa
            duration = len(audio) / orig_sr
            target_length = int(duration * target_sr)
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, np.arange(len(audio)), audio)

    async def transcribe(
        self, audio_bytes: bytes, language: str = "en"
    ) -> TranscriptionResult:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio data
            language: Language code (default: "en" for English)

        Returns:
            TranscriptionResult with the transcribed text
        """
        if not self._is_loaded:
            await self.load_model()

        try:
            # Preprocess audio
            audio_array = self.preprocess_audio(audio_bytes)

            # Skip if audio is too short (less than 100ms)
            min_samples = int(self.sample_rate * 0.1)
            if len(audio_array) < min_samples:
                return TranscriptionResult(
                    text="", is_final=True, confidence=0.0, language=language
                )

            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None, self._transcribe_sync, audio_array, language
            )

            return TranscriptionResult(
                text=text.strip(),
                is_final=True,
                confidence=1.0,
                language=language,
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return TranscriptionResult(
                text="", is_final=True, confidence=0.0, language=language
            )

    def _transcribe_sync(self, audio_array: np.ndarray, language: str) -> str:
        """Synchronous transcription (runs in thread pool)."""
        try:
            # Moshi ASR transcription
            # Adjust this based on actual Moshi API
            if hasattr(self._model, "transcribe"):
                result = self._model.transcribe(audio_array)
                return result.get("text", "")
            else:
                # Whisper fallback
                return self._transcribe_whisper(audio_array, language)
        except Exception as e:
            logger.error(f"Sync transcription failed: {e}")
            return ""

    def _transcribe_whisper(self, audio_array: np.ndarray, language: str) -> str:
        """Transcribe using Whisper model (fallback)."""
        input_features = self._processor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        ).input_features.to(self._device)

        with torch.no_grad():
            predicted_ids = self._model.generate(
                input_features,
                language=language,
                task="transcribe",
            )

        transcription = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return transcription[0] if transcription else ""

    async def transcribe_stream(
        self, audio_stream: AsyncGenerator[bytes, None], language: str = "en"
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """
        Stream transcription for real-time audio.

        Args:
            audio_stream: Async generator yielding audio chunks
            language: Language code

        Yields:
            TranscriptionResult objects with partial and final transcriptions
        """
        if not self._is_loaded:
            await self.load_model()

        audio_buffer = b""
        chunk_size = int(
            self.sample_rate * (self.chunk_duration_ms / 1000) * 2
        )  # 2 bytes per sample

        async for audio_chunk in audio_stream:
            audio_buffer += audio_chunk

            # Process when we have enough audio
            if len(audio_buffer) >= chunk_size:
                result = await self.transcribe(audio_buffer, language)
                if result.text:
                    yield result
                audio_buffer = b""

        # Process remaining audio
        if audio_buffer:
            result = await self.transcribe(audio_buffer, language)
            result.is_final = True
            yield result


# Global service instance (singleton pattern)
_asr_service: Optional[ASRService] = None


def get_asr_service() -> ASRService:
    """Get or create the global ASR service instance."""
    global _asr_service
    if _asr_service is None:
        _asr_service = ASRService()
    return _asr_service
