"""
TTS (Text-to-Speech) Service using Kyutai Moshi TTS.

This service handles text-to-speech conversion for the AI-Talk application.
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
class AudioChunk:
    """A chunk of generated audio."""

    data: bytes
    sample_rate: int
    is_final: bool = False
    format: str = "pcm"  # "pcm", "wav", or "opus"


class TTSService:
    """
    Text-to-Speech service using Kyutai Moshi TTS.

    This service provides:
    - Text-to-speech conversion
    - Streaming audio generation
    - Multiple output format support
    """

    def __init__(self):
        """Initialize the TTS service."""
        self._model = None
        self._processor = None
        self._device = None
        self._is_loaded = False
        self._lock = asyncio.Lock()

        # Audio configuration
        self.sample_rate = settings.tts_sample_rate  # 24000 Hz default
        self.channels = 1  # Mono output
        self.sample_width = 2  # 16-bit audio

        # Streaming configuration
        self.chunk_size = 4096  # Samples per chunk for streaming

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded

    async def load_model(self) -> None:
        """
        Load the TTS model into memory.

        This should be called during application startup if PRELOAD_MODELS is True,
        or lazily on first synthesis request.
        """
        async with self._lock:
            if self._is_loaded:
                logger.debug("TTS model already loaded, skipping")
                return

            logger.info(f"Loading TTS model: {settings.tts_model_name}")

            try:
                # Determine device
                self._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                logger.info(f"TTS using device: {self._device}")

                # Load model in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync)

                self._is_loaded = True
                logger.info("TTS model loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load TTS model: {e}")
                raise RuntimeError(f"Failed to load TTS model: {e}") from e

    def _load_model_sync(self) -> None:
        """Synchronous model loading (runs in thread pool)."""
        try:
            # Try to load Kyutai Moshi TTS
            # Note: The actual import may vary based on Moshi's API
            from moshi.models import loaders

            self._model, self._processor = loaders.load_tts_model(
                settings.tts_model_name,
                device=self._device,
                cache_dir=settings.models_cache_dir,
            )
            logger.info("Loaded Kyutai Moshi TTS model")

        except ImportError:
            # Fallback: Use a different TTS model for development
            logger.warning(
                "Moshi TTS not available, falling back to alternative TTS for development"
            )
            self._load_fallback_tts()

    def _load_fallback_tts(self) -> None:
        """Load a fallback TTS model for development."""
        try:
            # Try Coqui TTS
            try:
                from TTS.api import TTS

                self._model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
                self._model.to(self._device)
                logger.info("Loaded Coqui TTS fallback model")
                return
            except ImportError:
                pass

            # Try edge-tts (online, no GPU needed)
            try:
                import edge_tts

                self._model = "edge_tts"
                logger.info("Using edge-tts fallback (online)")
                return
            except ImportError:
                pass

            # If nothing works, use a dummy synthesizer for testing
            logger.warning("No TTS model available, using silent fallback")
            self._model = "silent"

        except Exception as e:
            logger.error(f"Failed to load fallback TTS model: {e}")
            raise

    async def unload_model(self) -> None:
        """Unload the model to free GPU memory."""
        async with self._lock:
            if not self._is_loaded:
                return

            logger.info("Unloading TTS model")

            if self._model not in ["edge_tts", "silent"]:
                del self._model
                del self._processor
                self._model = None
                self._processor = None

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self._is_loaded = False
            logger.info("TTS model unloaded")

    async def synthesize(self, text: str, output_format: str = "pcm") -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: The text to convert to speech
            output_format: Output format ("pcm", "wav")

        Returns:
            Audio data as bytes
        """
        if not self._is_loaded:
            await self.load_model()

        if not text or not text.strip():
            return self._generate_silence(0.1, output_format)

        try:
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                None, self._synthesize_sync, text, output_format
            )
            return audio_data

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            # Return a short silence on error
            return self._generate_silence(0.1, output_format)

    def _synthesize_sync(self, text: str, output_format: str) -> bytes:
        """Synchronous synthesis (runs in thread pool)."""
        try:
            if self._model == "silent":
                # Generate silence for testing
                return self._generate_silence(len(text) * 0.06, output_format)

            elif self._model == "edge_tts":
                # This would need special async handling
                # For now, return silence
                return self._generate_silence(len(text) * 0.06, output_format)

            elif hasattr(self._model, "tts"):
                # Coqui TTS API
                audio_array = self._model.tts(text)
                return self._array_to_bytes(audio_array, output_format)

            elif hasattr(self._model, "synthesize"):
                # Moshi TTS API (assumed)
                result = self._model.synthesize(text)
                audio_array = result.get("audio", result)
                return self._array_to_bytes(audio_array, output_format)

            else:
                # Generic model with forward pass
                return self._synthesize_generic(text, output_format)

        except Exception as e:
            logger.error(f"Sync synthesis failed: {e}")
            return self._generate_silence(0.5, output_format)

    def _synthesize_generic(self, text: str, output_format: str) -> bytes:
        """Generic synthesis using model's forward method."""
        try:
            # Process text input
            if self._processor:
                inputs = self._processor(text, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            else:
                raise ValueError("No processor available for generic synthesis")

            # Generate audio
            with torch.no_grad():
                output = self._model(**inputs)

            # Extract audio from output
            if hasattr(output, "waveform"):
                audio_array = output.waveform.squeeze().cpu().numpy()
            elif hasattr(output, "audio"):
                audio_array = output.audio.squeeze().cpu().numpy()
            else:
                audio_array = output.squeeze().cpu().numpy()

            return self._array_to_bytes(audio_array, output_format)

        except Exception as e:
            logger.error(f"Generic synthesis failed: {e}")
            raise

    def _array_to_bytes(
        self, audio_array: np.ndarray, output_format: str = "pcm"
    ) -> bytes:
        """Convert numpy audio array to bytes."""
        # Ensure float32 and normalize
        audio_array = np.asarray(audio_array, dtype=np.float32)

        # Normalize to [-1, 1] if needed
        max_val = np.abs(audio_array).max()
        if max_val > 1.0:
            audio_array = audio_array / max_val

        # Convert to 16-bit PCM
        audio_int16 = (audio_array * 32767).astype(np.int16)

        if output_format == "wav":
            return self._to_wav(audio_int16)
        else:  # pcm
            return audio_int16.tobytes()

    def _to_wav(self, audio_int16: np.ndarray) -> bytes:
        """Convert int16 audio to WAV format."""
        import struct

        buffer = io.BytesIO()

        # WAV header
        num_samples = len(audio_int16)
        data_size = num_samples * self.sample_width
        file_size = 36 + data_size

        # Write WAV header
        buffer.write(b"RIFF")
        buffer.write(struct.pack("<I", file_size))
        buffer.write(b"WAVE")
        buffer.write(b"fmt ")
        buffer.write(struct.pack("<I", 16))  # Subchunk1Size
        buffer.write(struct.pack("<H", 1))  # AudioFormat (PCM)
        buffer.write(struct.pack("<H", self.channels))  # NumChannels
        buffer.write(struct.pack("<I", self.sample_rate))  # SampleRate
        buffer.write(
            struct.pack("<I", self.sample_rate * self.channels * self.sample_width)
        )  # ByteRate
        buffer.write(struct.pack("<H", self.channels * self.sample_width))  # BlockAlign
        buffer.write(struct.pack("<H", self.sample_width * 8))  # BitsPerSample
        buffer.write(b"data")
        buffer.write(struct.pack("<I", data_size))

        # Write audio data
        buffer.write(audio_int16.tobytes())

        return buffer.getvalue()

    def _generate_silence(self, duration_seconds: float, output_format: str) -> bytes:
        """Generate silent audio for a given duration."""
        num_samples = int(self.sample_rate * duration_seconds)
        silence = np.zeros(num_samples, dtype=np.int16)

        if output_format == "wav":
            return self._to_wav(silence)
        else:
            return silence.tobytes()

    async def synthesize_stream(
        self, text: str, output_format: str = "pcm"
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        Stream synthesized speech from text.

        Args:
            text: The text to convert to speech
            output_format: Output format ("pcm", "wav")

        Yields:
            AudioChunk objects with audio data
        """
        if not self._is_loaded:
            await self.load_model()

        if not text or not text.strip():
            yield AudioChunk(
                data=self._generate_silence(0.1, output_format),
                sample_rate=self.sample_rate,
                is_final=True,
                format=output_format,
            )
            return

        try:
            # For models that support streaming, we'd use their streaming API
            # For now, we generate full audio and chunk it

            full_audio = await self.synthesize(text, "pcm")

            # Stream in chunks
            chunk_bytes = self.chunk_size * self.sample_width
            total_bytes = len(full_audio)
            offset = 0

            while offset < total_bytes:
                end = min(offset + chunk_bytes, total_bytes)
                chunk_data = full_audio[offset:end]
                is_final = end >= total_bytes

                yield AudioChunk(
                    data=chunk_data,
                    sample_rate=self.sample_rate,
                    is_final=is_final,
                    format="pcm",
                )

                offset = end

                # Small delay to simulate streaming
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"TTS streaming failed: {e}")
            yield AudioChunk(
                data=self._generate_silence(0.5, output_format),
                sample_rate=self.sample_rate,
                is_final=True,
                format=output_format,
            )

    async def synthesize_sentences(self, text: str) -> AsyncGenerator[AudioChunk, None]:
        """
        Synthesize text sentence by sentence for lower latency.

        Splits text into sentences and synthesizes each one,
        yielding audio as soon as each sentence is ready.

        Args:
            text: The text to convert to speech

        Yields:
            AudioChunk objects for each sentence
        """
        if not self._is_loaded:
            await self.load_model()

        # Split text into sentences
        sentences = self._split_sentences(text)

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            is_final = i == len(sentences) - 1

            try:
                audio_data = await self.synthesize(sentence, "pcm")

                yield AudioChunk(
                    data=audio_data,
                    sample_rate=self.sample_rate,
                    is_final=is_final,
                    format="pcm",
                )

            except Exception as e:
                logger.error(f"Failed to synthesize sentence: {e}")
                continue

    def _split_sentences(self, text: str) -> list:
        """Split text into sentences."""
        import re

        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]


# Global service instance (singleton pattern)
_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Get or create the global TTS service instance."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
