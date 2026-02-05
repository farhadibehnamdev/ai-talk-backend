"""
TTS (Text-to-Speech) Service using Kyutai TTS.

This service handles text-to-speech conversion using the
kyutai/tts-1.6b-en_fr model for the AI-Talk application.

Model: https://huggingface.co/kyutai/tts-1.6b-en_fr
- 1.8B parameters (despite the 1.6b name)
- Supports English and French
- Streaming text-to-speech generation
- Voice conditioning through pre-computed embeddings
- CFG distillation for improved speed (no need to double batch size)
- 1.28 second audio shift w.r.t. text

Hardware Requirements:
- ~4-5 GB VRAM
- CUDA 12.1+ recommended
"""

import asyncio
import io
import struct
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional

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
    duration_ms: float = 0.0


@dataclass
class VoiceConfig:
    """Configuration for voice synthesis."""

    voice_id: str = "default"
    speed: float = 1.0
    pitch: float = 1.0
    language: str = "en"


class TTSService:
    """
    Text-to-Speech service using Kyutai TTS.

    This service provides:
    - Text-to-speech conversion
    - Streaming audio generation
    - Multiple voice support via pre-computed embeddings
    - Multiple output format support (PCM, WAV)

    Hardware Requirements:
    - ~4-5 GB VRAM for the 1.8B model
    - CUDA 12.1+ recommended
    """

    def __init__(self):
        """Initialize the TTS service."""
        self._model = None
        self._mimi = None  # Mimi audio codec
        self._text_tokenizer = None
        self._voice_embeddings: Dict[str, torch.Tensor] = {}
        self._device = None
        self._dtype = None
        self._is_loaded = False
        self._lock = asyncio.Lock()
        self._use_moshi = False  # Flag to track which backend is used

        # Audio configuration
        self.sample_rate = settings.tts_sample_rate  # 24000 Hz default
        self.channels = 1  # Mono output
        self.sample_width = 2  # 16-bit audio

        # Model configuration
        self.model_name = settings.tts_model_name
        self.cache_dir = settings.models_cache_dir
        self.default_voice = settings.tts_voice
        self.audio_shift_ms = settings.tts_audio_shift_ms  # 1280ms

        # Streaming configuration
        self.chunk_size = 4096  # Samples per chunk for streaming
        self.frame_rate = 12.5  # Hz (model's frame rate)

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

            logger.info(f"Loading TTS model: {self.model_name}")

            try:
                # Determine device and dtype
                self._device = torch.device(
                    settings.tts_device
                    if settings.tts_device
                    else ("cuda" if torch.cuda.is_available() else "cpu")
                )
                self._dtype = getattr(torch, settings.tts_dtype, torch.float16)
                logger.info(f"TTS using device: {self._device}, dtype: {self._dtype}")

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
        # Try multiple loading strategies in order of preference

        # Strategy 1: Try Moshi library (primary)
        if self._try_load_moshi():
            return

        # Strategy 2: Try native loading with transformers
        if self._try_load_transformers():
            return

        # Strategy 3: Fallback for development
        logger.warning("Primary TTS models not available, using fallback")
        self._load_fallback_tts()

    def _try_load_moshi(self) -> bool:
        """Try loading with Moshi library."""
        try:
            import sentencepiece
            from huggingface_hub import hf_hub_download
            from moshi.models import loaders

            logger.info("Attempting to load Kyutai TTS with Moshi library...")

            # Load the TTS model
            self._model = loaders.load_tts(
                self.model_name,
                device=self._device,
                dtype=self._dtype,
            )

            # Load the Mimi audio codec (required for decoding audio tokens)
            self._mimi = loaders.load_mimi(
                device=self._device,
                dtype=self._dtype,
            )

            # Load text tokenizer
            tokenizer_path = hf_hub_download(
                self.model_name,
                "tokenizer.model",
                cache_dir=self.cache_dir,
            )
            self._text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

            # Try to load voice embeddings
            self._load_voice_embeddings()

            self._use_moshi = True
            logger.info(f"Loaded Kyutai TTS with Moshi library: {self.model_name}")
            return True

        except ImportError as e:
            logger.debug(f"Moshi import error: {e}")
            return False
        except Exception as e:
            logger.debug(f"Failed to load with Moshi: {e}")
            return False

    def _try_load_transformers(self) -> bool:
        """Try loading with transformers library."""
        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info("Attempting to load Kyutai TTS with transformers...")

            self._text_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=settings.trust_remote_code,
            )

            self._model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=self._dtype,
                cache_dir=self.cache_dir,
                trust_remote_code=settings.trust_remote_code,
            ).to(self._device)

            self._use_moshi = False
            logger.info(f"Loaded Kyutai TTS with transformers: {self.model_name}")
            return True

        except ImportError as e:
            logger.debug(f"Transformers import error: {e}")
            return False
        except Exception as e:
            logger.debug(f"Failed to load with transformers: {e}")
            return False

    def _load_voice_embeddings(self) -> None:
        """Load pre-computed voice embeddings from kyutai/tts-voices repository."""
        try:
            from huggingface_hub import hf_hub_download, list_repo_files

            voices_repo = "kyutai/tts-voices"

            # List available voice files
            try:
                voice_files = [
                    f for f in list_repo_files(voices_repo) if f.endswith(".pt")
                ]
                logger.info(f"Found {len(voice_files)} voice embeddings")

                # Load a few default voices
                for voice_file in voice_files[:5]:  # Load first 5 voices
                    try:
                        voice_path = hf_hub_download(
                            voices_repo,
                            voice_file,
                            cache_dir=self.cache_dir,
                        )
                        voice_name = voice_file.replace(".pt", "").replace("/", "_")
                        self._voice_embeddings[voice_name] = torch.load(
                            voice_path, map_location=self._device
                        )
                        logger.debug(f"Loaded voice: {voice_name}")
                    except Exception as e:
                        logger.debug(f"Failed to load voice {voice_file}: {e}")

            except Exception as e:
                logger.debug(f"Could not list voice files: {e}")

        except ImportError:
            logger.debug("huggingface_hub not available for voice loading")
        except Exception as e:
            logger.debug(f"Failed to load voice embeddings: {e}")

    def _load_fallback_tts(self) -> None:
        """Load a fallback TTS solution for development."""
        try:
            # Try Coqui TTS
            try:
                from TTS.api import TTS

                self._model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
                if self._device.type == "cuda":
                    self._model.to(self._device)
                self._use_moshi = False
                logger.info("Loaded Coqui TTS fallback model")
                return
            except ImportError:
                logger.debug("Coqui TTS not available")

            # Try edge-tts (online, no GPU needed)
            try:
                import edge_tts

                self._model = "edge_tts"
                self._use_moshi = False
                logger.info("Using edge-tts fallback (online)")
                return
            except ImportError:
                logger.debug("edge-tts not available")

            # If nothing works, use a silent synthesizer for testing
            logger.warning("No TTS model available, using silent fallback")
            self._model = "silent"
            self._use_moshi = False

        except Exception as e:
            logger.error(f"Failed to load fallback TTS: {e}")
            raise RuntimeError(
                "No TTS model could be loaded. Please install moshi or another TTS library."
            ) from e

    async def unload_model(self) -> None:
        """Unload the model to free GPU memory."""
        async with self._lock:
            if not self._is_loaded:
                return

            logger.info("Unloading TTS model")

            # Clean up model components
            if self._model not in ["edge_tts", "silent"] and self._model is not None:
                del self._model
                self._model = None

            if self._mimi is not None:
                del self._mimi
                self._mimi = None

            if self._text_tokenizer is not None:
                del self._text_tokenizer
                self._text_tokenizer = None

            # Clear voice embeddings
            self._voice_embeddings.clear()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._is_loaded = False
            logger.info("TTS model unloaded")

    def get_available_voices(self) -> List[str]:
        """Get list of available voice IDs."""
        voices = list(self._voice_embeddings.keys())
        if not voices:
            voices = ["default"]
        return voices

    async def synthesize(
        self,
        text: str,
        voice_config: Optional[VoiceConfig] = None,
        output_format: str = "pcm",
    ) -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: The text to convert to speech
            voice_config: Optional voice configuration
            output_format: Output format ("pcm", "wav")

        Returns:
            Audio data as bytes
        """
        if not self._is_loaded:
            await self.load_model()

        if not text or not text.strip():
            return self._generate_silence(0.1, output_format)

        voice_config = voice_config or VoiceConfig()

        try:
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                None, self._synthesize_sync, text, voice_config, output_format
            )
            return audio_data

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            # Return a short silence on error
            return self._generate_silence(0.1, output_format)

    def _synthesize_sync(
        self,
        text: str,
        voice_config: VoiceConfig,
        output_format: str,
    ) -> bytes:
        """Synchronous synthesis (runs in thread pool)."""
        try:
            if self._model == "silent":
                # Generate silence for testing
                duration = len(text) * 0.06  # ~60ms per character
                return self._generate_silence(duration, output_format)

            elif self._model == "edge_tts":
                # Edge TTS requires async, generate silence as placeholder
                duration = len(text) * 0.06
                return self._generate_silence(duration, output_format)

            elif self._use_moshi:
                # Kyutai TTS with Moshi library
                return self._synthesize_moshi(text, voice_config, output_format)

            elif hasattr(self._model, "tts"):
                # Coqui TTS API
                audio_array = self._model.tts(text)
                return self._array_to_bytes(np.array(audio_array), output_format)

            else:
                # Generic transformers-based synthesis
                return self._synthesize_transformers(text, voice_config, output_format)

        except Exception as e:
            logger.error(f"Sync synthesis failed: {e}")
            return self._generate_silence(0.5, output_format)

    def _synthesize_moshi(
        self,
        text: str,
        voice_config: VoiceConfig,
        output_format: str,
    ) -> bytes:
        """Synthesize using Kyutai TTS with Moshi library."""
        try:
            # Tokenize text
            text_tokens = self._text_tokenizer.encode(text)
            text_tensor = torch.tensor(
                [text_tokens], dtype=torch.long, device=self._device
            )

            # Get voice embedding if available
            voice_embedding = None
            if voice_config.voice_id in self._voice_embeddings:
                voice_embedding = self._voice_embeddings[voice_config.voice_id]
            elif "default" in self._voice_embeddings:
                voice_embedding = self._voice_embeddings["default"]

            with torch.no_grad():
                # Generate audio tokens
                if voice_embedding is not None:
                    audio_tokens = self._model.generate(
                        text_tensor,
                        voice_embedding=voice_embedding,
                    )
                else:
                    audio_tokens = self._model.generate(text_tensor)

                # Decode audio tokens to waveform using Mimi
                audio_waveform = self._mimi.decode(audio_tokens)

                # Convert to numpy
                audio_array = audio_waveform.squeeze().cpu().numpy()

            return self._array_to_bytes(audio_array, output_format)

        except Exception as e:
            logger.error(f"Moshi TTS synthesis failed: {e}")
            raise

    def _synthesize_transformers(
        self,
        text: str,
        voice_config: VoiceConfig,
        output_format: str,
    ) -> bytes:
        """Synthesize using transformers model."""
        try:
            # Tokenize input
            if hasattr(self._text_tokenizer, "encode"):
                inputs = self._text_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                ).to(self._device)
            else:
                raise ValueError("Tokenizer not properly initialized")

            with torch.no_grad():
                # Generate
                if hasattr(self._model, "generate"):
                    outputs = self._model.generate(**inputs)
                else:
                    outputs = self._model(**inputs)

                # Extract audio
                if hasattr(outputs, "waveform"):
                    audio_array = outputs.waveform.squeeze().cpu().numpy()
                elif hasattr(outputs, "audio"):
                    audio_array = outputs.audio.squeeze().cpu().numpy()
                elif isinstance(outputs, torch.Tensor):
                    audio_array = outputs.squeeze().cpu().numpy()
                else:
                    raise ValueError(f"Unknown output format: {type(outputs)}")

            return self._array_to_bytes(audio_array, output_format)

        except Exception as e:
            logger.error(f"Transformers TTS synthesis failed: {e}")
            raise

    def _array_to_bytes(
        self,
        audio_array: np.ndarray,
        output_format: str = "pcm",
    ) -> bytes:
        """Convert numpy audio array to bytes."""
        # Ensure float32 and normalize
        audio_array = np.asarray(audio_array, dtype=np.float32)

        # Normalize to [-1, 1] if needed
        max_val = np.abs(audio_array).max()
        if max_val > 1.0:
            audio_array = audio_array / max_val

        # Apply any additional normalization if enabled
        if settings.audio_normalize:
            # RMS normalization to -3dB
            rms = np.sqrt(np.mean(audio_array**2))
            if rms > 0:
                target_rms = 0.707  # -3dB
                audio_array = audio_array * (target_rms / rms)
                # Clip to prevent clipping
                audio_array = np.clip(audio_array, -1.0, 1.0)

        # Convert to 16-bit PCM
        audio_int16 = (audio_array * 32767).astype(np.int16)

        if output_format == "wav":
            return self._to_wav(audio_int16)
        else:  # pcm
            return audio_int16.tobytes()

    def _to_wav(self, audio_int16: np.ndarray) -> bytes:
        """Convert int16 audio to WAV format."""
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
        self,
        text: str,
        voice_config: Optional[VoiceConfig] = None,
        output_format: str = "pcm",
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        Stream synthesized speech from text.

        Args:
            text: The text to convert to speech
            voice_config: Optional voice configuration
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
                duration_ms=100.0,
            )
            return

        try:
            # For true streaming with Moshi, we'd use the streaming API
            # For now, generate full audio and stream in chunks
            full_audio = await self.synthesize(text, voice_config, "pcm")

            # Calculate chunk parameters
            chunk_bytes = self.chunk_size * self.sample_width
            total_bytes = len(full_audio)
            chunk_duration_ms = (self.chunk_size / self.sample_rate) * 1000
            offset = 0

            while offset < total_bytes:
                end = min(offset + chunk_bytes, total_bytes)
                chunk_data = full_audio[offset:end]
                is_final = end >= total_bytes

                actual_samples = len(chunk_data) // self.sample_width
                actual_duration_ms = (actual_samples / self.sample_rate) * 1000

                yield AudioChunk(
                    data=chunk_data,
                    sample_rate=self.sample_rate,
                    is_final=is_final,
                    format="pcm",
                    duration_ms=actual_duration_ms,
                )

                offset = end

                # Small delay to simulate real-time streaming
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"TTS streaming failed: {e}")
            yield AudioChunk(
                data=self._generate_silence(0.5, output_format),
                sample_rate=self.sample_rate,
                is_final=True,
                format=output_format,
                duration_ms=500.0,
            )

    async def synthesize_sentences(
        self,
        text: str,
        voice_config: Optional[VoiceConfig] = None,
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        Synthesize text sentence by sentence for lower latency.

        Splits text into sentences and synthesizes each one,
        yielding audio as soon as each sentence is ready.

        Args:
            text: The text to convert to speech
            voice_config: Optional voice configuration

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
                audio_data = await self.synthesize(sentence, voice_config, "pcm")

                # Calculate duration
                num_samples = len(audio_data) // self.sample_width
                duration_ms = (num_samples / self.sample_rate) * 1000

                yield AudioChunk(
                    data=audio_data,
                    sample_rate=self.sample_rate,
                    is_final=is_final,
                    format="pcm",
                    duration_ms=duration_ms,
                )

            except Exception as e:
                logger.error(f"Failed to synthesize sentence: {e}")
                continue

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences for incremental synthesis."""
        import re

        # Split on sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Also split on long pauses (commas followed by long segments)
        result = []
        for sentence in sentences:
            # If sentence is very long, split on commas too
            if len(sentence) > 150:
                parts = re.split(r"(?<=,)\s+", sentence)
                result.extend(parts)
            else:
                result.append(sentence)

        return [s.strip() for s in result if s.strip()]

    async def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "is_loaded": self._is_loaded,
            "device": str(self._device) if self._device else None,
            "dtype": str(self._dtype) if self._dtype else None,
            "backend": "moshi" if self._use_moshi else "other",
            "sample_rate": self.sample_rate,
            "available_voices": self.get_available_voices(),
            "audio_shift_ms": self.audio_shift_ms,
        }


# Global service instance (singleton pattern)
_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Get or create the global TTS service instance."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
