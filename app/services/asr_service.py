"""
ASR (Automatic Speech Recognition) Service using Kyutai STT.

This service handles real-time speech-to-text conversion using the
kyutai/stt-2.6b-en model for the AI-Talk application.

Model: https://huggingface.co/kyutai/stt-2.6b-en
- 2.6B parameters
- English-only
- 2.5 second streaming delay
- Robust to noisy conditions
- Produces transcripts with capitalization and punctuation
"""

import asyncio
import io
import re
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
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None


class ASRService:
    """
    Automatic Speech Recognition service using Kyutai STT.

    This service provides:
    - Audio preprocessing and format conversion
    - Real-time streaming transcription
    - Batch transcription for complete audio files

    Hardware Requirements:
    - ~6-8 GB VRAM for the 2.6B model
    - CUDA 12.1+ recommended
    """

    def __init__(self):
        """Initialize the ASR service."""
        self._model = None
        self._mimi = None  # Mimi audio tokenizer
        self._text_tokenizer = None
        self._device = None
        self._dtype = None
        self._is_loaded = False
        self._lock = asyncio.Lock()
        self._use_transformers = False  # Flag to track which backend is used

        # Audio configuration
        self.sample_rate = settings.asr_sample_rate  # 16000 Hz default
        self._model_sample_rate = (
            None  # Set during model loading (e.g. 24000 for Kyutai)
        )
        self.chunk_duration_ms = settings.audio_chunk_ms
        self.stream_delay_ms = settings.asr_stream_delay_ms  # 2500ms for 2.6b model

        # Model configuration
        self.model_name = settings.asr_model_name
        self.cache_dir = settings.models_cache_dir
        self._prefer_moshi = settings.asr_prefer_moshi
        self._enable_kyutai_transformers = settings.asr_enable_kyutai_transformers
        self._kyutai_attn_implementation = settings.asr_kyutai_attn_implementation
        self._whisper_fallback_model = settings.asr_whisper_fallback_model
        self._silence_rms_threshold = settings.asr_silence_rms_threshold
        self._silence_peak_threshold = settings.asr_silence_peak_threshold
        self._filter_noise_transcripts = settings.asr_filter_noise_transcripts
        self._noise_max_duration_s = settings.asr_noise_max_duration_s
        self._drop_common_false_positives = settings.asr_drop_common_false_positives
        self._common_false_positive_words = {
            w.strip().lower()
            for w in settings.asr_common_false_positive_words.split(",")
            if w.strip()
        }

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

            logger.info(f"Loading ASR model: {self.model_name}")

            try:
                # Determine device and dtype
                self._device = torch.device(
                    settings.asr_device
                    if settings.asr_device
                    else ("cuda" if torch.cuda.is_available() else "cpu")
                )
                self._dtype = getattr(torch, settings.asr_dtype, torch.float16)
                logger.info(f"ASR using device: {self._device}, dtype: {self._dtype}")

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
        model_name_lower = self.model_name.lower()

        # If the configured model is a Whisper model, load it directly
        if "whisper" in model_name_lower:
            logger.info(f"Whisper model configured: {self.model_name}")
            self._load_whisper_fallback(model_name=self.model_name)
            return

        is_kyutai_model = "kyutai" in model_name_lower

        # Kyutai model loading strategy:
        # 1. Moshi by default (more stable in mixed CUDA stacks)
        # 2. Transformers only when explicitly enabled
        # 3. Whisper fallback
        if is_kyutai_model:
            if self._prefer_moshi and self._try_load_moshi():
                return

            if self._enable_kyutai_transformers and self._try_load_transformers():
                return

            if not self._prefer_moshi and self._try_load_moshi():
                return

            if not self._enable_kyutai_transformers:
                logger.warning(
                    "Kyutai transformers backend is disabled "
                    "(ASR_ENABLE_KYUTAI_TRANSFORMERS=false). "
                    "Falling back to Whisper for stability."
                )
            else:
                logger.warning(
                    "Kyutai STT backends failed (Moshi/transformers). "
                    "Falling back to Whisper."
                )

            self._load_whisper_fallback()
            return

        # Non-Kyutai strategy:
        # 1. Transformers
        # 2. Moshi
        # 3. Whisper fallback
        if self._try_load_transformers():
            return

        if self._try_load_moshi():
            return

        logger.warning("Primary ASR models not available, using Whisper fallback")
        self._load_whisper_fallback()

    def _try_load_transformers(self) -> bool:
        """Try loading Kyutai STT with native transformers support."""
        try:
            from transformers import (
                KyutaiSpeechToTextForConditionalGeneration,
                KyutaiSpeechToTextProcessor,
            )
        except ImportError:
            try:
                import transformers

                version = tuple(map(int, transformers.__version__.split(".")[:2]))
                if version < (4, 53):
                    logger.info(
                        f"Transformers {transformers.__version__} < 4.53.0, "
                        f"KyutaiSpeechToText not available"
                    )
                    return False
            except Exception:
                pass
            logger.debug(
                "KyutaiSpeechToText classes not available in this transformers version"
            )
            return False

        try:
            logger.info("Attempting to load Kyutai STT with native transformers...")

            model_name = self.model_name
            trfs_model = (
                model_name.rstrip("/") + "-trfs"
                if not model_name.endswith("-trfs")
                else model_name
            )

            logger.info(f"Loading Kyutai STT processor: {trfs_model}")
            self._processor = KyutaiSpeechToTextProcessor.from_pretrained(
                trfs_model,
                cache_dir=self.cache_dir,
            )

            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

            attn_impl = (self._kyutai_attn_implementation or "eager").strip().lower()
            if attn_impl not in {"eager", "sdpa", "flash_attention_2"}:
                logger.warning(
                    f"Invalid ASR_KYUTAI_ATTN_IMPLEMENTATION='{attn_impl}', using 'eager'"
                )
                attn_impl = "eager"

            load_kwargs = {
                "cache_dir": self.cache_dir,
                "trust_remote_code": settings.trust_remote_code,
                "attn_implementation": attn_impl,
            }
            if self._device and self._device.type == "cuda":
                load_kwargs["device_map"] = "auto"
                load_kwargs["torch_dtype"] = self._dtype
            else:
                load_kwargs["torch_dtype"] = torch.float32

            logger.info(f"Loading Kyutai STT model: {trfs_model}")
            self._model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
                trfs_model,
                **load_kwargs,
            )
            if self._device and self._device.type != "cuda":
                self._model = self._model.to(self._device)

            if hasattr(self._model, "generation_config"):
                cache_impl = getattr(
                    self._model.generation_config, "cache_implementation", None
                )
                if cache_impl == "sliding_window":
                    self._model.generation_config.cache_implementation = "static"
                    logger.info(
                        "ASR generation cache changed: "
                        "sliding_window -> static (stability mode)"
                    )

            self._use_transformers = True
            self._model_sample_rate = 24000
            logger.info(f"Loaded Kyutai STT with transformers: {trfs_model}")
            logger.info(f"  Processor type: {type(self._processor).__name__}")
            logger.info(f"  Model type: {type(self._model).__name__}")
            logger.info(f"  Model device: {self._model.device}")
            logger.info(f"  Model sample rate: {self._model_sample_rate} Hz")
            return True

        except Exception as e:
            logger.debug(f"Failed to load Kyutai STT with transformers: {e}")
            return False

    def _try_load_moshi(self) -> bool:
        """Try loading with Moshi library."""
        try:
            # Import moshi components
            import sentencepiece
            from huggingface_hub import hf_hub_download
            from moshi.models import loaders

            logger.info("Attempting to load Kyutai STT with Moshi library...")

            # Load the STT model using Moshi's loaders
            # The moshi library provides streaming STT capabilities
            self._model = loaders.load_stt(
                self.model_name,
                device=self._device,
                dtype=self._dtype,
            )

            # Load the Mimi audio codec (required for tokenizing audio)
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

            self._use_transformers = False
            logger.info(f"Loaded Kyutai STT with Moshi library: {self.model_name}")
            return True

        except ImportError as e:
            logger.debug(f"Moshi import error: {e}")
            return False
        except Exception as e:
            logger.debug(f"Failed to load with Moshi: {e}")
            return False

    def _load_whisper_fallback(self, model_name: str = None) -> None:
        """Load Whisper as a fallback ASR model."""
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            if model_name is None:
                model_name = self._whisper_fallback_model
            logger.info(f"Loading Whisper model: {model_name}")

            self._processor = WhisperProcessor.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
            )
            self._model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=self._dtype,
                cache_dir=self.cache_dir,
            ).to(self._device)

            self._use_transformers = True
            self._model_sample_rate = 16000
            logger.info("Loaded Whisper fallback model for ASR")

        except Exception as e:
            logger.error(f"Failed to load fallback ASR model: {e}")
            raise RuntimeError(
                "No ASR model could be loaded. Please install moshi or ensure "
                "transformers >= 4.53.0 is available."
            ) from e

    async def unload_model(self) -> None:
        """Unload the model to free GPU memory."""
        async with self._lock:
            if not self._is_loaded:
                return

            logger.info("Unloading ASR model")

            # Clean up all model components
            if self._model is not None:
                del self._model
                self._model = None

            if self._mimi is not None:
                del self._mimi
                self._mimi = None

            if hasattr(self, "_processor") and self._processor is not None:
                del self._processor
                self._processor = None

            if self._text_tokenizer is not None:
                del self._text_tokenizer
                self._text_tokenizer = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._is_loaded = False
            logger.info("ASR model unloaded")

    def _is_encoded_audio(self, audio_bytes: bytes) -> bool:
        """
        Detect if audio bytes are in an encoded container format (Opus/WebM/OGG)
        rather than raw PCM.
        """
        if len(audio_bytes) < 4:
            return False
        # WebM/Matroska magic bytes
        if audio_bytes[:4] == b"\x1a\x45\xdf\xa3":
            return True
        # OGG magic bytes
        if audio_bytes[:4] == b"OggS":
            return True
        # RIFF/WAV
        if audio_bytes[:4] == b"RIFF":
            return True
        # fLaC
        if audio_bytes[:4] == b"fLaC":
            return True
        # Also detect by heuristic: if raw PCM int16 samples have extreme alternating
        # values, it's likely encoded data misread as PCM
        if len(audio_bytes) >= 40:
            test_samples = np.frombuffer(audio_bytes[:40], dtype=np.int16)
            # Check if successive samples have huge jumps (encoded data signature)
            diffs = np.abs(np.diff(test_samples.astype(np.float32)))
            avg_diff = np.mean(diffs)
            if avg_diff > 20000:
                logger.debug(
                    f"Audio heuristic: avg_diff={avg_diff:.0f}, "
                    f"likely encoded (not raw PCM)"
                )
                return True
        return False

    def _decode_with_ffmpeg(self, audio_bytes: bytes) -> np.ndarray:
        """
        Decode encoded audio (Opus/WebM/OGG/etc.) to raw PCM using ffmpeg.

        Returns:
            Numpy array of audio samples normalized to [-1, 1]
        """
        import subprocess

        try:
            # Use ffmpeg to convert any audio format to raw PCM 16-bit mono 16kHz
            process = subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    "pipe:0",  # Read from stdin
                    "-f",
                    "s16le",  # Output raw PCM 16-bit little-endian
                    "-acodec",
                    "pcm_s16le",  # PCM codec
                    "-ac",
                    "1",  # Mono
                    "-ar",
                    str(self.sample_rate),  # Target sample rate
                    "-v",
                    "error",  # Suppress verbose output
                    "pipe:1",  # Write to stdout
                ],
                input=audio_bytes,
                capture_output=True,
                timeout=10,
            )

            if process.returncode != 0:
                stderr = process.stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"ffmpeg failed: {stderr}")

            pcm_bytes = process.stdout
            if len(pcm_bytes) == 0:
                logger.warning("ffmpeg produced empty output")
                return np.array([], dtype=np.float32)

            audio_array = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0

            logger.debug(
                f"ffmpeg decoded: {len(audio_bytes)} encoded bytes -> "
                f"{len(audio_array)} samples ({len(audio_array) / self.sample_rate:.2f}s)"
            )
            return audio_array

        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Install it with: apt-get install ffmpeg"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg timed out decoding audio")

    def preprocess_audio(self, audio_bytes: bytes) -> np.ndarray:
        """
        Preprocess audio bytes into the format expected by the model.

        Supports both raw PCM (16-bit, mono) and encoded formats (Opus/WebM/OGG)
        which are automatically decoded via ffmpeg.

        Args:
            audio_bytes: Audio bytes (raw PCM or encoded format)

        Returns:
            Numpy array of audio samples normalized to [-1, 1]
        """
        if len(audio_bytes) == 0:
            return np.array([], dtype=np.float32)

        # Step 1: Check if this is encoded audio (Opus/WebM/OGG) and decode it
        if self._is_encoded_audio(audio_bytes):
            logger.debug(
                f"Detected encoded audio ({len(audio_bytes)} bytes), "
                f"decoding with ffmpeg..."
            )
            try:
                return self._decode_with_ffmpeg(audio_bytes)
            except Exception as e:
                logger.error(f"ffmpeg decode failed: {e}")
                # Don't fall back to raw PCM — interpreting encoded data as PCM
                # produces garbled noise that Whisper hallucinates on.
                logger.warning(
                    "Encoded audio could not be decoded. "
                    "This chunk may be missing the WebM header (stale buffer)."
                )
                return np.array([], dtype=np.float32)

        # Step 2: Try as raw PCM 16-bit
        try:
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes[:-1]
            if len(audio_bytes) == 0:
                return np.array([], dtype=np.float32)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception:
            pass

        # Step 3: Try using soundfile for other formats (WAV, FLAC, etc.)
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
            # Simple linear interpolation resampling without librosa
            duration = len(audio) / orig_sr
            target_length = int(duration * target_sr)
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, np.arange(len(audio)), audio)

    def _compute_audio_stats(self, audio_array: np.ndarray) -> tuple[float, float, float]:
        """Return duration (s), RMS, and peak for normalized audio."""
        if audio_array is None or len(audio_array) == 0:
            return 0.0, 0.0, 0.0
        duration_sec = len(audio_array) / float(self.sample_rate)
        rms = float(np.sqrt(np.mean(np.square(audio_array, dtype=np.float32))))
        peak = float(np.max(np.abs(audio_array)))
        return duration_sec, rms, peak

    def _is_low_energy_audio(self, audio_array: np.ndarray) -> bool:
        """Detect likely non-speech chunks based on decoded waveform energy."""
        _, rms, peak = self._compute_audio_stats(audio_array)
        return rms < self._silence_rms_threshold and peak < self._silence_peak_threshold

    def _is_low_information_text(self, text: str) -> bool:
        """Detect common short hallucination patterns from low-energy audio."""
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        normalized = re.sub(r"[^a-z0-9' ]", "", normalized).strip()
        if not normalized:
            return True

        words = normalized.split()
        if not words:
            return True

        if len(words) == 1 and words[0] in {"you", "yeah", "uh", "um", "hmm", "mm"}:
            return True

        if len(words) <= 3 and len(set(words)) == 1:
            return True

        return False

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

            duration_sec, rms, peak = self._compute_audio_stats(audio_array)
            low_energy = self._is_low_energy_audio(audio_array)
            if low_energy:
                logger.debug(
                    f"Skipping transcription: low-energy audio "
                    f"(duration={duration_sec:.2f}s, rms={rms:.5f}, peak={peak:.5f})"
                )
                return TranscriptionResult(
                    text="", is_final=True, confidence=0.0, language=language
                )

            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None, self._transcribe_sync, audio_array, language
            )

            # Hard filter for common standalone false positives (e.g. repeated "you").
            if self._drop_common_false_positives and duration_sec <= self._noise_max_duration_s:
                normalized = re.sub(r"\s+", " ", text.strip().lower())
                normalized = re.sub(r"[^a-z0-9' ]", "", normalized).strip()
                if normalized in self._common_false_positive_words:
                    logger.info(
                        f"Dropping common false-positive transcript: '{text.strip()}' "
                        f"(duration={duration_sec:.2f}s)"
                    )
                    text = ""

            # Filter typical short hallucinations from weak/short chunks (e.g. "you")
            if (
                self._filter_noise_transcripts
                and duration_sec <= self._noise_max_duration_s
                and rms < (self._silence_rms_threshold * 2.0)
                and self._is_low_information_text(text)
            ):
                logger.info(
                    f"Dropping likely noise transcript: '{text.strip()}' "
                    f"(duration={duration_sec:.2f}s, rms={rms:.5f}, peak={peak:.5f})"
                )
                text = ""

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
            if self._use_transformers:
                return self._transcribe_transformers(audio_array, language)
            else:
                return self._transcribe_moshi(audio_array)
        except Exception as e:
            logger.error(f"Sync transcription failed: {type(e).__name__}: {e!r}")
            import traceback

            logger.error(f"Sync transcription traceback:\n{traceback.format_exc()}")
            return ""

    def _transcribe_moshi(self, audio_array: np.ndarray) -> str:
        """Transcribe using Moshi STT model."""
        try:
            # Convert audio to tensor
            audio_tensor = (
                torch.from_numpy(audio_array)
                .unsqueeze(0)
                .to(device=self._device, dtype=self._dtype)
            )

            # Tokenize audio using Mimi
            with torch.no_grad():
                # Mimi expects audio at 24kHz, resample if needed
                if self.sample_rate != 24000:
                    import torchaudio.functional as F

                    audio_tensor = F.resample(audio_tensor, self.sample_rate, 24000)

                # Encode audio to tokens
                audio_tokens = self._mimi.encode(audio_tensor)

                # Run STT model
                text_tokens = self._model.generate(audio_tokens)

                # Decode text tokens
                text = self._text_tokenizer.decode(text_tokens.cpu().tolist())

            return text.strip()

        except Exception as e:
            logger.error(f"Moshi transcription failed: {e}")
            raise

    def _get_model_input_device(self) -> torch.device:
        """Resolve a stable device for model inputs."""
        hf_device_map = getattr(self._model, "hf_device_map", None)
        if isinstance(hf_device_map, dict) and hf_device_map:
            for mapped in hf_device_map.values():
                if mapped in {"cpu", "meta", "disk"}:
                    continue
                if isinstance(mapped, int):
                    return torch.device(f"cuda:{mapped}")
                if isinstance(mapped, str) and mapped.startswith("cuda"):
                    return torch.device(mapped)

        try:
            return next(self._model.parameters()).device
        except Exception:
            return self._device if self._device is not None else torch.device("cpu")

    def _transcribe_transformers(self, audio_array: np.ndarray, language: str) -> str:
        """Transcribe using transformers model (Kyutai STT or Whisper).

        For Kyutai STT, follows the official HuggingFace docs:
        https://huggingface.co/docs/transformers/en/model_doc/kyutai_speech_to_text
        """
        try:
            is_kyutai = "kyutai" in str(type(self._model)).lower()
            is_whisper = "whisper" in str(type(self._model)).lower()

            # --- Kyutai STT path (official docs) ---
            if is_kyutai:
                # Resample to 24kHz if needed (Kyutai expects 24kHz)
                model_sr = self._model_sample_rate or 24000
                if self.sample_rate != model_sr:
                    audio_array = self._resample(
                        audio_array, self.sample_rate, model_sr
                    )
                    logger.debug(
                        f"Resampled audio from {self.sample_rate}Hz to {model_sr}Hz, "
                        f"new length: {len(audio_array)} samples"
                    )

                # Processor call per official docs: pass the audio array.
                inputs = self._processor(audio=audio_array)

                logger.debug(f"Kyutai processor output keys: {list(inputs.keys())}")

                # For sharded models, use the first CUDA shard as input device.
                input_device = self._get_model_input_device()
                inputs = inputs.to(input_device)

                # Generate transcription
                generate_kwargs = {}
                cache_impl = None
                if hasattr(self._model, "generation_config"):
                    cache_impl = getattr(
                        self._model.generation_config, "cache_implementation", None
                    )
                if cache_impl in {None, "sliding_window"}:
                    generate_kwargs["cache_implementation"] = "static"

                with torch.no_grad():
                    output_tokens = self._model.generate(**inputs, **generate_kwargs)

                # Decode tokens to text
                transcription = self._processor.batch_decode(
                    output_tokens, skip_special_tokens=True
                )
                return transcription[0] if transcription else ""

            # --- Whisper path ---
            elif is_whisper:
                inputs = self._processor(
                    audio_array,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                )

                logger.debug(f"Whisper processor output keys: {list(inputs.keys())}")

                input_features = inputs.input_features.to(
                    device=self._device, dtype=self._dtype
                )

                whisper_generate_kwargs = {
                    "language": language,
                    "task": "transcribe",
                    "temperature": 0.0,
                    "condition_on_prev_tokens": False,
                }

                with torch.no_grad():
                    try:
                        predicted_ids = self._model.generate(
                            input_features,
                            **whisper_generate_kwargs,
                        )
                    except TypeError:
                        # Keep compatibility with older transformers kwargs.
                        predicted_ids = self._model.generate(
                            input_features,
                            language=language,
                            task="transcribe",
                        )

                transcription = self._processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )
                return transcription[0] if transcription else ""

            # --- Generic fallback path ---
            else:
                inputs = self._processor(
                    audio_array,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                )

                available_keys = list(inputs.keys())
                logger.debug(f"Generic processor output keys: {available_keys}")

                # Find the input tensor
                input_tensor = None
                input_key = None
                for key in ["input_features", "input_values", "audio_values"]:
                    if key in inputs:
                        input_tensor = inputs[key].to(
                            device=self._device, dtype=self._dtype
                        )
                        input_key = key
                        break

                if input_tensor is None:
                    raise ValueError(
                        f"Processor returned unexpected keys: {available_keys}. "
                        "Expected 'input_features' or 'input_values'."
                    )

                logger.debug(
                    f"Using input key '{input_key}', tensor shape: {input_tensor.shape}"
                )

                with torch.no_grad():
                    if hasattr(self._model, "generate"):
                        predicted_ids = self._model.generate(
                            **{input_key: input_tensor}
                        )
                    else:
                        outputs = self._model(input_tensor)
                        predicted_ids = outputs.logits.argmax(dim=-1)

                transcription = self._processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )
                return transcription[0] if transcription else ""

        except Exception as e:
            logger.error(
                f"Transformers transcription failed: {type(e).__name__}: {e!r}"
            )
            import traceback

            logger.error(f"Transformers traceback:\n{traceback.format_exc()}")
            raise

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
        # Chunk size based on configured duration
        chunk_size = int(
            self.sample_rate * (self.chunk_duration_ms / 1000) * 2
        )  # 2 bytes per sample (16-bit)

        # Accumulator for continuous context
        accumulated_audio = b""
        last_transcript = ""

        async for audio_chunk in audio_stream:
            audio_buffer += audio_chunk
            accumulated_audio += audio_chunk

            # Process when we have enough audio
            if len(audio_buffer) >= chunk_size:
                # Transcribe accumulated audio for better context
                result = await self.transcribe(accumulated_audio, language)

                # Only yield if we have new content
                if result.text and result.text != last_transcript:
                    # Mark as partial (not final) during streaming
                    result.is_final = False
                    last_transcript = result.text
                    yield result

                audio_buffer = b""

        # Process remaining audio as final
        if accumulated_audio:
            result = await self.transcribe(accumulated_audio, language)
            result.is_final = True
            yield result

    async def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "is_loaded": self._is_loaded,
            "device": str(self._device) if self._device else None,
            "dtype": str(self._dtype) if self._dtype else None,
            "backend": "transformers" if self._use_transformers else "moshi",
            "sample_rate": self.sample_rate,
            "stream_delay_ms": self.stream_delay_ms,
        }


# Global service instance (singleton pattern)
_asr_service: Optional[ASRService] = None


def get_asr_service() -> ASRService:
    """Get or create the global ASR service instance."""
    global _asr_service
    if _asr_service is None:
        _asr_service = ASRService()
    return _asr_service
