"""
LLM (Large Language Model) Service using vLLM with Qwen2.5.

This service handles conversation generation for the AI-Talk English tutor.
"""

import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Optional

from loguru import logger

from app.config import settings
from app.prompts import UserLevel, get_tutor_system_prompt


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class ConversationContext:
    """Context for an ongoing conversation."""

    messages: List[Message] = field(default_factory=list)
    user_level: UserLevel = UserLevel.INTERMEDIATE
    system_prompt: str = ""

    def __post_init__(self):
        if not self.system_prompt:
            self.system_prompt = get_tutor_system_prompt(self.user_level)

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self.messages.append(Message(role="assistant", content=content))

    def get_user_utterances(self) -> List[str]:
        """Get all user messages for analysis."""
        return [msg.content for msg in self.messages if msg.role == "user"]

    def format_for_llm(self) -> List[dict]:
        """Format conversation for LLM input."""
        formatted = [{"role": "system", "content": self.system_prompt}]
        for msg in self.messages:
            formatted.append({"role": msg.role, "content": msg.content})
        return formatted

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()


class LLMService:
    """
    LLM Service for conversation generation using vLLM.

    This service provides:
    - Fast inference with vLLM
    - Streaming token generation
    - Conversation context management
    - English tutoring capabilities
    """

    def __init__(self):
        """Initialize the LLM service."""
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._lock = asyncio.Lock()

        # Model configuration
        self.model_name = settings.llm_model_name
        self.max_model_len = settings.llm_max_model_len
        self.gpu_memory_utilization = settings.llm_gpu_memory_utilization

        # Generation defaults
        self.default_max_tokens = 256
        self.default_temperature = 0.7
        self.default_top_p = 0.9

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded

    async def load_model(self) -> None:
        """
        Load the LLM model into memory using vLLM.

        This should be called during application startup if PRELOAD_MODELS is True,
        or lazily on first generation request.
        """
        async with self._lock:
            if self._is_loaded:
                logger.debug("LLM model already loaded, skipping")
                return

            logger.info(f"Loading LLM model: {self.model_name}")

            try:
                # Load model in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync)

                self._is_loaded = True
                logger.info("LLM model loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load LLM model: {e}")
                raise RuntimeError(f"Failed to load LLM model: {e}") from e

    def _load_model_sync(self) -> None:
        """Synchronous model loading (runs in thread pool)."""
        try:
            from vllm import LLM

            self._model = LLM(
                model=self.model_name,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                trust_remote_code=True,
                download_dir=settings.models_cache_dir,
            )
            logger.info(f"vLLM loaded with model: {self.model_name}")

        except ImportError as e:
            logger.warning(f"vLLM not available: {e}, falling back to transformers")
            self._load_transformers_fallback()

    def _load_transformers_fallback(self) -> None:
        """Load model using transformers as fallback (for development without vLLM)."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("Loaded model with transformers fallback")

        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            raise

    async def unload_model(self) -> None:
        """Unload the model to free GPU memory."""
        async with self._lock:
            if not self._is_loaded:
                return

            logger.info("Unloading LLM model")

            import torch

            del self._model
            self._model = None

            if self._tokenizer:
                del self._tokenizer
                self._tokenizer = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._is_loaded = False
            logger.info("LLM model unloaded")

    async def generate(
        self,
        context: ConversationContext,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Generate a complete response for the conversation.

        Args:
            context: The conversation context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            The generated response text
        """
        if not self._is_loaded:
            await self.load_model()

        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature or self.default_temperature
        top_p = top_p or self.default_top_p

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._generate_sync,
                context,
                max_tokens,
                temperature,
                top_p,
            )
            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "I'm sorry, I had trouble generating a response. Could you please try again?"

    def _generate_sync(
        self,
        context: ConversationContext,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Synchronous generation (runs in thread pool)."""
        messages = context.format_for_llm()

        try:
            # Try vLLM generation
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["User:", "\n\nUser", "<|im_end|>", "<|endoftext|>"],
            )

            # Format as chat template
            prompt = self._format_chat_prompt(messages)

            outputs = self._model.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()

            return generated_text

        except (ImportError, AttributeError):
            # Fallback to transformers
            return self._generate_transformers(messages, max_tokens, temperature, top_p)

    def _format_chat_prompt(self, messages: List[dict]) -> str:
        """Format messages into a chat prompt string."""
        # Qwen2.5 chat template format
        prompt_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        # Add assistant prefix for generation
        prompt_parts.append("<|im_start|>assistant\n")

        return "\n".join(prompt_parts)

    def _generate_transformers(
        self,
        messages: List[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Generate using transformers (fallback)."""
        import torch

        # Use chat template if available
        if hasattr(self._tokenizer, "apply_chat_template"):
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self._model.device)
        else:
            prompt = self._format_chat_prompt(messages)
            input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(
                self._model.device
            )

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0][input_ids.shape[1] :]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()

    async def generate_stream(
        self,
        context: ConversationContext,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response for the conversation.

        Args:
            context: The conversation context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Yields:
            Generated tokens one at a time
        """
        if not self._is_loaded:
            await self.load_model()

        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature or self.default_temperature
        top_p = top_p or self.default_top_p

        try:
            # Check if we have vLLM with streaming support
            try:
                from vllm import SamplingParams

                sampling_params = SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=["User:", "\n\nUser", "<|im_end|>", "<|endoftext|>"],
                )

                messages = context.format_for_llm()
                prompt = self._format_chat_prompt(messages)

                # Use vLLM streaming if available
                async for token in self._stream_vllm(prompt, sampling_params):
                    yield token

            except (ImportError, AttributeError):
                # Fallback to non-streaming generation
                response = await self.generate(context, max_tokens, temperature, top_p)
                # Simulate streaming by yielding chunks
                words = response.split()
                for word in words:
                    yield word + " "
                    await asyncio.sleep(0.02)  # Small delay to simulate streaming

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield "I'm sorry, I had trouble generating a response."

    async def _stream_vllm(
        self, prompt: str, sampling_params
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from vLLM."""
        # vLLM streaming implementation
        # Note: vLLM's streaming API may vary by version
        loop = asyncio.get_event_loop()

        # For vLLM, we need to use the async engine for true streaming
        # This is a simplified version that chunks the output
        try:
            outputs = await loop.run_in_executor(
                None,
                lambda: self._model.generate([prompt], sampling_params),
            )

            generated_text = outputs[0].outputs[0].text

            # Yield token by token (simulated from complete output)
            # For true streaming, you'd use vLLM's AsyncLLMEngine
            buffer = ""
            for char in generated_text:
                buffer += char
                if char in " \n" or len(buffer) > 4:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)

            if buffer:
                yield buffer

        except Exception as e:
            logger.error(f"vLLM streaming error: {e}")
            raise

    def create_context(
        self,
        user_level: UserLevel = UserLevel.INTERMEDIATE,
    ) -> ConversationContext:
        """
        Create a new conversation context.

        Args:
            user_level: The user's English proficiency level

        Returns:
            A new ConversationContext instance
        """
        return ConversationContext(user_level=user_level)


# Global service instance (singleton pattern)
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
