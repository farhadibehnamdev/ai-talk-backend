"""
AI-Talk Backend - Main Application Entry Point

This is the main FastAPI application that serves the AI-Talk English learning
conversation system. It provides:
- WebSocket endpoint for real-time audio conversation
- Health check endpoints
- Model lifecycle management (loading on startup)
- GPU memory validation and monitoring

Hardware Requirements:
- NVIDIA GPU with 24GB+ VRAM (RTX 4090, A6000, etc.)
- CUDA 12.1+
- Python 3.11+

Models:
- STT: kyutai/stt-2.6b-en (~6-8 GB VRAM)
- TTS: kyutai/tts-1.6b-en_fr (~4-5 GB VRAM)
- LLM: Qwen/Qwen2.5-7B-Instruct-AWQ (~4-5 GB VRAM)
"""

import asyncio
import sys
from contextlib import asynccontextmanager
from typing import Dict

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api import websocket_router
from app.config import settings
from app.services.asr_service import get_asr_service
from app.services.llm_service import get_llm_service
from app.services.tts_service import get_tts_service


def validate_gpu() -> Dict:
    """
    Validate GPU availability and memory for running all models.

    Returns:
        Dictionary with GPU validation results
    """
    result = {
        "cuda_available": False,
        "gpu_name": None,
        "total_memory_gb": 0,
        "free_memory_gb": 0,
        "required_memory_gb": settings.total_model_vram_estimate_gb,
        "sufficient_memory": False,
        "warnings": [],
    }

    if not torch.cuda.is_available():
        result["warnings"].append(
            "CUDA not available - models will run on CPU (very slow)"
        )
        return result

    result["cuda_available"] = True
    result["gpu_name"] = torch.cuda.get_device_name(0)

    # Get memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_memory = (
        torch.cuda.get_device_properties(0).total_memory
        - torch.cuda.memory_allocated(0)
    ) / (1024**3)

    result["total_memory_gb"] = round(total_memory, 2)
    result["free_memory_gb"] = round(free_memory, 2)

    # Check if we have enough memory
    required = result["required_memory_gb"]
    available = total_memory - settings.gpu_memory_reserved_gb

    result["sufficient_memory"] = available >= required

    if not result["sufficient_memory"]:
        result["warnings"].append(
            f"GPU has {total_memory:.1f}GB but models require ~{required:.1f}GB. "
            f"Consider using smaller models or a larger GPU."
        )

    # Check CUDA version
    cuda_version = torch.version.cuda
    if cuda_version:
        cuda_major = int(cuda_version.split(".")[0])
        if cuda_major < 12:
            result["warnings"].append(
                f"CUDA {cuda_version} detected. CUDA 12.1+ is recommended for best performance."
            )

    return result


async def load_models_sequential():
    """
    Load models sequentially to manage GPU memory better.

    This is more memory-efficient than parallel loading as it allows
    each model to allocate memory before the next one starts.
    """
    asr_service = get_asr_service()
    llm_service = get_llm_service()
    tts_service = get_tts_service()

    # Load LLM first (uses vLLM with gpu_memory_utilization)
    logger.info("Loading LLM model (1/3)...")
    try:
        await llm_service.load_model()
        logger.info("✓ LLM model loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load LLM model: {e}")
        raise

    # Clear cache before loading next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load ASR model
    logger.info("Loading ASR/STT model (2/3)...")
    try:
        await asr_service.load_model()
        logger.info("✓ ASR model loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load ASR model: {e}")
        raise

    # Clear cache before loading next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load TTS model
    logger.info("Loading TTS model (3/3)...")
    try:
        await tts_service.load_model()
        logger.info("✓ TTS model loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load TTS model: {e}")
        raise

    # Final memory status
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(
            f"GPU Memory: {allocated:.2f}GB / {total:.2f}GB ({(allocated / total) * 100:.1f}%)"
        )


async def load_models_parallel():
    """
    Load models in parallel for faster startup.

    Use this only if you have plenty of GPU memory (48GB+).
    """
    asr_service = get_asr_service()
    llm_service = get_llm_service()
    tts_service = get_tts_service()

    logger.info("Loading all models in parallel...")

    await asyncio.gather(
        asr_service.load_model(),
        llm_service.load_model(),
        tts_service.load_model(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events:
    - Startup: Validate GPU, load AI models if PRELOAD_MODELS is True
    - Shutdown: Unload models and cleanup resources
    """
    # Startup
    logger.info("=" * 60)
    logger.info("AI-Talk Backend Starting...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug Mode: {settings.debug}")
    logger.info("=" * 60)

    # Validate GPU
    logger.info("Validating GPU...")
    gpu_info = validate_gpu()

    if gpu_info["cuda_available"]:
        logger.info(f"GPU: {gpu_info['gpu_name']}")
        logger.info(
            f"GPU Memory: {gpu_info['total_memory_gb']}GB total, {gpu_info['free_memory_gb']}GB free"
        )
        logger.info(f"Required Memory: ~{gpu_info['required_memory_gb']}GB")

        if gpu_info["sufficient_memory"]:
            logger.info("✓ GPU memory is sufficient for all models")
        else:
            logger.warning("⚠ GPU memory may be insufficient - loading may fail")
    else:
        logger.warning("⚠ No CUDA GPU detected - running on CPU will be very slow!")

    for warning in gpu_info["warnings"]:
        logger.warning(f"⚠ {warning}")

    # Load models
    if settings.preload_models:
        logger.info("=" * 60)
        logger.info("Preloading AI models...")
        logger.info(f"  STT: {settings.asr_model_name}")
        logger.info(f"  LLM: {settings.llm_model_name}")
        logger.info(f"  TTS: {settings.tts_model_name}")
        logger.info("=" * 60)

        try:
            # Use sequential loading for better memory management
            if settings.enable_memory_efficient_loading:
                await load_models_sequential()
            else:
                await load_models_parallel()

            logger.info("=" * 60)
            logger.info("✓ All models loaded successfully!")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Failed to preload models: {e}")
            logger.warning("Models will be loaded on first request (may cause timeout)")
    else:
        logger.info("Model preloading disabled - models will load on first request")

    logger.info("")
    logger.info("AI-Talk Backend Ready!")
    logger.info(
        f"WebSocket endpoint: ws://{settings.host}:{settings.port}/ws/conversation"
    )
    logger.info(f"Health check: http://{settings.host}:{settings.port}/health")
    logger.info(
        f"API docs: http://{settings.host}:{settings.port}/docs"
        if settings.debug
        else "API docs: disabled"
    )
    logger.info("")

    yield  # Application runs here

    # Shutdown
    logger.info("=" * 60)
    logger.info("AI-Talk Backend Shutting Down...")
    logger.info("=" * 60)

    try:
        asr_service = get_asr_service()
        llm_service = get_llm_service()
        tts_service = get_tts_service()

        # Unload models sequentially to avoid memory issues
        logger.info("Unloading TTS model...")
        await tts_service.unload_model()

        logger.info("Unloading ASR model...")
        await asr_service.unload_model()

        logger.info("Unloading LLM model...")
        await llm_service.unload_model()

        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("✓ All models unloaded successfully")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

    logger.info("AI-Talk Backend Stopped")
    logger.info("=" * 60)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="AI-Talk Backend",
        description="Real-time conversational AI for English learning with STT + LLM + TTS",
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(websocket_router, tags=["conversation"])

    return app


# Create the app instance
app = create_app()


@app.get("/", tags=["root"])
async def root():
    """Root endpoint - returns basic API information."""
    return {
        "name": "AI-Talk Backend",
        "version": "0.2.0",
        "description": "Real-time conversational AI for English learning",
        "models": {
            "stt": settings.asr_model_name,
            "llm": settings.llm_model_name,
            "tts": settings.tts_model_name,
        },
        "endpoints": {
            "websocket": "/ws/conversation",
            "health": "/health",
            "ready": "/ready",
            "status": "/status",
            "gpu": "/gpu",
            "docs": "/docs" if settings.debug else "disabled",
        },
    }


@app.get("/health", tags=["monitoring"])
async def health():
    """
    Basic health check endpoint.
    Returns 200 if the server is running.
    """
    return {"status": "healthy"}


@app.get("/ready", tags=["monitoring"])
async def ready():
    """
    Readiness check endpoint.
    Returns 200 only if all models are loaded and ready.
    """
    asr_service = get_asr_service()
    llm_service = get_llm_service()
    tts_service = get_tts_service()

    all_ready = all(
        [
            asr_service.is_loaded,
            llm_service.is_loaded,
            tts_service.is_loaded,
        ]
    )

    if not all_ready:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "models": {
                    "asr": asr_service.is_loaded,
                    "llm": llm_service.is_loaded,
                    "tts": tts_service.is_loaded,
                },
            },
        )

    return {"status": "ready", "all_models_loaded": True}


@app.get("/status", tags=["monitoring"])
async def status():
    """
    Detailed status endpoint showing service states and model information.
    """
    asr_service = get_asr_service()
    llm_service = get_llm_service()
    tts_service = get_tts_service()

    # Get detailed model info if available
    asr_info = (
        await asr_service.get_model_info()
        if hasattr(asr_service, "get_model_info")
        else {"loaded": asr_service.is_loaded}
    )
    llm_info = (
        await llm_service.get_model_info()
        if hasattr(llm_service, "get_model_info")
        else {"loaded": llm_service.is_loaded}
    )
    tts_info = (
        await tts_service.get_model_info()
        if hasattr(tts_service, "get_model_info")
        else {"loaded": tts_service.is_loaded}
    )

    return {
        "status": "running",
        "environment": settings.environment,
        "version": "0.2.0",
        "models": {
            "asr": asr_info,
            "llm": llm_info,
            "tts": tts_info,
        },
        "config": {
            "preload_models": settings.preload_models,
            "memory_efficient_loading": settings.enable_memory_efficient_loading,
            "asr_sample_rate": settings.asr_sample_rate,
            "tts_sample_rate": settings.tts_sample_rate,
            "llm_max_model_len": settings.llm_max_model_len,
            "llm_gpu_memory_utilization": settings.llm_gpu_memory_utilization,
        },
    }


@app.get("/gpu", tags=["monitoring"])
async def gpu_status():
    """
    GPU status and memory information.
    """
    gpu_info = validate_gpu()

    # Add real-time memory info if CUDA is available
    if torch.cuda.is_available():
        gpu_info["memory_allocated_gb"] = round(
            torch.cuda.memory_allocated(0) / (1024**3), 2
        )
        gpu_info["memory_reserved_gb"] = round(
            torch.cuda.memory_reserved(0) / (1024**3), 2
        )
        gpu_info["memory_utilization_percent"] = round(
            (
                torch.cuda.memory_allocated(0)
                / torch.cuda.get_device_properties(0).total_memory
            )
            * 100,
            1,
        )

    return gpu_info


@app.post("/models/load", tags=["models"])
async def load_models():
    """
    Manually trigger model loading.
    Useful if preload_models is disabled or models failed to load on startup.
    """
    asr_service = get_asr_service()
    llm_service = get_llm_service()
    tts_service = get_tts_service()

    results = {
        "asr": {"success": False, "error": None},
        "llm": {"success": False, "error": None},
        "tts": {"success": False, "error": None},
    }

    # Load sequentially
    try:
        if not llm_service.is_loaded:
            await llm_service.load_model()
        results["llm"]["success"] = True
    except Exception as e:
        results["llm"]["error"] = str(e)

    try:
        if not asr_service.is_loaded:
            await asr_service.load_model()
        results["asr"]["success"] = True
    except Exception as e:
        results["asr"]["error"] = str(e)

    try:
        if not tts_service.is_loaded:
            await tts_service.load_model()
        results["tts"]["success"] = True
    except Exception as e:
        results["tts"]["error"] = str(e)

    all_success = all(r["success"] for r in results.values())

    return {
        "status": "success" if all_success else "partial_failure",
        "results": results,
    }


@app.post("/models/unload", tags=["models"])
async def unload_models():
    """
    Manually unload all models to free GPU memory.
    """
    asr_service = get_asr_service()
    llm_service = get_llm_service()
    tts_service = get_tts_service()

    await tts_service.unload_model()
    await asr_service.unload_model()
    await llm_service.unload_model()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"status": "success", "message": "All models unloaded"}


def configure_logging():
    """Configure loguru logging based on settings."""
    # Remove default handler
    logger.remove()

    # Add console handler with appropriate level
    log_format = settings.log_format

    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.log_level,
        colorize=True,
    )

    # Add file handler for production
    if settings.is_production:
        logger.add(
            "logs/ai-talk-{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format=log_format,
        )


if __name__ == "__main__":
    configure_logging()

    # Print startup banner
    logger.info("")
    logger.info("=" * 60)
    logger.info("  AI-Talk Backend")
    logger.info("  Real-time Conversational AI for English Learning")
    logger.info("=" * 60)
    logger.info("")
    logger.info(f"  Server: http://{settings.host}:{settings.port}")
    logger.info(f"  Environment: {settings.environment}")
    logger.info("")
    logger.info("  Models:")
    logger.info(f"    STT: {settings.asr_model_name}")
    logger.info(f"    LLM: {settings.llm_model_name}")
    logger.info(f"    TTS: {settings.tts_model_name}")
    logger.info("")
    logger.info("=" * 60)
    logger.info("")

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        ws_ping_interval=settings.ws_ping_interval,
        ws_ping_timeout=settings.ws_ping_timeout,
    )
