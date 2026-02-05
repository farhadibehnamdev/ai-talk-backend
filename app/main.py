"""
AI-Talk Backend - Main Application Entry Point

This is the main FastAPI application that serves the AI-Talk English learning
conversation system. It provides:
- WebSocket endpoint for real-time audio conversation
- Health check endpoints
- Model lifecycle management (loading on startup)
"""

import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api import websocket_router
from app.config import settings
from app.services.asr_service import get_asr_service
from app.services.llm_service import get_llm_service
from app.services.tts_service import get_tts_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events:
    - Startup: Load AI models if PRELOAD_MODELS is True
    - Shutdown: Unload models and cleanup resources
    """
    # Startup
    logger.info("=" * 50)
    logger.info("AI-Talk Backend Starting...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug Mode: {settings.debug}")
    logger.info("=" * 50)

    if settings.preload_models:
        logger.info("Preloading AI models...")

        try:
            # Load models concurrently
            asr_service = get_asr_service()
            llm_service = get_llm_service()
            tts_service = get_tts_service()

            await asyncio.gather(
                asr_service.load_model(),
                llm_service.load_model(),
                tts_service.load_model(),
            )

            logger.info("All models loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to preload models: {e}")
            logger.warning("Models will be loaded on first request")
    else:
        logger.info("Model preloading disabled - models will load on first request")

    logger.info("AI-Talk Backend Ready!")
    logger.info(
        f"WebSocket endpoint: ws://{settings.host}:{settings.port}/ws/conversation"
    )

    yield  # Application runs here

    # Shutdown
    logger.info("AI-Talk Backend Shutting Down...")

    try:
        asr_service = get_asr_service()
        llm_service = get_llm_service()
        tts_service = get_tts_service()

        await asyncio.gather(
            asr_service.unload_model(),
            llm_service.unload_model(),
            tts_service.unload_model(),
        )

        logger.info("All models unloaded successfully")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

    logger.info("AI-Talk Backend Stopped")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="AI-Talk Backend",
        description="Real-time conversational AI for English learning",
        version="0.1.0",
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
        "version": "0.1.0",
        "description": "Real-time conversational AI for English learning",
        "endpoints": {
            "websocket": "/ws/conversation",
            "health": "/health",
            "ready": "/ready",
            "docs": "/docs" if settings.debug else "disabled",
        },
    }


@app.get("/status", tags=["monitoring"])
async def status():
    """
    Detailed status endpoint showing service states.
    """
    asr_service = get_asr_service()
    llm_service = get_llm_service()
    tts_service = get_tts_service()

    return {
        "status": "running",
        "environment": settings.environment,
        "models": {
            "asr": {
                "loaded": asr_service.is_loaded,
                "model": settings.asr_model_name,
            },
            "llm": {
                "loaded": llm_service.is_loaded,
                "model": settings.llm_model_name,
            },
            "tts": {
                "loaded": tts_service.is_loaded,
                "model": settings.tts_model_name,
            },
        },
        "config": {
            "preload_models": settings.preload_models,
            "asr_sample_rate": settings.asr_sample_rate,
            "tts_sample_rate": settings.tts_sample_rate,
        },
    }


def configure_logging():
    """Configure loguru logging based on settings."""
    import sys

    # Remove default handler
    logger.remove()

    # Add console handler with appropriate level
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.log_level,
        colorize=True,
    )

    # Add file handler for production
    if settings.environment == "production":
        logger.add(
            "logs/ai-talk-{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format=log_format,
        )


if __name__ == "__main__":
    configure_logging()

    logger.info(f"Starting server on {settings.host}:{settings.port}")

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
