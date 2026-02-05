"""
API module for AI-Talk backend.

This module contains the API endpoints:
- WebSocket endpoint for real-time conversation
- Health check endpoint
"""

from .websocket import router as websocket_router

__all__ = [
    "websocket_router",
]
