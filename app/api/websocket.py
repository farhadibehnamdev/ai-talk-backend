"""
WebSocket API for real-time conversation.

This module handles the WebSocket connection for the AI-Talk conversation flow:
1. Receive audio from client
2. Process through ASR (Speech-to-Text)
3. Send transcript to client
4. Generate LLM response (streaming)
5. Send response text to client
6. Generate TTS audio (streaming)
7. Send audio to client
8. On conversation end, generate analysis and feedback
"""

import asyncio
import io
import json
import struct
import time
import uuid
import wave
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from app.config import settings
from app.prompts import UserLevel
from app.services import AnalysisService, ASRService, LLMService, TTSService
from app.services.analysis_service import get_analysis_service
from app.services.asr_service import get_asr_service
from app.services.llm_service import ConversationContext, get_llm_service
from app.services.tts_service import get_tts_service

router = APIRouter()


class MessageType(str, Enum):
    """Types of messages sent over WebSocket."""

    # Client -> Server
    AUDIO = "audio"
    END_CONVERSATION = "end_conversation"
    SET_LEVEL = "set_level"
    PING = "ping"

    # Server -> Client
    TRANSCRIPT = "transcript"
    AI_TEXT = "ai_text"
    AI_AUDIO = "ai_audio"
    TURN_COMPLETE = "turn_complete"
    ANALYSIS = "analysis"
    ERROR = "error"
    PONG = "pong"
    CONNECTED = "connected"


@dataclass
class ConversationSession:
    """Represents an active conversation session."""

    session_id: str
    context: ConversationContext
    user_utterances: List[str] = field(default_factory=list)
    ai_responses: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

    def add_user_utterance(self, text: str) -> None:
        """Add a user utterance to the session."""
        self.user_utterances.append(text)
        self.context.add_user_message(text)

    def add_ai_response(self, text: str) -> None:
        """Add an AI response to the session."""
        self.ai_responses.append(text)
        self.context.add_assistant_message(text)


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.sessions: dict[str, ConversationSession] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"Client connected: {session_id}")

    def disconnect(self, session_id: str) -> None:
        """Remove a WebSocket connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
        logger.info(f"Client disconnected: {session_id}")

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get or create a conversation session."""
        if session_id not in self.sessions:
            llm_service = get_llm_service()
            context = llm_service.create_context(UserLevel.INTERMEDIATE)
            self.sessions[session_id] = ConversationSession(
                session_id=session_id,
                context=context,
            )
        return self.sessions[session_id]

    def is_connected(self, session_id: str) -> bool:
        """Check if a session is still connected."""
        return session_id in self.active_connections

    async def send_json(self, session_id: str, data: dict) -> bool:
        """Send JSON data to a specific client. Returns False if send failed."""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(data)
                return True
            except Exception as e:
                logger.error(f"Failed to send JSON to {session_id}: {e}")
                self.disconnect(session_id)
                return False
        return False

    async def send_bytes(self, session_id: str, data: bytes) -> bool:
        """Send binary data to a specific client. Returns False if send failed."""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_bytes(data)
                return True
            except Exception as e:
                logger.error(f"Failed to send bytes to {session_id}: {e}")
                self.disconnect(session_id)
                return False
        return False


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/conversation")
async def conversation_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time conversation."""
    session_id = str(uuid.uuid4())
    turn_task: Optional[asyncio.Task] = None
    silence_watcher_task: Optional[asyncio.Task] = None

    try:
        await manager.connect(websocket, session_id)

        # Send connection confirmation
        await manager.send_json(
            session_id,
            {
                "type": MessageType.CONNECTED,
                "session_id": session_id,
            },
        )

        # Get services
        asr_service = get_asr_service()
        llm_service = get_llm_service()
        tts_service = get_tts_service()
        analysis_service = get_analysis_service()

        # Get or create session
        session = manager.get_session(session_id)

        # Turn detection and interruption settings.
        audio_buffer = bytearray()
        audio_lock = asyncio.Lock()
        last_audio_at = 0.0
        barge_in_buffered_bytes = 0
        barge_in_audio_buffer = bytearray()
        min_audio_length = settings.ws_min_audio_buffer_bytes
        max_audio_length = settings.ws_max_audio_buffer_bytes
        speech_end_silence_sec = settings.ws_speech_end_silence_ms / 1000.0
        barge_in_min_bytes = settings.ws_barge_in_min_bytes

        async def process_turn_async(turn_audio: bytes) -> None:
            """Run one ASR->LLM->TTS turn in a cancellable task."""
            nonlocal turn_task
            try:
                await process_audio_turn(
                    session_id=session_id,
                    session=session,
                    audio_data=turn_audio,
                    asr_service=asr_service,
                    llm_service=llm_service,
                    tts_service=tts_service,
                )
            except asyncio.CancelledError:
                logger.info(f"Turn cancelled for {session_id} (barge-in)")
                raise
            except Exception:
                logger.exception(f"Unhandled turn task error for {session_id}")
            finally:
                turn_task = None

        async def maybe_start_turn(force: bool = False) -> bool:
            """Start a turn when buffered audio is ready."""
            nonlocal turn_task, barge_in_buffered_bytes, barge_in_audio_buffer, last_audio_at
            if turn_task is not None and not turn_task.done():
                return False

            async with audio_lock:
                if len(audio_buffer) == 0:
                    return False

                if not force:
                    if len(audio_buffer) < min_audio_length:
                        return False
                    if (
                        last_audio_at <= 0
                        or (time.time() - last_audio_at) < speech_end_silence_sec
                    ):
                        return False

                turn_audio = bytes(audio_buffer)
                audio_buffer.clear()
                barge_in_buffered_bytes = 0
                barge_in_audio_buffer.clear()
                last_audio_at = 0.0

            logger.debug(
                f"Starting turn for {session_id}: {len(turn_audio)} buffered bytes"
            )
            turn_task = asyncio.create_task(process_turn_async(turn_audio))
            return True

        async def silence_watcher() -> None:
            """Background task to flush turns after speech-end silence."""
            while manager.is_connected(session_id):
                await asyncio.sleep(0.1)
                await maybe_start_turn(force=False)

        silence_watcher_task = asyncio.create_task(silence_watcher())

        while manager.is_connected(session_id):
            try:
                message = await websocket.receive()

                if message.get("type") == "websocket.disconnect":
                    logger.info(f"Received disconnect message for {session_id}")
                    break

                if "bytes" in message:
                    audio_data = message["bytes"] or b""
                    if len(audio_data) == 0:
                        continue

                    # Drop stale barge-in candidate once AI turn has completed.
                    if (
                        (turn_task is None or turn_task.done())
                        and barge_in_buffered_bytes > 0
                    ):
                        barge_in_buffered_bytes = 0
                        barge_in_audio_buffer.clear()

                    # Barge-in: if user keeps sending audio while AI is responding,
                    # cancel the current turn and switch to user speech.
                    if turn_task is not None and not turn_task.done():
                        barge_in_audio_buffer.extend(audio_data)
                        barge_in_buffered_bytes += len(audio_data)
                        if barge_in_buffered_bytes < barge_in_min_bytes:
                            continue

                        logger.info(
                            f"Barge-in detected for {session_id}: "
                            f"{barge_in_buffered_bytes} bytes during AI response"
                        )
                        turn_task.cancel()
                        try:
                            await turn_task
                        except asyncio.CancelledError:
                            pass
                        turn_task = None
                        audio_data = bytes(barge_in_audio_buffer)
                        barge_in_buffered_bytes = 0
                        barge_in_audio_buffer.clear()

                        # Signal interruption so frontend can stop playback.
                        await manager.send_json(
                            session_id,
                            {"type": MessageType.TURN_COMPLETE, "interrupted": True},
                        )

                    async with audio_lock:
                        audio_buffer.extend(audio_data)
                        last_audio_at = time.time()
                        buffered_len = len(audio_buffer)

                    # Safety flush for very long utterances.
                    if buffered_len >= max_audio_length:
                        await maybe_start_turn(force=True)

                elif "text" in message:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")

                    if msg_type == MessageType.PING:
                        await manager.send_json(session_id, {"type": MessageType.PONG})

                    elif msg_type == MessageType.SET_LEVEL:
                        level_str = data.get("level", "intermediate")
                        try:
                            level = UserLevel(level_str)
                            session.context.user_level = level
                            session.context.system_prompt = (
                                get_llm_service().create_context(level).system_prompt
                            )
                            logger.info(f"Session {session_id} level set to {level}")
                        except ValueError:
                            await manager.send_json(
                                session_id,
                                {
                                    "type": MessageType.ERROR,
                                    "message": f"Invalid level: {level_str}",
                                },
                            )

                    elif msg_type == MessageType.END_CONVERSATION:
                        # Process remaining audio and wait for turn completion.
                        await maybe_start_turn(force=True)
                        active_turn = turn_task
                        if active_turn is not None:
                            try:
                                await active_turn
                            except asyncio.CancelledError:
                                pass

                        await send_analysis(
                            session_id=session_id,
                            session=session,
                            analysis_service=analysis_service,
                        )

                    elif msg_type == "process_audio":
                        # Explicit request to flush buffered audio now.
                        await maybe_start_turn(force=True)

            except WebSocketDisconnect:
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from {session_id}: {e}")
                await manager.send_json(
                    session_id,
                    {"type": MessageType.ERROR, "message": "Invalid JSON message"},
                )
            except Exception as e:
                error_msg = str(e)
                if "disconnect" in error_msg.lower() or "close" in error_msg.lower():
                    logger.info(f"Connection lost for {session_id}: {error_msg}")
                    break
                logger.error(f"Error processing message from {session_id}: {error_msg}")
                if not await manager.send_json(
                    session_id,
                    {"type": MessageType.ERROR, "message": error_msg},
                ):
                    break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
    finally:
        if silence_watcher_task is not None:
            silence_watcher_task.cancel()
            try:
                await silence_watcher_task
            except asyncio.CancelledError:
                pass

        if turn_task is not None and not turn_task.done():
            turn_task.cancel()
            try:
                await turn_task
            except asyncio.CancelledError:
                pass

        manager.disconnect(session_id)


async def process_audio_turn(
    session_id: str,
    session: ConversationSession,
    audio_data: bytes,
    asr_service: ASRService,
    llm_service: LLMService,
    tts_service: TTSService,
) -> None:
    """
    Process a complete audio turn through the ASR -> LLM -> TTS pipeline.

    Args:
        session_id: The session identifier
        session: The conversation session
        audio_data: The audio bytes to process
        asr_service: ASR service instance
        llm_service: LLM service instance
        tts_service: TTS service instance
    """
    try:
        # Step 1: ASR - Convert speech to text
        logger.debug(f"Processing {len(audio_data)} bytes of audio for {session_id}")

        # --- Audio diagnostic logging ---
        try:
            # Interpret as raw 16-bit PCM mono
            trimmed = audio_data if len(audio_data) % 2 == 0 else audio_data[:-1]
            if len(trimmed) > 0:
                samples = np.frombuffer(trimmed, dtype=np.int16).astype(np.float32)
                duration_sec = len(samples) / 16000.0
                rms = np.sqrt(np.mean(samples**2))
                peak = np.max(np.abs(samples))
                non_zero = np.count_nonzero(samples)
                pct_non_zero = (
                    (non_zero / len(samples)) * 100 if len(samples) > 0 else 0
                )

                logger.info(
                    f"[AUDIO DIAG] session={session_id} | "
                    f"bytes={len(audio_data)} | samples={len(samples)} | "
                    f"duration={duration_sec:.2f}s | "
                    f"RMS={rms:.1f} | peak={peak:.0f} | "
                    f"non_zero={pct_non_zero:.1f}%"
                )

                if rms < 10:
                    logger.warning(
                        f"[AUDIO DIAG] Audio appears to be SILENCE or near-silent "
                        f"(RMS={rms:.1f}). Mic might not be working or sending empty frames."
                    )
                elif rms > 30000:
                    logger.warning(
                        f"[AUDIO DIAG] Audio appears to be CLIPPING/NOISE "
                        f"(RMS={rms:.1f}). Check audio encoding format."
                    )
                else:
                    logger.info(
                        f"[AUDIO DIAG] Audio levels look NORMAL (RMS={rms:.1f}). "
                        f"Speech likely present."
                    )

                # Save first audio chunk as WAV for manual inspection (once per session)
                debug_dir = Path("/tmp/audio_debug")
                debug_file = debug_dir / f"{session_id}.wav"
                if not debug_file.exists():
                    try:
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        with wave.open(str(debug_file), "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)  # 16-bit
                            wf.setframerate(16000)
                            wf.writeframes(trimmed)
                        logger.info(
                            f"[AUDIO DIAG] Saved debug WAV: {debug_file} "
                            f"({duration_sec:.2f}s) — download and play to verify"
                        )
                    except Exception as wav_err:
                        logger.debug(
                            f"[AUDIO DIAG] Could not save debug WAV: {wav_err}"
                        )
        except Exception as diag_err:
            logger.debug(f"[AUDIO DIAG] Diagnostic failed: {diag_err}")
        # --- End audio diagnostics ---

        transcript_result = await asr_service.transcribe(audio_data)

        if not transcript_result.text:
            logger.debug(f"Empty transcript for {session_id}")
            return

        # Send transcript to client
        if not await manager.send_json(
            session_id,
            {
                "type": MessageType.TRANSCRIPT,
                "text": transcript_result.text,
                "is_final": transcript_result.is_final,
            },
        ):
            return

        # Add to conversation
        session.add_user_utterance(transcript_result.text)

        # Step 2: LLM - Generate response (streaming)
        logger.debug(f"Generating LLM response for {session_id}")

        full_response = ""
        async for token in llm_service.generate_stream(session.context):
            full_response += token
            if not await manager.send_json(
                session_id,
                {
                    "type": MessageType.AI_TEXT,
                    "text": token,
                    "is_complete": False,
                },
            ):
                return

        # Send completion signal for text
        if not await manager.send_json(
            session_id,
            {
                "type": MessageType.AI_TEXT,
                "text": "",
                "is_complete": True,
                "full_text": full_response,
            },
        ):
            return

        # Add AI response to conversation
        session.add_ai_response(full_response)

        # Step 3: TTS - Convert response to speech (streaming)
        logger.debug(f"Generating TTS audio for {session_id}")

        # Signal audio stream starting
        if not await manager.send_json(
            session_id,
            {"type": "ai_audio_start", "sample_rate": tts_service.sample_rate},
        ):
            return

        # Stream audio chunks
        async for audio_chunk in tts_service.synthesize_stream(full_response):
            if not await manager.send_bytes(session_id, audio_chunk.data):
                return

        # Signal turn complete
        await manager.send_json(
            session_id,
            {"type": MessageType.TURN_COMPLETE},
        )

        logger.debug(f"Turn complete for {session_id}")

    except asyncio.CancelledError:
        logger.info(f"Audio turn interrupted for {session_id}")
        raise
    except Exception as e:
        logger.error(f"Error in audio turn processing for {session_id}: {e}")
        if manager.is_connected(session_id):
            await manager.send_json(
                session_id,
                {
                    "type": MessageType.ERROR,
                    "message": f"Failed to process audio: {str(e)}",
                },
            )


async def send_analysis(
    session_id: str,
    session: ConversationSession,
    analysis_service: AnalysisService,
) -> None:
    """
    Generate and send conversation analysis/feedback.

    Args:
        session_id: The session identifier
        session: The conversation session
        analysis_service: Analysis service instance
    """
    try:
        logger.info(f"Generating analysis for {session_id}")

        # Generate analysis
        feedback = await analysis_service.analyze_conversation(session.user_utterances)

        # Send analysis to client
        if not await manager.send_json(
            session_id,
            {
                "type": MessageType.ANALYSIS,
                "data": feedback.to_dict(),
            },
        ):
            return

        logger.info(
            f"Analysis sent for {session_id}: "
            f"grammar={feedback.grammar_score}, final={feedback.final_score}"
        )

    except Exception as e:
        logger.error(f"Error generating analysis for {session_id}: {e}")
        if manager.is_connected(session_id):
            await manager.send_json(
                session_id,
                {
                    "type": MessageType.ERROR,
                    "message": f"Failed to generate analysis: {str(e)}",
                },
            )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ai-talk-backend",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint.

    Verifies that all required services are loaded and ready.
    """
    asr_service = get_asr_service()
    llm_service = get_llm_service()
    tts_service = get_tts_service()

    return {
        "status": "ready",
        "services": {
            "asr": asr_service.is_loaded,
            "llm": llm_service.is_loaded,
            "tts": tts_service.is_loaded,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
