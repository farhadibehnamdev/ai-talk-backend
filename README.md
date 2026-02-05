# AI-Talk Backend

Real-time conversational AI backend for English learning, featuring Speech-to-Text (ASR), Large Language Model (LLM) conversation, and Text-to-Speech (TTS) capabilities.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      WebSocket API                          │
│                    /ws/conversation                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   ASR       │───▶│    LLM      │───▶│    TTS      │
│ Kyutai STT  │    │  Qwen2.5    │    │ Kyutai TTS  │
│             │    │  (vLLM)     │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
     Audio              Text              Audio
     Input             Response           Output
```

## 📋 Requirements

- Python 3.11+
- NVIDIA GPU with CUDA 12.1+ (recommended: RTX 4090 or A6000)
- 24GB+ VRAM for all models
- Docker with NVIDIA Container Toolkit (for containerized deployment)

## 🚀 Quick Start

### Option 1: Local Development (with GPU)

1. **Create virtual environment:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate   # Windows
   ```

2. **Install dependencies:**
   ```bash
   # Install PyTorch with CUDA
   pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

   # Install vLLM
   pip install vllm==0.6.6.post1

   # Install other dependencies
   pip install -r requirements.txt

   # Install Moshi (optional - will use fallback if unavailable)
   pip install git+https://github.com/kyutai-labs/moshi.git
   ```

3. **Configure environment:**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Run the server:**
   ```bash
   python -m app.main
   # or
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Option 2: Docker Deployment

1. **Build and run with Docker Compose:**
   ```bash
   cd ..  # Go to project root
   docker-compose up -d ai-backend
   ```

2. **Check logs:**
   ```bash
   docker-compose logs -f ai-backend
   ```

3. **Stop services:**
   ```bash
   docker-compose down
   ```

## 📁 Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration management
│   ├── api/
│   │   ├── __init__.py
│   │   └── websocket.py     # WebSocket endpoint handlers
│   ├── services/
│   │   ├── __init__.py
│   │   ├── asr_service.py   # Speech-to-Text service
│   │   ├── llm_service.py   # LLM conversation service
│   │   ├── tts_service.py   # Text-to-Speech service
│   │   └── analysis_service.py  # Conversation analysis
│   └── prompts/
│       ├── __init__.py
│       └── tutor_prompts.py # English tutor prompts
├── Dockerfile
├── requirements.txt
├── env.example
└── README.md
```

## 🔌 API Endpoints

### WebSocket: `/ws/conversation`

Real-time bidirectional communication for conversation.

#### Client → Server Messages

| Type | Format | Description |
|------|--------|-------------|
| Audio | Binary (bytes) | PCM 16-bit, 16kHz, mono audio data |
| End Conversation | `{"type": "end_conversation"}` | Request conversation analysis |
| Set Level | `{"type": "set_level", "level": "beginner\|intermediate\|advanced"}` | Change difficulty |
| Ping | `{"type": "ping"}` | Keep-alive ping |

#### Server → Client Messages

| Type | Format | Description |
|------|--------|-------------|
| Connected | `{"type": "connected", "session_id": "..."}` | Connection confirmed |
| Transcript | `{"type": "transcript", "text": "...", "is_final": bool}` | User speech transcription |
| AI Text | `{"type": "ai_text", "text": "...", "is_complete": bool}` | AI response (streaming) |
| AI Audio | Binary (bytes) | PCM 16-bit, 24kHz audio |
| Audio Start | `{"type": "ai_audio_start", "sample_rate": 24000}` | Audio stream starting |
| Turn Complete | `{"type": "turn_complete"}` | AI finished responding |
| Analysis | `{"type": "analysis", "data": {...}}` | Conversation feedback |
| Error | `{"type": "error", "message": "..."}` | Error occurred |
| Pong | `{"type": "pong"}` | Ping response |

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/ready` | Readiness check (model status) |
| GET | `/status` | Detailed service status |
| GET | `/docs` | Swagger UI (debug mode only) |

## ⚙️ Configuration

Environment variables (see `env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `DEBUG` | `false` | Enable debug mode |
| `ENVIRONMENT` | `development` | Environment name |
| `LLM_MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct-AWQ` | LLM model |
| `LLM_GPU_MEMORY_UTILIZATION` | `0.6` | GPU memory fraction for LLM |
| `ASR_MODEL_NAME` | `kyutai/moshi-asr` | ASR model |
| `TTS_MODEL_NAME` | `kyutai/moshi-tts` | TTS model |
| `PRELOAD_MODELS` | `true` | Load models on startup |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed CORS origins |
| `LOG_LEVEL` | `INFO` | Logging level |

## 📊 Analysis Response Format

When conversation ends, the analysis includes:

```json
{
  "type": "analysis",
  "data": {
    "grammar_score": 85,
    "final_score": 82,
    "mistakes": [
      {
        "original": "I goed to store",
        "correction": "I went to the store",
        "explanation": "'Go' is irregular: go → went"
      }
    ],
    "strengths": [
      "Good vocabulary usage",
      "Clear pronunciation"
    ],
    "suggestions": [
      "Practice irregular past tense verbs",
      "Try using more complex sentences"
    ]
  }
}
```

## 🔧 Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
# Format code
black app/
isort app/

# Type checking
mypy app/
```

### Adding New Features

1. Create service in `app/services/`
2. Add API endpoint in `app/api/`
3. Update `app/main.py` if needed
4. Add tests in `tests/`

## 🚀 Deployment on Vast.ai

1. **Rent a GPU instance:**
   - Recommended: RTX 4090 (24GB) or A6000 (48GB)
   - Docker support enabled
   - 50GB+ disk space

2. **SSH into instance and clone repository**

3. **Run with Docker:**
   ```bash
   docker-compose up -d
   ```

4. **Set up Cloudflare Tunnel for HTTPS:**
   ```bash
   # Install cloudflared
   curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
   chmod +x cloudflared
   
   # Authenticate and create tunnel
   ./cloudflared tunnel login
   ./cloudflared tunnel create ai-talk
   ./cloudflared tunnel route dns ai-talk api.yourdomain.com
   
   # Run tunnel
   ./cloudflared tunnel run ai-talk
   ```

## 📝 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request