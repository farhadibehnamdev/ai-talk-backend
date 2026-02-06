# AI-Talk Backend

Real-time conversational AI backend for English learning, featuring Speech-to-Text (STT), Large Language Model (LLM) conversation, and Text-to-Speech (TTS) capabilities.

## 🎯 Features

- **Real-time Speech-to-Text**: Streaming transcription using Kyutai STT 2.6B
- **Intelligent Conversations**: English tutoring powered by Qwen 2.5 7B
- **Natural Speech Synthesis**: High-quality TTS using Kyutai TTS 1.8B
- **WebSocket API**: Low-latency bidirectional communication
- **GPU Optimized**: Efficient memory management for 24GB GPUs

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      WebSocket API                          │
│                    /ws/conversation                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      STT        │───▶│      LLM        │───▶│      TTS        │
│ kyutai/stt-     │    │ Qwen2.5-7B-     │    │ kyutai/tts-     │
│ 2.6b-en         │    │ Instruct-AWQ    │    │ 1.6b-en_fr      │
│ (~6-8 GB VRAM)  │    │ (~4-5 GB VRAM)  │    │ (~4-5 GB VRAM)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
     Audio                  Text                  Audio
     Input                Response               Output
```

## 📋 Hardware Requirements

### GPU Requirements

| GPU | VRAM | Suitability |
|-----|------|-------------|
| **RTX 4090** | 24 GB | ✅ Excellent - Recommended |
| **RTX 3090/3090 Ti** | 24 GB | ✅ Very Good |
| **RTX A6000** | 48 GB | ✅ Best - Plenty of headroom |
| **RTX 4080** | 16 GB | ⚠️ Tight - May require tuning |
| **RTX 3080 (10GB)** | 10 GB | ❌ Not sufficient |

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU VRAM** | 20 GB | 24 GB+ |
| **System RAM** | 32 GB | 64 GB |
| **CPU** | 8 cores | 16+ cores |
| **Storage** | 50 GB SSD | 100+ GB NVMe SSD |
| **CUDA** | 12.1+ | 12.4+ |
| **Python** | 3.11+ | 3.11 |

### Model VRAM Breakdown

| Model | Parameters | VRAM Usage |
|-------|------------|------------|
| `kyutai/stt-2.6b-en` | 2.6B | ~6-8 GB |
| `kyutai/tts-1.6b-en_fr` | 1.8B | ~4-5 GB |
| `Qwen/Qwen2.5-7B-Instruct-AWQ` | 7.61B (4-bit) | ~4-5 GB |
| **Total** | - | **~16-20 GB** |

## 🚀 Quick Start

### Option 1: Automated Setup

```bash
# Clone and enter the project
cd ai-talk-backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Run the setup script
python setup.py
```

The setup script will:
1. Check system requirements (Python, CUDA, GPU memory)
2. Download all required models from Hugging Face
3. Verify the installation

### Option 2: Manual Setup

#### Step 1: Install PyTorch with CUDA

```bash
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

#### Step 2: Install vLLM

```bash
pip install vllm==0.6.6.post1
```

#### Step 3: Install Moshi (Kyutai STT/TTS)

```bash
pip install git+https://github.com/kyutai-labs/moshi.git
```

#### Step 4: Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

#### Step 5: Download Models

```bash
# Using the setup script
python setup.py --download

# Or manually with Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download('kyutai/stt-2.6b-en')
snapshot_download('kyutai/tts-1.6b-en_fr')
snapshot_download('Qwen/Qwen2.5-7B-Instruct-AWQ')
snapshot_download('kyutai/tts-voices')
"
```

#### Step 6: Configure Environment

```bash
cp env.example .env
# Edit .env with your configuration
```

#### Step 7: Run the Server

```bash
python -m app.main
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Option 3: Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d ai-backend

# Check logs
docker-compose logs -f ai-backend

# Stop services
docker-compose down
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file from `env.example`:

```bash
cp env.example .env
```

#### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `DEBUG` | `false` | Enable debug mode |
| `ENVIRONMENT` | `development` | Environment name |

#### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct-AWQ` | LLM model |
| `LLM_MAX_MODEL_LEN` | `4096` | Maximum context length |
| `LLM_GPU_MEMORY_UTILIZATION` | `0.35` | GPU memory fraction for LLM |
| `ASR_MODEL_NAME` | `kyutai/stt-2.6b-en` | STT model |
| `ASR_SAMPLE_RATE` | `16000` | Audio input sample rate |
| `ASR_PREFER_MOSHI` | `true` | Prefer Moshi backend for Kyutai STT stability |
| `ASR_ENABLE_KYUTAI_TRANSFORMERS` | `false` | Enable Kyutai STT via transformers backend |
| `ASR_KYUTAI_ATTN_IMPLEMENTATION` | `eager` | Attention impl for Kyutai transformers backend |
| `ASR_WHISPER_FALLBACK_MODEL` | `openai/whisper-large-v3` | Whisper fallback model when Kyutai fails |
| `ASR_SILENCE_RMS_THRESHOLD` | `0.006` | Skip very weak decoded chunks |
| `ASR_SILENCE_PEAK_THRESHOLD` | `0.08` | Skip low-peak decoded chunks |
| `ASR_FILTER_NOISE_TRANSCRIPTS` | `true` | Drop weak short low-information transcripts |
| `ASR_NOISE_MAX_DURATION_S` | `2.0` | Max duration for low-information filtering |
| `TTS_MODEL_NAME` | `kyutai/tts-1.6b-en_fr` | TTS model |
| `TTS_SAMPLE_RATE` | `24000` | Audio output sample rate |

#### Memory Management

| Variable | Default | Description |
|----------|---------|-------------|
| `PRELOAD_MODELS` | `true` | Load models on startup |
| `ENABLE_MEMORY_EFFICIENT_LOADING` | `true` | Sequential model loading |
| `GPU_MEMORY_RESERVED_GB` | `2.0` | Reserved GPU memory |
| `MODELS_CACHE_DIR` | `./models` | Model cache directory |

### GPU Memory Optimization

For a 24GB GPU, use these recommended settings:

```env
LLM_GPU_MEMORY_UTILIZATION=0.35  # ~8.4 GB for LLM
# STT will use ~6-8 GB
# TTS will use ~4-5 GB
# Total: ~18-21 GB
```

## 📁 Project Structure

```
ai-talk-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration management
│   ├── api/
│   │   ├── __init__.py
│   │   └── websocket.py     # WebSocket endpoint handlers
│   ├── services/
│   │   ├── __init__.py
│   │   ├── asr_service.py   # Speech-to-Text service (Kyutai STT)
│   │   ├── llm_service.py   # LLM conversation service (Qwen + vLLM)
│   │   ├── tts_service.py   # Text-to-Speech service (Kyutai TTS)
│   │   └── analysis_service.py  # Conversation analysis
│   └── prompts/
│       ├── __init__.py
│       └── tutor_prompts.py # English tutor prompts
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── setup.py                 # Setup and model download script
├── env.example
└── README.md
```

## 🔌 API Endpoints

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/ready` | Readiness check (models loaded) |
| GET | `/status` | Detailed service status |
| GET | `/gpu` | GPU memory information |
| POST | `/models/load` | Manually load models |
| POST | `/models/unload` | Unload models (free memory) |
| GET | `/docs` | Swagger UI (debug mode only) |

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

# Linting
ruff check app/
```

### Checking GPU Status

```bash
# Via API
curl http://localhost:8000/gpu

# Via Python
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')"
```

### Model Loading Test

```bash
# Test basic loading
python setup.py --test

# Full server test
python -m app.main
```

## 🚀 Deployment on Vast.ai / RunPod

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

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Check current GPU usage
nvidia-smi

# Reduce LLM memory utilization in .env
LLM_GPU_MEMORY_UTILIZATION=0.30

# Enable memory efficient loading
ENABLE_MEMORY_EFFICIENT_LOADING=true
```

### Models Not Loading

```bash
# Check model status
curl http://localhost:8000/status

# Manually trigger model loading
curl -X POST http://localhost:8000/models/load

# Check logs
docker-compose logs -f ai-backend
```

### vLLM Issues

If vLLM fails to load, the system will automatically fall back to transformers:

```bash
# Check if vLLM is working
python -c "import vllm; print(vllm.__version__)"

# Reinstall if needed
pip uninstall vllm
pip install vllm==0.6.6.post1
```

### Moshi Import Errors

```bash
# Reinstall from GitHub
pip uninstall moshi
pip install git+https://github.com/kyutai-labs/moshi.git
```

## 📚 Model Documentation

- **Kyutai STT**: [huggingface.co/kyutai/stt-2.6b-en](https://huggingface.co/kyutai/stt-2.6b-en)
- **Kyutai TTS**: [huggingface.co/kyutai/tts-1.6b-en_fr](https://huggingface.co/kyutai/tts-1.6b-en_fr)
- **Qwen 2.5 AWQ**: [huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ)
- **Moshi GitHub**: [github.com/kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)

## 📝 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
