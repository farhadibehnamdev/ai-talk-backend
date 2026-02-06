# AI-Talk Backend — Deployment Guide

Complete step-by-step guide for deploying the AI-Talk backend on **bare metal** (Vast.ai, RunPod, local machine) and **Docker**.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Option 1: Bare Metal (Without Docker)](#option-1-bare-metal-without-docker)
- [Option 2: Docker Deployment](#option-2-docker-deployment)
- [Configuration Reference](#configuration-reference)
- [Model Options](#model-options)
- [Verifying the Deployment](#verifying-the-deployment)
- [Running in Background](#running-in-background)
- [Updating the Code](#updating-the-code)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

| Component      | Minimum        | Recommended       |
| -------------- | -------------- | ----------------- |
| **GPU VRAM**   | 20 GB          | 24 GB+ (RTX 4090) |
| **System RAM** | 32 GB          | 64 GB             |
| **Storage**    | 50 GB SSD      | 100 GB NVMe SSD   |
| **CUDA**       | 12.1+          | 12.4+             |
| **Python**     | 3.11+          | 3.11              |

### VRAM Breakdown by Model

| Model                              | Parameters    | VRAM     |
| ---------------------------------- | ------------- | -------- |
| `openai/whisper-medium` (ASR)      | 769M          | ~1.5 GB  |
| `kyutai/stt-2.6b-en-trfs` (ASR)   | 2.6B          | ~6-8 GB  |
| `Qwen/Qwen2.5-7B-Instruct-AWQ`    | 7.6B (4-bit)  | ~4-5 GB  |
| Coqui TTS (fallback TTS)           | ~80M          | ~0.4 GB  |
| `kyutai/tts-1.6b-en_fr` (TTS)     | 1.8B          | ~4-5 GB  |

> **Tip:** Using Whisper + Coqui TTS instead of Kyutai STT/TTS reduces total VRAM from ~16-20 GB to ~6-7 GB.

### Software Requirements

- **NVIDIA GPU drivers** (550+ recommended)
- **CUDA Toolkit 12.1+**
- **ffmpeg** (required for decoding browser audio)
- **Python 3.11+**
- **Git**

---

## Option 1: Bare Metal (Without Docker)

Best for **Vast.ai**, **RunPod**, or any Linux machine with a GPU.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-talk-backend.git
cd ai-talk-backend
```

### Step 2: Verify GPU and System Dependencies

```bash
# Check NVIDIA driver and CUDA
nvidia-smi

# Check Python version (need 3.11+)
python3 --version

# Check ffmpeg (required for decoding browser audio)
which ffmpeg

# If ffmpeg is missing:
apt-get update && apt-get install -y ffmpeg
```

### Step 3: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

> **Important:** Always activate the venv before running any commands below.

### Step 4: Install PyTorch with CUDA

```bash
pip install --upgrade pip setuptools wheel
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is available:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 5: Install vLLM

```bash
pip install vllm==0.6.6.post1
```

### Step 6: Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### Step 7: Install TTS Fallback (Recommended)

The Kyutai TTS model requires `transformers >= 4.54` which may not be available yet. Install a fallback:

```bash
# Option A: Coqui TTS (offline, uses GPU, good quality)
pip install TTS

# Option B: Edge TTS (online, no GPU needed, requires internet)
pip install edge-tts
```

### Step 8: (Optional) Install Moshi Library

Only needed if you want to use the native Moshi library for Kyutai models:

```bash
pip install git+https://github.com/kyutai-labs/moshi.git
```

### Step 9: Configure Environment

```bash
cp env.example .env
```

Edit `.env` with your preferred settings. **Recommended for most setups:**

```env
# Server
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=production

# ASR — Choose ONE:
# Option A: Whisper (reliable, works out of the box)
ASR_MODEL_NAME=openai/whisper-medium
# Option B: Kyutai STT (requires transformers >= 4.53, may segfault on some versions)
# ASR_MODEL_NAME=kyutai/stt-2.6b-en-trfs

ASR_SAMPLE_RATE=16000

# LLM
LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-AWQ
LLM_GPU_MEMORY_UTILIZATION=0.35

# TTS — Kyutai TTS will auto-fallback to Coqui/edge-tts if not supported
TTS_MODEL_NAME=kyutai/tts-1.6b-en_fr
TTS_SAMPLE_RATE=24000

# Model loading
PRELOAD_MODELS=true
MODELS_CACHE_DIR=./models

# CORS — set to your frontend URL
CORS_ORIGINS=http://localhost:3000,https://your-frontend.vercel.app

# Logging
LOG_LEVEL=INFO
```

### Step 10: Start the Server

```bash
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

First startup will **download models** from HuggingFace (~10-30 minutes depending on internet speed). Subsequent starts are much faster.

Watch the logs for:

```
✓ ASR model loaded
✓ LLM model loaded
✓ TTS model loaded
✓ All models loaded successfully!
AI-Talk Backend Ready!
Uvicorn running on http://0.0.0.0:8000
```

### Step 11: Verify

```bash
# In a second terminal:
curl http://localhost:8000/health
# Expected: {"status":"healthy", ...}

curl http://localhost:8000/ready
# Expected: {"status":"ready","services":{"asr":true,"llm":true,"tts":true}, ...}
```

---

## Option 2: Docker Deployment

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-talk-backend.git
cd ai-talk-backend
```

### Step 2: Install NVIDIA Container Toolkit

Required for GPU access inside Docker containers:

```bash
# Add the NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Step 3: Configure Environment

```bash
cp env.example .env
```

Edit `.env` — same settings as bare metal, but change the models cache dir:

```env
MODELS_CACHE_DIR=/app/models

# ASR
ASR_MODEL_NAME=openai/whisper-medium

# LLM
LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-AWQ
LLM_GPU_MEMORY_UTILIZATION=0.35

# TTS
TTS_MODEL_NAME=kyutai/tts-1.6b-en_fr

# CORS
CORS_ORIGINS=http://localhost:3000,https://your-frontend.vercel.app
```

### Step 4: Build and Start

```bash
# Build the image (takes 10-20 minutes first time)
docker compose build

# Start in detached mode
docker compose up -d
```

### Step 5: Monitor Startup

```bash
# Watch logs (models download on first run — can take 10-30 min)
docker compose logs -f ai-backend
```

Wait for:

```
✓ All models loaded successfully!
AI-Talk Backend Ready!
```

### Step 6: Verify

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### Docker Management Commands

```bash
# Stop the service
docker compose down

# Restart after code changes
docker compose down
docker compose up -d --build

# View logs
docker compose logs -f ai-backend

# Shell into the container
docker compose exec ai-backend bash

# Check GPU inside container
docker compose exec ai-backend nvidia-smi

# Remove everything including downloaded models (WARNING: re-downloads on next start)
docker compose down -v
```

---

## Configuration Reference

### Environment Variables

All variables can be set in `.env` or passed directly.

#### Server

| Variable        | Default       | Description                  |
| --------------- | ------------- | ---------------------------- |
| `HOST`          | `0.0.0.0`    | Server bind address          |
| `PORT`          | `8000`        | Server port                  |
| `DEBUG`         | `false`       | Enable debug mode / Swagger  |
| `ENVIRONMENT`   | `development` | `development` or `production`|

#### ASR (Speech-to-Text)

| Variable            | Default               | Description                        |
| ------------------- | --------------------- | ---------------------------------- |
| `ASR_MODEL_NAME`    | `kyutai/stt-2.6b-en`  | HuggingFace model ID               |
| `ASR_SAMPLE_RATE`   | `16000`               | Input audio sample rate (Hz)       |
| `ASR_DEVICE`        | *(auto)*              | `cuda:0`, `cpu`, or auto-detect    |
| `ASR_DTYPE`         | `float16`             | `float16`, `bfloat16`, `float32`   |

#### LLM (Language Model)

| Variable                       | Default                          | Description                     |
| ------------------------------ | -------------------------------- | ------------------------------- |
| `LLM_MODEL_NAME`              | `Qwen/Qwen2.5-7B-Instruct-AWQ`  | HuggingFace model ID            |
| `LLM_MAX_MODEL_LEN`           | `4096`                           | Maximum context length          |
| `LLM_GPU_MEMORY_UTILIZATION`  | `0.35`                           | Fraction of GPU memory for LLM  |
| `LLM_MAX_TOKENS`              | `256`                            | Max tokens per response         |
| `LLM_TEMPERATURE`             | `0.7`                            | Generation temperature          |

#### TTS (Text-to-Speech)

| Variable          | Default                  | Description                        |
| ----------------- | ------------------------ | ---------------------------------- |
| `TTS_MODEL_NAME`  | `kyutai/tts-1.6b-en_fr`  | HuggingFace model ID               |
| `TTS_SAMPLE_RATE` | `24000`                  | Output audio sample rate (Hz)      |
| `TTS_DEVICE`      | *(auto)*                 | `cuda:0`, `cpu`, or auto-detect    |

#### Model Loading

| Variable                          | Default    | Description                              |
| --------------------------------- | ---------- | ---------------------------------------- |
| `PRELOAD_MODELS`                  | `true`     | Load models on startup                   |
| `MODELS_CACHE_DIR`                | `./models` | Where to store downloaded models         |
| `ENABLE_MEMORY_EFFICIENT_LOADING` | `true`     | Load models sequentially to manage VRAM  |
| `TRUST_REMOTE_CODE`               | `true`     | Trust remote code in HF models           |

#### CORS

| Variable       | Default                | Description                              |
| -------------- | ---------------------- | ---------------------------------------- |
| `CORS_ORIGINS` | `http://localhost:3000`| Comma-separated allowed origins          |

---

## Model Options

### ASR Models

| Model ID                      | Size   | VRAM    | Quality    | Speed   | Notes                                          |
| ----------------------------- | ------ | ------- | ---------- | ------- | ---------------------------------------------- |
| `openai/whisper-medium`       | 769M   | ~1.5 GB | ★★★★☆     | Fast    | **Recommended.** Reliable, battle-tested.      |
| `openai/whisper-large-v3`     | 1.5B   | ~3 GB   | ★★★★★     | Medium  | Best accuracy, more VRAM.                      |
| `openai/whisper-small`        | 244M   | ~0.5 GB | ★★★☆☆     | Fastest | Lightweight, lower accuracy.                   |
| `kyutai/stt-2.6b-en-trfs`    | 2.6B   | ~6-8 GB | ★★★★★     | Medium  | Requires `transformers >= 4.53`. May segfault. |

### TTS Models

| Model                         | VRAM    | Quality    | Notes                                               |
| ----------------------------- | ------- | ---------- | --------------------------------------------------- |
| Coqui TTS (tacotron2-DDC)    | ~0.4 GB | ★★★☆☆     | **Recommended fallback.** `pip install TTS`          |
| edge-tts                      | 0 GB    | ★★★★☆     | Online only (Microsoft). `pip install edge-tts`      |
| `kyutai/tts-1.6b-en_fr`      | ~4-5 GB | ★★★★★     | Requires `transformers >= 4.54` (not yet released).  |

### Recommended Configurations

**Low VRAM (16 GB):**

```env
ASR_MODEL_NAME=openai/whisper-small
LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-AWQ
LLM_GPU_MEMORY_UTILIZATION=0.30
# Install: pip install edge-tts
```

**Standard (24 GB — RTX 4090):**

```env
ASR_MODEL_NAME=openai/whisper-medium
LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-AWQ
LLM_GPU_MEMORY_UTILIZATION=0.35
# Install: pip install TTS
```

**High VRAM (48 GB — A6000):**

```env
ASR_MODEL_NAME=kyutai/stt-2.6b-en-trfs
LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-AWQ
LLM_GPU_MEMORY_UTILIZATION=0.35
TTS_MODEL_NAME=kyutai/tts-1.6b-en_fr
```

---

## Verifying the Deployment

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Expected:

```json
{"status": "healthy", "service": "ai-talk-backend", "timestamp": "..."}
```

### 2. Readiness Check

```bash
curl http://localhost:8000/ready
```

Expected (all `true`):

```json
{"status": "ready", "services": {"asr": true, "llm": true, "tts": true}, "timestamp": "..."}
```

### 3. GPU Status

```bash
curl http://localhost:8000/gpu
```

### 4. WebSocket Test

Use a WebSocket client (e.g. [websocat](https://github.com/vi/websocat)):

```bash
# Install websocat
# apt-get install websocat  OR  cargo install websocat

websocat ws://localhost:8000/ws/conversation
```

You should receive:

```json
{"type": "connected", "session_id": "..."}
```

### 5. Full Pipeline Test

Send a ping:

```json
{"type": "ping"}
```

Expected response:

```json
{"type": "pong"}
```

---

## Running in Background

### Option A: nohup (simplest)

```bash
cd ~/ai-talk-backend
source venv/bin/activate
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# View logs
tail -f server.log

# Find and stop the process
ps aux | grep uvicorn
kill <PID>
```

### Option B: tmux / screen (recommended for Vast.ai)

```bash
# Start a tmux session
tmux new -s backend

# Inside tmux:
cd ~/ai-talk-backend
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Detach: press Ctrl+B, then D
# Re-attach later:
tmux attach -t backend
```

### Option C: systemd (production servers)

Create `/etc/systemd/system/ai-talk.service`:

```ini
[Unit]
Description=AI-Talk Backend
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/ai-talk-backend
Environment=PATH=/root/ai-talk-backend/venv/bin:/usr/local/bin:/usr/bin
ExecStart=/root/ai-talk-backend/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-talk
sudo systemctl start ai-talk

# Check status
sudo systemctl status ai-talk

# View logs
journalctl -u ai-talk -f
```

---

## Updating the Code

### Bare Metal

```bash
cd ~/ai-talk-backend
source venv/bin/activate

# Stop the server (Ctrl+C or kill the process)
# Then:
git pull origin main
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
cd ~/ai-talk-backend
git pull origin main
docker compose down
docker compose up -d --build
```

> **Note:** `--build` rebuilds the image with the new code. Downloaded models persist in the Docker volume and won't be re-downloaded.

---

## Troubleshooting

### Server won't start: "Address already in use"

The previous server process is still running.

```bash
# Find what's using port 8000
lsof -i :8000
# or
ss -tlnp | grep 8000

# Kill it
kill -9 <PID>

# Or kill all uvicorn processes
pkill -9 -f uvicorn
```

### CUDA Out of Memory

```bash
# Check GPU usage
nvidia-smi

# Kill zombie GPU processes
kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)

# Reduce LLM memory in .env
LLM_GPU_MEMORY_UTILIZATION=0.25
```

### Models not downloading / HuggingFace timeout

```bash
# Set HF cache directory explicitly
export HF_HOME=./models
export TRANSFORMERS_CACHE=./models

# If behind a firewall, try setting a mirror:
export HF_ENDPOINT=https://hf-mirror.com

# Manual download
python -c "
from huggingface_hub import snapshot_download
snapshot_download('openai/whisper-medium', cache_dir='./models')
snapshot_download('Qwen/Qwen2.5-7B-Instruct-AWQ', cache_dir='./models')
"
```

### ASR produces empty transcripts

1. **Check audio format:** The frontend must send audio over WebSocket. The backend auto-detects Opus/WebM encoded audio and decodes it via ffmpeg.

2. **Verify ffmpeg is installed:**
   ```bash
   which ffmpeg
   # If missing: apt-get install -y ffmpeg
   ```

3. **Check the audio diagnostic logs** — look for `[AUDIO DIAG]`:
   - `RMS < 10` → Silence. Mic not working or muted.
   - `RMS > 30000` → Noise/clipping. Wrong audio encoding.
   - `RMS 1000-8000` → Normal speech detected.

4. **Debug audio files** are saved to `/tmp/audio_debug/<session-id>.wav`. Download and play them:
   ```bash
   # From your local machine:
   scp -P <PORT> root@<VAST_IP>:/tmp/audio_debug/*.wav .
   ```

### Kyutai STT segfaults

The Kyutai STT model (`kyutai/stt-2.6b-en-trfs`) may segfault depending on the `transformers` version. **Switch to Whisper:**

```bash
# Edit .env
echo 'ASR_MODEL_NAME=openai/whisper-medium' >> .env

# Restart the server
```

### TTS produces silence

The Kyutai TTS model requires `transformers >= 4.54`. Install a fallback:

```bash
source venv/bin/activate

# Coqui TTS (offline)
pip install TTS

# OR edge-tts (online, needs internet)
pip install edge-tts

# Restart the server
```

### Docker: "could not select device driver nvidia"

NVIDIA Container Toolkit is not installed:

```bash
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

### Docker: "docker-compose: command not found"

Use the newer `docker compose` (with a space) instead:

```bash
# Old (v1, may not be installed):
docker-compose up -d

# New (v2, built into Docker):
docker compose up -d
```

### WebSocket disconnect error loop

If you see repeated errors like:

```
Cannot call "receive" once a disconnect message has been received
```

This is fixed in the current codebase. Pull the latest code and restart:

```bash
git pull origin main
# Then restart the server
```

### Checking What's Running

```bash
# Check server process
ps aux | grep uvicorn

# Check GPU processes
nvidia-smi

# Check Docker containers
docker ps

# Check port usage
ss -tlnp | grep 8000
```

---

## Architecture

```
Frontend (Browser)
    │
    │  WebSocket (wss://)
    │  Audio: Opus/WebM encoded
    │
    ▼
┌──────────────────────────────────────────────┐
│              AI-Talk Backend                  │
│                                              │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   │
│  │  ASR    │──▶│  LLM    │──▶│  TTS    │   │
│  │         │   │         │   │         │   │
│  │ Whisper │   │ Qwen2.5 │   │ Coqui   │   │
│  │ or      │   │ 7B AWQ  │   │ or      │   │
│  │ Kyutai  │   │ (vLLM)  │   │ Kyutai  │   │
│  └─────────┘   └─────────┘   └─────────┘   │
│                                              │
│  Audio → ffmpeg → PCM → ASR → text          │
│  text → LLM → response                      │
│  response → TTS → audio → client            │
└──────────────────────────────────────────────┘
```

### WebSocket Protocol

**Client → Server:**
- Binary data: Audio bytes (Opus/WebM encoded from browser `MediaRecorder`)
- JSON: `{"type": "end_conversation"}` — End and get analysis
- JSON: `{"type": "set_level", "level": "beginner|intermediate|advanced"}`
- JSON: `{"type": "ping"}`

**Server → Client:**
- JSON: `{"type": "connected", "session_id": "..."}`
- JSON: `{"type": "transcript", "text": "...", "is_final": true}`
- JSON: `{"type": "ai_text", "text": "...", "is_complete": false}` (streaming tokens)
- JSON: `{"type": "ai_audio_start", "sample_rate": 24000}`
- Binary: Audio PCM data
- JSON: `{"type": "turn_complete"}`
- JSON: `{"type": "analysis", "data": {...}}`
- JSON: `{"type": "error", "message": "..."}`
