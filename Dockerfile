# AI-Talk Backend Dockerfile
# Multi-stage build for optimized image size

# ===========================================
# Stage 1: Base image with CUDA support
# ===========================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# ===========================================
# Stage 2: Builder stage for dependencies
# ===========================================
FROM base AS builder

WORKDIR /build

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support first
RUN pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM (requires specific installation)
RUN pip install vllm==0.6.6.post1

# Install remaining dependencies
RUN pip install -r requirements.txt

# Install Moshi from GitHub (if available)
RUN pip install git+https://github.com/kyutai-labs/moshi.git || echo "Moshi installation skipped - will use fallback"

# ===========================================
# Stage 3: Production image
# ===========================================
FROM base AS production

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r aiuser && useradd -r -g aiuser aiuser

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/data \
    && chown -R aiuser:aiuser /app

# Copy application code
COPY --chown=aiuser:aiuser ./app /app/app
COPY --chown=aiuser:aiuser ./env.example /app/.env.example

# Set environment variables
ENV HOST=0.0.0.0 \
    PORT=8000 \
    ENVIRONMENT=production \
    MODELS_CACHE_DIR=/app/models \
    LOG_LEVEL=INFO \
    PRELOAD_MODELS=true

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER aiuser

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
