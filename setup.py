#!/usr/bin/env python3
"""
AI-Talk Backend Setup Script

This script helps set up the AI-Talk backend by:
1. Validating system requirements (Python, CUDA, GPU memory)
2. Downloading required models from Hugging Face
3. Verifying model files
4. Testing basic model loading

Usage:
    python setup.py              # Run all checks and download models
    python setup.py --check      # Only check requirements
    python setup.py --download   # Only download models
    python setup.py --verify     # Verify downloaded models
    python setup.py --test       # Test model loading

Requirements:
    - Python 3.11+
    - NVIDIA GPU with 24GB+ VRAM
    - CUDA 12.1+
    - ~30GB disk space for models
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Model configurations
MODELS = {
    "stt": {
        "name": "kyutai/stt-2.6b-en",
        "description": "Kyutai STT 2.6B English (Speech-to-Text)",
        "vram_gb": 7.0,
        "disk_gb": 5.5,
    },
    "tts": {
        "name": "kyutai/tts-1.6b-en_fr",
        "description": "Kyutai TTS 1.8B English/French (Text-to-Speech)",
        "vram_gb": 4.5,
        "disk_gb": 4.0,
    },
    "llm": {
        "name": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "description": "Qwen 2.5 7B Instruct AWQ (4-bit quantized LLM)",
        "vram_gb": 5.0,
        "disk_gb": 4.5,
    },
    "tts_voices": {
        "name": "kyutai/tts-voices",
        "description": "Kyutai TTS Voice Embeddings",
        "vram_gb": 0.0,
        "disk_gb": 0.5,
    },
}

# Minimum requirements
MIN_PYTHON_VERSION = (3, 11)
MIN_CUDA_VERSION = (12, 1)
MIN_GPU_MEMORY_GB = 20
RECOMMENDED_GPU_MEMORY_GB = 24


class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls):
        """Disable colors for non-TTY outputs."""
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = ""
        cls.MAGENTA = cls.CYAN = cls.WHITE = cls.BOLD = cls.RESET = ""


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")


def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_warning(text: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def print_error(text: str):
    """Print an error message."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_info(text: str):
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version meets requirements."""
    version = sys.version_info[:2]
    version_str = f"{version[0]}.{version[1]}"

    if version >= MIN_PYTHON_VERSION:
        return True, f"Python {version_str}"
    else:
        return (
            False,
            f"Python {version_str} (required: {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+)",
        )


def check_cuda_available() -> Tuple[bool, str]:
    """Check if CUDA is available."""
    try:
        import torch

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            return True, f"CUDA {cuda_version}"
        else:
            return False, "CUDA not available"
    except ImportError:
        return False, "PyTorch not installed"


def check_cuda_version() -> Tuple[bool, str]:
    """Check if CUDA version meets requirements."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA not available"

        cuda_version = torch.version.cuda
        if cuda_version:
            major, minor = map(int, cuda_version.split(".")[:2])
            if (major, minor) >= MIN_CUDA_VERSION:
                return True, f"CUDA {cuda_version}"
            else:
                return (
                    False,
                    f"CUDA {cuda_version} (required: {MIN_CUDA_VERSION[0]}.{MIN_CUDA_VERSION[1]}+)",
                )
        return False, "Could not determine CUDA version"
    except ImportError:
        return False, "PyTorch not installed"


def check_gpu_memory() -> Tuple[bool, str]:
    """Check if GPU has enough memory."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA not available"

        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        if total_memory >= RECOMMENDED_GPU_MEMORY_GB:
            return True, f"{gpu_name} ({total_memory:.1f}GB)"
        elif total_memory >= MIN_GPU_MEMORY_GB:
            return (
                True,
                f"{gpu_name} ({total_memory:.1f}GB) - Warning: {RECOMMENDED_GPU_MEMORY_GB}GB+ recommended",
            )
        else:
            return (
                False,
                f"{gpu_name} ({total_memory:.1f}GB) - Minimum {MIN_GPU_MEMORY_GB}GB required",
            )
    except ImportError:
        return False, "PyTorch not installed"


def check_disk_space(path: str = ".") -> Tuple[bool, str]:
    """Check if there's enough disk space for models."""
    total_required = sum(m["disk_gb"] for m in MODELS.values())

    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)

        if free_gb >= total_required * 1.5:  # 50% buffer
            return True, f"{free_gb:.1f}GB free (need ~{total_required:.1f}GB)"
        elif free_gb >= total_required:
            return (
                True,
                f"{free_gb:.1f}GB free (need ~{total_required:.1f}GB) - Low space warning",
            )
        else:
            return (
                False,
                f"{free_gb:.1f}GB free (need ~{total_required:.1f}GB)",
            )
    except Exception as e:
        return False, f"Could not check disk space: {e}"


def check_package_installed(package: str) -> bool:
    """Check if a Python package is installed."""
    try:
        __import__(package)
        return True
    except ImportError:
        return False


def check_required_packages() -> Dict[str, bool]:
    """Check if required packages are installed."""
    packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "vllm": "vLLM",
        "huggingface_hub": "Hugging Face Hub",
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "soundfile": "SoundFile",
        "librosa": "Librosa",
    }

    results = {}
    for package, name in packages.items():
        results[name] = check_package_installed(package)

    return results


def get_model_cache_dir() -> Path:
    """Get the Hugging Face cache directory."""
    # Check environment variable first
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")

    if cache_dir:
        return Path(cache_dir)

    # Default location
    if platform.system() == "Windows":
        return Path.home() / ".cache" / "huggingface" / "hub"
    else:
        return Path.home() / ".cache" / "huggingface" / "hub"


def check_model_downloaded(model_id: str) -> bool:
    """Check if a model is already downloaded."""
    cache_dir = get_model_cache_dir()

    # Convert model ID to cache folder name
    model_folder = "models--" + model_id.replace("/", "--")
    model_path = cache_dir / model_folder

    if model_path.exists():
        # Check if there are any snapshot folders
        snapshots_dir = model_path / "snapshots"
        if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
            return True

    return False


def download_model(model_id: str, description: str) -> bool:
    """Download a model from Hugging Face."""
    print_info(f"Downloading {description}...")
    print(f"  Model: {model_id}")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            model_id,
            resume_download=True,
            local_files_only=False,
        )
        print_success(f"Downloaded {model_id}")
        return True

    except Exception as e:
        print_error(f"Failed to download {model_id}: {e}")
        return False


def run_system_checks() -> bool:
    """Run all system requirement checks."""
    print_header("System Requirements Check")

    all_passed = True
    warnings = []

    # Python version
    passed, msg = check_python_version()
    if passed:
        print_success(msg)
    else:
        print_error(msg)
        all_passed = False

    # CUDA availability
    passed, msg = check_cuda_available()
    if passed:
        print_success(msg)
    else:
        print_warning(msg)
        warnings.append("CUDA not available - models will run on CPU (very slow)")

    # CUDA version
    passed, msg = check_cuda_version()
    if passed:
        print_success(msg)
    elif "not available" not in msg.lower():
        print_warning(msg)
        warnings.append(msg)

    # GPU memory
    passed, msg = check_gpu_memory()
    if passed:
        if "Warning" in msg:
            print_warning(msg)
            warnings.append("GPU memory is below recommended 24GB")
        else:
            print_success(msg)
    else:
        print_error(msg)
        all_passed = False

    # Disk space
    passed, msg = check_disk_space()
    if passed:
        if "Low space" in msg:
            print_warning(msg)
            warnings.append("Low disk space")
        else:
            print_success(msg)
    else:
        print_error(msg)
        all_passed = False

    # Required packages
    print("\nRequired packages:")
    packages = check_required_packages()
    for name, installed in packages.items():
        if installed:
            print_success(f"  {name}")
        else:
            print_warning(f"  {name} - not installed")
            if name in ["PyTorch", "vLLM"]:
                warnings.append(f"{name} not installed")

    # Summary
    print()
    if all_passed:
        if warnings:
            print_warning("System check passed with warnings:")
            for w in warnings:
                print(f"  • {w}")
        else:
            print_success("All system requirements met!")
    else:
        print_error("System requirements check failed!")
        print("Please address the issues above before continuing.")

    return all_passed


def run_model_download() -> bool:
    """Download all required models."""
    print_header("Model Download")

    # Check HuggingFace Hub
    if not check_package_installed("huggingface_hub"):
        print_error("huggingface_hub not installed!")
        print("Run: pip install huggingface-hub")
        return False

    print(f"Cache directory: {get_model_cache_dir()}\n")

    # Calculate total VRAM needed
    total_vram = sum(m["vram_gb"] for m in MODELS.values())
    total_disk = sum(m["disk_gb"] for m in MODELS.values())
    print(f"Total VRAM required: ~{total_vram:.1f}GB")
    print(f"Total disk space required: ~{total_disk:.1f}GB\n")

    all_success = True

    for key, model in MODELS.items():
        model_id = model["name"]
        description = model["description"]

        # Check if already downloaded
        if check_model_downloaded(model_id):
            print_success(f"{description} - already downloaded")
            continue

        # Download
        if not download_model(model_id, description):
            all_success = False

    print()
    if all_success:
        print_success("All models downloaded successfully!")
    else:
        print_error("Some models failed to download. Please try again.")

    return all_success


def verify_models() -> bool:
    """Verify all models are downloaded and accessible."""
    print_header("Model Verification")

    all_verified = True

    for key, model in MODELS.items():
        model_id = model["name"]
        description = model["description"]

        if check_model_downloaded(model_id):
            print_success(f"{description}")
        else:
            print_error(f"{description} - not found")
            all_verified = False

    print()
    if all_verified:
        print_success("All models verified!")
    else:
        print_error("Some models are missing. Run: python setup.py --download")

    return all_verified


def test_model_loading() -> bool:
    """Test loading a model to verify everything works."""
    print_header("Model Loading Test")

    print_info("Testing basic model loading (this may take a few minutes)...")

    # Test PyTorch and CUDA
    print("\n1. Testing PyTorch and CUDA...")
    try:
        import torch

        if torch.cuda.is_available():
            # Simple CUDA test
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            y = x * 2
            assert y.sum().item() == 12.0
            print_success("PyTorch CUDA working")

            # Print GPU info
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"   Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB"
            )
        else:
            print_warning("CUDA not available, using CPU")

    except Exception as e:
        print_error(f"PyTorch test failed: {e}")
        return False

    # Test Transformers
    print("\n2. Testing Transformers library...")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct-AWQ", trust_remote_code=True
        )
        test_text = "Hello, this is a test."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print_success("Transformers working")

    except Exception as e:
        print_error(f"Transformers test failed: {e}")
        return False

    # Test vLLM (optional)
    print("\n3. Testing vLLM (optional)...")
    try:
        import vllm

        print_success(f"vLLM {vllm.__version__} available")
    except ImportError:
        print_warning("vLLM not installed - will use transformers fallback")
    except Exception as e:
        print_warning(f"vLLM check failed: {e}")

    # Test Moshi (optional)
    print("\n4. Testing Moshi library (optional)...")
    try:
        import moshi

        print_success("Moshi library available")
    except ImportError:
        print_warning("Moshi not installed - will use fallback ASR/TTS")
    except Exception as e:
        print_warning(f"Moshi check failed: {e}")

    print()
    print_success("Basic tests passed!")
    print_info("For full model loading test, run: python -m app.main")

    return True


def print_setup_instructions():
    """Print manual setup instructions."""
    print_header("Setup Instructions")

    print(
        """
1. Install PyTorch with CUDA:
   pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

2. Install vLLM:
   pip install vllm==0.6.6.post1

3. Install Moshi (for Kyutai STT/TTS):
   pip install git+https://github.com/kyutai-labs/moshi.git

4. Install other requirements:
   pip install -r requirements.txt

5. Download models:
   python setup.py --download

6. Create .env file:
   cp env.example .env

7. Run the server:
   python -m app.main
"""
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI-Talk Backend Setup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py              Run all checks and download models
  python setup.py --check      Only check system requirements
  python setup.py --download   Download required models
  python setup.py --verify     Verify downloaded models
  python setup.py --test       Test model loading
  python setup.py --help       Show this help message
        """,
    )

    parser.add_argument(
        "--check", action="store_true", help="Only check system requirements"
    )
    parser.add_argument(
        "--download", action="store_true", help="Download required models"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify downloaded models"
    )
    parser.add_argument("--test", action="store_true", help="Test model loading")
    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )

    args = parser.parse_args()

    # Disable colors if requested or not a TTY
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    print_header("AI-Talk Backend Setup")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")

    # Determine what to run
    run_all = not any([args.check, args.download, args.verify, args.test])

    success = True

    if args.check or run_all:
        if not run_system_checks():
            success = False
            if run_all:
                print_warning("Continuing despite check failures...\n")

    if args.download or run_all:
        if not run_model_download():
            success = False

    if args.verify or run_all:
        if not verify_models():
            success = False

    if args.test or run_all:
        if not test_model_loading():
            success = False

    # Final summary
    print_header("Setup Summary")

    if success:
        print_success("Setup completed successfully!")
        print_info("You can now run: python -m app.main")
    else:
        print_warning("Setup completed with some issues.")
        print_info("Review the warnings above and address any critical issues.")

    print_setup_instructions()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
