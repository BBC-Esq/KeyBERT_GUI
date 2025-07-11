"""
Core utility functions that do not depend on the GUI.
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

# CUDA helpers
def is_nvidia_gpu_available() -> bool:
    """Return True iff `nvidia-smi` is callable."""
    try:
        subprocess.run(
            ["nvidia-smi"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def set_cuda_paths() -> None:
    """Pre-pend NVIDIA DLL paths shipped with `nvidia` wheels (Windows-only)."""
    venv_base = Path(os.environ.get("VIRTUAL_ENV", Path(sys.executable).parent.parent))
    nvidia_root = venv_base / "Lib" / "site-packages" / "nvidia"
    bin_paths = [
        nvidia_root / "cuda_runtime" / "bin",
        nvidia_root / "cublas" / "bin",
        nvidia_root / "cudnn" / "bin",
        nvidia_root / "cuda_nvrtc" / "bin",
        nvidia_root / "cuda_nvcc" / "bin",
    ]
    os.environ["PATH"] = os.pathsep.join(str(p) for p in bin_paths) + os.pathsep + os.environ.get("PATH", "")
    # CUDA_PATH is optional but helps certain libraries
    os.environ.setdefault("CUDA_PATH", str(nvidia_root / "cuda_runtime"))

# Parameter validation
def validate_keyword_params(p: dict) -> list[str]:
    """Return a list of human-readable error messages (empty if OK)."""
    errors: list[str] = []

    min_n, max_n = p["keyphrase_ngram_range"]
    if min_n < 1 or max_n < 1 or min_n > max_n:
        errors.append("Invalid n-gram range")

    if p["top_n"] < 1:
        errors.append("Top N must be ≥ 1")

    if p.get("use_mmr"):
        div = p["diversity"]
        if not (0.0 <= div <= 1.0):
            errors.append("Diversity must be between 0 and 1")

    if p.get("use_maxsum"):
        if p["nr_candidates"] < p["top_n"]:
            errors.append("nr_candidates must be ≥ Top N")

    return errors

# Batch processing validation
def validate_batch_params(directory: str, output_path: str) -> list[str]:
    """Validate batch processing parameters."""
    errors: list[str] = []
    
    if not directory.strip():
        errors.append("Please select a directory to process")
    elif not Path(directory).exists():
        errors.append("Selected directory does not exist")
    elif not Path(directory).is_dir():
        errors.append("Selected path is not a directory")
    
    if not output_path.strip():
        errors.append("Please specify an output file path")
    else:
        output_file = Path(output_path)
        if not output_file.suffix.lower() == '.json':
            errors.append("Output file must have .json extension")
        
        # Check if output directory exists
        output_dir = output_file.parent
        if not output_dir.exists():
            errors.append(f"Output directory does not exist: {output_dir}")
    
    return errors