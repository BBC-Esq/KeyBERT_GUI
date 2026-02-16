from __future__ import annotations

import os
import sys
import json
import subprocess
from pathlib import Path

def is_nvidia_gpu_available() -> bool:
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
    os.environ.setdefault("CUDA_PATH", str(nvidia_root / "cuda_runtime"))


def validate_keyword_params(p: dict) -> list[str]:
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


def validate_batch_params(directory: str, output_path: str) -> list[str]:
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

        output_dir = output_file.parent
        if not output_dir.exists():
            errors.append(f"Output directory does not exist: {output_dir}")

    return errors


class SettingsManager:

    def __init__(self, app_name: str = "KeyBERT_GUI"):
        self.app_name = app_name
        self.settings_file = self._get_settings_path()
        self.default_settings = {
            "window_geometry": {
                "width": 900,
                "height": 800,
                "x": None,
                "y": None
            },
            "extraction_params": {
                "ngram_min": 1,
                "ngram_max": 2,
                "stop_words": "",
                "diversification": "None",
                "diversity": 0.5,
                "candidates": 20,
                "top_n": 5,
                "use_default_keybert": False
            },
            "paths": {
                "last_file_dir": "",
                "last_batch_dir": "",
                "last_batch_output": "",
                "custom_model_path": ""
            }
        }

    def _get_settings_path(self) -> Path:
        if sys.platform == "win32":
            base_dir = Path(os.environ.get("APPDATA", Path.home()))
        elif sys.platform == "darwin":
            base_dir = Path.home() / "Library" / "Application Support"
        else:
            base_dir = Path.home() / ".config"

        app_dir = base_dir / self.app_name
        app_dir.mkdir(parents=True, exist_ok=True)
        return app_dir / "settings.json"

    def load_settings(self) -> dict:
        if not self.settings_file.exists():
            return self.default_settings.copy()

        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                settings = self.default_settings.copy()
                self._deep_update(settings, loaded)
                return settings
        except Exception:
            return self.default_settings.copy()

    def save_settings(self, settings: dict) -> bool:
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def _deep_update(self, base: dict, updates: dict) -> None:
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value