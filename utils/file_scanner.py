from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def scan_directory_for_documents(directory: str | Path) -> list[Path]:
    dir_path = Path(directory)

    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    found_files = []

    for file_path in dir_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            found_files.append(file_path)

    return sorted(found_files)


def get_file_counts_by_type(files: list[Path]) -> dict[str, int]:
    counts = {".txt": 0, ".pdf": 0, ".docx": 0}

    for file_path in files:
        ext = file_path.suffix.lower()
        if ext in counts:
            counts[ext] += 1

    return counts


def estimate_processing_time(file_count: int) -> str:
    if file_count == 0:
        return "No files to process"
    elif file_count <= 5:
        return "Less than 1 minute"
    elif file_count <= 20:
        return "1-5 minutes"
    elif file_count <= 50:
        return "5-15 minutes"
    else:
        return "15+ minutes"