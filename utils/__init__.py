"""
Utils package for KeyBERT GUI application.
"""

from .core import (
    is_nvidia_gpu_available,
    set_cuda_paths,
    validate_keyword_params,
    validate_batch_params,
)

from .file_scanner import (
    scan_directory_for_documents,
    get_file_counts_by_type,
    estimate_processing_time,
    SUPPORTED_EXTENSIONS,
)

__all__ = [
    "is_nvidia_gpu_available",
    "set_cuda_paths", 
    "validate_keyword_params",
    "validate_batch_params",
    "scan_directory_for_documents",
    "get_file_counts_by_type",
    "estimate_processing_time",
    "SUPPORTED_EXTENSIONS",
]