"""
JSON output formatting utilities for batch processing results.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class BatchResultFormatter:
    """Handles formatting and saving batch processing results to JSON."""
    
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []
        self.metadata = {
            "processing_timestamp": datetime.now().isoformat(),
            "total_files_processed": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "processing_parameters": {}
        }
    
    def add_result(
        self,
        file_path: Path,
        extracted_text: str | None = None,
        keywords: list[tuple[str, float]] | None = None,
        error: str | None = None
    ) -> None:
        """
        Add a processing result for a single file.
        
        Args:
            file_path: Path to the processed file
            extracted_text: Extracted text content (None if extraction failed)
            keywords: List of (keyword, score) tuples (None if extraction failed)
            error: Error message if processing failed
        """
        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_extension": file_path.suffix.lower(),
            "success": error is None,
            "extracted_text": extracted_text,
            "keywords": [{"phrase": kw, "score": score} for kw, score in keywords] if keywords else [],
            "error": error,
            "processed_at": datetime.now().isoformat()
        }
        
        self.results.append(result)
        self.metadata["total_files_processed"] += 1
        
        if error is None:
            self.metadata["successful_extractions"] += 1
        else:
            self.metadata["failed_extractions"] += 1
    
    def set_processing_parameters(self, params: dict[str, Any]) -> None:
        """Store the processing parameters used."""
        self.metadata["processing_parameters"] = params.copy()
    
    def save_to_file(self, output_path: str | Path) -> None:
        """
        Save results to JSON file.
        
        Args:
            output_path: Path where to save the JSON file
            
        Raises:
            IOError: If file cannot be written
        """
        output_data = {
            "metadata": self.metadata,
            "results": self.results
        }
        
        output_path = Path(output_path)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise IOError(f"Failed to save results to {output_path}: {e}")
    
    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics for the batch processing."""
        return {
            "total_files": self.metadata["total_files_processed"],
            "successful": self.metadata["successful_extractions"],
            "failed": self.metadata["failed_extractions"],
            "success_rate": (
                self.metadata["successful_extractions"] / self.metadata["total_files_processed"] * 100
                if self.metadata["total_files_processed"] > 0 else 0
            )
        }