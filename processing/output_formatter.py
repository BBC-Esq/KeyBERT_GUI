from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class BatchResultFormatter:

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
        self.metadata["processing_parameters"] = params.copy()

    def save_to_file(self, output_path: str | Path) -> None:
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
        return {
            "total_files": self.metadata["total_files_processed"],
            "successful": self.metadata["successful_extractions"],
            "failed": self.metadata["failed_extractions"],
            "success_rate": (
                self.metadata["successful_extractions"] / self.metadata["total_files_processed"] * 100
                if self.metadata["total_files_processed"] > 0 else 0
            )
        }


class SingleResultFormatter:

    @staticmethod
    def format_as_json(
        keywords: list[tuple[str, float]],
        source_file: str | None = None,
        source_text: str | None = None,
        parameters: dict | None = None
    ) -> str:
        result = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "source_file": source_file,
                "processing_parameters": parameters or {}
            },
            "keywords": [
                {"phrase": kw, "score": score} 
                for kw, score in keywords
            ]
        }

        if source_text:
            result["source_text"] = source_text
        
        return json.dumps(result, indent=2, ensure_ascii=False)

    @staticmethod
    def format_as_csv(keywords: list[tuple[str, float]]) -> str:
        lines = ["Keyword,Score"]
        for kw, score in keywords:
            kw_escaped = kw.replace('"', '""')
            lines.append(f'"{kw_escaped}",{score:.6f}')
        return "\n".join(lines)

    @staticmethod
    def save_to_file(
        output_path: str | Path,
        keywords: list[tuple[str, float]],
        source_file: str | None = None,
        source_text: str | None = None,
        parameters: dict | None = None
    ) -> None:
        output_path = Path(output_path)
        ext = output_path.suffix.lower()

        try:
            if ext == '.json':
                content = SingleResultFormatter.format_as_json(
                    keywords, source_file, source_text, parameters
                )
            elif ext == '.csv':
                content = SingleResultFormatter.format_as_csv(keywords)
            else:
                raise ValueError(f"Unsupported format: {ext}")

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            raise IOError(f"Failed to save results to {output_path}: {e}")