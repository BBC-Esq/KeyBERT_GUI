from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from keybert import KeyBERT
from PySide6.QtCore import QThread, Signal
from sentence_transformers import SentenceTransformer


@dataclass(slots=True)
class KeywordParams:
    keyphrase_ngram_range: tuple[int, int] = (1, 2)
    stop_words: list[str] | None = None
    top_n: int = 5
    use_maxsum: bool = False
    use_mmr: bool = False
    diversity: float = 0.5
    nr_candidates: int = 20

class KeywordExtractorWorker(QThread):

    result = Signal(list)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        document: str,
        model: KeyBERT,
        params: dict,
    ) -> None:
        super().__init__()
        self.document = document
        self.model = model
        self.params = KeywordParams(**params)

    def run(self) -> None:
        try:
            kws = self._extract()
            self.result.emit(kws)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def _extract(self):
        p = self.params
        if p.use_maxsum:
            return self.model.extract_keywords(
                self.document,
                keyphrase_ngram_range=p.keyphrase_ngram_range,
                stop_words=p.stop_words,
                use_maxsum=True,
                nr_candidates=p.nr_candidates,
                top_n=p.top_n,
            )
        if p.use_mmr:
            return self.model.extract_keywords(
                self.document,
                keyphrase_ngram_range=p.keyphrase_ngram_range,
                stop_words=p.stop_words,
                use_mmr=True,
                diversity=p.diversity,
                top_n=p.top_n,
            )
        return self.model.extract_keywords(
            self.document,
            keyphrase_ngram_range=p.keyphrase_ngram_range,
            stop_words=p.stop_words,
            top_n=p.top_n,
        )

class KeyBERTProcessor:

    def __init__(self) -> None:
        self.default_model_name = "all-MiniLM-L12-v2"
        self._use_cuda = torch.cuda.is_available()
        self._device = "cuda" if self._use_cuda else "cpu"

        self._default_sentence_model: SentenceTransformer | None = None
        self._custom_model: SentenceTransformer | None = None
        self._custom_model_name: str | None = None
        self._kw_model: KeyBERT | None = None

        self._load_default_model()
        self._update_kw_model(self._default_sentence_model)

    def get_keybert_model(self) -> KeyBERT:
        if self._kw_model is None:
            raise RuntimeError("KeyBERT model not initialized")
        return self._kw_model

    def load_custom_model(self, model_path: str | os.PathLike) -> None:
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise ValueError("Model path is not a directory")

        if not any(model_path.glob(name) for name in ("config.json", "config_sentence_transformers.json")):
            raise ValueError("No Sentence-Transformers config found in the directory")

        self._custom_model = SentenceTransformer(
            str(model_path), device=self._device, trust_remote_code=True
        )
        self._custom_model_name = model_path.name
        self._update_kw_model(self._custom_model)

    def update_keybert_model(self, use_default_keybert: bool) -> None:
        if use_default_keybert:
            self._kw_model = KeyBERT()
        else:
            self._update_kw_model(
                self._custom_model or self._default_sentence_model
            )

    def has_custom_model(self) -> bool:
        return self._custom_model is not None

    def get_custom_model_name(self) -> str | None:
        return self._custom_model_name

    def get_default_model_name(self) -> str:
        return self.default_model_name

    def _load_default_model(self) -> None:
        self._default_sentence_model = SentenceTransformer(
            self.default_model_name, device=self._device, trust_remote_code=True
        )

    def _update_kw_model(self, model: SentenceTransformer | None) -> None:
        self._kw_model = KeyBERT(model=model)
