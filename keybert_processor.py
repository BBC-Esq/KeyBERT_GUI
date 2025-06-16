import os
import torch
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


class KeywordExtractorWorker(QThread):
    result = Signal(list)
    error = Signal(str)
    finished = Signal()

    def __init__(self, doc, kw_model, params):
        super().__init__()
        self.doc = doc
        self.kw_model = kw_model
        self.params = params

    def run(self):
        try:
            if self.params['use_maxsum']:
                keywords = self.kw_model.extract_keywords(
                    self.doc,
                    keyphrase_ngram_range=self.params['keyphrase_ngram_range'],
                    stop_words=self.params['stop_words'],
                    use_maxsum=True,
                    nr_candidates=self.params['nr_candidates'],
                    top_n=self.params['top_n']
                )
            elif self.params['use_mmr']:
                keywords = self.kw_model.extract_keywords(
                    self.doc,
                    keyphrase_ngram_range=self.params['keyphrase_ngram_range'],
                    stop_words=self.params['stop_words'],
                    use_mmr=True,
                    diversity=self.params['diversity'],
                    top_n=self.params['top_n']
                )
            else:
                keywords = self.kw_model.extract_keywords(
                    self.doc,
                    keyphrase_ngram_range=self.params['keyphrase_ngram_range'],
                    stop_words=self.params['stop_words'],
                    top_n=self.params['top_n']
                )
            self.result.emit(keywords)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class KeyBERTProcessor:
    def __init__(self):
        self.default_model_name = 'all-MiniLM-L12-v2'
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'

        self.default_sentence_model = None
        self.custom_model = None
        self.custom_model_name = None
        self.kw_model = None

        self._load_default_model()
        self._initialize_keybert_model()

    def _load_default_model(self):
        try:
            self.default_sentence_model = SentenceTransformer(
                self.default_model_name,
                device=self.device,
                trust_remote_code=True,
            )
        except Exception as e:
            raise Exception(f"Failed to load the default model: {str(e)}")

    def _initialize_keybert_model(self):
        self.kw_model = KeyBERT(model=self.default_sentence_model)

    def load_custom_model(self, model_path):
        if not os.path.isdir(model_path):
            raise ValueError("Model path is not a directory")
        try:
            self.custom_model = SentenceTransformer(
                model_path,
                device=self.device,
                trust_remote_code=True,
            )
            self.custom_model_name = os.path.basename(model_path)
            self.kw_model = KeyBERT(model=self.custom_model)
        except Exception as e:
            raise Exception(f"Failed to load custom model: {str(e)}")

    def update_keybert_model(self, use_default_keybert=False):
        if use_default_keybert:
            self.kw_model = KeyBERT()
        else:
            if self.custom_model:
                self.kw_model = KeyBERT(model=self.custom_model)
            else:
                self.kw_model = KeyBERT(model=self.default_sentence_model)

    def get_keybert_model(self):
        return self.kw_model

    def has_custom_model(self):
        return self.custom_model is not None

    def get_custom_model_name(self):
        return self.custom_model_name

    def get_default_model_name(self):
        return self.default_model_name

    def get_device_info(self):
        return {
            'device': self.device,
            'cuda_available': self.use_cuda,
            'torch_version': torch.__version__
        }

    def extract_keywords_sync(self, text, **params):
        try:
            default_params = {
                'keyphrase_ngram_range': (1, 2),
                'stop_words': None,
                'top_n': 5,
                'use_maxsum': False,
                'use_mmr': False,
                'diversity': 0.5,
                'nr_candidates': 20
            }
            default_params.update(params)

            if default_params['use_maxsum']:
                return self.kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=default_params['keyphrase_ngram_range'],
                    stop_words=default_params['stop_words'],
                    use_maxsum=True,
                    nr_candidates=default_params['nr_candidates'],
                    top_n=default_params['top_n']
                )
            elif default_params['use_mmr']:
                return self.kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=default_params['keyphrase_ngram_range'],
                    stop_words=default_params['stop_words'],
                    use_mmr=True,
                    diversity=default_params['diversity'],
                    top_n=default_params['top_n']
                )
            else:
                return self.kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=default_params['keyphrase_ngram_range'],
                    stop_words=default_params['stop_words'],
                    top_n=default_params['top_n']
                )
        except Exception as e:
            raise Exception(f"Keyword extraction failed: {str(e)}")

    def batch_extract_keywords(self, texts, **params):
        results = []
        for text in texts:
            try:
                keywords = self.extract_keywords_sync(text, **params)
                results.append(keywords)
            except Exception as e:
                results.append([])
        return results

    def get_model_info(self):
        info = {
            'default_model': self.default_model_name,
            'custom_model': self.custom_model_name if self.custom_model else None,
            'current_model_type': None,
            'device': self.device,
            'cuda_available': self.use_cuda
        }

        if hasattr(self.kw_model, 'model') and self.kw_model.model is not None:
            if self.custom_model and self.kw_model.model == self.custom_model:
                info['current_model_type'] = 'custom'
            elif self.kw_model.model == self.default_sentence_model:
                info['current_model_type'] = 'default_sentence_transformer'
            else:
                info['current_model_type'] = 'unknown_sentence_transformer'
        else:
            info['current_model_type'] = 'default_keybert'
        
        return info

    def validate_extraction_params(self, params):
        errors = []

        if 'keyphrase_ngram_range' in params:
            min_n, max_n = params['keyphrase_ngram_range']
            if min_n < 1 or max_n < 1 or min_n > max_n:
                errors.append("Invalid ngram range: min and max must be >= 1 and min <= max")

        if 'top_n' in params and params['top_n'] < 1:
            errors.append("top_n must be >= 1")

        if params.get('use_mmr') and 'diversity' in params:
            diversity = params['diversity']
            if not (0 <= diversity <= 1):
                errors.append("diversity must be between 0 and 1")

        if params.get('use_maxsum') and 'nr_candidates' in params:
            nr_candidates = params['nr_candidates']
            top_n = params.get('top_n', 5)
            if nr_candidates < top_n:
                errors.append("nr_candidates must be >= top_n")
        
        return errors

    def get_available_models(self):

        return [
            'all-MiniLM-L12-v2',
            'all-MiniLM-L6-v2',
            'all-mpnet-base-v2',
            'paraphrase-MiniLM-L6-v2',
            'paraphrase-multilingual-MiniLM-L12-v2'
        ]

    def switch_default_model(self, model_name):
        try:
            new_model = SentenceTransformer(
                model_name,
                device=self.device,
                trust_remote_code=True,
            )
            self.default_sentence_model = new_model
            self.default_model_name = model_name

            if not self.custom_model:
                self.kw_model = KeyBERT(model=self.default_sentence_model)

        except Exception as e:
            raise Exception(f"Failed to switch to model {model_name}: {str(e)}")

    def clear_custom_model(self):
        self.custom_model = None
        self.custom_model_name = None
        self.kw_model = KeyBERT(model=self.default_sentence_model)

    def get_memory_usage(self):
        if torch.cuda.is_available():
            return {
                'gpu_memory_allocated': torch.cuda.memory_allocated(),
                'gpu_memory_reserved': torch.cuda.memory_reserved(),
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024**2
            }
        return {'message': 'CUDA not available - memory info not accessible'}