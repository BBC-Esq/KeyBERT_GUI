import pytest

from processing.keybert import KeyBERTProcessor, KeywordParams
from processing.extraction import TextExtractorWorker

# KeyBERTProcessor – smoke test (CPU only)
def test_default_model_loads():
    kp = KeyBERTProcessor()
    assert kp.get_keybert_model() is not None

# TextExtractorWorker helper (runs synchronously for the test)
def test_extract_txt(tmp_path):
    sample = tmp_path / "hello.txt"
    sample.write_text("hello world")
    worker = TextExtractorWorker(sample)
    assert worker._extract_from_file(sample) == "hello world"
