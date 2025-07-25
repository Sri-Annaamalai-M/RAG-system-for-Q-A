import pytest
import os
from src.document_processor import DocumentProcessor

@pytest.fixture
def processor():
    return DocumentProcessor('config.yaml')

def test_load_pdf(processor):
    # Create a dummy PDF for testing (requires actual PDF for real test)
    test_pdf = os.path.join(processor.raw_dir, "test.pdf")
    if not os.path.exists(test_pdf):
        pytest.skip("No test PDF available")
    text = processor.load_pdf(test_pdf)
    assert isinstance(text, str)
    assert len(text) > 0

def test_chunk_text(processor):
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks, metadata = processor.chunk_text(text, "Test Paper", max_sentences=2)
    assert len(chunks) == 2
    assert len(metadata) == 2
    assert metadata[0]['paper_title'] == "Test Paper"
    assert "This is sentence one. This is sentence two." in chunks[0]