import os
import json
import PyPDF2
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import yaml

# Ensure 'punkt' and 'punkt_tab' are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', force=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', force=True)

class DocumentProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.raw_dir = self.config['data']['raw_dir']
        self.chunks_dir = self.config['data']['chunks_dir']
        self.embeddings_dir = self.config['data']['embeddings_dir']
        self.embedding_model = SentenceTransformer(self.config['models']['embedding_model'])
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)

    def load_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    def chunk_text(self, text, paper_title, max_sentences=5):
        # Always use English for sentence tokenization
        sentences = nltk.sent_tokenize(text, language='english')
        chunks = []
        metadata = []
        for i in range(0, len(sentences), max_sentences):
            chunk = " ".join(sentences[i:i + max_sentences])
            chunks.append(chunk)
            metadata.append({
                'paper_title': paper_title,
                'chunk_id': f"{paper_title}_{i//max_sentences}",
                'page_number': None  # PDF page extraction can be added if needed
            })
        return chunks, metadata

    def process_pdfs(self):
        all_chunks = []
        all_metadata = []
        for pdf_file in os.listdir(self.raw_dir):
            if pdf_file.endswith('.pdf'):
                paper_title = pdf_file.replace('.pdf', '')
                pdf_path = os.path.join(self.raw_dir, pdf_file)
                text = self.load_pdf(pdf_path)
                chunks, metadata = self.chunk_text(text, paper_title)
                all_chunks.extend(chunks)
                all_metadata.extend(metadata)
        
        # Save chunks and metadata with UTF-8 encoding
        for i, (chunk, meta) in enumerate(zip(all_chunks, all_metadata)):
            with open(os.path.join(self.chunks_dir, f"chunk_{i}.txt"), 'w', encoding='utf-8') as f:
                f.write(chunk)
            with open(os.path.join(self.chunks_dir, f"chunk_{i}_meta.json"), 'w', encoding='utf-8') as f:
                json.dump(meta, f)
        
        return all_chunks, all_metadata

    def vectorize_chunks(self, chunks):
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        return embeddings

    def build_faiss_index(self, embeddings):
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, os.path.join(self.embeddings_dir, 'index.faiss'))
        return index