
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import yaml

class Retriever:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.chunks_dir = self.config['data']['chunks_dir']
        self.embeddings_dir = self.config['data']['embeddings_dir']
        self.top_k = self.config['retriever']['top_k']
        self.embedding_model = SentenceTransformer(self.config['models']['embedding_model'])
        self.index = faiss.read_index(self.config['retriever']['faiss_index'])
        
    def retrieve(self, query):
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
        distances, indices = self.index.search(query_embedding.astype(np.float32), self.top_k)
        
        retrieved_chunks = []
        retrieved_metadata = []
        for idx in indices[0]:
            with open(os.path.join(self.chunks_dir, f"chunk_{idx}.txt"), 'r', encoding='utf-8') as f:
                chunk = f.read()
            with open(os.path.join(self.chunks_dir, f"chunk_{idx}_meta.json"), 'r', encoding='utf-8') as f:
                meta = json.load(f)
            retrieved_chunks.append(chunk)
            retrieved_metadata.append(meta)
        
        return retrieved_chunks, retrieved_metadata
