import faiss
import numpy as np
from typing import List


class FAISSVectorStore:
    def __init__(self, embedding_dim: int):
        
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.texts = []

    def add_embeddings(self, embeddings: np.ndarray, texts: List[str]):
        
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])

        return results
