from sentence_transformers import SentenceTransformer
from typing import List


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"): #alternative 1 : allenai/scibert_scivocab_uncased
                                                              #alternative 2: sentence-transformers/scibert-base-nli
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]):
        
        return self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )
