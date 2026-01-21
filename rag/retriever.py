import numpy as np
from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSVectorStore


class Retriever:
    def __init__(self, embedder: Embedder, vectorstore: FAISSVectorStore):
        self.embedder = embedder
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int = 5):
    
        
        query_embedding = self.embedder.embed_texts([query])

        
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        
        results = self.vectorstore.search(query_embedding, top_k=top_k)

        return results
