from pathlib import Path
import time
import os

from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSVectorStore
from rag.retriever import Retriever
from rag.generator import Generator
from monitoring.metrics import log_request


USE_GROQ = os.getenv("USE_GROQ", "true").lower() == "true"

if USE_GROQ:
    from rag.groq_llm import GroqLLM
else:
    from rag.ollama_llm import OllamaLLM


class RAGService:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

        self.chunks = self._load_chunks()

        self.embedder = Embedder()
        embeddings = self.embedder.embed_texts(self.chunks)

        self.vectorstore = FAISSVectorStore(
            embedding_dim=embeddings.shape[1]
        )
        self.vectorstore.add_embeddings(embeddings, self.chunks)

        self.retriever = Retriever(self.embedder, self.vectorstore)

        
        if USE_GROQ:
            self.llm = GroqLLM()
        else:
            self.llm = OllamaLLM()

        self.generator = Generator(self.llm)

    def _load_chunks(self):
        chunks_dir = Path("data/processed/chunks")
        chunks = []

        for file in chunks_dir.glob("*_chunks.txt"):
            text = file.read_text(encoding="utf-8", errors="ignore")
            for block in text.split("[CHUNK"):
                block = block.strip()
                if not block:
                    continue
                content = block.split("]\n", 1)
                if len(content) == 2:
                    chunks.append(content[1].strip())

        if not chunks:
            raise RuntimeError("No chunks found. Run ingestion first.")

        return chunks

    def query(self, question: str) -> dict:
        start = time.time()

        
        retrieval_start = time.time()
        retrieved_chunks = self.retriever.retrieve(
            question, top_k=self.top_k
        )
        retrieval_time = time.time() - retrieval_start

        
        generation_start = time.time()
        answer = self.generator.generate(
            retrieved_chunks, question
        )
        generation_time = time.time() - generation_start

        total_time = time.time() - start

        # my loos to save
        try:
            log_request({
                "question": question,
                "answer": answer,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "model": getattr(self.llm, "model", "unknown"),
                "top_k": self.top_k,
                "num_contexts": len(retrieved_chunks)
            })
        except Exception as e:
            print("Logging failed:", e)

        return {
            "question": question,
            "answer": answer,
            "contexts": retrieved_chunks
        }

    
    def query_stream(self, question: str): #Streaming
        retrieved_chunks = self.retriever.retrieve(
            question, top_k=self.top_k
        )

        for token in self.generator.stream(
            retrieved_chunks, question
        ):
            yield token
