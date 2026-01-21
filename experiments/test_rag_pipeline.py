from pathlib import Path

from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSVectorStore
from rag.retriever import Retriever
from rag.generator import Generator
from rag.mock_llm import MockLLM


def load_chunks():
    """
    Load chunk texts from processed chunk files.
    """
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

    return chunks


def main():
    print("🔹 Loading chunks...")
    chunks = load_chunks()
    print(f"✅ Loaded {len(chunks)} chunks")

    
    embedder = Embedder()
    embeddings = embedder.embed_texts(chunks)

    vectorstore = FAISSVectorStore(embedding_dim=embeddings.shape[1])
    vectorstore.add_embeddings(embeddings, chunks)

    retriever = Retriever(embedder, vectorstore)
    generator = Generator(MockLLM())

    
    question = "How are Trojan attacks optimized in large language models?"

    print("\n Retrieving relevant chunks...")
    retrieved_chunks = retriever.retrieve(question, top_k=3)

    print("\n Retrieved chunks:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"\n--- Chunk {i} ---")
        print(chunk[:500])  # preview only

    print("\n Generating answer...")
    answer = generator.generate(retrieved_chunks, question)

    print("\n Final Answer:")
    print(answer)


if __name__ == "__main__":
    main()
