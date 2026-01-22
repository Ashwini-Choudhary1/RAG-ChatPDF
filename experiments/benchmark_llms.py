import time
import statistics
from pathlib import Path

from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSVectorStore
from rag.retriever import Retriever
from rag.generator import Generator
from rag.ollama_llm import OllamaLLM


QUESTION = "How are Trojan attacks optimized in large language models?"
TOP_K = 3
RUNS = 5


def load_chunks():
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


def setup_retrieval():
    chunks = load_chunks()

    embedder = Embedder()
    embeddings = embedder.embed_texts(chunks)

    vectorstore = FAISSVectorStore(embedding_dim=embeddings.shape[1])
    vectorstore.add_embeddings(embeddings, chunks)

    retriever = Retriever(embedder, vectorstore)
    retrieved_chunks = retriever.retrieve(QUESTION, top_k=TOP_K)

    return retrieved_chunks


def benchmark_model(model_name: str, context_chunks):
    llm = OllamaLLM(model=model_name)
    generator = Generator(llm)

    latencies = []
    answers = []

    for i in range(RUNS):
        start = time.perf_counter()
        answer = generator.generate(context_chunks, QUESTION)
        end = time.perf_counter()

        latencies.append(end - start)
        answers.append(answer)

        print(f"[{model_name}] Run {i+1} latency: {latencies[-1]:.2f}s")

    return latencies, answers


def main():
    print(" Preparing retrieval context (shared across models)...")
    context_chunks = setup_retrieval()

    print("\n Benchmarking LLaMA 3")
    llama_lat, llama_answers = benchmark_model("llama3", context_chunks)

    print("\n Benchmarking Mistral")
    mistral_lat, mistral_answers = benchmark_model("mistral", context_chunks)

    print("\n Latency Summary")
    print(f"LLaMA 3 → avg: {statistics.mean(llama_lat):.2f}s | std: {statistics.stdev(llama_lat):.2f}")
    print(f"Mistral → avg: {statistics.mean(mistral_lat):.2f}s | std: {statistics.stdev(mistral_lat):.2f}")

    print("\n Sample Answers (for qualitative analysis)")
    print("\n--- LLaMA 3 (Run 1) ---\n", llama_answers[0])
    print("\n--- Mistral (Run 1) ---\n", mistral_answers[0])


if __name__ == "__main__":
    main()
