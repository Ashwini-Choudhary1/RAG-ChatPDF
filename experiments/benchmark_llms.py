import time
import statistics
import json
import csv
from pathlib import Path
from datetime import datetime

from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSVectorStore
from rag.retriever import Retriever
from rag.generator import Generator
from rag.ollama_llm import OllamaLLM


QUESTION = "How are Trojan attacks optimized in large language models?"
TOP_K = 3
RUNS = 5

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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

    records = []

    for run_id in range(1, RUNS + 1):
        start = time.perf_counter()
        answer = generator.generate(context_chunks, QUESTION)
        end = time.perf_counter()

        latency = end - start

        records.append({
            "model": model_name,
            "run": run_id,
            "latency_seconds": latency,
            "is_cold_start": run_id == 1,
            "question": QUESTION,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat()
        })

        print(f"[{model_name}] Run {run_id} latency: {latency:.2f}s")

    return records


def save_results(all_records):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    
    json_path = RESULTS_DIR / f"benchmark_results_{timestamp}.json" # to save data in json files
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

    
    csv_path = RESULTS_DIR / f"benchmark_results_{timestamp}.csv" # to store data in csv files
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "run",
                "latency_seconds",
                "is_cold_start",
                "question",
                "timestamp"
            ]
        )
        writer.writeheader()
        for r in all_records:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print(f"\n Results saved to:")
    print(f" - {json_path}")
    print(f" - {csv_path}")


def main():
    print(" Preparing retrieval context (shared across models)...")
    context_chunks = setup_retrieval()

    all_records = []

    print("\n Benchmarking LLaMA 3")
    all_records.extend(benchmark_model("llama3", context_chunks))

    print("\n Benchmarking Mistral")
    all_records.extend(benchmark_model("mistral", context_chunks))

    save_results(all_records)

    
    for model in ["llama3", "mistral"]:
        latencies = [r["latency_seconds"] for r in all_records if r["model"] == model] # to print summary to console only
        print(
            f"{model} → avg: {statistics.mean(latencies):.2f}s | "
            f"std: {statistics.stdev(latencies):.2f}"
        )


if __name__ == "__main__":
    main()
