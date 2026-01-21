from pathlib import Path
from typing import List


def estimate_tokens(text: str) -> int:
    return len(text.split())


CLEAN_TEXT_DIR = Path("data/processed/cleaned_text")
CHUNKS_DIR = Path("data/processed/chunks")

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


CHUNK_SIZE = 400    # this will be the chunk size  
CHUNK_OVERLAP = 50   # for overlapping so llm does not lose the semantic meaning


def chunk_text(paragraphs: List[str]) -> List[str]:
    
    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        
        if current_tokens + para_tokens > CHUNK_SIZE:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            
            overlap_words = chunk_text.split()[-CHUNK_OVERLAP:]
            current_chunk = [" ".join(overlap_words), para]
            current_tokens = estimate_tokens(current_chunk[0]) + para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_all_files():
    files = list(CLEAN_TEXT_DIR.glob("*.txt"))

    if not files:
        print("No cleaned text files found.")
        return

    for file in files:
        print(f"Chunking: {file.name}")

        text = file.read_text(encoding="utf-8", errors="ignore")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = chunk_text(paragraphs)

        
        output_file = CHUNKS_DIR / f"{file.stem}_chunks.txt"
        with output_file.open("w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"[CHUNK {i}]\n")
                f.write(chunk + "\n\n")

        print(f"Saved {len(chunks)} chunks to {output_file}")


if __name__ == "__main__":
    process_all_files()
