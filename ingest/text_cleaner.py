import re
from pathlib import Path

RAW_TEXT_DIR = Path("data/processed/extracted_text")
CLEAN_TEXT_DIR = Path("data/processed/cleaned_text")

CLEAN_TEXT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([.,;:])([A-Za-z])", r"\1 \2", text)

    lines = text.split("\n")
    merged_lines = []

    buffer = ""
    for line in lines:
        line = line.strip()

        if not line:
            if buffer:
                merged_lines.append(buffer)
                buffer = ""
            continue

        if buffer and not buffer.endswith((".", ":", "?", "!", ")")):
            buffer += " " + line
        else:
            if buffer:
                merged_lines.append(buffer)
            buffer = line

    if buffer:
        merged_lines.append(buffer)

    text = "\n\n".join(merged_lines)

    cleaned_blocks = []
    for block in text.split("\n\n"):
        alpha_ratio = sum(c.isalpha() for c in block) / max(len(block), 1)
        if alpha_ratio < 0.2:
            continue
        cleaned_blocks.append(block)

    return "\n\n".join(cleaned_blocks)


def process_all_texts():
    files = list(RAW_TEXT_DIR.glob("*.txt"))

    if not files:
        print("No extracted text files found.")
        return

    for file in files:
        print(f" Cleaning: {file.name}")

        raw_text = file.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw_text)

        output_file = CLEAN_TEXT_DIR / file.name
        output_file.write_text(cleaned, encoding="utf-8")

        print(f"Saved cleaned text to {output_file}")


if __name__ == "__main__":
    process_all_texts()
