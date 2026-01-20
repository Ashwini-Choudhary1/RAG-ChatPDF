from pathlib import Path
from pypdf import PdfReader

raw_pdf_dir = Path("data/raw_pdfs")
ext_text_dir = Path("data/processed/extracted_text")

ext_text_dir.mkdir(parents=True,exist_ok=True)

def ext_txt_pdf(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)
    all_txt=[]

    for pg_number , page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text:
                all_txt.append(text)
        except Exception as e :
            print(f"Error extracting the page {pg_number} from {pdf_path.name} :{e}")
    return "\n".join(all_txt)

def process_all_pdfs():
    """
    Process all PDFs in data/raw_pdfs and save extracted text.
    """
    pdf_files = list(raw_pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in data/raw_pdfs/")
        return

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")

        extracted_text = ext_txt_pdf(pdf_file)

        output_file = ext_text_dir / f"{pdf_file.stem}.txt"
        output_file.write_text(extracted_text, encoding="utf-8")

        print(f" Saved extracted text to {output_file}")


if __name__ == "__main__":
    process_all_pdfs()