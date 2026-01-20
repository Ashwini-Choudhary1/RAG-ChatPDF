# 📄 RAG-ChatPDF

RAG-ChatPDF is a local Retrieval-Augmented Generation (RAG) system that allows you to ask questions over full PDF documents, especially academic research papers.

The system parses PDFs, stores their semantic representations in a vector database, and uses a Large Language Model (LLM) to generate answers grounded only in retrieved document content.

# Why RAG-ChatPDF?

Large Language Models:

Cannot remember your PDFs

Hallucinate on niche academic topics

Have limited context windows

Cannot scale to large document collections
---------------------------------------------------------------------------
# RAG-ChatPDF solves this by:

Keeping documents outside the LLM

Retrieving only relevant content

Injecting knowledge dynamically at query time

Producing grounded, explainable answers
----------------------------------------------------------------------------
# How It Works (High Level)

PDF Files
   ↓
Text Extraction
   ↓
Text Cleaning & Safety Filtering
   ↓
Semantic Chunking
   ↓
Vector Database
   ↓
---------------------
User Question
   ↓
Semantic Retrieval
   ↓
Relevant Passages
   ↓
LLM Reasoning
   ↓
Answer from PDFs

# Project Structure

RAG-ChatPDF/
├── data/
│   ├── raw_pdfs/              # Original PDF documents
│   ├── processed/
│   │   ├── extracted_text/    # Raw extracted text
│   │   └── cleaned_text/      # Cleaned text (after preprocessing)
│   └── metadata/              # Paper metadata
│
├── ingest/
│   ├── pdf_loader.py          # PDF text extraction
│   ├── text_cleaner.py        # Cleaning & safety filtering
│   └── chunker.py             # Semantic chunking (coming next)
│
├── embeddings/
│   └── embedder.py            # Text → vector embeddings (upcoming)
│
├── vectorstore/
│   └── faiss_store.py         # Vector database logic (upcoming)
│
├── rag/
│   ├── retriever.py           # Retrieval logic (upcoming)
│   ├── prompt.py              # Prompt grounding (upcoming)
│   └── generator.py           # LLM interface (upcoming)
│
├── api/
│   └── main.py                # FastAPI backend (upcoming)
│
├── requirements.txt
├── README.md
└── .gitignore

