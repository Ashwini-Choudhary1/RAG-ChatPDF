# RAG-ChatPDF
RAG-ChatPDF is a local Retrieval-Augmented Generation (RAG) system designed for querying academic research papers and full PDF documents.

By storing semantic representations in a vector database, the system ensures that the Large Language Model (LLM) generates answers grounded strictly in your document content, eliminating hallucinations and overcoming context window limits.

# Project Structure

```
RAG-ChatPDF/
├── data/
│   ├── raw_pdfs/           # Original PDF documents
│   ├── processed/
│   │   ├── extracted_text/ # Raw extracted text
│   │   └── cleaned_text/   # Cleaned text (after preprocessing)
│   └── metadata/           # Paper metadata
│
├── ingest/
│   ├── pdf_loader.py       # PDF text extraction
│   ├── text_cleaner.py     # Cleaning & safety filtering
│   └── chunker.py          # Semantic chunking (Coming Soon)
│
├── embeddings/
│   └── embedder.py         # Text → vector embeddings
│
├── vectorstore/
│   └── faiss_store.py      # FAISS vector database logic
│
├── rag/
│   ├── retriever.py        # Vector similarity retrieval
│   ├── prompt.py           # Prompt grounding logic
│   └── generator.py        # LLM interface abstraction
│
├── api/
│   └── main.py             # FastAPI backend
│
├── requirements.txt
├── README.md
└── .gitignore

PDF → Extraction → Cleaning → Chunking → Embeddings → FAISS
                                                     ↓
                                              Retriever
                                                     ↓
                                               Generator (LLM)
                                                     ↓
                                             FastAPI (Streaming)
```

# Why RAG-ChatPDF?
Standard LLMs face several hurdles when dealing with specific research:

Static Memory: They cannot "remember" my private or local PDFs.

Hallucinations: They may invent facts about niche academic topics.

Context Limits: They cannot process hundreds of pages at once.

Scalability: They struggle with large-scale document collections.

# RAG-ChatPDF solves this by:

Keeping documents stored efficiently outside the LLM.

Retrieving only the most relevant sections for every query.

Injecting knowledge dynamically at runtime.

Producing grounded, explainable answers with source attribution.






