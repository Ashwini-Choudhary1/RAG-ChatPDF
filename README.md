# RAG-ChatPDF
RAG-ChatPDF is a local Retrieval-Augmented Generation (RAG) system designed for querying academic research papers and full PDF documents.

By storing semantic representations in a vector database, the system ensures that the Large Language Model (LLM) generates answers grounded strictly in your document content, eliminating hallucinations and overcoming context window limits.

# Why RAG-ChatPDF?
Standard LLMs face several hurdles when dealing with specific research:

Static Memory: They cannot "remember" your private or local PDFs.

Hallucinations: They may invent facts about niche academic topics.

Context Limits: They cannot process hundreds of pages at once.

Scalability: They struggle with large-scale document collections.

# RAG-ChatPDF solves this by:

Keeping documents stored efficiently outside the LLM.

Retrieving only the most relevant sections for every query.

Injecting knowledge dynamically at runtime.

Producing grounded, explainable answers with source attribution.

#  How It Works
The system follows a pipeline from raw data to generated insight:

Ingestion: PDF Files → Text Extraction → Cleaning & Safety Filtering.

# Project Structure

RAG-ChatPDF/
├── data/
│   ├── raw_pdfs/           # Original PDF documents
│   ├── processed/
│   │   ├── extracted_text/ # Raw extracted text
│   │   └── cleaned_text/   # Cleaned text (after preprocessing)
│   └── metadata/           # Paper metadata
├── ingest/
│   ├── pdf_loader.py       # PDF text extraction
│   ├── text_cleaner.py     # Cleaning & safety filtering
│   └── chunker.py          # Semantic chunking (Coming Soon)
├── embeddings/
│   └── embedder.py         # Text → vector embeddings (Upcoming)
├── vectorstore/
│   └── faiss_store.py      # Vector database logic (Upcoming)
├── rag/
│   ├── retriever.py        # Retrieval logic (Upcoming)
│   ├── prompt.py           # Prompt grounding (Upcoming)
│   └── generator.py        # LLM interface (Upcoming)
├── api/
│   └── main.py             # FastAPI backend (Upcoming)
├── requirements.txt
├── README.md
└── .gitignore

Indexing: Semantic Chunking → Vector Database (FAISS).

Retrieval: User Question → Semantic Search → Relevant Passages.

Generation: Relevant Passages + Prompt → LLM Reasoning → Final Answer.
