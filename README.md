# Recent Update

Added Streaming and connected with Groq API to work faster


# RAG-ChatPDF
RAG-ChatPDF is a local Retrieval-Augmented Generation (RAG) system designed for querying academic research papers and full PDF documents.

By storing semantic representations in a vector database, the system ensures that the Large Language Model (LLM) generates answers grounded strictly in your document content, eliminating hallucinations and overcoming context window limits.

# Project Structure

```
RAG-ChatPDF/
│
├── data/
│   ├── raw_pdfs/
│   ├── processed/
│   └── metadata/
│
├── ingest/
│   ├── pdf_loader.py
│   ├── text_cleaner.py
│   └── chunker.py
│
├── embeddings/
│   └── embedder.py
│
├── vectorstore/
│   └── faiss_store.py
│
├── rag/
│   ├── retriever.py
│   ├── generator.py
│   ├── groq_llm.py
│   ├── ollama_llm.py
│   └── rag_service.py
│
├── monitoring/
│   └── metrics.py
│
├── api/
│   └── main.py
│
├── requirements.txt
├── README.md
└── .gitignore

```

```
# Work Flow

                ┌──────────────┐
                │   PDF Files  │
                └──────┬───────┘
                       ↓
                Text Extraction
                       ↓
                    Cleaning
                       ↓
                    Chunking
                       ↓
                   Embeddings
                       ↓
                FAISS Vector DB
                       ↓
                   Retriever
                       ↓
                Prompt Builder
                       ↓
        ┌──────────────┴──────────────┐
        │                             │
     Groq LLM                     Ollama LLM/Mistral/Llama3
        │                             │
        └──────────────┬──────────────┘
                       ↓
                FastAPI (Streaming)
                       ↓
                  Production


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






