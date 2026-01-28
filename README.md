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

Indexing: Semantic Chunking → Vector Database (FAISS).

Retrieval: User Question → Semantic Search → Relevant Passages.

Generation: Relevant Passages + Prompt → LLM Reasoning → Final Answer.
