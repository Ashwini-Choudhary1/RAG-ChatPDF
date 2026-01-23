from fastapi import FastAPI
from pydantic import BaseModel

from rag.rag_service import RAGService


app = FastAPI(
    title="RAG-ChatPDF API",
    description="Query academic PDFs using Retrieval-Augmented Generation",
    version="1.0"
)

rag_service = RAGService(top_k=3)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: list[str]


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    return rag_service.query(request.question)
