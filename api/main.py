from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.rag_service import RAGService


app = FastAPI(
    title="RAG-ChatPDF API",
    description="Query academic PDFs using Retrieval-Augmented Generation",
    version="1.0"
)

rag_service = RAGService(top_k=3)



@app.get("/health")
def health():
    return {"status": "ok"}



@app.get("/model")
def model_info():
    return {
        "model": rag_service.llm.model_name
    }


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    contexts: list[str]



@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    return rag_service.query(request.question)



@app.post("/query-stream")
def query_stream(request: QueryRequest):
    return StreamingResponse(
        rag_service.query_stream(request.question),
        media_type="text/plain"
    )



@app.post("/retrieve")
def retrieve_only(request: QueryRequest):
    contexts = rag_service.retrieve_only(request.question)
    return {"contexts": contexts}
