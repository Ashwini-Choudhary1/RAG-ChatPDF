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


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: list[str]



@app.post("/query", response_model=QueryResponse) # Blocking endpoint (unchanged)
def query_rag(request: QueryRequest):
    return rag_service.query(request.question)



@app.post("/query-stream") # Streaming endpoint 
def query_stream(request: QueryRequest):

    
    retrieved_chunks = rag_service.retriever.retrieve(
        request.question,
        top_k=rag_service.top_k
    )

    
    def token_generator():
        for token in rag_service.generator.stream(
            retrieved_chunks,
            request.question
        ):
            yield token

    return StreamingResponse(
        token_generator(),
        media_type="text/plain"
    )
