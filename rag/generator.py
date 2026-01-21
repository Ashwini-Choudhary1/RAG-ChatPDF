from typing import List


class Generator:
    def __init__(self, llm_client):
        
        self.llm_client = llm_client

    def build_prompt(self, context_chunks: List[str], question: str) -> str:
        
        context = "\n\n".join(context_chunks)

        prompt = f"""
You are an assistant answering questions based ONLY on the provided context.

Context:
{context}

Question:
{question}

Answer using only the information from the context above.
"""

        return prompt.strip()

    def generate(self, context_chunks: List[str], question: str) -> str:
        
        prompt = self.build_prompt(context_chunks, question)

        response = self.llm_client.generate(prompt)
        return response
