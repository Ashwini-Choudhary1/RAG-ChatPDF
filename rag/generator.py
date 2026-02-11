from typing import List


class Generator:
    def __init__(self, llm_client):
        self.llm = llm_client
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
        return self.llm_client.generate(prompt)

    def stream(self, context_chunks: List[str], question: str):
        prompt = self.build_prompt(context_chunks, question)
        for token in self.llm_client.stream(prompt):
            yield token
