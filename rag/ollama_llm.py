import requests
import json
import os


class OllamaLLM:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
        self.base_url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.base_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]

    def stream(self, prompt: str):
        response = requests.post(
            self.base_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True
            },
            stream=True
        )

        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    yield data["response"]
