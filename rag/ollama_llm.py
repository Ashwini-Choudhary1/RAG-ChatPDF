import subprocess
import os


class OllamaLLM:
    def __init__(self, model: str | None = None):
        
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3") # if model not defined it will take ollama as default model

    def generate(self, prompt: str) -> str:
        result = subprocess.run(
            ["ollama", "run", self.model],
            input=prompt,
            text=True,
            capture_output=True
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        return result.stdout.strip()
