import subprocess
import os


class OllamaLLM:
    def __init__(self, model: str | None = None):
        self.model_name = model or os.getenv("OLLAMA_MODEL", "llama3")
        self.model = self.model_name  # optional alias

    def generate(self, prompt: str) -> str:
        result = subprocess.run(
            ["ollama", "run", self.model_name],
            input=prompt,
            text=True,
            capture_output=True
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        return result.stdout.strip()
