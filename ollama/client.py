from dataclasses import dataclass
import yaml
import httpx

@dataclass
class OllamaClient:
    base_url: str

    @classmethod
    def from_config(cls, config_path: str) -> "OllamaClient":
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(base_url=config["ollama"]["base_url"])

    def embed(self, text: str, model: str) -> list[float]:
        """Generate embeddings for the given text using the specified model."""
        url = f"{self.base_url}/api/embed"
        payload = {
            "model": model,
            "input": text
        }

        response = httpx.Client().post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        return data.get("embedding", [])