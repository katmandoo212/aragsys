from ollama.client import OllamaClient

def test_ollama_client_creation():
    client = OllamaClient("http://localhost:11434")
    assert client.base_url == "http://localhost:11434"
