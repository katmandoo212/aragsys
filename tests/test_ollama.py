from ollama.client import OllamaClient
import tempfile
import os

def test_ollama_client_creation():
    client = OllamaClient("http://localhost:11434")
    assert client.base_url == "http://localhost:11434"

def test_ollama_client_from_config(tmp_path):
    yaml_content = """
ollama:
  base_url: "http://custom-url:11434"
"""
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml_content)

    from ollama.client import OllamaClient
    client = OllamaClient.from_config(str(config_file))
    assert client.base_url == "http://custom-url:11434"
