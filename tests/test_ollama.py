import pytest
from ollama.client import OllamaClient
import tempfile
import os
from unittest.mock import Mock, patch
import httpx

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


def test_ollama_embed_returns_vector(tmp_path):
    # Create mock config
    config_file = tmp_path / "models.yaml"
    config_file.write_text('ollama:\n  base_url: "http://localhost:11434"')

    client = OllamaClient.from_config(str(config_file))

    with patch('httpx.Client.post') as mock_post:
        # Mock the embedding response
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response

        vector = client.embed("test query", "bge-m3:latest")

        assert vector == [0.1, 0.2, 0.3]
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)


def test_ollama_embed_connection_error(tmp_path):
    config_file = tmp_path / "models.yaml"
    config_file.write_text('ollama:\n  base_url: "http://localhost:11434"')

    client = OllamaClient.from_config(str(config_file))

    with patch('httpx.Client.post') as mock_post:
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(httpx.ConnectError):
            client.embed("test", "bge-m3:latest")


def test_ollama_client_has_generation_models():
    """OllamaClient config includes generation models."""
    from ollama.client import OllamaClient

    client = OllamaClient.from_config("config/models.yaml")
    assert hasattr(client, 'generation_model')
    assert client.generation_model is not None
