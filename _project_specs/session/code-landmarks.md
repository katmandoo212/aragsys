# Code Landmarks

Quick reference to important parts of the codebase.

## Entry Points
| Location | Purpose |
|----------|---------|
| tests/test_integration.py | Round-trip verification test |
| main.py | (Placeholder) CLI entry point |

## Core Business Logic
| Location | Purpose |
|----------|---------|
| registry/technique_registry.py | Technique discovery and metadata |
| pipeline/builder.py | Pipeline composition |
| nodes/technique_node.py | Workflow node wrapper |
| ollama/client.py | Ollama API client |

## Configuration
| Location | Purpose |
|----------|---------|
| config/techniques.yaml | Technique definitions |
| config/pipelines.yaml | Pipeline compositions |
| config/models.yaml | Ollama and model settings |
| pyproject.toml | Dependencies and project metadata |

## Key Patterns
| Pattern | Example Location | Notes |
|---------|------------------|-------|
| Dataclass for metadata | registry/technique_registry.py:7-12 | TechniqueMetadata |
| Registry pattern | registry/technique_registry.py:14-37 | Dynamic loading |
| Builder pattern | pipeline/builder.py:5-12 | Pipeline composition |
| PocketFlow Node | nodes/technique_node.py:6-18 | prep/exec/post |
| Factory method | ollama/client.py:12-18 | from_config() |

## Testing
| Location | Purpose |
|----------|---------|
| tests/test_registry.py | Registry unit tests |
| tests/test_nodes.py | Node unit tests |
| tests/test_builder.py | Builder unit tests |
| tests/test_ollama.py | Ollama client unit tests |
| tests/test_integration.py | Round-trip integration test |

## Gotchas & Non-Obvious Behavior
| Location | Issue | Notes |
|----------|-------|-------|
| registry/technique_registry.py:30-31 | Enabled default is True | If not specified, technique is enabled |
| nodes/technique_node.py:15-17 | prep returns tuple | (query, docs) not dict |
| pipeline/builder.py:23 | Technique lookup fails if missing | Raises ValueError currently |
| ollama/client.py:15 | Config requires nested structure | config["ollama"]["base_url"] |

## External Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| pocketflow | >=0.0.3 | Workflow orchestration |
| pydantic | >=2.12.5 | Data validation (not yet used) |
| pyyaml | >=6.0.3 | YAML parsing |
| httpx | >=0.28.1 | HTTP requests (future Ollama calls) |
| pytest | >=9.0.2 | Testing framework |