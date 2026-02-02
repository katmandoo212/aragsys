# Decision Log

Track key decisions for future reference. Never delete entries.

---

## [2026-02-01] Architecture Foundation - MVP Scope

**Decision**: Focus on architecture plumbing first, not actual retrieval

**Context**: Initial project kick-off

**Options Considered**:
1. Build working end-to-end RAG with actual retrieval
2. Focus on architecture foundation with mocked techniques
3. Implement GraphRAG features first

**Choice**: Architecture foundation with mocked techniques

**Reasoning**:
- Establishes Registry, Builder, and Node patterns before complexity
- Testable and extensible structure is more valuable than broken retrieval
- Enables adding real techniques incrementally
- Aligns with SOLID principles and YAGNI

**Trade-offs**:
- No actual retrieval functionality in MVP
- More upfront architecture work

**References**:
- docs/plans/2026-02-01-kickoff-mvp-design.md
- registry/technique_registry.py - Registry pattern
- pipeline/builder.py - Builder pattern

---

## [2026-02-01] Python Tooling - uv

**Decision**: Use uv for Python package management

**Context**: Setting up project tooling

**Options Considered**:
1. pip + virtualenv
2. Poetry
3. uv

**Choice**: uv

**Reasoning**:
- Fast package installation and resolution
- Modern, well-maintained
- Better lock file handling than Poetry
- Industry direction for Python tooling

**References**:
- pyproject.toml - Dependencies

---

## [2026-02-01] Workflow Orchestrator - PocketFlow

**Decision**: Use PocketFlow for node-based workflow

**Context**: Choosing workflow orchestration library

**Options Considered**:
1. PocketFlow
2. LangChain
3. Custom implementation

**Choice**: PocketFlow

**Reasoning**:
- Lightweight and focused
- Python-native
- Prep/exec/post pattern matches our needs
- No overhead from larger frameworks

**References**:
- nodes/technique_node.py - PocketFlow Node implementation

---

## [2026-02-01] LLM Provider - Ollama

**Decision**: Use Ollama for local LLM inference

**Context**: Choosing LLM and embedding provider

**Options Considered**:
1. OpenAI API (cloud)
2. Anthropic API (cloud)
3. Ollama (local)

**Choice**: Ollama

**Reasoning**:
- Local inference (no API costs)
- Privacy (data stays local)
- Self-hostable for scientific/clinical use cases
- Supports multiple open-source models

**Trade-offs**:
- Requires local GPU/CPU resources
- Models smaller than top-tier cloud offerings

**References**:
- ollama/client.py - OllamaClient implementation
- config/models.yaml - Model configuration

---

## [2026-02-01] Query Model - glm-4.7:cloud

**Decision**: Use glm-4.7:cloud as default query model

**Context**: Selecting primary LLM for generation

**Options Considered**:
1. llama3.2
2. mistral
3. glm-4.7:cloud

**Choice**: glm-4.7:cloud

**Reasoning**:
- User preference (explicit requirement)
- Configurable per pipeline

**References**:
- config/models.yaml - Query model settings

---

## [2026-02-01] Embedding Models - Multi-Model Support

**Decision**: Support 5 different embedding models

**Context**: Choosing embedding models for vector search

**Options Considered**:
1. Single best model
2. Multiple models for different use cases

**Choice**: Multiple models

**Reasoning**:
- Different models excel at different tasks
- General purpose (nomic-embed-text-v2-moe)
- Multilingual (qwen3-embedding)
- Lightweight (granite-embedding)
- Dense retrieval (bge-m3)
- Complex queries (mxbai-embed-large)

**References**:
- config/models.yaml - Available embedding models

---

## [2026-02-01] Configuration - YAML-Driven

**Decision**: Use YAML for all configuration

**Context**: Choosing configuration format

**Options Considered**:
1. YAML
2. JSON
3. TOML
4. Python code

**Choice**: YAML

**Reasoning**:
- Human-readable and editable
- Industry standard for config
- Easy to comment
- No need to rebuild config changes

**References**:
- config/techniques.yaml - Technique definitions
- config/pipelines.yaml - Pipeline compositions
- config/models.yaml - Ollama settings