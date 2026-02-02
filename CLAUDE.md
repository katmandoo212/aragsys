# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project State

This repository contains the **Scientific Agentic RAG Framework** - a pluggable, extensible Retrieval-Augmented Generation framework for multi-hop reasoning over academic and clinical literature.

### Current Status

**Phase:** MVP Architecture Complete (2026-02-01)

The MVP establishes a three-layer architecture with:
- YAML-driven configuration (techniques, pipelines, models)
- TechniqueRegistry with dynamic loading and filtering
- PipelineBuilder for composing workflows
- TechniqueNode for PocketFlow integration
- OllamaClient for LLM/embedding model access

**Next phases** will implement actual retrieval, generation, and multi-hop reasoning capabilities.

## Planned Architecture

The project implements an iterative RAG system using:
- **Python 3.14+** with uv for package management
- **PocketFlow** for node-based workflow orchestration
- **Ollama** for local LLM inference
- **YAML** for all configuration

### Core Design Patterns
- **Registry Pattern:** Dynamic technique loading from YAML configuration
- **Strategy Pattern:** Swappable retrieval strategies
- **Builder Pattern:** Pipeline composition
- **Duck typing interface** for flexibility

### Implementation Phases

1. **Phase 1 (MVP - Complete):** Architecture foundation with mocked techniques
2. **Phase 2:** Naive RAG with dense retrieval
3. **Phase 3:** Advanced retrieval (HyDE, Multi-Query, Hybrid search)
4. **Phase 4:** Precision (Reranking, Contextual Compression)
5. **Phase 5:** GraphRAG with multi-hop reasoning
6. **Phase 6:** Agentic self-correction with iterative refinement

## Key Files

### Configuration
- `config/techniques.yaml` - Technique definitions and their configs
- `config/pipelines.yaml` - Pipeline compositions
- `config/models.yaml` - Ollama and model settings

### Core Components
- `registry/technique_registry.py` - TechniqueRegistry, TechniqueMetadata
- `nodes/technique_node.py` - Generic TechniqueNode for PocketFlow
- `pipeline/builder.py` - PipelineBuilder
- `ollama/client.py` - OllamaClient

### Tests
- `tests/test_registry.py` - Registry tests (7 tests)
- `tests/test_nodes.py` - Node tests (2 tests)
- `tests/test_builder.py` - Builder tests (2 tests)
- `tests/test_ollama.py` - Ollama client tests (2 tests)
- `tests/test_integration.py` - Round-trip integration test (1 test)

**Total: 14 tests, all passing**

## Commands

```bash
# Run tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_registry.py -v

# Install dependencies
uv sync
```

## Session Management

**IMPORTANT:** Follow session-management.md skill. Update session state at natural breakpoints.

### After Every Task Completion
Ask yourself:
1. Was a decision made? → Log to `_project_specs/session/decisions.md`
2. Did this take >10 tool calls? → Full checkpoint to `_project_specs/session/current-state.md`
3. Is a major feature complete? → Create archive entry in `_project_specs/session/archive/`
4. Otherwise → Quick update to `_project_specs/session/current-state.md`

### Checkpoint Triggers
**Quick Update** (current-state.md):
- After any todo completion
- After small changes

**Full Checkpoint** (current-state.md + decisions.md):
- After significant changes
- After ~20 tool calls
- After any decision
- When switching focus areas

**Archive** (archive/ + full checkpoint):
- End of session
- Major feature complete
- Context feels heavy (~50+ tool calls)

### Session Start Protocol
When beginning work:
1. Read `_project_specs/session/current-state.md`
2. Review recent `decisions.md` entries if needed
3. Continue from "Next Steps"

### Session End Protocol
Before ending or when context limit approaches:
1. Create archive: `_project_specs/session/archive/YYYY-MM-DD.md`
2. Update current-state.md with handoff format
3. Ensure next steps are specific and actionable

## Coding Standards

- **Type hints** on all function signatures
- **Duck typing interface** - all techniques follow protocol
- **Dependency injection** - all dependencies injected via YAML config
- **YAGNI** - only build what's needed for current phase
- **TDD** - write tests before implementation