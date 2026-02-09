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

---

## [2026-02-05] Phase 2 - Vector Store - ChromaDB

**Decision**: Use ChromaDB for vector storage

**Context**: Phase 2 Naive RAG implementation

**Options Considered**:
1. ChromaDB
2. Qdrant
3. Pinecone (cloud)

**Choice**: ChromaDB

**Reasoning**:
- Lightweight and embedded (no separate service)
- Python-native
- Easy to set up for local development
- Good for single-user research tool

**Trade-offs**:
- Less scalable than cloud solutions
- Limited advanced indexing features

**References**:
- utils/vector_store.py - ChromaDB wrapper
- config/models.yaml - Vector store settings

---

## [2026-02-07] Phase 3 - PDF Parsing - pdfplumber

**Decision**: Use pdfplumber over pypdf for PDF extraction

**Context**: Phase 3 Document Formats implementation

**Options Considered**:
1. pdfplumber
2. pypdf
3. PyMuPDF

**Choice**: pdfplumber

**Reasoning**:
- Better table extraction
- More robust layout analysis
- Preserves document structure better
- Well-maintained

**References**:
- utils/pdf_chunker.py - PDF processing with pdfplumber

---

## [2026-02-07] Phase 4 - Graph Store - Neo4j

**Decision**: Use Neo4j for entity graph storage

**Context**: Phase 4 Advanced Retrieval implementation

**Options Considered**:
1. Neo4j
2. ArangoDB
3. Custom in-memory graph

**Choice**: Neo4j

**Reasoning**:
- Mature graph database
- Cypher query language is expressive
- Good performance for multi-hop queries
- Industry standard for graph use cases

**Trade-offs**:
- Requires separate service installation
- More complex setup than embedded options

**References**:
- stores/neo4j_store.py - Neo4j store implementation
- config/neo4j.yaml - Neo4j connection settings

---

## [2026-02-07] Phase 4 - Hybrid Search - Reciprocal Rank Fusion

**Decision**: Use RRF for combining dense and sparse retrieval

**Context**: HybridTechnique implementation

**Options Considered**:
1. Reciprocal Rank Fusion (RRF)
2. Weighted score average
3. Learning to rank

**Choice**: RRF

**Reasoning**:
- Simple and effective
- No training required
- Standard approach for hybrid search
- Configurable k parameter

**References**:
- techniques/hybrid.py - RRF implementation

---

## [2026-02-07] Phase 5 - Reranking - LLM-based scoring

**Decision**: Use LLM prompt for reranking instead of cross-encoder

**Context**: RerankTechnique implementation

**Options Considered**:
1. LLM prompt scoring
2. Cross-encoder model
3. Learnable reranker

**Choice**: LLM prompt scoring

**Reasoning**:
- Uses existing Ollama infrastructure
- Simpler than cross-encoder integration
- More flexible (can explain reasoning)
- Works well with current LLM models

**Trade-offs**:
- Slower than cross-encoder
- Less precise than dedicated reranker models

**References**:
- techniques/rerank.py - LLM-based reranking

---

## [2026-02-07] Phase 6 - Graph Techniques - Entity-based retrieval

**Decision**: Implement three complementary graph techniques

**Context**: Phase 6 GraphRAG implementation

**Options Considered**:
1. Single complex graph technique
2. Three modular techniques (entity, multihop, expand)
3. User-customizable graph queries

**Choice**: Three modular techniques

**Reasoning**:
- Each technique has distinct use case
- Composable with other techniques
- Follows existing technique pattern
- Easier to test and maintain

**References**:
- techniques/graph_entity.py - Entity-based retrieval
- techniques/graph_multihop.py - Multi-hop reasoning
- techniques/graph_expand.py - Relationship expansion

---

## [2026-02-07] Phase 7 - Generation - Answer dataclass

**Decision**: Create dedicated Answer dataclass for generation responses

**Context**: Phase 7 Generation implementation

**Options Considered**:
1. Return dict
2. Return tuple
3. Create Answer dataclass

**Choice**: Answer dataclass

**Reasoning**:
- Type-safe and IDE-friendly
- Clear structure (content, metadata, citations)
- Extensible for future fields
- Follows Python best practices

**References**:
- utils/answer.py - Answer dataclass definition

---

## [2026-02-07] Phase 7 - Generation Strategies - Three approaches

**Decision**: Implement three complementary generation strategies

**Context**: Phase 7 Generation implementation

**Options Considered**:
1. Single configurable technique
2. Three techniques (simple, context, CoT)
3. Technique with pluggable strategies

**Choice**: Three techniques

**Reasoning**:
- Different use cases benefit from different strategies
- Simple for basic queries, Context for citations, CoT for reasoning
- Composable with retrieval techniques
- Matches existing pattern

**References**:
- techniques/generate_simple.py - Basic generation
- techniques/generate_context.py - Citation-aware generation
- techniques/generate_cot.py - Chain-of-thought generation