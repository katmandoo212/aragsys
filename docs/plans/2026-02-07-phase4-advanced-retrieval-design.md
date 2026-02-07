# Phase 4: Advanced Retrieval with Neo4j - Design

**Status:** Design Complete
**Date:** 2026-02-07
**Previous Phase:** Phase 3 - Document Formats (54 tests passing)

## Overview

Phase 4 introduces three advanced retrieval techniques and transitions the storage layer from ChromaDB to Neo4j as a unified store. The key changes are:

1. **HyDE Technique** - Generates hypothetical answers before retrieval, improving semantic matching
2. **Multi-Query Technique** - Expands queries into multiple perspectives for better coverage
3. **Hybrid Search Technique** - Combines dense (vector) and sparse (full-text) retrieval with RRF
4. **Neo4j Integration** - Replaces ChromaDB, storing documents, embeddings, and entities
5. **Basic NER** - Extracts people, organizations, locations from document chunks

The architecture maintains the existing Registry Pattern and duck-typing interfaces, making these new techniques plug-and-play with the existing system.

## Architecture Components

### 1. Neo4j Store Layer (`stores/neo4j_store.py`)

**Neo4jStore** replaces `VectorStore` with expanded capabilities:
- `add_documents()` - Upserts Document nodes with content, metadata, and embedding vector
- `add_entities()` - Creates Entity nodes (Person, Organization, Location) and `(:Document)-[:CONTAINS]->(:Entity)` relationships
- `vector_search()` - Dense retrieval using Neo4j's vector index (similarity search on embeddings)
- `fulltext_search()` - Sparse retrieval using Neo4j's full-text indexes
- `hybrid_search()` - Combines both using Reciprocal Rank Fusion (RRF)

**Neo4j Schema:**
```cypher
(:Document {id, content, metadata: {}, embedding: [...]})
(:Entity {id, name, type})
(:Document)-[:CONTAINS]->(:Entity)
```

### 2. New Retrieval Techniques (`techniques/`)

**HyDETechnique** - Uses LLM to generate a hypothetical answer, embeds it, then retrieves documents matching that hypothetical embedding.

**MultiQueryTechnique** - Generates 3-5 query variations using LLM, retrieves for each, deduplicates, and ranks by combined score.

**HybridTechnique** - Calls both `vector_search()` and `fulltext_search()`, merges results with RRF formula: `score = 1/(k+rank_dense) + 1/(k+rank_sparse)`

### 3. Entity Extraction (`utils/entity_extractor.py`)

**EntityExtractor** - Simple rule-based NER using spaCy for Person, Organization, Location extraction. Returns list of `{text, label, start, end}` tuples.

## Data Flow

### Document Ingestion Flow
```
1. TextChunker chunks file → list of (content, metadata)
2. EntityExtractor extracts entities from each chunk
3. OllamaClient generates embeddings for each chunk
4. Neo4jStore.add_documents() creates Document nodes with embeddings
5. Neo4jStore.add_entities() creates Entity nodes and relationships
```

### HyDE Retrieval Flow
```
1. User provides query
2. HyDETechnique calls LLM: "Generate a hypothetical answer for: {query}"
3. Embed hypothetical answer → hypothetical_vector
4. Neo4jStore.vector_search(hypothetical_vector, top_k) → results
5. Return Documents with similarity scores
```

### Multi-Query Retrieval Flow
```
1. User provides query
2. MultiQueryTechnique calls LLM: "Generate 3 diverse queries for: {query}"
3. For each generated query:
   - Embed query
   - Neo4jStore.vector_search(query_vector, top_k)
4. Deduplicate results (same document ID)
5. Sum scores from duplicate matches
6. Return ranked unique Documents
```

### Hybrid Retrieval Flow
```
1. User provides query
2. HybridTechnique generates:
   - query_vector → vector_search() → dense_results (ranked)
   - fulltext_query → fulltext_search() → sparse_results (ranked)
3. Apply RRF to merge: score = 1/(60+rank_dense) + 1/(60+rank_sparse)
4. Return Documents with combined scores
```

## Configuration and Dependencies

### pyproject.toml Additions
```toml
dependencies = [
    # Existing...
    "neo4j>=5.0.0",         # Neo4j Python driver
    "spacy>=3.8.0",         # NER for entity extraction
]
```

### New config files
- `config/neo4j.yaml` - Neo4j connection settings
- `config/entities.yaml` - Entity extraction settings

### config/neo4j.yaml
```yaml
neo4j:
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "password"
  database: "rag"
vector_index:
  name: "document_embeddings"
  dimension: 1024  # Based on bge-m3 model
fulltext_index:
  name: "document_fulltext"
```

### config/entities.yaml
```yaml
extraction:
  model: "en_core_web_sm"
  entity_types: ["PERSON", "ORG", "GPE"]
  min_confidence: 0.7
```

### techniques.yaml extensions
```yaml
techniques:
  hyde:
    class: techniques.hyde.HyDETechnique
    enabled: true
    config:
      embedding_model: "bge-m3:latest"
      generation_model: "glm-4.7:cloud"
      top_k: 5

  multi_query:
    class: techniques.multi_query.MultiQueryTechnique
    enabled: true
    config:
      embedding_model: "bge-m3:latest"
      generation_model: "glm-4.7:cloud"
      num_queries: 3
      top_k: 5

  hybrid:
    class: techniques.hybrid.HybridTechnique
    enabled: true
    config:
      embedding_model: "bge-m3:latest"
      top_k: 10
      rrf_k: 60  # Constant for RRF formula
```

## Testing Strategy

### Test Files to Create
- `tests/test_neo4j_store.py` - Neo4j CRUD, vector search, fulltext search, RRF
- `tests/test_entity_extractor.py` - NER extraction, edge cases
- `tests/test_hyde_technique.py` - Hypothetical generation, embedding, retrieval
- `tests/test_multi_query_technique.py` - Query generation, deduplication, ranking
- `tests/test_hybrid_technique.py` - Dense + sparse merge, RRF calculation

### Neo4j Test Setup
```python
# tests/conftest.py - Add Neo4j fixture
@pytest.fixture(scope="session")
def neo4j_driver():
    # Use test database, clean on setup
    # Return driver instance
```

### Test Coverage Targets
- **Neo4jStore**: ~15 tests (CRUD, search types, RRF, entity relationships)
- **EntityExtractor**: ~8 tests (basic extraction, empty input, entity types)
- **HyDETechnique**: ~6 tests (hypothetical generation, retrieval flow)
- **MultiQueryTechnique**: ~8 tests (query generation, dedup, ranking)
- **HybridTechnique**: ~6 tests (dense/sparse paths, RRF formula)

**Expected Phase 4 Test Count**: ~43 new tests + 54 existing = **~97 total**

### Integration Test
- `tests/test_phase4_integration.py` - End-to-end: ingest document with NER, retrieve with all three techniques

## Error Handling and Edge Cases

### Neo4j Store
- Connection failures → Retry with exponential backoff
- Index not found → Auto-create indexes on first use
- Duplicate documents → Use MERGE (upsert) to avoid duplicates
- Empty search results → Return empty list, log warning

### Entity Extraction
- spaCy model not downloaded → Auto-download on first use
- No entities found → Proceed without entities, log info
- Malformed text → Skip chunk, continue with others

### HyDE Technique
- LLM generation failure → Fall back to original query embedding
- Empty hypothetical response → Use original query
- Timeout after 30s → Fallback to naive retrieval

### Multi-Query Technique
- Fewer queries generated than requested → Use what was generated
- No unique results after deduplication → Return what's available
- LLM API failure → Fall back to single query (naive path)

### Hybrid Technique
- Vector search fails → Use full-text only
- Full-text search fails → Use vector only
- Both fail → Return empty list, log error

## Migration Path from ChromaDB to Neo4j

1. **Phase 4a**: Add Neo4jStore alongside existing VectorStore
2. **Phase 4b**: Update techniques to use Neo4jStore (config-driven)
3. **Phase 4c**: Keep ChromaDB for backwards compatibility (optional migration script)
4. **Phase 5**: Deprecate ChromaDB path

## Summary

### New Files
- `stores/neo4j_store.py`
- `utils/entity_extractor.py`
- `techniques/hyde.py`
- `techniques/multi_query.py`
- `techniques/hybrid.py`
- `config/neo4j.yaml`
- `config/entities.yaml`

### Modified Files
- `pyproject.toml` (neo4j, spacy)
- `config/techniques.yaml` (3 new techniques)
- `techniques/__init__.py`

### New Tests
~43 tests across 6 test files

### Total Tests Phase 4
~97 tests (54 existing + 43 new)