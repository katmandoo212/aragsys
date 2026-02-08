# Phase 5: Precision (Reranking, Contextual Compression) Design

**Date:** 2026-02-07

## Overview

Phase 5 adds precision improvements through two new techniques: **RerankTechnique** and **CompressTechnique**. Both follow the existing technique pattern (retrieve method, config-driven) and can be composed with Phase 4 retrieval techniques.

**RerankTechnique** takes initial retrieval results (from any technique) and re-scores them using a cross-encoder model via Ollama. The cross-encoder evaluates query-document relevance more accurately than bi-encoder similarity scores.

**CompressTechnique** reduces token usage by filtering retrieved documents to only the most relevant segments. It uses a hybrid approach: first, keyword-based filtering extracts segments containing query terms; then optionally, an LLM refines to keep only semantically relevant content.

Both techniques work in a chain: `Retrieve → Rerank → Compress → Generate`. They're registered in `techniques.yaml` and can be used independently or together in pipelines.

## Architecture

### Components

1. **`techniques/rerank.py`** - RerankTechnique implementation
2. **`techniques/compress.py`** - CompressTechnique implementation
3. **`config/precision.yaml`** - Shared precision settings
4. **`config/techniques.yaml`** - Updated with new technique entries

### Integration Pattern

Both techniques follow the existing pattern:
- `retrieve(query: str) -> list[dict]` method
- Configuration-driven via YAML
- Can be composed in pipelines
- Optional base technique dependency (can accept pre-retrieved results)

## RerankTechnique

### Dependencies
- `ollama_client: OllamaClient` - for running the scoring model
- `base_technique: BaseTechnique` - optional, underlying retrieval technique

### Configuration
```yaml
config:
  scoring_model: "bge-reranker-v2:latest"
  top_k: 5
  score_threshold: 0.5
```

### Implementation Approach
1. `retrieve(query)` gets initial results from base technique (or accepts pre-retrieved)
2. For each query-document pair, call LLM with scoring prompt
3. Prompt asks model to rate relevance on 0-1 scale
4. Sort by relevance score, keep top_k
5. Return re-ranked results with new scores

### Scoring Prompt
```
Rate the relevance of this document to the query on a scale of 0.0 to 1.0.
Query: {query}
Document: {document_content}
Relevance score:
```

### Return Format
```python
[
    {
        "content": "...",
        "metadata": {...},
        "relevance_score": 0.85
    },
    ...
]
```

## CompressTechnique

### Dependencies
- `ollama_client: OllamaClient` - for optional LLM refinement
- `base_technique: BaseTechnique` - optional, for getting initial results

### Configuration
```yaml
config:
  use_llm_refinement: true
  min_keyword_matches: 1
  segment_length: 200
  top_k_segments: 3
  refinement_model: "glm-4.7:cloud"
```

### Implementation Approach (Two-Stage)

**Stage 1: Keyword Extraction**
1. Extract query terms (remove stop words, get unique words)
2. Split document into segments (sentences or fixed-length chunks)
3. Score segments by keyword overlap
4. Keep segments meeting `min_keyword_matches` threshold

**Stage 2: Optional LLM Refinement**
1. Pass filtered segments to LLM with extraction prompt
2. Prompt asks to keep only query-relevant content
3. Return refined segments

### Extraction Prompt
```
From these segments, extract only content directly relevant to the query.
Query: {query}
Segments:
{segments}
Relevant content:
```

### Return Format
```python
[
    {
        "content": "...",  # compressed content
        "metadata": {...},
        "relevance_score": 0.75
    },
    ...
]
```

## Configuration Files

### `config/precision.yaml`
```yaml
precision:
  keyword_extraction:
    stop_words: ["the", "a", "an", "is", "are", "was", "were"]
    segment_length: 200
    min_keyword_matches: 1
  llm_refinement:
    enabled: true
    model: "glm-4.7:cloud"
    max_segments: 3
```

### `config/techniques.yaml` (Additions)
```yaml
rerank:
  class: techniques.rerank.RerankTechnique
  enabled: true
  config:
    scoring_model: "bge-reranker-v2:latest"
    top_k: 5
    score_threshold: 0.5

compress:
  class: techniques.compress.CompressTechnique
  enabled: true
  config:
    use_llm_refinement: true
    top_k_segments: 3
```

## Testing Strategy

### RerankTechnique Tests (4 tests)
1. Basic reranking with mock LLM responses
2. Score threshold filtering
3. Empty results handling
4. Configuration loading

### CompressTechnique Tests (5 tests)
1. Keyword extraction and filtering
2. LLM refinement (with mock)
3. Fallback without LLM refinement
4. Document with no matches
5. Configuration options

### Integration
- Update `techniques/__init__.py` to export new techniques
- Total new tests: 9
- Expected final test count: 74 + 9 = 83

## Dependencies

No new Python package dependencies required. Uses existing:
- `ollama` (via OllamaClient)
- YAML config loading

## Next Steps

1. Create detailed implementation plan with TDD
2. Set up isolated worktree
3. Execute implementation