# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
# Project Context

For additional context about [specific topic], see @.claude/llm-context.md

## Project State

This repository is in the **design/planning phase**. No implementation code exists yet. This repository contains design documentation for a planned "Scientific Agentic RAG Framework" that will implement multi-hop reasoning over academic and clinical literature.

## Planned Architecture

The project aims to implement an iterative RAG system using:
- **Python 3.13+**
- **PocketFlow** for node-based workflow orchestration
- **Vector Stores:** ChromaDB or Qdrant
- **Graph Store:** Neo4j for entity relationships
- **Hybrid Retrieval:** Dense embeddings + BM25 (Reciprocal Rank Fusion)
- **Reranking:** Cross-encoder models
- **Evaluation:** RAGAS or DeepEval

### Core Design Patterns
- **Registry Pattern:** Dynamic RAG technique loading from YAML configuration
- **Strategy Pattern:** Swappable retrieval strategies
- **Chain of Responsibility:** Retrieve → Rerank → Compress → Generate
- **Decorator Pattern:** Wrapping retrievers with enhancements
- **SOLID Principles:** SRP, DI, and duck typing prioritized

## Implementation Phases

Implementation follows a strict 5-phase roadmap:

1. **Phase 1 (Naive RAG):** Baseline with dense retrieval
2. **Phase 2 (Advanced Retrieval):** Hybrid search + HyDE + Multi-Query
3. **Phase 3 (Precision):** Reranking + Contextual Compression
4. **Phase 4 (GraphRAG):** Multi-hop reasoning with Neo4j
5. **Phase 5 (Agentic):** Self-correction loop with iterative refinement

## Key Files to Create

When implementing, create these directories and files:

```
scientific-rag/
├── config/
│   ├── techniques.yaml           # RAG technique registry
│   ├── pipelines.yaml            # Pipeline compositions
│   ├── models.yaml               # LLM/embedding configs
│   ├── vector_stores.yaml        # Vector store configs
│   └── evaluation.yaml           # Metric configs
├── techniques/                   # RAG technique implementations
│   ├── base.py                   # BaseTechnique protocol
│   ├── naive_rag.py
│   ├── hyde.py
│   ├── multi_query.py
│   ├── hybrid_search.py
│   ├── reranker.py
│   ├── contextual_compression.py
│   └── graph_traversal.py
├── stores/                       # Storage adapters
│   ├── vector_store.py
│   ├── chroma_adapter.py
│   ├── qdrant_adapter.py
│   └── neo4j_adapter.py
├── registry/
│   └── technique_registry.py     # Dynamic YAML loading
├── nodes/                        # PocketFlow nodes
│   ├── retrieval_node.py
│   ├── generation_node.py
│   ├── critique_node.py
│   └── graph_query_node.py
├── flows/                        # PocketFlow flows
│   ├── naive_flow.py
│   ├── advanced_flow.py
│   └── agentic_flow.py
├── pipeline/
│   ├── builder.py                # Builder pattern
│   └── orchestrator.py
├── utils/
│   ├── call_llm.py
│   ├── embeddings.py
│   ├── chunking.py
│   └── config_loader.py
├── evaluation/
│   ├── metrics.py
│   ├── faithfulness.py
│   ├── contextual_recall.py
│   └── noise_robustness.py
├── main.py
├── requirements.txt
└── README.md
```

## Commands (Once Implemented)

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python main.py --config config/pipelines.yaml --query "your query"

# Run tests
pytest tests/

# Evaluate
python -m evaluation.evaluate --dataset benchmarks/scientific_qa.json --pipeline agentic

# Visualize flow
python -m utils.visualize_flow --flow agentic
```

## PocketFlow Node Pattern

All nodes follow the prep/exec/post pattern:

```python
class RetrievalNode(Node):
    def prep(self, shared):
        return shared["query"], shared.get("iteration", 0)

    def exec(self, prep_result):
        query, iteration = prep_result
        return retrieve_with_techniques(query)

    def post(self, shared, prep_res, exec_res):
        shared["retrieved_docs"] = exec_res
        return "critique"  # Next node
```

## Coding Standards

- **No try-except in utility functions** - let PocketFlow's retry handle errors
- **Duck typing interface** - all techniques implement BaseTechnique protocol
- **Dependency injection** - all dependencies injected via YAML config
- **SRP** - each technique has one clear purpose

## Technique Registry Pattern

RAG techniques are loaded dynamically from YAML:

```python
# config/techniques.yaml
techniques:
  naive_rag:
    class: techniques.naive_rag.NaiveRAG
    enabled: true
    config:
      chunk_size: 500
      top_k: 5
```

The registry dynamically imports and instantiates classes at runtime.

## Additional Resources

- PocketFlow: https://the-pocket.github.io/PocketFlow/
- SOLID Principles: https://en.wikipedia.org/wiki/SOLID