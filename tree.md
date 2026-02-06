```
scientific-rag/
├── config/
│   ├── techniques.yaml           # RAG technique registry
│   ├── vector_stores.yaml         # Vector store configurations
│   ├── models.yaml                # LLM and embedding model configs
│   ├── pipelines.yaml             # Pipeline composition definitions
│   └── evaluation.yaml            # Evaluation metric configurations
├── docs/
│   └── design.md                  # High-level PocketFlow design doc
├── techniques/                    # Standalone RAG technique implementations
│   ├── __init__.py
│   ├── base.py                    # Abstract base interface
│   ├── naive_rag.py              # Phase 1: Baseline retrieval
│   ├── hyde.py                    # Phase 2: Hypothetical Document Embeddings
│   ├── multi_query.py            # Phase 2: Query decomposition
│   ├── hybrid_search.py          # Phase 2: Vector + BM25 fusion
│   ├── reranker.py               # Phase 3: Cross-encoder reranking
│   ├── contextual_compression.py # Phase 3: Context pruning
│   └── graph_traversal.py        # Phase 4: Knowledge graph exploration
├── stores/                        # Storage adapters (Dependency Injection)
│   ├── __init__.py
│   ├── vector_store.py           # Abstract vector store interface
│   ├── chroma_adapter.py         # ChromaDB implementation
│   ├── qdrant_adapter.py         # Qdrant implementation
│   └── neo4j_adapter.py          # Neo4j graph store
├── registry/
│   ├── __init__.py
│   └── technique_registry.py     # Registry for dynamic technique loading
├── pipeline/
│   ├── __init__.py
│   ├── builder.py                # Builder pattern for pipeline construction
│   └── orchestrator.py           # PocketFlow-based pipeline orchestration
├── nodes/                         # PocketFlow nodes
│   ├── __init__.py
│   ├── retrieval_node.py         # Handles retrieval stage
│   ├── rerank_node.py            # Handles reranking stage
│   ├── generation_node.py        # Handles LLM generation
│   ├── critique_node.py          # Self-correction validation
│   └── graph_query_node.py       # Graph traversal for multi-hop
├── flows/                         # PocketFlow flow definitions
│   ├── __init__.py
│   ├── naive_flow.py             # Phase 1: Simple linear flow
│   ├── advanced_flow.py          # Phases 2-3: Hybrid retrieval flow
│   ├── agentic_flow.py           # Phase 5: Self-correcting loop
│   └── flow_factory.py           # Factory for creating flows
├── utils/
│   ├── __init__.py
│   ├── call_llm.py               # LLM wrapper utility
│   ├── embeddings.py             # Embedding generation
│   ├── chunking.py               # Document chunking strategies
│   └── config_loader.py          # YAML configuration loader
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                # Strategy pattern for metrics
│   ├── faithfulness.py           # Hallucination detection
│   ├── contextual_recall.py      # Retrieval coverage
│   └── noise_robustness.py       # Irrelevant chunk filtering
├── benchmarks/
│   └── scientific_qa.json        # Evaluation dataset
├── tests/
│   ├── test_techniques/
│   ├── test_nodes/
│   ├── test_flows/
│   └── test_registry/
├── main.py                        # Entry point
├── requirements.txt
└── README.md
```