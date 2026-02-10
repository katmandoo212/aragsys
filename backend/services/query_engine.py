"""Query orchestration service."""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional, List, AsyncIterator
from registry.technique_registry import TechniqueRegistry
from ollama.client import OllamaClient
from utils.answer import Answer
from utils.document import Document


@dataclass
class QueryContext:
    """Context for query execution."""

    query_id: str
    query: str
    pipeline: str
    start_time: float = field(default_factory=lambda: datetime.now(UTC).timestamp())
    answer: Optional[Answer] = None
    retrieved_docs: List[Document] = field(default_factory=list)
    error: Optional[str] = None


class QueryEngine:
    """Service for orchestrating RAG queries."""

    def __init__(self, config_path: str = "config/techniques.yaml"):
        self.registry = TechniqueRegistry(config_path)
        self.ollama_client = OllamaClient.from_config("config/models.yaml")

    async def execute_query(
        self,
        query: str,
        pipeline: str = "naive_flow",
        max_context_docs: int = 5
    ) -> AsyncIterator[dict]:
        """Execute query and yield progress events."""
        from backend.models.query import ProgressEvent

        query_id = f"query_{datetime.now(UTC).timestamp()}"
        context = QueryContext(query_id=query_id, query=query, pipeline=pipeline)

        try:
            # Step 1: Embed query
            yield ProgressEvent(
                status="embedding_query",
                progress=10,
                message="Embedding query..."
            ).model_dump()

            # For demo, skip actual embedding and use mock
            # In real implementation: query_vector = self.ollama_client.embed(query)

            # Step 2: Retrieve documents
            yield ProgressEvent(
                status="retrieving",
                progress=30,
                message="Retrieving documents..."
            ).model_dump()

            # Mock retrieval for demo
            context.retrieved_docs = [
                Document(
                    content="This is a sample document chunk about clinical trials.",
                    metadata={"source": "sample1.pdf", "page": 1},
                    score=0.85
                ),
                Document(
                    content="Medical guidelines recommend certain treatments.",
                    metadata={"source": "sample2.md"},
                    score=0.78
                )
            ]

            # Step 3: Generate answer
            yield ProgressEvent(
                status="generating",
                progress=70,
                message="Generating answer..."
            ).model_dump()

            # For demo, use a mock response
            # In real implementation: use generation technique from pipeline
            context.answer = Answer(
                content=f"Based on the retrieved documents, I can provide information about '{query}'. The clinical trial data shows promising results.",
                metadata={"model": "glm-4.7:cloud", "pipeline": pipeline},
                citations=["sample1.pdf", "sample2.md"]
            )

            # Complete
            response_time = int((datetime.now(UTC).timestamp() - context.start_time) * 1000)

            yield ProgressEvent(
                status="complete",
                progress=100,
                message="Complete",
                data={
                    "answer_id": query_id,
                    "query": context.query,
                    "content": context.answer.content,
                    "citations": context.answer.citations,
                    "metadata": context.answer.metadata,
                    "retrieved_docs": len(context.retrieved_docs),
                    "response_time_ms": response_time,
                    "timestamp": datetime.now(UTC).isoformat()
                }
            ).model_dump()

        except Exception as e:
            yield ProgressEvent(
                status="error",
                progress=0,
                message="Error occurred",
                data={"error": str(e)}
            ).model_dump()