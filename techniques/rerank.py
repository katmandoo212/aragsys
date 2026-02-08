from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ollama.client import OllamaClient

@dataclass
class RerankTechnique:
    config: dict
    ollama_client: "OllamaClient"
    base_technique: Optional[object] = None
    scoring_model: str = field(init=False)
    top_k: int = field(init=False)
    score_threshold: float = field(init=False)

    def __post_init__(self):
        self.scoring_model = self.config.get("scoring_model", "bge-reranker-v2:latest")
        self.top_k = self.config.get("top_k", 5)
        self.score_threshold = self.config.get("score_threshold", 0.5)

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve and rerank results using cross-encoder scoring."""
        # Get initial results from base technique
        if self.base_technique:
            results = self.base_technique.retrieve(query)
        else:
            return []

        if not results:
            return []

        # Score each document
        scored_results = []
        for result in results:
            score = self._score_document(query, result["content"])
            if score >= self.score_threshold:
                result["relevance_score"] = score
                scored_results.append(result)

        # Sort by score and keep top_k
        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_results[:self.top_k]

    def _score_document(self, query: str, document: str) -> float:
        """Score document relevance using cross-encoder prompt."""
        prompt = (
            f"Rate the relevance of this document to the query on a scale of 0.0 to 1.0.\n"
            f"Query: {query}\n"
            f"Document: {document}\n"
            f"Relevance score:"
        )
        response = self.ollama_client.generate(prompt, self.scoring_model)
        try:
            return float(response.strip())
        except ValueError:
            return 0.0