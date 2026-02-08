from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
import re

if TYPE_CHECKING:
    from ollama.client import OllamaClient

@dataclass
class CompressTechnique:
    config: dict
    ollama_client: "OllamaClient"
    base_technique: Optional[object] = None
    use_llm_refinement: bool = field(init=False)
    min_keyword_matches: int = field(init=False)
    top_k_segments: int = field(init=False)
    refinement_model: str = field(init=False)

    def __post_init__(self):
        self.use_llm_refinement = self.config.get("use_llm_refinement", True)
        self.min_keyword_matches = self.config.get("min_keyword_matches", 1)
        self.top_k_segments = self.config.get("top_k_segments", 3)
        self.refinement_model = self.config.get("refinement_model", "glm-4.7:cloud")

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve and compress documents to relevant segments."""
        # Get initial results from base technique
        if self.base_technique:
            results = self.base_technique.retrieve(query)
        else:
            return []

        if not results:
            return []

        compressed_results = []
        for result in results:
            compressed = self._compress_document(query, result["content"])
            if compressed:
                result["content"] = compressed
                compressed_results.append(result)

        return compressed_results

    def _compress_document(self, query: str, content: str) -> str:
        """Compress document using keyword extraction and optional LLM refinement."""
        # Extract query terms (simple tokenization, lowercase)
        query_terms = set(re.findall(r'\w+', query.lower()))

        # Split into segments (simple sentence-based)
        segments = re.split(r'[.!?]+', content)
        segments = [s.strip() for s in segments if s.strip()]

        # Score segments by keyword overlap
        scored_segments = []
        for segment in segments:
            segment_terms = set(re.findall(r'\w+', segment.lower()))
            overlap = len(query_terms & segment_terms)

            if overlap >= self.min_keyword_matches:
                scored_segments.append((segment, overlap))

        # Sort by overlap and keep top_k
        scored_segments.sort(key=lambda x: x[1], reverse=True)
        top_segments = [s[0] for s in scored_segments[:self.top_k_segments]]

        if not top_segments:
            return content

        # Optional LLM refinement
        if self.use_llm_refinement:
            return self._llm_refine(query, top_segments)

        # Join segments without LLM
        return ". ".join(top_segments)

    def _llm_refine(self, query: str, segments: list[str]) -> str:
        """Refine segments using LLM to keep only relevant content."""
        segments_text = "\n".join(f"- {s}" for s in segments)
        prompt = (
            f"From these segments, extract only content directly relevant to the query.\n"
            f"Query: {query}\n"
            f"Segments:\n{segments_text}\n"
            f"Relevant content:"
        )
        return self.ollama_client.generate(prompt, self.refinement_model)